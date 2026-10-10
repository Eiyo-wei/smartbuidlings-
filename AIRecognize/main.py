# ============================================================
# 文件名: main.py
# 功能: 多危险源图片AI识别服务 + 塔吊安全预测 (Flask API)
# 运行: python main.py (需在 myenv 环境下)
# 端口: 5000
# ============================================================

from flask import Flask, request, jsonify
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ═══ 新增：塔吊预测依赖 ═══
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------- 1. 初始化Flask应用 ----------
app = Flask(__name__)

# ---------- 2. 加载全局模型 (服务启动时只加载一次) ----------
print("=" * 50)
print("正在加载AI模型，请稍候...")

# 2.1 安全帽模型 (ModelScope)
helmet_model = pipeline(
    Tasks.domain_specific_object_detection,
    model='iic/cv_tinynas_object-detection_damoyolo_safety-helmet',
    trust_remote_code=True
)
print("✓ 安全帽模型加载完成")

# 2.2 吸烟模型 (ModelScope)
smoking_model = pipeline(
    Tasks.domain_specific_object_detection,
    model='iic/cv_tinynas_object-detection_damoyolo_cigarette',
    trust_remote_code=True
)
print("✓ 吸烟模型加载完成")

# 2.3 明火模型 (ModelScope)
fire_model = pipeline(
    Tasks.domain_specific_object_detection,
    model='iic/cv_tinynas_object-detection_damoyolo_smokefire',
    trust_remote_code=True
)
print("✓ 明火模型加载完成")

# 2.4 安全衣模型 (YOLO) — 需要提前下载 last.pt 放到 model/ 目录
safety_vest_model = YOLO("model/last.pt")
print("✓ 安全衣模型加载完成")

# ═══ 新增：加载塔吊预测模型 ═══
# 定义损失函数（与训练时一致，用于加载模型）
def weighted_focal_loss(y_true, y_pred):
    alpha = tf.where(tf.equal(y_true, 2), 0.5,
                    tf.where(tf.equal(y_true, 1), 0.4, 0.1))
    gamma = 2.0
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    pt = tf.exp(-ce_loss)
    return tf.reduce_mean(alpha * tf.pow(1 - pt, gamma) * ce_loss)

class CraneSafetyPredictor:
    """塔吊安全状态预测器"""
    def __init__(self, model_path='enhanced_crane_model_v4.h5'):
        self.model = load_model(
            model_path,
            custom_objects={'weighted_focal_loss': weighted_focal_loss}
        )
        scaler_params = np.load('scaler_params.npy', allow_pickle=True).item()
        self.mean = scaler_params['mean']
        self.scale = scaler_params['scale']
        self.classes = np.load('label_classes.npy', allow_pickle=True)
        self.seq_length = 20
        print(f"✓ 塔吊预测模型加载完成，支持类别：{list(self.classes)}")

    def preprocess_data(self, crane_data_df):
        required_features = ['weight', 'amplitude', 'obliguity', 
                             'windSpeed', 'moment', 'height', 'rotation']
        if not all(col in crane_data_df.columns for col in required_features):
            raise ValueError(f"缺少必需特征列，需包含：{required_features}")
        scaled_data = (crane_data_df[required_features].values - self.mean) / self.scale
        return scaled_data.reshape(1, self.seq_length, len(required_features))

    def predict_last_10(self, crane_data_df):
        sequence = self.preprocess_data(crane_data_df)
        pred_probs = self.model.predict(sequence, verbose=0)[0]
        pred_class_idx = np.argmax(pred_probs)
        pred_class = self.classes[pred_class_idx]
        
        last_10_results = []
        for idx in range(10, 20):
            last_10_results.append({
                "data_index": idx + 1,
                "safety_status": pred_class,
                "confidence": round(float(pred_probs[pred_class_idx]), 4),
                "prob_distribution": {
                    "安全": round(float(pred_probs[list(self.classes).index("安全")]), 4) if "安全" in self.classes else 0.0,
                    "预警": round(float(pred_probs[list(self.classes).index("预警")]), 4) if "预警" in self.classes else 0.0,
                    "危险": round(float(pred_probs[list(self.classes).index("危险")]), 4) if "危险" in self.classes else 0.0
                }
            })
        
        status_counts = {
            "安全": sum(1 for res in last_10_results if res["safety_status"] == "安全"),
            "预警": sum(1 for res in last_10_results if res["safety_status"] == "预警"),
            "危险": sum(1 for res in last_10_results if res["safety_status"] == "危险")
        }
        return last_10_results, status_counts

crane_predictor = CraneSafetyPredictor()
print("✓ 塔吊预测器初始化完成")

print("所有模型加载完毕，服务已就绪")
print("=" * 50)


# ---------- 3. 检测配置 ----------
DETECTION_CONFIG = {
    1: {"name": "未戴安全帽", "confidence_threshold": 0.3},
    2: {"name": "未穿安全衣", "base_confidence": 0.3, "unsafe_confidence": 0.3},
    3: {"name": "吸烟检测", "confidence_threshold": 0.3},
    6: {"name": "明火检测", "confidence_threshold": 0.3},
}


def format_box(box, score):
    return {
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
        "confidence": float(score)
    }


# ---------- 4. 四个检测处理函数 ----------
def process_helmet(image):
    result = helmet_model(image)
    valid_boxes = []
    for label, box, score in zip(result['labels'], result['boxes'], result['scores']):
        if score >= DETECTION_CONFIG[1]['confidence_threshold'] and label == 'no safety hat':
            valid_boxes.append(format_box(box, score))
    return {
        "type": 1,
        "detection_name": DETECTION_CONFIG[1]["name"],
        "total_objects": len(result['scores']),
        "valid_count": len(valid_boxes),
        "boxes": valid_boxes
    }
     
def process_safety_vest(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = safety_vest_model(img, conf=DETECTION_CONFIG[2]['base_confidence'])
    valid_boxes = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            class_name = safety_vest_model.names[int(box.cls[0])]
            if class_name == "nor_cloths" and conf >= DETECTION_CONFIG[2]['unsafe_confidence']:
                valid_boxes.append(format_box(box.xyxy[0].tolist(), conf))
    return {
        "type": 2,
        "detection_name": DETECTION_CONFIG[2]["name"],
        "total_objects": sum(len(r.boxes) for r in results),
        "valid_count": len(valid_boxes),
        "boxes": valid_boxes
    }

def process_smoking(image):
    result = smoking_model(image)
    valid_boxes = []
    for label, box, score in zip(result['labels'], result['boxes'], result['scores']):
        if score >= DETECTION_CONFIG[3]['confidence_threshold'] and label in ['cigarette', 'smoking']:
            valid_boxes.append(format_box(box, score))
    return {
        "type": 3,
        "detection_name": DETECTION_CONFIG[3]["name"],
        "total_objects": len(result['scores']),
        "valid_count": len(valid_boxes),
        "boxes": valid_boxes
    }

def process_fire(image):
    result = fire_model(image)
    valid_boxes = []
    for label, box, score in zip(result['labels'], result['boxes'], result['scores']):
        if score >= DETECTION_CONFIG[6]['confidence_threshold'] and label.lower() == "fire":
            valid_boxes.append(format_box(box, score))
    return {
        "type": 6,
        "detection_name": DETECTION_CONFIG[6]["name"],
        "total_objects": len(result['scores']),
        "valid_count": len(valid_boxes),
        "boxes": valid_boxes
    }

def detect_objects(image, detection_type):
    if detection_type == 1:
        return process_helmet(image)
    elif detection_type == 2:
        return process_safety_vest(image)
    elif detection_type == 3:
        return process_smoking(image)
    elif detection_type == 6:
        return process_fire(image)
    else:
        raise ValueError(f"不支持的检测类型: {detection_type}")


# ---------- 5. API路由 ----------
@app.route('/detect', methods=['POST'])
def detect():
    try:
        detection_types = list(map(int, request.form.getlist('type')))
        image_file = request.files.get('image')
        if not image_file or image_file.filename == '':
            return jsonify({"error": "无效的图片输入，需上传图片文件"}), 400
        if not detection_types:
            return jsonify({"error": "未指定检测类型"}), 400
        image = Image.open(image_file.stream)
        results = [detect_objects(image, dt) for dt in detection_types]
        return jsonify(results)
    except ValueError as e:
        return jsonify({"error": f"参数错误: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500


# ═══ 塔吊安全预测接口 ═══
@app.route('/crane_safety', methods=['POST'])
def crane_safety_predict():
    try:
        request_data = request.get_json()
        if not request_data or "crane_data" not in request_data:
            return jsonify({
                "status": "error",
                "error": "请求格式错误，需包含'crane_data'字段（20条塔吊运行数据列表）"
            }), 400
        
        crane_data_list = request_data["crane_data"]
        if len(crane_data_list) != 20:
            return jsonify({
                "status": "error",
                "error": f"塔吊数据数量错误，需传入20条，实际传入{len(crane_data_list)}条"
            }), 400
        
        crane_data_df = pd.DataFrame(crane_data_list)
        last_10_results, status_counts = crane_predictor.predict_last_10(crane_data_df)
        
        return jsonify({
            "status": "success",
            "request_info": {
                "safety_count": status_counts["安全"],
                "warning_count": status_counts["预警"],
                "danger_count": status_counts["危险"]
            },
            "last_10_safety_results": last_10_results
        })
    except ValueError as e:
        return jsonify({"status": "error", "error": f"数据预处理失败：{str(e)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"预测失败：{str(e)}"}), 500


# ---------- 6. 健康检查 ----------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "AI危险源识别 + 塔吊预测服务"})


# ---------- 7. 启动服务 ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)