# 智慧工地安全监控系统（SmartBuildings）

一套面向施工现场的安全监控综合平台，集成了 **物联设备数据采集**、**3D 数字孪生可视化**、**AI 视频危险源识别** 与 **塔吊安全智能预测**。

## 系统架构

```
┌─────────────┐   TCP(8888)   ┌──────────────────────┐   HTTP    ┌─────────────────────┐
│ 工地物联设备  │ ────────────> │  Spring Boot 后端服务  │ ────────> │  Python AI 服务(5000) │
│ 塔吊/环境/喷淋 │               │  Netty TCP + REST API │ <──────── │  图像识别 + 塔吊预测   │
└─────────────┘               └──────────────────────┘           └─────────────────────┘
                                      │    │                            ▲
                                MySQL    Redis                    HLS 视频流抓帧
                                      │    │                            │
                               ┌──────┴────┴──────┐            ┌────────┴────────┐
                               │  Vue3 前端管理端   │            │  摄像头(WVP/HLS) │
                               │  Three.js 3D 大屏 │            └─────────────────┘
                               └──────────────────┘
```

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 后端服务 | Java 17、Spring Boot 3.3.3、MyBatis-Plus、Netty（TCP 数据接入）、Redis、MySQL、SpringDoc OpenAPI |
| 前端管理端（ConsSafetySystem） | Vue 3、Vite、Element Plus、ECharts、Vuex、Vue Router、TailwindCSS |
| 3D 可视化（tcpClient） | Node.js、Three.js（GLB 场景模型、轨道控制器） |
| **AI 识别（AIRecognize）** | Python、Flask、TensorFlow/Keras、ModelScope(DAMO-YOLO)、Ultralytics YOLOv8、scikit-learn、pandas |

## 目录结构

```
smartbuildings/
├── AIRecognize/            # ★ AI 识别服务（本仓库重点）
│   ├── main.py             # Flask 服务入口：图像危险源检测 + 塔吊安全预测
│   ├── tower_crane_train.py# 塔吊安全预测模型训练脚本 (CNN+BiLSTM+Attention)
│   ├── test.py             # ModelScope 安全帽模型快速验证脚本
│   ├── enhanced_crane_model_v4.h5   # 训练好的塔吊预测模型
│   ├── scaler_params.npy   # 标准化参数（训练时保存）
│   ├── label_classes.npy   # 标签类别映射（安全/预警/危险）
│   ├── tower_crane_data.csv# 塔吊运行历史数据（训练集）
│   ├── model/last.pt       # YOLOv8 安全衣检测权重
│   └── requirements.txt    # Python 依赖
├── src/                    # Spring Boot 后端源码
├── ConsSafetySystem/       # Vue3 前端管理端
├── tcpClient/              # Three.js 3D 可视化客户端 + 模拟数据生成器
└── pom.xml                 # Maven 配置
```

---

# ★ AI 识别模块（AIRecognize）

AI 模块是一个基于 **Flask 的独立微服务**（默认端口 `5000`），为整个平台提供两类智能能力：

1. **多危险源图像识别** —— 对摄像头抓拍画面进行目标检测，识别未戴安全帽、未穿安全衣、吸烟、明火四类安全隐患；
2. **塔吊安全状态预测** —— 基于塔吊最近 20 条运行数据的时序特征，用深度学习模型预测当前作业处于「安全 / 预警 / 危险」哪种状态。

## 1. 多危险源图像识别

### 1.1 模型组成

| 检测项 | type | 模型 | 来源 |
| --- | --- | --- | --- |
| 未戴安全帽 | 1 | DAMO-YOLO（TinyNAS） | ModelScope `iic/cv_tinynas_object-detection_damoyolo_safety-helmet` |
| 未穿安全衣 | 2 | YOLOv8 自训练权重 | 本地 `model/last.pt`，识别 `nor_cloths`（未穿反光衣）类别 |
| 吸烟行为 | 3 | DAMO-YOLO（TinyNAS） | ModelScope `iic/cv_tinynas_object-detection_damoyolo_cigarette` |
| 明火识别 | 6 | DAMO-YOLO（TinyNAS） | ModelScope `iic/cv_tinynas_object-detection_damoyolo_smokefire` |

所有模型在服务启动时**一次性加载到内存**，避免每次请求重复加载。四个检测项通过 `type` 参数自由组合，一次请求可同时执行多种检测（例如同时查未戴安全帽 + 未穿安全衣）。

### 1.2 检测接口 `POST /detect`

- 请求格式：`multipart/form-data`
- 参数：
  - `image`：图片文件（必填）
  - `type`：检测类型，可重复传多个（1/2/3/6，必填）

请求示例：

```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@site_photo.jpg" \
  -F "type=1" -F "type=2"
```

响应示例（数组，每个元素对应一种检测类型）：

```json
[
  {
    "type": 1,
    "detection_name": "未戴安全帽",
    "total_objects": 5,
    "valid_count": 1,
    "boxes": [
      { "x1": 120.5, "y1": 88.2, "x2": 210.3, "y2": 190.7, "confidence": 0.87 }
    ]
  },
  {
    "type": 2,
    "detection_name": "未穿安全衣",
    "total_objects": 3,
    "valid_count": 0,
    "boxes": []
  }
]
```

- `total_objects`：该模型检测出的全部目标数；
- `valid_count` / `boxes`：置信度 ≥ 0.3 且命中**违规类别**（如 `no safety hat`、`nor_cloths`、`cigarette`、`fire`）的目标，即有效告警框。

### 1.3 阈值配置

检测阈值统一在 `main.py` 的 `DETECTION_CONFIG` 中维护，当前全部为 `0.3`，可按现场误报情况调整。

## 2. 塔吊安全状态预测（重点）

### 2.1 问题定义

塔吊在吊装作业中，超载（重量/力矩）、大风、倾斜等因素会引发安全事故。本模块将塔吊安全评估建模为**时序分类问题**：

- **输入**：最近 20 个采样时刻的 7 维运行特征
  `weight（吊重）、amplitude（幅度）、obliguity（倾斜度）、windSpeed（风速）、moment（力矩）、height（高度）、rotation（回转角）`
- **输出**：三分类 —— `安全` / `预警` / `危险`，以及对应的置信度概率分布

### 2.2 模型结构（CNN + BiLSTM + Attention）

由 `tower_crane_train.py` 构建，核心结构：

```
Input(20, 7)
 → Masking → Conv1D(64, k=3) → BN → Conv1D(128, k=5) → MaxPool   # 1D 卷积提取局部形态
 → Bidirectional LSTM(192, return_sequences=True)                # 双向 LSTM 捕获长期时序依赖
 → Attention(query/value) + Concatenate                          # 注意力聚焦关键时刻（如超载瞬间）
 → Bidirectional LSTM(96, dropout=0.2) → BN
 → Dense(128, L2) → Dropout(0.4) → Dense(3, softmax)
```

针对**安全样本远多于危险样本**的类别不平衡问题，训练时采用三重策略：

| 策略 | 说明 |
| --- | --- |
| 加权 Focal Loss | 自定义损失 `weighted_focal_loss`，类别权重：安全 0.1 / 预警 0.4 / 危险 0.5，`γ=2` 抑制易分样本 |
| 固定类别权重 | `class_weight = {安全:1.0, 预警:4.0, 危险:8.0}`，让危险类获得 8 倍关注度 |
| 噪声数据增强 | 危险样本加更强高斯噪声（σ≥0.05），其余类别随机 50% 概率加噪（σ=0.03），提升泛化能力 |

另实现了 `DangerFocusCallback`：每个 epoch 结束监控**危险类召回率**，低于 0.2 时自动将学习率下调 30%，防止模型"牺牲危险类换整体准确率"。

训练产物（随仓库提供，可直接用于推理）：

- `enhanced_crane_model_v4.h5` — 模型权重
- `scaler_params.npy` — StandardScaler 的均值/标准差（推理时必须用同一套参数）
- `label_classes.npy` — 标签编码映射

### 2.3 预测接口 `POST /crane_safety`

- 请求格式：`application/json`
- 参数：`crane_data`，**恰好 20 条**塔吊运行数据组成的数组，每条包含上述 7 个英文字段

请求示例：

```bash
curl -X POST http://localhost:5000/crane_safety \
  -H "Content-Type: application/json" \
  -d '{"crane_data": [ {"weight":1.2,"amplitude":30.5,"obliguity":0.4,"windSpeed":3.1,"moment":45.2,"height":25.0,"rotation":120.0}, "... 共20条 ..." ]}'
```

响应示例：

```json
{
  "status": "success",
  "request_info": { "safety_count": 8, "warning_count": 2, "danger_count": 0 },
  "last_10_safety_results": [
    {
      "data_index": 11,
      "safety_status": "安全",
      "confidence": 0.9312,
      "prob_distribution": { "安全": 0.9312, "预警": 0.0512, "危险": 0.0176 }
    }
  ]
}
```

- 服务端会校验数据条数与特征列完整性，不符合直接返回 400；
- `request_info` 汇总最近 10 个时刻的状态统计，`last_10_safety_results` 为逐时刻明细，方便前端绘制趋势。

### 2.4 健康检查 `GET /health`

```json
{ "status": "ok", "service": "AI危险源识别 + 塔吊预测服务" }
```

## 3. 与后端服务的联动

AI 服务并非孤立运行，Spring Boot 后端在两个方向上消费它：

1. **视频 AI 巡检**（[AICameraService.java](src/main/java/com/qst/smartbuildings/service/AICameraService.java)）
   - 服务启动时拉取所有启用了 AI 监控的摄像头，通过 JavaCV/FFmpeg 抓取 HLS 视频流；
   - **每 5 秒**抓一帧，编码为 JPEG 后按摄像头配置的 `ai_monitor_type` 组合 `type` 参数，调用 `/detect`；
   - 对检测结果做**跨帧去重**（同一摄像头同类告警数量未上升不重复告警），有效告警写入 AI 告警表供前端展示；
   - 抓流失败的摄像头放入失败队列，**每 30 秒**自动重试恢复。

2. **塔吊实时预测**（塔吊 TCP 数据处理器）
   - Netty TCP 服务（端口 8888）接收塔吊上报的实时运行数据，累积 20 条后调用 `/crane_safety`，预测结果参与告警判定。

相关配置位于 `src/main/resources/application.yml`：

```yaml
camera:
  ai:
    baseUrl: http://localhost:5000/detect
tower:
  ai:
    baseUrl: http://localhost:5000/crane_safety
```

## 4. 环境准备与运行

### 4.1 安装依赖

> 建议 Python 3.8~3.10 + 独立虚拟环境（ requirements 中 TensorFlow 2.10 / torch 2.0.1 / mmcv-full 对版本敏感，需与 Python 版本匹配）。

```bash
cd AIRecognize
python -m venv myenv
myenv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 4.2 准备模型文件

| 文件 | 获取方式 |
| --- | --- |
| ModelScope 三个检测模型 | 首次运行自动从 ModelScope 下载（需外网，首次较慢） |
| `model/last.pt` | YOLOv8 安全衣自训练权重，需提前放入 `model/` 目录 |
| `enhanced_crane_model_v4.h5` 等 | 随仓库提供；如需重训见下节 |

### 4.3 启动服务

```bash
python main.py
# 服务监听 0.0.0.0:5000，启动日志会依次打印各模型加载完成信息
```

### 4.4 重新训练塔吊预测模型（可选）

将新的塔吊运行数据 CSV（需含 7 个特征列与「安全状态」中文标签列）放到 `AIRecognize` 目录下，执行：

```bash
python tower_crane_train.py
```

脚本按时间顺序 8:2 划分训练/验证集，训练 30 个 epoch（早停耐心值 15），结束后重新生成 `.h5` 模型与 scaler/label 文件。随机种子固定为 42，结果可复现。

## 5. 注意事项

- ModelScope 模型加载需要 `trust_remote_code=True`，请确保部署环境可访问 ModelScope 仓库；
- `torch` 固定使用 2.0.1，是为了规避 2.6+ 版本对权重加载的安全限制（YOLO 权重反序列化）；
- 首次启动需要下载/加载 5 个模型，耗时较长，属正常现象；
- AI 服务与后端解耦，可独立部署在不同机器，只需修改 `application.yml` 中的两个 `baseUrl`。
