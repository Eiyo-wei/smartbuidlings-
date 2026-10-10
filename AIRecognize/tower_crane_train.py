# ============================================================
# 文件名: tower_crane_train.py
# 功能: 塔吊安全状态预测模型训练 (CNN + LSTM + Attention)
# 运行: python tower_crane_train.py (需在 myenv 环境下)
# 依赖: tower_crane_data.csv 需在同一目录下
# 输出: enhanced_crane_model_v4.h5, scaler_params.npy, label_classes.npy
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout, 
    Masking, BatchNormalization, Conv1D, 
    MaxPooling1D, Attention, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

# ---------- 1. 设置全局随机种子 (保证结果可复现) ----------
np.random.seed(42)
tf.random.set_seed(42)


# ---------- 2. 定义损失函数 (需与模型绑定) ----------
def weighted_focal_loss(y_true, y_pred):
    """
    加权 Focal Loss，专门处理类别不平衡问题
    权重分配：安全类 0.1，预警类 0.4，危险类 0.5
    """
    alpha = tf.where(tf.equal(y_true, 2), 0.5,   # 危险类
                    tf.where(tf.equal(y_true, 1), 0.4, 0.1))  # 预警类, 安全类
    gamma = 2.0
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    pt = tf.exp(-ce_loss)
    return tf.reduce_mean(alpha * tf.pow(1 - pt, gamma) * ce_loss)


# ---------- 3. 数据预处理函数 ----------
def create_enhanced_sequences(data, seq_length=20, noise_level=0.03):
    """生成带噪声增强的时序数据，增加危险样本的特征区分度"""
    # 使用英文字段名，与 CSV 数据对应
    features = data[['weight', 'amplitude', 'obliguity', 
                     'windSpeed', 'moment', 'height', 'rotation']].values
    
    # 标准化处理 (保存均值和标准差供后续预测使用)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    np.save('scaler_params.npy', {'mean': scaler.mean_, 'scale': scaler.scale_})
    
    # 生成增强序列
    X, y = [], []
    for i in range(len(scaled_features) - seq_length):
        seq = scaled_features[i:i+seq_length]
        current_label = data['安全状态编码'].values[i+seq_length]
        
        # 对危险样本增加更多噪声以提高泛化能力
        if current_label == 2:  # 危险类
            noise = np.random.normal(0, max(noise_level, 0.05), seq.shape)
            seq = np.clip(seq + noise, -3, 3)
        elif np.random.rand() < 0.5:  # 其他类别随机加噪
            noise = np.random.normal(0, noise_level, seq.shape)
            seq = np.clip(seq + noise, -3, 3)
            
        X.append(seq)
        y.append(current_label)
    
    return np.array(X), np.array(y).flatten()


# ---------- 4. 固定类别权重 ----------
def get_fixed_weights():
    """固定类别权重，确保危险类获得足够关注"""
    # 标签编码：0=安全，1=预警，2=危险
    return {0: 1.0, 1: 4.0, 2: 8.0}


# ---------- 5. 构建模型 ----------
def build_attention_model(input_shape, num_classes):
    """构建带注意力机制的 CNN + LSTM 混合模型"""
    # 输入层
    inputs = Input(shape=input_shape)
    
    # 特征提取层 (1D卷积捕获局部特征)
    x = Masking(mask_value=-100.0)(inputs)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # 双向 LSTM 层 (捕获长期时序依赖)
    lstm_out = Bidirectional(LSTM(192, return_sequences=True))(x)
    
    # 注意力机制 (关注关键时刻)
    query = Dense(96)(lstm_out)
    value = Dense(96)(lstm_out)
    attention = Attention()([query, value])
    x = Concatenate(axis=-1)([lstm_out, attention])
    
    # 分类层
    x = Bidirectional(LSTM(96, dropout=0.2, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # 定义模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 编译 (使用自定义损失函数)
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0005),
        loss=weighted_focal_loss,
        metrics=['accuracy']
    )
    return model


# ---------- 6. 自定义回调：危险类重点监控 ----------
class DangerFocusCallback(Callback):
    """当危险类召回率过低时，自动降低学习率"""
    def __init__(self, val_data, danger_class=2, min_recall=0.2):
        super().__init__()
        self.X_val, self.y_val = val_data
        self.danger_class = danger_class
        self.min_recall = min_recall
        
    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        recall = recall_score(self.y_val, y_pred, labels=[self.danger_class], average=None)[0]
        print(f" - val_danger_recall: {recall:.4f}")
        
        if recall < self.min_recall and epoch > 5:
            current_lr = self.model.optimizer.learning_rate.numpy()
            new_lr = current_lr * 0.7
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            print(f"  危险类召回率过低，调整学习率至: {new_lr:.6f}")


# ---------- 7. 主训练流程 ----------
def main():
    print("=" * 50)
    print("开始塔吊安全预测模型训练...")
    
    # 1. 加载数据
    df = pd.read_csv('tower_crane_data.csv')
    print(f"✓ 加载数据: {len(df)} 条记录")
    
    # 2. 标签编码
    le = LabelEncoder()
    df['安全状态编码'] = le.fit_transform(df['安全状态'])
    print("✓ 类别编码映射:", dict(zip(le.classes_, range(len(le.classes_)))))
    
    # 3. 生成时序序列
    X, y = create_enhanced_sequences(df, seq_length=20, noise_level=0.03)
    print(f"✓ 生成时序序列: X.shape={X.shape}, y.shape={y.shape}")
    
    # 4. 划分训练集和验证集 (按时间顺序 8:2)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"✓ 训练集: {len(X_train)} 条, 验证集: {len(X_val)} 条")
    
    # 5. 获取类别权重
    class_weights = get_fixed_weights()
    print("✓ 类别权重分配:", class_weights)
    
    # 6. 构建模型
    model = build_attention_model(
        input_shape=(20, X_train.shape[2]),
        num_classes=len(le.classes_)
    )
    model.summary()
    
    # 7. 训练模型
    print("\n开始训练 (预计 10-30 分钟)...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=256,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=15, monitor='val_danger_recall', mode='max', 
                         restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            DangerFocusCallback(val_data=(X_val, y_val), danger_class=2)
        ],
        verbose=2
    )
    
    # 8. 评估模型
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\n" + "=" * 50)
    print("验证集详细评估报告:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    
    # 9. 保存模型和辅助文件
    model.save('enhanced_crane_model_v4.h5')
    np.save('label_classes.npy', le.classes_)
    print("\n✓ 模型已保存为 enhanced_crane_model_v4.h5")
    print("✓ 标签映射已保存为 label_classes.npy")
    print("✓ 标准化参数已保存为 scaler_params.npy")
    print("=" * 50)
    print("训练完成！")


if __name__ == '__main__':
    main()