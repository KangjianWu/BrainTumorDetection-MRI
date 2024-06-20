import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from unet_model import build_unet_model
import matplotlib.pyplot as plt

# 加载预处理后的数据
train_images = np.load('../data/processed/train_images.npy')
train_masks = np.load('../data/processed/train_masks')  # 需要提供生成的分割掩码
test_images = np.load('../data/processed/test_images.npy')
test_masks = np.load('../data/processed/test_masks.npy')

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

# 构建并训练模型
input_shape = (224, 224, 3)
model = build_unet_model(input_shape)

# 设置训练参数
epochs = 10
batch_size = 32

# 训练模型并保存历史
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

# 保存训练好的模型，使用 Keras 原生格式
model.save('models/unet/unet_model.keras')

# 保存训练历史
history_path = 'models/unet/unet_history.npy'
if not os.path.exists(os.path.dirname(history_path)):
    os.makedirs(os.path.dirname(history_path))
np.save(history_path, history.history)

# 可视化训练历史
if not os.path.exists('models/unet'):
    os.makedirs('models/unet')

# 绘制并保存准确率图像
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('models/unet/accuracy.png')
plt.close()

# 绘制并保存损失图像
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('models/unet/loss.png')
plt.close()
