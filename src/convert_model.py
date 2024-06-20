import tensorflow as tf

# 加载 TensorFlow SavedModel 格式的模型
model = tf.keras.models.load_model('path_to_saved_model')  # 替换为实际的模型路径

# 保存为 .h5 格式
model.save('path_to_pretrained_unet_model.h5')

# 或者保存为 .keras 格式
model.save('path_to_pretrained_unet_model.keras')
