import tensorflow as tf
from tensorflow.keras.utils import plot_model

model = tf.keras.applications.EfficientNetB2()

# 绘制网络结构
plot_model(model, to_file='efficientnet_b2.png', show_shapes=True)