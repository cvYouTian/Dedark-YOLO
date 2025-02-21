

def get_anchors(anchors_path):
    with open(anchors_path, 'r') as f:
        anchors = f.readline()
        print()

    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


import tensorflow as tf
import numpy as np

# 定义参数
batch_size = 2
output_size = 10
anchor_per_scale = 3
stride = 8

# 定义 anchors
anchors = np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32)

# 定义 conv_raw_dwdh
conv_raw_dwdh = tf.random.normal([batch_size, output_size, output_size, anchor_per_scale, 2])

# 计算预测的边界框宽度和高度
pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride

# 打印结果以验证
print("pred_wh shape:", pred_wh.shape)
print("pred_wh:\n", pred_wh)