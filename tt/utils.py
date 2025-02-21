import os
import numpy as np


def get_anchors(anchors_path):
    with open(anchors_path, 'r') as f:
        anchors = f.readline()
        print()

    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)
