import torch
from torchsummary import summary
import cv2
from PIL import Image
from torch import nn
import torch.nn.functional as F
from typing import Union
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from pathlib import Path


class loss:
    def __init__(self, model):
        device = next(model.parameters()).device
        h = model.args


