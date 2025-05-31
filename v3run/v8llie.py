import torch
import cv2
from PIL import Image
from torch import nn
from torchvision import transforms
from core.config_lowlight import cfg, args
import core.common as common
from utils.tal import make_anchors, dist2bbox
import os
from core.yolov8_asff import YOLOv8l

image_path = "/home/youtian/Documents/pro/pyCode/Dedark-YOLO/darkdet/experiments_lowlight/exp_58/yolo3-ori/192.168.31.30_20241020_104014_3678654_2.jpg"
print(os.path.abspath(image_path))
# image_path = np.full((640, 640, 3),255, dtype=np.uint8)

net = YOLOv8l(80, 16)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
net.to(device)
net.eval()
transform = transforms.Compose([transforms.Resize((640, 640)),
                                transforms.ToTensor()])

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if image is None:
    raise ValueError("not process image: {}".format(image))

image = Image.fromarray(image)
image = transform(image)

batch_input = image.unsqueeze(0).to(device)

with torch.no_grad():
    # tensor: [1, 84, 8400], list[tensor]: [x15, x18, x21]
    out = net(batch_input)

print(out)
