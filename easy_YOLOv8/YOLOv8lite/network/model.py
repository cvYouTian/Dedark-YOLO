import torch
from torchsummary import summary
import cv2
from PIL import Image
from torch import nn
import torch.nn.functional as F
from typing import Union, Type
from torchvision import transforms
from pathlib import Path

from YOLOv8lite.Utils import make_anchors, dist2bbox


def auto_padding(kernel_size, pad=None):
    if pad is None:
        pad = kernel_size // 2 if isinstance(kernel_size, int) else (x // 2 for x in kernel_size)
    return pad


class Conv(nn.Module):
    def __init__(self, input_chanel,
                 output_chanel,
                 kernel_size: Union[int, tuple, list],
                 stride: int,
                 padding: Union[int, None]):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_chanel,
                              out_channels=output_chanel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=auto_padding(kernel_size, padding),
                              bias=False)
        self.BN = nn.BatchNorm2d(output_chanel)

    def forward(self, x):
        return F.silu(self.BN(self.conv(x)))


class SPPF(nn.Module):
    """这里的sppf结构作者使用了中间的隐藏层

    """

    def __init__(self, in_channel=512, out_channel=512):
        super(SPPF, self).__init__()
        c_ = in_channel // 2
        self.cv1 = Conv(in_channel, c_, 1, 1, padding=auto_padding(1, None))
        self.cv2 = Conv(c_ * 4, out_channel, 1, 1, padding=auto_padding(1, None))
        self.m = nn.MaxPool2d(5, 1, padding=auto_padding(5, None))

    def forward(self, x):
        x0 = self.cv1(x)
        x1 = self.m(x0)
        x2 = self.m(x1)
        return self.cv2(torch.cat((x0, x1, x2, self.m(x2)), 1))


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 shortcut: bool = True,
                 kernel_size: Union[int, tuple, list] = 3):
        super(Bottleneck, self).__init__()
        self.shortcut = shortcut
        self.conv1 = Conv(input_chanel=in_channel,
                          output_chanel=out_channel // 2,
                          kernel_size=kernel_size,
                          stride=1,
                          padding=auto_padding(kernel_size, None))

        self.conv2 = Conv(input_chanel=out_channel // 2,
                          output_chanel=out_channel,
                          kernel_size=kernel_size,
                          stride=1,
                          padding=auto_padding(kernel_size, None))

    def forward(self, x):
        if self.shortcut:
            x0 = self.conv1(x)
            x1 = self.conv2(x0)
            logits = x + x1
        else:
            x0 = self.conv1(x)
            logits = self.conv2(x0)
        return logits


class C2f(nn.Module):
    def __init__(self,
                 shortcut: bool,
                 block: int,
                 in_channel,
                 out_channel):
        super(C2f, self).__init__()
        self.conv1 = Conv(input_chanel=in_channel,
                          output_chanel=out_channel,
                          kernel_size=1,
                          stride=1,
                          padding=auto_padding(1, None))

        self.conv2 = Conv(input_chanel=out_channel // 2 * (block + 2),
                          output_chanel=out_channel,
                          kernel_size=1,
                          stride=1,
                          padding=auto_padding(1, None))

        self.bottleneck = nn.ModuleList(
            [Bottleneck(out_channel // 2, out_channel // 2, shortcut) for _ in range(block)])

    def forward(self, x):
        module = list(self.conv1(x).chunk(2, 1))
        module.extend(m(module[-1]) for m in self.bottleneck)
        x = self.conv2(torch.cat(module, 1))
        return x


class DFL(nn.Module):
    """glenn, 经过DFL之后直接变成了四点的定位坐标

    Args:
        c1[]: c1 is reg_max
    """

    def __init__(self, c1=16):
        super().__init__()
        # 单层的感知机
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        # 这里定义了0-15的16个参数
        x = torch.arange(c1, dtype=torch.float)
        # 给卷积核赋值
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # [1, 64, 8400]
        b, c, a = x.shape
        # 第一次的定位信息纬度变化， 这里是16维在前， 4点信息在后
        x = x.view(b, 4, self.c1, a).transpose(2, 1)
        print(x.softmax(1))
        # 先在16个纬度上实现softmax，再使用单层感知机（16, 1）实现16个只

        # [1, 4, 8400]
        return self.conv(x.softmax(1)).view(b, 4, a)


# TODO: Detect中没有使用官网的参数初始化函数
class Detect(nn.Module):
    def __init__(self, nc, reg_max):
        super().__init__()

        self.strides = torch.empty(0)
        self.anchors = torch.empty(0)

        # 分类的数量 + 定位信息的数量
        self.nc = nc
        # 检测头的数量
        self.stride = torch.tensor([8, 16, 32], dtype=torch.float32, device="cuda:0")
        self.reg_max = reg_max
        # self.no 不包含位置关系只是一个纯数字
        self.no = self.reg_max * 4 + nc 
        self.dfl = DFL(reg_max)
        # 定位中也不包含位置信息
        self.bbox_low = nn.Sequential(Conv(256, 64, 3, 1, 1),
                                  Conv(64, 64, 3, 1, 1),
                                  nn.Conv2d(64, self.reg_max * 4, 1, 1, 0))

        self.cls_low = nn.Sequential(Conv(256, 256, 3, 1, 1),
                                 Conv(256, 256, 3, 1, 1),
                                 nn.Conv2d(256, self.nc, 1, 1, 0))

        self.bbox_mid = nn.Sequential(Conv(512, 64, 3, 1, 1),
                                  Conv(64, 64, 3, 1, 1),
                                  nn.Conv2d(64, self.reg_max * 4, 1, 1, 0))

        self.cls_mid = nn.Sequential(Conv(512, 256, 3, 1, 1),
                                 Conv(256, 256, 3, 1, 1),
                                 nn.Conv2d(256, self.nc, 1, 1, 0))

        self.bbox_high = nn.Sequential(Conv(512, 64, 3, 1, 1),
                                  Conv(64, 64, 3, 1, 1),
                                  nn.Conv2d(64, self.reg_max * 4, 1, 1, 0))

        self.cls_high = nn.Sequential(Conv(512, 256, 3, 1, 1),
                                 Conv(256, 256, 3, 1, 1),
                                 nn.Conv2d(256, self.nc, 1, 1, 0))

    def forward(self, x):
        # x: [feat80, feat40, feat20]

        shape = x[0].shape
        bbox_low = self.bbox_low(x[0])
        cls_low = self.cls_low(x[0])
        # [1, 64 + 80, 80, 80]
        # 组合时定位在前， 分类通道在后
        low_feat = torch.cat((bbox_low, cls_low), 1)

        bbox_mid = self.bbox_mid(x[1])
        cls_mid = self.cls_mid(x[1])
        # [1, 64 + 80, 40, 40]
        mid_feat = torch.cat((bbox_mid, cls_mid), 1)

        bbox_high = self.bbox_high(x[2])
        cls_high = self.cls_high(x[2])
        # [1, 64 + 80, 20, 20]
        high_feat = torch.cat((bbox_high, cls_high), 1)

        x = [low_feat, mid_feat, high_feat]

        if self.training:
            return x

        # 如果不是训练还需要下面的操作

        # [2, 8400]  [1, 8400]
        self.anchors, self.strides = (x.transpose(0, 1)
                                      for x in make_anchors(x, self.stride, 0.5))
        # [1, 80+64, 8400] : 三个尺寸所有的像素点信息, 即每个像素点都要预测出80个类别信息和16组的坐标信息
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        # box [1, 64, 8400] cls [1, 80, 8400]
        # 拆分时和153行的位置信息对应，即定位信息在前， 分类信息在后
        box, cls = x_cat.split(split_size=(self.reg_max * 4, self.nc), dim=1)
        # dfl:[1, 4, 8400], anchors: [1, 2, 8400] -> [1, 4, 8400] 最后的bbox坐标
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        # cls这里使用的是sigmiod函数，做了一下归一化 -> [1, 84, 8400]
        y = torch.cat((dbox, cls.sigmoid()), 1)

        # y:[1, 4+80, 8400] x:[[1, 64+80, 80, 80], [1, 64+80, 40, 40], [1, 64+80, 20, 20]]
        return y, x


class YOLOv8l(nn.Module):
    def __init__(self, nc, reg_max):
        super(YOLOv8l, self).__init__()
        self.nc = nc
        self.reg_max = reg_max

        # backbone
        self.conv_0 = Conv(3, 64, 3, 2,
                           padding=auto_padding(3, None))

        self.conv_1 = Conv(64, 128, 3, 2,
                           padding=auto_padding(3, None))

        self.c2f_2 = C2f(True, 3, 128, 128)

        self.conv_3 = Conv(128, 256, 3, 2,
                           padding=auto_padding(3, None))

        self.c2f_4 = C2f(True, 6, 256, 256)

        self.conv_5 = Conv(256, 512, 3, 2,
                           padding=auto_padding(3, None))

        self.c2f_6 = C2f(True, 6, 512, 512)

        self.conv_7 = Conv(512, 512, 3, 2,
                           padding=auto_padding(3, None))

        self.c2f_8 = C2f(True, 3, 512, 512)

        self.sppf_9 = SPPF(512, 512)

        # head
        self.upsample = nn.Upsample(None, 2, "nearest")

        self.c2f_12 = C2f(False, 3, 1024, 512)

        self.c2f_15 = C2f(False, 3, 768, 256)

        self.conv_16 = Conv(256, 256, 3, 2,
                            padding=auto_padding(3, None))

        self.c2f_18 = C2f(False, 3, 768, 512)

        self.conv_19 = Conv(512, 512, 3, 2,
                            padding=auto_padding(3, None))

        self.c2f_21 = C2f(False, 3, 1024, 512)

        self.det = Detect(nc, reg_max)

    def forward(self, x):
        # backbone
        p1 = self.conv_0(x)
        p2 = self.conv_1(p1)
        x2 = self.c2f_2(p2)
        p3 = self.conv_3(x2)
        x4 = self.c2f_4(p3)
        p4 = self.conv_5(x4)
        x6 = self.c2f_6(p4)
        p5 = self.conv_7(x6)
        x8 = self.c2f_8(p5)
        sppf = self.sppf_9(x8)

        # head
        x10 = self.upsample(sppf)
        x11 = torch.cat((x6, x10), 1)
        x12 = self.c2f_12(x11)
        x13 = self.upsample(x12)
        x14 = torch.cat((x4, x13), 1)
        x15 = self.c2f_15(x14)

        x16 = self.conv_16(x15)
        x17 = torch.cat((x12, x16), 1)
        x18 = self.c2f_18(x17)

        x19 = self.conv_19(x18)
        x20 = torch.cat((sppf, x19), 1)

        x21 = self.c2f_21(x20)

        return self.det([x15, x18, x21])


if __name__ == '__main__':
    image_path = "/home/youtian/Pictures/Screenshots/bird.png"
    net = YOLOv8l(80, 16)

    net.to("cuda:0")
    net.eval()
    transform = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.ToTensor()])

    # image = Image.open(image_path)
    image = cv2.imread(image_path)
    image = Image.fromarray(image)
    image = transform(image)

    batch_input = image.unsqueeze(0).to("cuda:0")

    with torch.no_grad():
        # tensor: [1, 84, 8400], list[tensor]: [x15, x18, x21]
        out = net(batch_input)

    print(out)

    # feature_visualization(out, "nn.Conv2d, nn.Conv2d, nn.Conv2d", 5)
    # summary(net, input_size=(3, 640, 640))
