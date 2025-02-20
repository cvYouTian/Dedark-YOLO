import random
from pydoc import ispath

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import tqdm
from jieba.lac_small.predict import batch_size
from modelscope.msdatasets.dataset_cls.custom_datasets import DataLoader

from dataset import CustomDataset, CfgDataloader
from ffmpeg import output
from sympy.physics.vector import outer
from torch.nn import init
import math
from config_lowlight import args, cfg
from model import gpu_id


# 写日志操作
def write_mes(msg, log_name=None, show=True, mode='a'):
    get_end = lambda line: '' if line.endswith('\n') else '\n'
    if show:
        if isinstance(msg, str):
            print(msg, end=get_end(msg))
        elif isinstance(msg, (list, tuple)):
            for line in msg:
                print(line, end=get_end(line))  # might be a different thing
        else:
            print(msg)

    if log_name is not None:
        with open(log_name, mode) as f:
            f.writelines(msg)


class DBL(nn.Module):
    def __init__(self, ic, oc, ks=3,downsample=False):
        super(DBL, self).__init__()
        if downsample:
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, stride=2, padding=ks // 2, bias=False)
        else:
            self.conv = nn.Conv2d(ic, oc, kernel_size=ks, stride=1, padding=ks // 2, bias=False)
        self.bn = nn.BatchNorm2d(oc)
        self.leakrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakrelu(x)
        return x

class resunit(nn.Module):
    def __init__(self, ic, mc, oc):
        super(resunit, self).__init__()
        self.conv1 = DBL(ic, mc, ks=1)
        self.conv2 = DBL(mc, oc, ks=3)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + shortcut

class resn(nn.Module):
    def __init__(self, n, ic, mc, oc):
        super(resn, self).__init__()
        self.res = nn.Sequential(*[resunit(oc, mc, oc) for _ in range(n)])

    def forward(self, x):
        x = self.res(x)
        return x


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = DBL(3, 32)
        self.conv2 = DBL(32, 64, downsample=True)
        self.res1 = resn(1, 64, 32, 64)

        self.conv3 = DBL(64, 128, downsample=True)
        self.res2 = resn(2, 128, 64, 128)

        self.conv4 = DBL(128, 256, downsample=True)
        self.res3 = resn(8, 256, 128, 256)

        self.conv5 = DBL(256, 512, downsample=True)
        self.res4 = resn(8, 512, 256, 512)

        self.conv6 = DBL(512, 1024, downsample=True)
        self.res5 = resn(4, 1024, 512, 1024)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)
        route_1 = x = self.res3(x)
        x = self.conv5(x)
        route_2 = x = self.res4(x)
        x = self.conv6(x)
        out_put = self.res5(x)

        return route_1, route_2, out_put

class Neck(nn.Module):
    def __init__(self, ic, mc, oc):
        super(Neck, self).__init__()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)

        return x

class SubNet(nn.Module):
    def __init__(self,  cfg=cfg):
        super(SubNet, self).__init__()
        self.output_dim = cfg.num_filter_parameters
        channels = 6

        # 卷积层定义
        self.conv_layers = nn.Sequential(
            # 第0层: 3x3 卷积，输入通道3，输出通道16，下采样
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            # 第1层: 3x3 卷积，通道数翻倍
            nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            # 第2层: 3x3 卷积，保持通道数
            nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            # 第3层: 3x3 卷积，保持通道数
            nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),

            # 第4层: 3x3 卷积，保持通道数
            nn.Conv2d(2 * channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, self.output_dim)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 调整输入尺寸到256x256
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        # 卷积部分
        x = self.conv_layers(x)
        # 展平特征
        x = torch.flatten(x, start_dim=1)
        # 全连接部分
        x = self.fc_layers(x)
        return x

# YOLOv3 Model
class YOLOV3(nn.Module):
    def __init__(self, num_class, anchors, strides, input_size=416, isp_flag=False):
        super(YOLOV3, self).__init__()
        self.num_class = num_class
        self.anchors = anchors
        self.strides = strides
        self.isp_flag = isp_flag
        # backbone
        self.darknet = Darknet53()
        # neck
        self.conv_lbranch = nn.Sequential(
            DBL(1024, 512, 1),
            DBL(512, 1024),
            DBL(1024, 512, 1),
            DBL(512, 1024),
            DBL(1024, 512, 1),)
        self.branch = DBL(512, 1024)
        self.conv_lbox = nn.Conv2d(1024, 3 * (self.num_class + 5), 1)


        self.conv_mbranch = nn.Sequential(

        )

        # head

        self.head_s = DetectionHead(256, len(anchors[0]), num_classes)
        self.head_m = DetectionHead(256, len(anchors[0]), num_classes)
        self.head_l = DetectionHead(256, len(anchors[0]), num_classes)

        # 类似实现其他检测头
    def _filtered(self, input_processed, input_clean):
        # 实现前向传播逻辑, 定义微调的子网络
        # 这里处理的都是batch

        self.filter_params = input_processed
        filtered_pipline = []

        if self.isp_flag:
            # 实现子网络处理模块
            input_data = F.interpolate(input_processed, size=(256, 256), mode='bilinear', align_corners=False)
            fine_tune = SubNet(input_channels=3, cfg=cfg)(input_data)

            # 白平衡等滤波器
            filters = cfg.filters
            filters = [filter(input_processed) for filter in filters]
            filters_parameters = []

            for i, filter in enumerate(filters):
                # 传入一张处理后的图片和一张
                input_processed, filtered_param = filter.apply(input_processed, fine_tune)
                filters_parameters.append(filtered_param)
                filtered_pipline.append(input_processed)

            self.filter_params = filters_parameters
        self.image_filtered = input_processed
        self.filtered_pipline = filtered_pipline

        recovery_loss = torch.sum((self.image_filtered - input_clean) ** 2.0)

        return self.image_filtered, self.filtered_pipline, recovery_loss

    def forward(self, x, input_clean):
        # 滤波操作
        self.image_filtered, self.filtered_pipline, recovery_loss = self._filtered(x, input_clean)

        # backbone
        route_1, route_2, x = self.darknet(self.image_filtered)

        # 多尺度检测头
        out_s = self.head_s(x)
        out_m = self.head_m(x)
        out_l = self.head_l(x)

        return out_s, out_m, out_l

