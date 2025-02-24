import torch
import torch.nn as nn
import numpy as np
from utils import get_anchors
import torch.nn.functional as F
from config_lowlight import cfg


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

class DetectionHead(nn.Module):
    def __init__(self, num_class, anchors, stride):
        super(DetectionHead, self).__init__()
        self.num_class = num_class
        self.anchor_per_scale = len(anchors)
        self.stride = stride
        self.anchors = anchors

    def conv_shape(self, input_data):
        # [b, c, h, w]
        conv_shape = input_data.shape
        batch_size = conv_shape[0]
        output_height = conv_shape[2]
        output_width = conv_shape[3]

        # 确保 height 和 width 相等
        assert output_height == output_width, "Height and width must be equal"
        output_size = output_height

        return batch_size, output_size

    def forward(self, x):
        batch_size, size = self.conv_shape(x)
        # [b, c, h, w] -> [b, h, w, c]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [b, h, w, c] -> [b, h, w, an, p]
        x = x.view(batch_size, size, size, self.anchor_per_scale, 5 + self.num_class)

        # 中心点坐标特征
        conv_raw_dxdy = x[:, :, :, :, 0:2]  # dx, dy
        # 宽高特征
        conv_raw_dwdh = x[:, :, :, :, 2:4]  # dw, dh
        # 置信度特征
        conv_raw_conf = x[:, :, :, :, 4:5]  # confidence
        # 每个类别的概率特征
        conv_raw_prob = x[:, :, :, :, 5:]  # class probabilities

        # 定义列矩阵（每一列都是从0开始）[size, size]
        y = torch.arange(size, dtype=torch.float32).view(-1, 1).repeat(1, size)
        # 定义行矩阵（每一行都是从0开始）[size, size]
        x = torch.arange(size, dtype=torch.float32).view(1, -1).repeat(size, 1)

        # 定义一个[b, size, size, anchor, 2]的特征网格
        xy_grid = torch.stack([x, y], dim=-1)
        xy_grid = xy_grid[None, :, :, None, :].repeat(batch_size, 1, 1, self.anchor_per_scale, 1)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + xy_grid) * self.stride
        pred_wh = (torch.exp(conv_raw_dwdh) * self.anchors) * self.stride
        pred_xywh = torch.concat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)

        return torch.concat([pred_xywh, pred_conf, pred_prob], dim=-1)

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
    def __init__(self, num_class, isp_flag=False):
        super(YOLOV3, self).__init__()
        # 配置文件
        self.num_class = num_class
        # [3, 3, 2]的np数组
        self.anchors = get_anchors(cfg.YOLO.ANCHORS)
        # 定义下采样率[8, 16, 32]
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.isp_flag = isp_flag

        # backbone
        self.darknet = Darknet53()
        # neck
        self.conv_lbranch = nn.Sequential(
            DBL(1024, 512, 1),
            DBL(512, 1024),
            DBL(1024, 512, 1),
            DBL(512, 1024),
            DBL(1024, 512, 1))

        self.conv_lbox = nn.Sequential(
            DBL(512, 1024),
            nn.Conv2d(1024, 3 * (self.num_class + 5), 1))

        self.l_upsample = nn.Sequential(
            DBL(512, 256, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.conv_mbranch = nn.Sequential(
            DBL(768, 256, 1),
            DBL(256, 512),
            DBL(512, 256, 1),
            DBL(256, 512),
            DBL(512, 256, 1),
        )
        self.m_upsample = nn.Sequential(
            DBL(256, 128, 1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.conv_mbox = nn.Sequential(
            DBL(256, 512),
            nn.Conv2d(512, 3 * (self.num_class + 5), 1)
        )

        self.conv_sbranch = nn.Sequential(
            DBL(384, 128, 1),
            DBL(128, 256),
            DBL(256, 128, 1),
            DBL(128, 256),
            DBL(256, 128, 1)
        )
        self.conv_sbox = nn.Sequential(
            DBL(128, 256),
            nn.Conv2d(256, 3 * (self.num_class + 5), 1)
        )

        # head
        self.head_s = DetectionHead(num_class, self.anchors[0], self.strides[0])
        self.head_m = DetectionHead(num_class, self.anchors[1], self.strides[1])
        self.head_l = DetectionHead(num_class, self.anchors[2], self.strides[2])

    def _filtered(self, input_processed, input_clean):
        # 这里处理的都是batch
        self.filter_params = input_processed
        filtered_pipline = []

        if self.isp_flag:
            # 实现子网络处理模块
            input_data = F.interpolate(input_processed, size=(256, 256), mode='bilinear', align_corners=False)
            fine_tune = SubNet(cfg=cfg)
            fine_tune = fine_tune(input_data)

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
        image_filtered, filtered_pipline, recovery_loss = self._filtered(x, input_clean)
        # backbone
        route_1, route_2, x = self.darknet(image_filtered)

        # neck
        x = self.conv_lbranch(x)
        # [b,13,13,18]
        l_box = self.conv_lbox(x)

        x = self.l_upsample(x)
        x = torch.concat([route_2, x], 1)
        x = self.conv_mbranch(x)
        # [b,26,26,18]
        m_box = self.conv_mbox(x)

        x = self.m_upsample(x)
        x = torch.concat([route_1, x], 1)

        x = self.conv_sbranch(x)
        # [b,52,52,18]
        s_box = self.conv_sbox(x)

        # 多尺度检测头
        pred_sbbox = self.head_s(s_box)
        pred_mbbox = self.head_m(m_box)
        pred_lbbox = self.head_l(l_box)

        return pred_sbbox, pred_mbbox, pred_lbbox , recovery_loss



if __name__ == '__main__':
    model = YOLOV3(num_class=1, input_size=416, isp_flag=True)
    # print(model)
    for name, param in model.named_parameters():
        print(name, "\n")