"""
Block modules
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, PConv, SCConv
from .transformer import TransformerBlock


__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'PconvBottleneck', 'FasterC2f_N',
           'FasterC2f', 'PconvBottleneck_n', "SCConvBottleneck", "SCConv", "SCC2f", "SC_PW_Bottleneck", "SC_PW_C2f",
           "SC_Conv3_C2f", "SC_Conv3_Bottleneck", "Conv3_SC_C2f", "Conv3_SC_Bottleneck", "AsffTribeLevel",
           "AsffDoubLevel", "RFBblock", "MFRU")


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class AsffTribeLevel(nn.Module):
    def __init__(self, level):
        super(AsffTribeLevel, self).__init__()
        self.level = level
        self.dim = [512, 512, 256]
        self.inter_dim = self.dim[self.level]
        # 0 是最深的feature
        if level == 0:
            # TODO: 这里x和l版本的最后P5的输出不是1024，而是512， 做一根据ASFF（FM的维度不变的话就使用，
            #  或者减少的话就是就使用1*1卷积）这一特性，直接做下采样
            self.stride_level_1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level == 1:
            # FM的维度不变的话就使用，或者减少的话就是就使用1*1卷积
            # FM的维度升高需要使用3*3卷积he下采样
            # TODO 因为纬度没有变，所以可以直接拿到
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
            # 2 是最浅的feature
        elif level == 2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            # 添加了level1的将维
            self.compress_level_1 = add_conv(512, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 8   # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.level == 0:
            level_0_resized = x[0]
            level_1_resized = self.stride_level_1(x[1])
            level_2_downsampled_inter = F.max_pool2d(x[2], 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            # 直接拿到x[0]的信息， 因为纬度没有改变
            level_0_resized = F.interpolate(x[0], scale_factor=2, mode='nearest')
            level_1_resized = x[1]
            level_2_resized = self.stride_level_2(x[2])

        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x[0])
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x[1])
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x[2]

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        return out


class AsffDoubLevel(nn.Module):
    def __init__(self, level):
        super(AsffDoubLevel, self).__init__()
        self.level = level
        # self.width_coefficient = width_coefficient
        self.dim = [512, 256]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)

        compress_c = 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # note : 这里的x是两个
        # print(x[0].shape)
        if self.level == 0:
            level_0_resized = x[0]
            level_1_resized = self.stride_level_1(x[1])

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x[0])
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x[1]

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :]

        out = self.expand(fused_out_reduced)

        return out


# class MFRU(nn.Module):
#     """多尺度特征重构单元
#     使用SRU来设计
#     """
#     def __init__(self, level):
#         super(MFRU, self).__init__()
#         compress_c = 16
#
#         self.scconv = SCConv(512)
#         self.scconv_s = SCConv(256)
#
#         self.pwconv0 = nn.Conv2d(512, 256, 1, 1, 0)
#         self.pwconv1 = nn.Conv2d(512, 256, 1, 1, 0)
#         self.pwconv2 = nn.Conv2d(256, 256, 1, 1, 0)
#
#         self.weight_level_0 = nn.Conv2d(256, compress_c, 1, 1, 0)
#         self.weight_level_1 = nn.Conv2d(256, compress_c, 1, 1, 0)
#         self.weight_level_2 = nn.Conv2d(256, compress_c, 1, 1, 0)
#
#         self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)
#
#     def forward(self, x):
#         """
#         Args:
#             [20, 40 , 80]
#         """
#         level_0 = self.pwconv0(x[0])
#         level_0_resized = self.scconv(F.interpolate(level_0, scale_factor=4, mode='nearest'))
#
#         level_1 = self.pwconv1(x[1])
#         level_1_resized = self.scconv(F.interpolate(level_1, scale_factor=2, mode='nearest'))
#
#         level_2 = self.pwconv2(x[2])
#         level_2_resized = self.scconv_s(level_2)
#
#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         level_2_weight_v = self.weight_level_2(level_2_resized)
#
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
#         levels_weight = self.weight_levels(levels_weight_v)
#
#         levels_weight = F.softmax(levels_weight, dim=1)
#
#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#                             level_1_resized * levels_weight[:, 1:2, :, :] + \
#                             level_2_resized * levels_weight[:, 2:, :, :]
#
#         return fused_out_reduced

class MFRU(nn.Module):
    """多尺度特征重构单元
    使用SRU来设计
    """
    def __init__(self, level):
        super(MFRU, self).__init__()
        compress_c = 16

        # P9的输出【20，20， 512】
        self.scconv512 = SCConv(512)
        self.scconv256 = SCConv(256)
        # p6的输出【40，40， 512】
        self.pwconv = nn.Conv2d(512, 256, 1, 1, 0)

        # self.weight_level_0 = add_conv(256, compress_c, 1, 1)
        self.weight_level_0 = nn.Conv2d(256, compress_c, 1, 1, 0)
        # self.weight_level_1 = add_conv(256, compress_c, 1, 1)
        self.weight_level_1 = nn.Conv2d(256, compress_c, 1, 1, 0)
        self.weight_level_2 = nn.Conv2d(256, compress_c, 1, 1, 0)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)

    def forward(self, x):
        """
        Args:
            [20, 40 , 80]
        """
        # [20, 20, 256]
        level_0 = self.pwconv(self.scconv512(x[0]))
        # out [80, 80, 256]
        level_0_resized = F.interpolate(level_0, scale_factor=4, mode='nearest')
        # [40, 40, 256]
        level_1 = self.pwconv(self.scconv512(x[1]))
        # out [80, 80, 256]
        level_1_resized = F.interpolate(level_1, scale_factor=2, mode='nearest')
        # out [80, 80, 256]
        level_2_resized = self.scconv256(x[2])

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        # 通道维
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]
        # [80, 80, 256]
        out = self.scconv256(fused_out_reduced)

        return out
#
# class MFRUL(nn.Module):
#     """多尺度特征重构单元
#     使用SRU来设计
#     """
#     def __init__(self, level):
#         super(MFRUL, self).__init__()
#         compress_c = 16
#
#         self.scconv = SCConv(128)
#
#         self.pwconv0 = nn.Conv2d(512, 128, 1, 1, 0)
#         self.pwconv1 = nn.Conv2d(512, 128, 1, 1, 0)
#         self.pwconv2 = nn.Conv2d(256, 128, 1, 1, 0)
#
#         self.weight_level_0 = nn.Conv2d(128, compress_c, 1, 1, 0)
#         self.weight_level_1 = nn.Conv2d(128, compress_c, 1, 1, 0)
#         self.weight_level_2 = nn.Conv2d(128, compress_c, 1, 1, 0)
#
#         self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)
#
#     def forward(self, x):
#         """
#         Args:
#             [20, 40 , 80]
#         """
#         level_0 = self.pwconv0(x[0])
#         level_0_resized = self.scconv(F.interpolate(level_0, scale_factor=8, mode='nearest'))
#
#         level_1 = self.pwconv1(x[1])
#         level_1_resized = self.scconv(F.interpolate(level_1, scale_factor=4, mode='nearest'))
#
#         level_2 = self.pwconv2(x[2])
#         level_2_resized = self.scconv(F.interpolate(level_2, scale_factor=2, mode="nearest"))
#
#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         level_2_weight_v = self.weight_level_2(level_2_resized)
#
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
#         levels_weight = self.weight_levels(levels_weight_v)
#         # 通道维
#         levels_weight = F.softmax(levels_weight, dim=1)
#
#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#                             level_1_resized * levels_weight[:, 1:2, :, :] + \
#                             level_2_resized * levels_weight[:, 2:, :, :]
#         # [80, 80, 256]
#         out = fused_out_reduced
#
#         return out
#



class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        # conv层要求不跟踪梯度
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        # 1纬度是chanel纬度
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(c1=(2 + n) * self.c, c2=c2, k=1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class FasterC2f_N(C2f):
    """
    add FasterBlock with 2 convolutions
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)
        self.m = nn.ModuleList(PconvBottleneck_n(self.c, self.c, shortcut, g=g, k=((3, 3), (3, 3)),
                                                 e=1.0) for _ in range(n))


class FasterC2f(C2f):
    """
    add FasterBlock with 2 convolutions
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)
        self.m = nn.ModuleList(PconvBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)),
                                               e=1.0) for _ in range(n))


class SCC2f(C2f):
    """
    add FasterBlock with 2 SC-convolutions
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)
        self.m = nn.ModuleList(SCConvBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)),
                                               e=1.0) for _ in range(n))


class SC_PW_C2f(C2f):
    """
    add FasterBlock with 2 SC-convolutions
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)
        self.m = nn.ModuleList(SC_PW_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)),
                                               e=1.0) for _ in range(n))


class SC_Conv3_C2f(C2f):
    """
    add SCCBlock with SCC2f(SCConv+Conv3_BN_Silu)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)
        self.m = nn.ModuleList(SC_Conv3_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)),
                                               e=1.0) for _ in range(n))


class Conv3_SC_C2f(C2f):
    """
    add SCBlock with SCC2f(Conv3_BN_Silu+SCConv)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)
        self.m = nn.ModuleList(Conv3_SC_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)),
                                               e = 1.0) for _ in range(n))


class SC_PW_PW_C2f(C2f):
    """
    add SCBlock with SC-convolutions
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c = int(c2 * e)
        self.m = nn.ModuleList(SC_PW_PW_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)),
                                               e=1.0) for _ in range(n))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class PconvBottleneck(nn.Module):
    """add bottleneck of PConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.fasterblock = nn.Sequential(
            PConv(dim=c1, n_div=4),
            # 使用33卷机稳定
            Conv(c1=c1, c2=c_, k=3, s=1, p=1),
        )
        self.conv = nn.Conv2d(in_channels=c_, out_channels=c2, kernel_size=1, stride=1,
            padding=autopad(k=1, p=None, d=1), groups=g, bias=False)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.conv(self.fasterblock(x)) if self.add else self.conv(self.fasterblock(x))


class PconvBottleneck_n(nn.Module):
    """add bottleneck of PConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        # hidden channels
        c_ = int(c2 * e)
        self.fasterblock = nn.Sequential(
            PConv(dim=c1, n_div=4),
            Conv(c1=c1, c2=2*c_, k=1, s=1),
            nn.Conv2d(in_channels=2*c_, out_channels=c2, kernel_size=1, stride=1, padding=autopad(k=1, p=None, d=1),
                      groups=g, bias=False)
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.fasterblock(x) if self.add else self.fasterblock(x)


class SCConvBottleneck(nn.Module):
    """add bottleneck of PConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)
        self.SandCRblock = nn.Sequential(
            SCConv(c1),
            Conv(c1=c1, c2=c2, k=1, s=1),
            # nn.Conv2d(in_channels=2 * c_, out_channels=c2, kernel_size=1, stride=1, padding=autopad(k=1, p=None, d=1),
            #           groups=g, bias=False)
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.SandCRblock(x) if self.add else self.SandCRblock(x)


class SC_PW_Bottleneck(nn.Module):
    """add bottleneck of """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()

        self.SandCRblock = nn.Sequential(
            SCConv(c1),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, stride=1, padding=autopad(k=1, p=None, d=1),
                      groups=g, bias=True)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.SandCRblock(x) if self.add else self.SandCRblock(x)


class SC_Conv3_Bottleneck(nn.Module):
    """add bottleneck of SCConv_Conv3_Bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()

        self.SandCRblock = nn.Sequential(
            SCConv(c1),
            Conv(c1=c1, c2=c2, k=3, s=1, p=autopad(k=3, d=1), g=g)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.SandCRblock(x) if self.add else self.SandCRblock(x)


class SC_PW_PW_Bottleneck(nn.Module):
    """add bottleneck of SCConv + PWConv_BN_SiLU + PWConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()

        self.SandCRblock = nn.Sequential(
            SCConv(c1),
            Conv(c1=c1, c2=2*c1, k=1, s=1, p=autopad(k=1, d=1), g=g),
            nn.Conv2d(in_channels=2*c1, out_channels=c2, kernel_size=1, stride=1, padding=autopad(k=1, p=None, d=1),
                  groups=g, bias=False)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.SandCRblock(x) if self.add else self.SandCRblock(x)


class Conv3_SC_Bottleneck(nn.Module):
    """add bottleneck of SCConv_Conv3_Bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()

        self.SandCRblock = nn.Sequential(
            Conv(c1=c1, c2=c2, k=3, s=1, p=autopad(k=3, d=1), g=g),
            SCConv(c2),
            # nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=autopad(k=1, p=None, d=1),
            #           groups=g, bias=False)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.SandCRblock(x) if self.add else self.SandCRblock(x)


class RFBblock(nn.Module):
    # def __init__(self, in_ch, residual=False):
    def __init__(self, in_ch):
        super(RFBblock, self).__init__()
        inter_c = in_ch // 4
        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=2, padding=2)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=3, padding=3)
        )
        # self.residual = residual

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        out = torch.cat((x_0, x_1, x_2, x_3), 1)
        # if self.residual:
        #     out += x
        return out


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
