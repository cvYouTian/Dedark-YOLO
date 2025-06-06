# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, PconvBottleneck, PconvBottleneck_n, FasterC2f_N, FasterC2f, SCC2f,
                    SCConvBottleneck, SC_PW_C2f, SC_PW_Bottleneck, SC_Conv3_Bottleneck, SC_Conv3_C2f,
                    Conv3_SC_Bottleneck, Conv3_SC_C2f, SC_PW_PW_Bottleneck, SC_PW_PW_C2f, AsffDoubLevel,
                    AsffTribeLevel, RFBblock, MFRU)
from .llie import lowlight_recovery
from .common import ExtractParameters2
from .filter_cfg import (UsmFilter, GammaFilter, ImprovedWhiteBalanceFilter, ContrastFilter, ToneFilter)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, PConv, SCConv, FC)

from .head import Classify, Detect, Pose, RTDETRDecoder, Segment, AsffDetect

from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

__all__ = ("UsmFilter", "GammaFilter", "ImprovedWhiteBalanceFilter", "ContrastFilter", "ToneFilter",
           "ExtractParameters2", "lowlight_recovery",'Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv',
           'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'PConv',
           'PconvBottleneck', 'PconvBottleneck_n', 'FasterC2f_N', 'FasterC2f', "SCConv", "SCConvBottleneck", "SCC2f",
           "SC_PW_Bottleneck", "SC_PW_C2f", "SC_Conv3_Bottleneck", "SC_Conv3_C2f", "Conv3_SC_Bottleneck", "Conv3_SC_C2f",
           "SC_PW_PW_C2f", "SC_PW_PW_Bottleneck", "FC", "AsffTribeLevel", "AsffDoubLevel", "AsffDetect", "RFBblock",
           "MFRU")
