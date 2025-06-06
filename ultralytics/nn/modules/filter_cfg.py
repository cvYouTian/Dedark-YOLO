from easydict import EasyDict as edict
from .filtersB import DeDarkFilter, UsmFilter, GammaFilter, ImprovedWhiteBalanceFilter, ContrastFilter, ToneFilter
import argparse


parser = argparse.ArgumentParser(description='filter-param')
args = parser.parse_args()

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

###########################################################################
# Filter Parameters
###########################################################################
cfg.num_filter_parameters = 15

cfg.dedark_begin_param = 0

cfg.wb_begin_param = 1
cfg.gamma_begin_param = 4
cfg.tone_begin_param = 5
cfg.contrast_begin_param = 13
cfg.usm_begin_param = 14


cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.1, 1.0)
cfg.usm_range = (0.0, 5)



# Masking is DISABLED
cfg.masking = False
cfg.minimum_strength = 0.3
cfg.maximum_sharpness = 1
cfg.clamp = False

###########################################################################
# CNN Parameters
###########################################################################
cfg.source_img_size = 64
cfg.base_channels = 32
cfg.dropout_keep_prob = 0.5
# G and C use the same feed dict?
cfg.share_feed_dict = True
cfg.shared_feature_extractor = True
cfg.fc1_size = 128
cfg.bnw = False
# number of filters for the first convolutional layers for all networks
#                      (stochastic/deterministic policy, critic, value)
cfg.feature_extractor_dims = 4096

###########################################################################



DD = DeDarkFilter(cfg)
WB = ImprovedWhiteBalanceFilter(cfg)
GF = GammaFilter(cfg)
# TF = ToneFilter(cfg)
CF = ContrastFilter(cfg)
S = UsmFilter(cfg)

# no dedark
# cfg.filters = [WB, GF, TF, CF, S]

cfg.filters = [DD, WB, GF,  CF, S]