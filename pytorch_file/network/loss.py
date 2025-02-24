import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_file.configs.config_lowlight import args
from pytorch_file.Utils.utils import get_anchors
from pytorch_file.Utils.utils import read_class_names, write_mes


class YOLOLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化损失参数
        # [3, 3, 2]的np数组
        self.anchors = get_anchors(cfg.YOLO.ANCHORS)
        # 定义下采样率[8, 16, 32]
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchor_per_scale = args.YOLO.ANCHOR_PER_SCALE
        self.num_classes = len(read_class_names(cfg.YOLO.CLASSES))

    def focal(self, target, actual, alpha=1.0, gamma=2.0):
        """
        计算 Focal Loss。
        :param target: 真实标签，形状为 [batch_size, ...]。
        :param actual: 模型预测的概率值（经过 sigmoid 后的值），形状为 [batch_size, ...]。
        :param alpha: 平衡正负样本的权重。
        :param gamma: 调节难易样本权重的参数。
        :return: Focal Loss。
        """
        # 计算 Focal Loss
        focal_loss = alpha * torch.pow(torch.abs(target - actual), gamma)

        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = torch.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                               boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)

        boxes2 = torch.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                               boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

        boxes1 = torch.concat([torch.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            torch.maximum(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
        boxes2 = torch.concat([torch.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            torch.maximum(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = torch.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = torch.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = torch.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = torch.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]

        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = torch.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
        boxes2 = torch.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

        left_up = torch.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = torch.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        conv_shape = conv.shape
        batch_size = conv_shape[0]
        # 注意这里和tensorflow的区别
        output_size = conv_shape[2]

        input_size = stride * output_size

        conv = conv.permute(0,2,3,1).contiguous()
        conv = conv.view(batch_size, output_size, output_size,self.anchor_per_scale, 5 + self.num_classes)

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        # ????
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = self.bbox_giou(pred_xywh, label_xywh).unsqueeze(-1)
        # 将 input_size 转换为 float32 类型
        input_size = torch.tensor(input_size, dtype=torch.float32)

        # 计算边界框损失的权重
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        # 计算 IoU 和背景掩码
        iou = self.bbox_iou(pred_xywh[:, :, :, :, None, :], bboxes[:, None, None, None, :, :])
        max_iou = torch.max(iou, dim=-1, keepdim=True)
        respond_bgd = (1.0 - respond_bbox) * (max_iou < self.iou_loss_thresh).float()

        # 计算置信度损失
        conf_focal = self.focal(respond_bbox, pred_conf)
        conf_loss = conf_focal * (
                respond_bbox * F.binary_cross_entropy_with_logits(conv_raw_conf, respond_bbox, reduction='none')
                + respond_bgd * F.binary_cross_entropy_with_logits(conv_raw_conf, respond_bbox, reduction='none')
        )

        # 计算类别概率损失
        prob_loss = respond_bbox * F.binary_cross_entropy_with_logits(conv_raw_prob, label_prob, reduction='none')

        # 计算损失的均值
        giou_loss = torch.mean(torch.sum(giou_loss, dim=[1, 2, 3, 4]))
        conf_loss = torch.mean(torch.sum(conf_loss, dim=[1, 2, 3, 4]))
        prob_loss = torch.mean(torch.sum(prob_loss, dim=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def forward(self, conv, pred, label, true_bbox, recovery):
        loss_sbbox = self.loss_layer(conv[0], pred[0], label[0], true_bbox[0], anchors=self.anchors[0], stride=self.stride[0])
        loss_mbbox = self.loss_layer(conv[1], pred[1], label[1], true_bbox[1], anchors=self.anchors[1], stride=self.stride[1])
        loss_lbbox = self.loss_layer(conv[2], pred[2], label[2], true_bbox[2], anchors=self.anchors[2], stride=self.stride[2])

        # 实现 GIoU 损失、置信度损失、类别损失和恢复损失
        giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]
        conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]
        cls_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]
        recovery_loss = recovery

        return {
            'giou': giou_loss,
            'conf': conf_loss,
            'cls': cls_loss,
            'recovery': recovery_loss,
            'total': giou_loss + conf_loss + cls_loss + recovery_loss
        }