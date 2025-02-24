import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from config_lowlight import args, cfg
import time
from utils import get_anchors
import os
from utils import read_class_names, write_mes
from dataset import CustomDataset
from .model import YOLOV3


class YOLOTrainer:
    def __init__(self, args, cfg):
        # 3
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        # person
        self.classes = read_class_names(cfg.YOLO.CLASSES)
        # 1
        self.num_classes = len(self.classes)
        # 1e-2
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        # 1e-6
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        # 0
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        # 70
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        # 2
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        # NULL
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        # 时间戳
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        # 0.9995
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150
        self.train_logdir = "./data/log/train"

        # 数据加载
        self.train_loader = DataLoader(CustomDataset("train", cfg),
                                       batch_size=args.TRAIN.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(CustomDataset("test", cfg),
                                     batch_size=args.TEST.BATCH_SIZE, shuffle=False)

        # step数
        self.steps_per_period = len(self.train_loader)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化参数
        self.model = YOLOV3(num_class=len(read_class_names(args.YOLO.CLASSES)), isp_flag=True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.TRAIN.LEARN_RATE_INIT)
        self.criterion = YOLOLoss(cfg)  # 需要自定义损失函数

        # 学习率调度
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=args.TRAIN.LEARN_RATE_END)

    def train(self):
        if os.path.exists(self.initial_weight):
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.model.load_state_dict(torch.load(self.initial_weight))

        else:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(args.epochs):
            # 关于加载别的权重的功能还没有实现
            if epoch <= self.first_stage_epochs:
                # 冻结
                pass
            else:
                # 解冻
                pass

            self.model.train()
            train_epoch_loss = []
            pbar = tqdm(self.train_loader)
            for batch_idx, (input_data, label_sbbox, label_mbbox, label_lbbox,
                            true_sbboxes, true_mbboxes, true_lbboxes) in enumerate(pbar):
                # 低光照增强
                if args.lowlight_FLAG:
                    lowlight_param = torch.rand(1) * 5 + 5  # [5, 10]
                    enhanced_images = input_data ** lowlight_param.item()
                else:
                    print("not implement lowlight train!!")
                    enhanced_images = input_data

                enhanced_images = enhanced_images.to(self.device)

                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                label = [label_sbbox, label_mbbox, label_lbbox]
                # xywh
                true_sbboxes = true_sbboxes.to(self.device)
                true_mbboxes = true_mbboxes.to(self.device)
                true_lbboxes = true_lbboxes.to(self.device)
                true = [true_sbboxes, true_mbboxes, true_lbboxes]

                # 前向传播
                pred, recovery_loss = self.model(enhanced_images, input_data)

                # 计算损失
                loss_dict = self.criterion(conv, pred, label, true, recovery_loss)
                total_loss = loss_dict['total']

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # 记录日志...

            # 验证步骤
            self.model.eval()
            test_epoch_loss = []
            with torch.no_grad():

                for batch_idx, (input_data, label_sbbox, label_mbbox, label_lbbox,
                                true_sbboxes, true_mbboxes, true_lbboxes) in enumerate(self.val_loader):
                    if args.lowlight_FLAG:
                        lowlight_param = torch.rand(1) * 5 + 5  # [5, 10]
                        enhanced_images = input_data ** lowlight_param.item()
                    else:
                        print("not implement lowlight train!!")
                        enhanced_images = input_data

                    enhanced_images = enhanced_images.to(self.device)
                    label_sbbox = label_sbbox.to(self.device)
                    label_mbbox = label_mbbox.to(self.device)
                    label_lbbox = label_lbbox.to(self.device)
                    true_sbboxes = true_sbboxes.to(self.device)
                    true_mbboxes = true_mbboxes.to(self.device)
                    true_lbboxes = true_lbboxes.to(self.device)

                    # 前向传播
                    output = self.model(enhanced_images, input_data)
                    giou_loss, conf_loss, prob_loss, recovery_loss = self.model.compute_loss(
                        label_sbbox, label_mbbox, label_lbbox, true_sbboxes, true_mbboxes, true_lbboxes)
                    loss = giou_loss + conf_loss + prob_loss

                    test_epoch_loss.append(loss.item())


            # 保存模型

            # torch.save({
            #     'model': self.model.state_dict(),
            #     'optimizer': self.optimizer.state_dict(),
            # }, f'checkpoints/epoch_{epoch}.pth')

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = args.ckpt_dir + "/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)



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

    def focal(target, actual, alpha=1.0, gamma=2.0):
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


# 使用示例
if __name__ == '__main__':
    trainer = YOLOTrainer(args, cfg)
    trainer.train()

# if __name__ == '__main__':
#     # 配置 GPU
#     device = torch.device(f'cuda:{args.gpu_id}' if args.use_gpu else 'cpu')
#     # 配置文件
#     if args.use_gpu == 0:
#         gpu_id = '-1'
#     else:
#         gpu_id = args.gpu_id
#         gpu_list = list()
#         gpu_ids = gpu_id.split(',')
#         for i in range(len(gpu_ids)):
#             gpu_list.append('/gpu:%d' % int(i))
#     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
#
#     exp_folder = os.path.join(args.exp_dir, 'exp_{}'.format(args.exp_num))
#
#     set_ckpt_dir = args.ckpt_dir
#     args.ckpt_dir = os.path.join(exp_folder, set_ckpt_dir)
#     if not os.path.exists(args.ckpt_dir):
#         os.makedirs(args.ckpt_dir)
#
#     config_log = os.path.join(exp_folder, 'config.txt')
#     arg_dict = args.__dict__
#     msg = ['{}: {}\n'.format(k, v) for k, v in arg_dict.items()]
#     write_mes(msg, config_log, mode='w')

