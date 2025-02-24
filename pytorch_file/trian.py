import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from configs.config_lowlight import args, cfg
import time
from network.loss import YOLOLoss
from Utils.utils import get_anchors
import os
from Utils.utils import read_class_names, write_mes
from data.dataset import CustomDataset
from network.model import YOLOV3


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
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2,
                                                     eta_min=args.TRAIN.LEARN_RATE_END)

    def train(self):
        if os.path.exists(self.initial_weight):
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.model.load_state_dict(torch.load(self.initial_weight))

        else:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(args.epochs):
            # consume的功能未实现
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
