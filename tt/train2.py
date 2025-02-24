import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import shutil
import numpy as np
from tqdm import tqdm

class YoloTrain:
    def __init__(self):
        # 初始化配置
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FIRST_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT

        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150
        self.train_logdir = "./data/log/train"

        # 数据集
        self.trainset = Dataset('train')
        self.testset = Dataset('test')
        self.steps_per_period = len(self.trainset)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型
        self.model = YOLOV3().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate_init)
        self.global_step = torch.tensor(1.0, dtype=torch.float32, requires_grad=False)

        # 学习率调度
        self.warmup_steps = self.warmup_periods * self.steps_per_period
        self.train_steps = (self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period

        # 日志
        self.writer = SummaryWriter(self.train_logdir)

    def compute_lr(self):
        if self.global_step < self.warmup_steps:
            return self.global_step / self.warmup_steps * self.learn_rate_init
        else:
            return self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * (
                        1 + torch.cos((self.global_step - self.warmup_steps) / (self.train_steps - self.warmup_steps) * torch.tensor(np.pi)))

    def train(self):
        train_loader = DataLoader(self.trainset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(self.testset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

        # 加载预训练权重
        if os.path.exists(self.initial_weight):
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.model.load_state_dict(torch.load(self.initial_weight))
        else:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            self.model.train()
            train_epoch_loss = []
            pbar = tqdm(train_loader)
            for batch_idx, (input_data, label_sbbox, label_mbbox, label_lbbox, true_sbboxes, true_mbboxes, true_lbboxes) in enumerate(pbar):
                input_data = input_data.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                true_sbboxes = true_sbboxes.to(self.device)
                true_mbboxes = true_mbboxes.to(self.device)
                true_lbboxes = true_lbboxes.to(self.device)

                # 前向传播
                output = self.model(input_data, trainable=True)
                giou_loss, conf_loss, prob_loss, recovery_loss = self.model.compute_loss(
                    label_sbbox, label_mbbox, label_lbbox, true_sbboxes, true_mbboxes, true_lbboxes)
                loss = giou_loss + conf_loss + prob_loss

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 更新全局步数
                self.global_step += 1

                # 学习率更新
                lr = self.compute_lr()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # 记录日志
                train_epoch_loss.append(loss.item())
                pbar.set_description("train loss: %.2f" % (loss.item()))

            # 测试阶段
            self.model.eval()
            test_epoch_loss = []
            with torch.no_grad():
                for batch_idx, (input_data, label_sbbox, label_mbbox, label_lbbox, true_sbboxes, true_mbboxes, true_lbboxes) in enumerate(test_loader):
                    input_data = input_data.to(self.device)
                    label_sbbox = label_sbbox.to(self.device)
                    label_mbbox = label_mbbox.to(self.device)
                    label_lbbox = label_lbbox.to(self.device)
                    true_sbboxes = true_sbboxes.to(self.device)
                    true_mbboxes = true_mbboxes.to(self.device)
                    true_lbboxes = true_lbboxes.to(self.device)

                    # 前向传播
                    output = self.model(input_data, trainable=False)
                    giou_loss, conf_loss, prob_loss, recovery_loss = self.model.compute_loss(
                        label_sbbox, label_mbbox, label_lbbox, true_sbboxes, true_mbboxes, true_lbboxes)
                    loss = giou_loss + conf_loss + prob_loss

                    test_epoch_loss.append(loss.item())

            # 计算平均损失
            train_epoch_loss = np.mean(train_epoch_loss)
            test_epoch_loss = np.mean(test_epoch_loss)

            # 保存模型
            ckpt_file = args.ckpt_dir + "/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            torch.save(self.model.state_dict(), ckpt_file)

        self.writer.close()

# 示例运行代码
if __name__ == "__main__":
    trainer = YoloTrain()
    trainer.train()