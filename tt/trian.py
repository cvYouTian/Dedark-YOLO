import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config_lowlight import args, cfg
from utils import read_class_names
from dataset import CustomDataset
from .model import YOLOV3


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


class YOLOTrainer:
    def __init__(self, args):
        # 初始化参数
        self.model = YOLOV3(num_class=len(read_class_names(args.YOLO.CLASSES)), isp_flag=True).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.TRAIN.LEARN_RATE_INIT)
        self.criterion = YOLOLoss()  # 需要自定义损失函数

        # 数据加载
        self.train_loader = DataLoader(CustomDataset("train", cfg),
                                       batch_size=args.TRAIN.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(CustomDataset("test", cfg),
                                     batch_size=args.TEST.BATCH_SIZE, shuffle=False)

        # 学习率调度
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=args.TRAIN.LEARN_RATE_END)

    def train(self):
        for epoch in range(args.epochs):
            self.model.train()
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                # 低光照增强
                if args.lowlight_FLAG:
                    lowlight_param = torch.rand(1) * 5 + 5  # [5, 10]
                    enhanced_images = images ** lowlight_param.item()
                else:
                    enhanced_images = images

                # 前向传播
                pred_s, pred_m, pred_l, recovery = self.model(enhanced_images)

                # 计算损失
                loss_dict = self.criterion(pred_s, pred_m, pred_l, targets, recovery, images)
                total_loss = loss_dict['total']

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # 记录日志...

            # 验证步骤
            self.validate()

            # 保存模型
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, f'checkpoints/epoch_{epoch}.pth')


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化损失参数...

    def forward(self, pred_s, pred_m, pred_l, targets, recovery, clean_images):
        # 实现 GIoU 损失、置信度损失、类别损失和恢复损失
        giou_loss = ...
        conf_loss = ...
        cls_loss = ...
        recovery_loss = nn.MSELoss()(recovery, clean_images)

        return {
            'giou': giou_loss,
            'conf': conf_loss,
            'cls': cls_loss,
            'recovery': recovery_loss,
            'total': giou_loss + conf_loss + cls_loss + recovery_loss
        }


# 数据加载器
class LowlightDataset(torch.utils.data.Dataset):
    def __init__(self, ...):

    # 实现数据加载和预处理

    def __getitem__(self, idx):
        # 返回增强后的图像和标注
        return image, targets


# 使用示例
if __name__ == '__main__':
    trainer = YOLOTrainer(args)
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

