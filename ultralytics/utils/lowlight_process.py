import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np


def apply_lowlight_and_save(input_dir, output_dir, lowlight_param=7.5, batch_size=16, resize_size=(640, 640)):
    """
    批量处理图像，调整尺寸后应用低光变换并保存，文件名与原图一致。

    Args:
        input_dir (str): 输入图像目录
        output_dir (str): 输出图像目录
        lowlight_param (float): 低光变换参数，默认 7.5
        batch_size (int): 批量处理大小
        resize_size (tuple): 调整后的图像尺寸 (height, width)，默认 (640, 640)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    img_extensions = ('.jpg', '.jpeg', '.png')
    img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                 if f.lower().endswith(img_extensions)]

    if not img_paths:
        print("输入目录中未找到任何图像！")
        return

    # 初始化变换
    to_tensor = transforms.ToTensor()  # 转换为 [0, 1] 张量
    resize = transforms.Resize(resize_size)  # 调整图像尺寸
    images = []
    filenames = []
    batch_count = 0

    for idx, img_path in enumerate(img_paths):
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        # 调整图像尺寸并转换为张量
        img_resized = resize(img)  # 调整到指定尺寸
        img_tensor = to_tensor(img_resized)  # 形状：(3, resize_size[0], resize_size[1])
        images.append(img_tensor)
        filenames.append(os.path.basename(img_path))

        # 达到批量大小时处理
        if len(images) == batch_size or idx == len(img_paths) - 1:
            # 转换为批量张量
            batch_images = torch.stack(images)  # 形状：(batch_size, 3, H, W)

            # 应用低光变换
            with torch.no_grad():
                lowlight_images = torch.pow(batch_images, lowlight_param)

            # 保存处理后的图像
            lowlight_images = lowlight_images.cpu().numpy()  # 转换为 numpy
            for i in range(lowlight_images.shape[0]):
                img = lowlight_images[i].transpose(1, 2, 0)  # 转换为 (H, W, 3)
                img = (img * 255).astype(np.uint8)  # 转换为 [0, 255]
                img = img[:, :, ::-1]  # RGB 转 BGR（OpenCV 格式）
                save_path = os.path.join(output_dir, filenames[i])
                cv2.imwrite(save_path, img)
                print(f"已保存图像：{save_path}")

            # 清空当前批量
            images = []
            filenames = []
            batch_count += 1

    print(f"处理完成，共 {batch_count} 个批次，总计 {len(img_paths)} 张图像。")


if __name__ == "__main__":
    # 示例用法
    input_dir = "/home/youtian/Documents/pro/pyCode/datasets/xxxx/TLdevkit/TL2025/JPEGImages"  # 输入图像目录
    output_dir = "/home/youtian/Documents/pro/pyCode/datasets/darkpic"  # 输出图像目录
    lowlight_param = 7.5  # 低光参数
    batch_size = 16  # 批量大小
    resize_size = (640, 640)  # 调整后的图像尺寸

    apply_lowlight_and_save(input_dir, output_dir, lowlight_param, batch_size, resize_size)