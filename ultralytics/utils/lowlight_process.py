import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
from collections import defaultdict


def apply_lowlight_and_save(input_dir, output_dir, lowlight_param=7.5, batch_size=16):
    """
    批量处理图像，保留原图分辨率，应用低光变换并保存，文件名与原图一致。

    Args:
        input_dir (str): 输入图像目录
        output_dir (str): 输出图像目录
        lowlight_param (float): 低光变换参数，默认 7.5
        batch_size (int): 批量处理大小
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

    # 按分辨率分组图像
    size_groups = defaultdict(list)  # { (height, width): [(img_path, filename), ...] }
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        width, height = img.size  # 原图分辨率 (width, height)
        size_groups[(height, width)].append((img_path, os.path.basename(img_path)))

    total_images = len(img_paths)
    processed_images = 0
    batch_count = 0

    # 处理每个分辨率组
    for size, path_filename_pairs in size_groups.items():
        height, width = size
        print(f"处理分辨率: {width}x{height}")

        images = []
        filenames = []

        for idx, (img_path, filename) in enumerate(path_filename_pairs):
            # 加载图像并转换为张量
            img = Image.open(img_path).convert('RGB')
            img_tensor = to_tensor(img)  # 形状：(3, height, width)
            images.append(img_tensor)
            filenames.append(filename)

            # 达到批量大小或组内最后一张图像
            if len(images) == batch_size or idx == len(path_filename_pairs) - 1:
                # 转换为批量张量
                batch_images = torch.stack(images)  # 形状：(batch_size, 3, height, width)

                # 应用低光变换
                with torch.no_grad():
                    lowlight_images = torch.pow(batch_images, lowlight_param)

                # 保存处理后的图像
                lowlight_images = lowlight_images.cpu().numpy()  # 转换为 numpy
                for i in range(lowlight_images.shape[0]):
                    img = lowlight_images[i].transpose(1, 2, 0)  # 转换为 (H, W, 3)
                    img = (img * 255).astype(np.uint8)  # 转换为 [0, 255]
                    img = img[:, :, ::-1]  # RGB 转 BGR（OpenCV 格式）
                    img = np.ascontiguousarray(img)  # 确保内存连续
                    save_path = os.path.join(output_dir, filenames[i])
                    cv2.imwrite(save_path, img)
                    print(f"已保存图像：{save_path}")

                processed_images += len(images)
                batch_count += 1
                images = []
                filenames = []

    print(f"处理完成，共 {batch_count} 个批次，总计 {processed_images}/{total_images} 张图像。")


if __name__ == "__main__":
    # 示例用法
    input_dir = "/home/ncst/YOU/pro/Dedark-YOLO/tielu-yolo/images/test"  # 输入图像目录
    output_dir = "/home/ncst/YOU/pro/Dedark-YOLO/tielu-yolo/images/test_dark"  # 输出图像目录
    lowlight_param = 5.0  # 低光参数
    batch_size = 16  # 批量大小

    apply_lowlight_and_save(input_dir, output_dir, lowlight_param, batch_size)