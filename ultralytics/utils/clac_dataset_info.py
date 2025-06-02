from pathlib import Path
from typing import Dict, Union
import json
from collections import Counter
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义尺寸阈值常量 (基于相对面积)
SMALL_THRESHOLD = 0.005  # 0.5% of image area
MEDIUM_THRESHOLD = 0.10  # 10% of image area
OUTPUT_FILENAME = "dataset_status.json"


def calc_instance(label_path: Union[Path, str], class_map: Dict[int, str]) -> None:
    """计算YOLO格式数据集的类别分布和实例尺寸统计(基于相对尺寸)。

    统计内容包括:
    - 总图片数量
    - 每类出现的图片数量
    - 每类的实例总数
    - 按相对尺寸划分的实例数量(小/中/大)

    Args:
        label_path: YOLO格式标签文件夹路径
        class_map: 类别索引到名称的映射字典

    Raises:
        ValueError: 参数无效时抛出
        FileNotFoundError: 路径不存在时抛出
    """
    # 参数验证
    if not isinstance(class_map, dict) or not all(
            isinstance(k, int) and isinstance(v, str) for k, v in class_map.items()):
        raise ValueError("class_map must be a dictionary mapping integers to strings")

    label_path = Path(label_path)
    if not label_path.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_path}")

    logger.info(f"Starting analysis of labels in: {label_path}")

    # 初始化计数器
    counters = {
        'total_images': 0,  # 新增总图片计数器
        'images_per_class': Counter(),  # 重命名以更清晰
        'instances': Counter(),
        'size_distribution': {
            'small': Counter(),
            'medium': Counter(),
            'large': Counter()
        }
    }

    # 处理每个标签文件
    for label_file in label_path.rglob("*.txt"):
        counters['total_images'] += 1  # 每个文件代表一张图片

        try:
            with label_file.open('r') as f:
                lines = [line.strip() for line in f if line.strip()]

            # 解析所有有效标注
            annotations = []
            present_classes = set()  # 记录本图中出现的类别

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    if class_id not in class_map:
                        logger.warning(f"Unknown class ID {class_id} in {label_file}")
                        continue

                    w, h = map(float, parts[3:5])
                    relative_area = w * h
                    annotations.append((class_id, relative_area))
                    present_classes.add(class_id)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid annotation in {label_file}: {line} - {str(e)}")
                    continue

            if not annotations:
                continue

            # 更新每类出现的图片数
            counters['images_per_class'].update(class_map[cls] for cls in present_classes)

            # 更新实例级统计
            for class_id, area in annotations:
                class_name = class_map[class_id]
                counters['instances'].update([class_name])

                # 根据相对面积分类
                if area <= SMALL_THRESHOLD:
                    counters['size_distribution']['small'].update([class_name])
                    print(class_name, label_file)
                elif area <= MEDIUM_THRESHOLD:
                    counters['size_distribution']['medium'].update([class_name])
                else:
                    counters['size_distribution']['large'].update([class_name])

        except Exception as e:
            logger.error(f"Error processing {label_file}: {str(e)}", exc_info=True)
            continue

    # 准备结果并保存
    result = {
        "summary": {
            "total_images": counters['total_images'],
            "total_instances": sum(counters['instances'].values()),
            "size_thresholds": {
                "small": SMALL_THRESHOLD,
                "medium": MEDIUM_THRESHOLD
            }
        },
        "images_per_class": dict(counters['images_per_class']),
        "instances": dict(counters['instances']),
        "size_distribution": {
            "small": dict(counters['size_distribution']['small']),
            "medium": dict(counters['size_distribution']['medium']),
            "large": dict(counters['size_distribution']['large'])
        }
    }

    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Analysis completed. Results saved to {OUTPUT_FILENAME}")


if __name__ == "__main__":
    # 示例用法
    try:
        calc_instance(
            label_path="/home/youtian/Documents/pro/pyCode/datasets/tielu-yolo/labels/test",
            class_map={0: "person", 1: "debrisflow", 2: "rockfall"}
        )
    except Exception as e:
        logger.error(f"Failed to analyze dataset: {str(e)}", exc_info=True)
        raise

