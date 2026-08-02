import argparse
from pathlib import Path

import cv2
import numpy as np


# letterbox
def letterbox_image(img_path, target_size=640, save_path=None):
    """
    将图片通过 letterbox 方式调整为 target_size x target_size
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图片: {img_path}")
        return

    h, w = img.shape[:2]
    # 计算缩放比例
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建目标尺寸的纯色背景 (灰色 114,114,114)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    # 计算粘贴位置 (居中)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    if save_path:
        cv2.imwrite(str(save_path), canvas)
    return canvas


# resize_fill
def resize_fill(img_path, target_size=640, save_path=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    if save_path:
        cv2.imwrite(str(save_path), resized)


def batch_convert(input_dir, output_dir, target_size=640, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() in extensions:
            save_path = output_dir / img_path.name

            # 伸缩方式调用
            # letterbox_image(img_path, target_size, save_path)
            resize_fill(img_path, target_size, save_path)
            print(f"已转换: {img_path.name} -> {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resize all images in a directory to square images.")
    parser.add_argument("input_dir", type=Path, help="Directory containing source images")
    parser.add_argument("output_dir", type=Path, help="Directory for converted images")
    parser.add_argument("--size", type=int, default=640, help="Output width and height (default: 640)")
    args = parser.parse_args()
    batch_convert(args.input_dir, args.output_dir, args.size)
