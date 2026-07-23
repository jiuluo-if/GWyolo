import cv2
import numpy as np
from pathlib import Path


images_original = r"F:\img\gw5.0" # 原始文件夹
images_640 = r"F:\python\yolo\GW-YOLO-main\GW-YOLO-main\imgs\gw5.0" # 输出文件夹


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
    # 修改路径
    input_folder = images_original
    output_folder = images_640
    batch_convert(input_folder, output_folder)