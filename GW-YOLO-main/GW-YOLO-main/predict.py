import os
import cv2
from ultralytics import YOLO

# ========== 用户配置（请在此修改） ==========
WEIGHTS_PATH = r"F:\python\yolo\GW-YOLO-main\GW-YOLO-main\runs\segment\train\weights\best.pt"
SOURCE_PATH = r"F:\python\yolo\GW-YOLO-main\GW-YOLO-main\imgs\gw5.0"
OUTPUT_DIR = "runs/filtered/predict_5.0"       # 根输出目录，其下自动创建 labels/ 和 images/
CONF_THRESHOLD = 0.25
IMGSZ = 640
# ==========================================

CLASS_NAMES = ['chirp', 'noise']   # 用于显示标签

def save_results(filtered, output_dir, img_path, img_shape, orig_img):
    """保存标签文件到 labels/ 子目录，保存可视化图片到 images/ 子目录"""
    # 创建两个子目录
    label_dir = os.path.join(output_dir, "labels")
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)

    # ---- 保存标签文件 ----
    txt_path = os.path.join(label_dir, name + ".txt")
    img_w, img_h = img_shape
    with open(txt_path, 'w') as f:
        for det in filtered:
            x1, y1, x2, y2 = det['bbox']
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            f.write(f"{det['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {det['confidence']:.6f}\n")

    # ---- 绘制并保存可视化图片 ----
    img = orig_img.copy()
    h, w = img.shape[:2]
    for det in filtered:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cls_id = det['class_id']
        conf = det['confidence']
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)  # chirp红色, noise蓝色
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 标签文字背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 保存图片，保持原扩展名（若扩展名无效则默认 .jpg）
    if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
        ext = '.jpg'
    img_out_path = os.path.join(image_dir, name + ext)
    cv2.imwrite(img_out_path, img)

def main():
    model = YOLO(WEIGHTS_PATH)
    results = model.predict(source=SOURCE_PATH, conf=CONF_THRESHOLD, imgsz=IMGSZ, stream=False)

    for result in results:
        orig_h, orig_w = result.orig_shape
        orig_img = result.orig_img  # 原始BGR图像 (numpy array)

        best_chirp = None
        best_conf = -1.0
        noises = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                if cls_id == 0:  # chirp
                    if conf > best_conf:
                        best_conf = conf
                        best_chirp = {'class_id': 0, 'confidence': float(conf), 'bbox': box.tolist()}
                else:            # noise (class_id == 1)
                    noises.append({'class_id': 1, 'confidence': float(conf), 'bbox': box.tolist()})

        # 组装最终结果
        filtered = [best_chirp] + noises if best_chirp is not None else noises
        save_results(filtered, OUTPUT_DIR, result.path, (orig_w, orig_h), orig_img)

    print(f"预测完成，标签保存至 {OUTPUT_DIR}/labels，图片保存至 {OUTPUT_DIR}/images")

if __name__ == "__main__":
    main()