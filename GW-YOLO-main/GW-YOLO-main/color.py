import cv2
import numpy as np
from ultralytics import YOLO


path = r"F:\python\yolo\GW-YOLO-main\GW-YOLO-main\runs\segment\train3\weights\best.pt"
img_path = r"F:\python\yolo\GW-YOLO-main\GW-YOLO-main\train\images\1251930624_def6bd39b9a44cd065c913c306ee5d20_0_L1_png.rf.76a0c0dda0d75c6df3f62be298d96a55.jpg"
# 加载模型
model = YOLO(path)

# 读取图像
img = cv2.imread(img_path)
h, w = img.shape[:2]

# 执行预测
results = model(img)

# 用彩色掩码“涂抹”结果
if results[0].masks is not None:
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        # 随机生成颜色
        color = np.random.randint(0, 255, 3).tolist()
        mask_255 = (mask * 255).astype(np.uint8)
        # 将颜色填充到目标轮廓内
        color_mask[mask_255 > 128] = color

        # 可选：添加边界框和标签
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        cls = int(boxes.cls[i])
        conf = float(boxes.conf[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 半透明混合
    blended = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
    cv2.imshow("Colorful Masks", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
     # 若无掩码，则回退到仅绘制检测框
    annotated = results[0].plot()
    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()