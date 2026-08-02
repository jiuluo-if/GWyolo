import cv2
import numpy as np


# test
# path = r"F:\python\yolo\GW-YOLO-main\GW-YOLO-main\prodict_model_26n_100\0e454f82f22eed829dff65541d76f521_png.rf.f2e3c0550d22f3ec8808c7d0cbc03db4.jpg"


def detect_line_segments(image_path, min_line_length=300, max_line_gap=20, threshold=120):
    # 1. 读取图像并转为灰度图
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 2. 高斯模糊降噪（推荐，可以减少误检）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 边缘检测（直线检测前的关键步骤）
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # 4. 执行概率霍夫直线检测，返回线段端点坐标
    lines = cv2.HoughLinesP(
        edges,
        rho=1,  # 距离精度（像素）
        theta=np.pi / 180,  # 角度精度（弧度，即1度）
        threshold=threshold,  # 累加器阈值，越大检测越严格
        minLineLength=min_line_length,  # 最小线段长度（像素）
        maxLineGap=max_line_gap  # 最大允许断裂间隙（像素）
    )

    # 5. 在原图上绘制检测到的线段并打印端点坐标
    result_img = img.copy()
    x = []
    y = []
    if lines is not None:
        # print(f"共检测到 {len(lines)} 条线段:")
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            # 打印起点和终点坐标
            # print(f"线段 {i + 1}: 起点({x1}, {y1}) -> 终点({x2}, {y2})",end="")
            if x1 == x2:
                x.append(x1)
            elif y1 == y2:
                y.append(y1)
            # 在原图上绘制绿色线段
            # cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("未检测到任何线段")
    x.sort()
    y.sort()
    # return result_img, lines, x, y
    return x, y

# 调用函数
# if __name__ == "__main__":
#     # 替换为你的图片路径
#     output_image, detected_lines, path_x, path_y = detect_line_segments(path)
#
#     # 显示结果
#     cv2.imshow("Detected Lines", output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
