from ultralytics import YOLO
from line_find import detect_line_segments
import cv2

# 配置
path_model = r"F:\python\yolo\GW-YOLO-main\GW-YOLO-main\runs\segment\train\weights\best.pt"
path_img = r"F:\python\yolo\GW-YOLO-main\GW-YOLO-main\prodict_model_26n_100\bns_ext_loud_65c829ff70cf73ba8820f6fb0021fc9f_png.rf.4a862530d8aa36f1aae4067aba6ad53d.jpg"
len_time = 3.00
len_freq = 1000.00
output_image_path = r"F:\python\yolo\GW-YOLO-main\GW-YOLO-main\result"

# 加载模型
model = YOLO(path_model)
results = model(path_img)


# 映射函数
def map_xp2axis(len_xay, xp_xay, xp_xay_max, axis_xay, axis_xay_max):
    x, y = [0, 1]

    # 长度计算
    axis_size = [(axis_xay_max[x] - axis_xay[x]), (axis_xay_max[y] - axis_xay[y])]  # 坐标轴长度
    box_size = [(xp_xay[x] - axis_xay[x]), (axis_xay_max[y] - xp_xay_max[y])]  # 识别框长度
    # print(f"{axis_size, box_size}")
    # 比例映射
    real_size = []
    for i in (0, 1):
        if axis_size[i] == 0:
            # 处理零尺寸情况
            real_size.append(0)
        else:
            real_size.append(box_size[i] / axis_size[i] * len_xay[i])

    return real_size


# # 提取结果
# for result in results:
#     # 获取xyxy格式的边界框坐标 (单位: 像素)
#     boxes_xyxy = result.boxes.xyxy.cpu().numpy()
#     for box in boxes_xyxy:
#         x_min, y_min, x_max, y_max = box  # x_max 右侧坐标
#         print(f"最右边时间 (像素坐标): {x_max}")

# 遍历每张图像的结果
for result in results:
    path_x, path_y = detect_line_segments(result.path)
    axis_size_min = [path_x[0], path_y[0]]
    axis_size_max = [path_x[1], path_y[-1]]

    # 获取边界框对象
    boxes = result.boxes

    # 提取 xyxy 格式的边界框（像素单位）
    xyxy_tensor = boxes.xyxy.cpu().numpy()  # 形状: [N, 4]

    # 提取置信度和类别
    conf_tensor = boxes.conf.cpu().numpy()   # 形状: [N,]
    cls_tensor = boxes.cls.cpu().numpy()     # 形状: [N,]

    # 输出每个检测框信息
    for i in range(len(xyxy_tensor)):
        x1, y1, x2, y2 = xyxy_tensor[i]
        conf = conf_tensor[i]
        cls_id = int(cls_tensor[i])
        label = result.names[cls_id]

        # opencv
        img = cv2.imread(result.path, cv2.IMREAD_COLOR)

        # if label.lower() == 'chirp' and conf > 0.2:
        if label.lower() == 'chirp':
            real_size = map_xp2axis([len_time, len_freq], [x1, y1], [x2, y2], axis_size_min, axis_size_max)
            # print(f"{[len_time, len_freq], [x1, y1], [x2, y2], axis_size_min, axis_size_max}")
            print(f"检测到: {label}, "
                  f"边界框: ({x1:.1f}, {y1:.1f}) 到 ({x2:.1f}, {y2:.1f}), "
                  f"置信度: {conf:.2f}"
                  f"发生时间:{real_size[0]:.2f}s"
                  )

            # 绘制模块
            img_for_drawing = img.copy()  # 使用 .copy() 避免修改原始数据

            # cv2.line(图像, 起点坐标(x1,y1), 终点坐标(x2,y2), 颜色(B,G,R), 线条粗细)
            start_point = (int(x2), int(path_y[0]))  # 起点 (x, y)
            end_point = (int(x2), int(path_y[-1]))  # 终点 (x, y)
            color = (0, 0, 255)  # 红色 (B,G,R)
            thickness = 2  # 线条粗细
            cv2.line(img_for_drawing, start_point, end_point, color, thickness)

            cv2.imshow("Image Window", img_for_drawing)  # "Image Window" 是窗口标题
            cv2.waitKey(0)  # 等待任意键盘输入，0表示无限等待[reference:2]
            cv2.destroyAllWindows()  # 关闭所有由OpenCV创建的窗口[reference:3]
            # 保存图片
            # cv2.imwrite(output_image_path, img_for_drawing)

        else:
            print("检测未通过")

            # python analysis.py