import cv2
import numpy as np

# 加载模板图像并缩小
template = cv2.imread('template_2019.jpg')
scale_factor = 0.3
template = cv2.resize(template, (0, 0), fx=scale_factor, fy=scale_factor)  # 缩小模板图像
w, h = template.shape[1], template.shape[0]

# 打开视频文件
cap = cv2.VideoCapture('0705_2019.mp4')

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 设置跳帧数
n = 3
frame_count = 0

# 用于存储上一帧的匹配结果
prev_top_left = None
prev_bottom_right = None

# 初始置信度阈值
confidence_threshold = 0.32

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % n == 0:
        # 缩小视频帧
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # 转换为HSV颜色空间
        hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

        # 定义灰色的HSV范围
        lower_gray = np.array([0, 0, 110])
        upper_gray = np.array([180, 50, 160])

        # 定义红色的HSV范围
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # 创建掩膜，分离灰色
        mask_gray = cv2.inRange(hsv_frame, lower_gray, upper_gray)

        # 创建掩膜，分离红色
        mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

        # 合并红色掩膜
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 合并灰色和红色掩膜
        mask = cv2.bitwise_or(mask_gray, mask_red)

        # 将灰色部分设置为 (178, 178, 178)
        gray_color = np.array([178, 178, 178], dtype=np.uint8)
        small_frame[mask_gray > 0] = gray_color

        # 将红色部分设置为 (255, 0, 0)
        red_color = np.array([0, 0, 255], dtype=np.uint8)
        small_frame[mask_red > 0] = red_color

        # 将掩膜应用到原始图像上
        masked_frame = cv2.bitwise_and(small_frame, small_frame, mask=mask)

        # 形态学闭运算填补孔洞
        kernel = np.ones((5, 5), np.uint8)
        masked_frame_closed = cv2.morphologyEx(masked_frame, cv2.MORPH_CLOSE, kernel)

        # 将掩膜之外的部分设置为白色
        white_background = np.full(small_frame.shape, 255, dtype=np.uint8)
        inverse_mask = cv2.bitwise_not(mask)
        white_frame = cv2.bitwise_and(white_background, white_background, mask=inverse_mask)
        masked_frame_with_white_bg = cv2.add(masked_frame, white_frame)

        best_match_val = -1
        best_match_loc = None
        best_scale = 1.0

        # 多尺度模板匹配
        for scale in np.linspace(0.1, 1.3, 9):
            resized_template = cv2.resize(template, (int(w * scale), int(h * scale)))
            if resized_template.shape[0] > masked_frame_with_white_bg.shape[0] or resized_template.shape[1] > masked_frame_with_white_bg.shape[1]:
                continue

            res = cv2.matchTemplate(masked_frame_with_white_bg, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_scale = scale

        crop_template = template[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        for scale in np.linspace(1.2, 1.8, 4):
            resized_template = cv2.resize(crop_template, (int(w * scale), int(h * scale)))
            if resized_template.shape[0] > masked_frame_with_white_bg.shape[0] or resized_template.shape[1] > masked_frame_with_white_bg.shape[1]:
                continue

            res = cv2.matchTemplate(masked_frame_with_white_bg, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_scale = scale

        # 打印当前帧的最大置信度
        print(f"Frame {frame_count}: Max confidence: {best_match_val}")

        if best_match_loc is not None and best_match_val >= confidence_threshold:
            prev_top_left = (int(best_match_loc[0] / scale_factor), int(best_match_loc[1] / scale_factor))
            prev_bottom_right = (int((best_match_loc[0] + w * best_scale) / scale_factor), int((best_match_loc[1] + h * best_scale) / scale_factor))
            center_x = (prev_top_left[0] + prev_bottom_right[0]) // 2
            center_y = (prev_top_left[1] + prev_bottom_right[1]) // 2
            print(f"Frame {frame_count}: Center of the box: ({center_x}, {center_y})")
        else:
            prev_top_left = None
            prev_bottom_right = None

    # 绘制上一帧的匹配结果
    if prev_top_left is not None and prev_bottom_right is not None:
        cv2.rectangle(frame, prev_top_left, prev_bottom_right, (0, 255, 0), 2)
        center_x = (prev_top_left[0] + prev_bottom_right[0]) // 2
        center_y = (prev_top_left[1] + prev_bottom_right[1]) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # 显示结果
    cv2.imshow('Frame', frame)
    if frame_count % n == 0:
        cv2.imshow('Masked Frame with White Background', masked_frame_with_white_bg)

    # 按视频帧率延迟显示
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
