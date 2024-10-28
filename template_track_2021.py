import cv2
import numpy as np

# 加载模板图像
template = cv2.imread('template_2021.jpg', 0)
w, h = template.shape[::-1]

# 打开视频文件
cap = cv2.VideoCapture('0705_2021.mp4')
# cap = cv2.VideoCapture(0)

# 设置跳帧数
n = 4
frame_count = 0

# 用于存储上一帧的匹配结果
prev_top_left = None
prev_bottom_right = None

# 设置置信度阈值
confidence_threshold = 0.35

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % n == 0:
        # 转换为HSV颜色空间
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义黑色的HSV范围
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 100])

        # 创建掩膜，保留黑色区域
        mask = cv2.inRange(hsv_frame, lower_black, upper_black)

        # 反转掩膜，得到非黑色区域为黑色的掩膜
        mask_inv = cv2.bitwise_not(mask)

        # 将掩膜应用到原始图像上，将非黑色区域变为白色
        white_background = np.ones_like(frame) * 255
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        white_masked_frame = cv2.add(masked_frame, white_background, mask=mask_inv)

        # 转换掩膜后的图像为灰度图像
        gray_masked_frame = cv2.cvtColor(white_masked_frame, cv2.COLOR_BGR2GRAY)

        best_match_val = -1
        best_match_loc = None
        best_scale = 1.0

        # 多尺度模板匹配
        for scale in np.linspace(0.2, 1.2, 8):
            resized_template = cv2.resize(template, (int(w * scale), int(h * scale)))
            if resized_template.shape[0] > gray_masked_frame.shape[0] or resized_template.shape[1] > gray_masked_frame.shape[1]:
                continue

            res = cv2.matchTemplate(gray_masked_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_scale = scale
                
        crop_template = template[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        for scale in np.linspace(1.2, 1.8, 2):
            resized_template = cv2.resize(crop_template, (int(w * scale), int(h * scale)))
            if resized_template.shape[0] > gray_masked_frame.shape[0] or resized_template.shape[1] > gray_masked_frame.shape[1]:
                continue

            res = cv2.matchTemplate(gray_masked_frame, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_scale = scale
        
        # 打印当前帧的最大置信度
        print(f"Frame {frame_count}: Max confidence: {best_match_val}")

        if best_match_loc is not None and best_match_val >= confidence_threshold:
            prev_top_left = best_match_loc
            prev_bottom_right = (prev_top_left[0] + int(w * best_scale), prev_top_left[1] + int(h * best_scale))
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
        cv2.imshow('Masked Frame', white_masked_frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
