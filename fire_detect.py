import cv2
import numpy as np

# 打开视频文件
video_path = '0705_2023(1).mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

while True:
    # 读取每一帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色和白色的HSV范围
    lower_red1 = np.array([0, 30, 200])
    upper_red1 = np.array([100, 255, 255])
    lower_red2 = np.array([160, 30, 200])
    upper_red2 = np.array([180, 255, 255])

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # 创建红色和白色的掩膜
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # 合并掩膜
    mask = mask_red1 | mask_red2 | mask_white

    # 对掩膜进行形态学操作，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow("Masked", mask)

    # 找到掩膜中的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 处理每个轮廓
    for contour in contours:
        # 计算轮廓的中心
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 在帧上绘制红点
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
            # 在命令行打印位置
            print(f"Detected light at position: ({cx}, {cy})")

    # 显示结果帧
    cv2.imshow('Frame', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
