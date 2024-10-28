import cv2
import numpy as np

# 读取视频文件
cap = cv2.VideoCapture('0703.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    # 将帧转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 定义红色和粉红色的HSV范围
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([150, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    # 合并两个范围的掩膜
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 高斯模糊处理
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # 进行闭运算以填补孔洞
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 使用霍夫圆变换检测圆形
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=40, maxRadius=200)
    
    # 如果检测到圆形，则绘制圆形及圆心
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            # 输出圆心坐标
            print(f"Detected circle at: ({x}, {y}), radius: {r}")
            # 绘制圆形边界
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            # 绘制圆心
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # 红色圆心
    
    # 显示原始视频帧及检测结果
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    
    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
