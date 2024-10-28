import cv2
import numpy as np
from pyzbar.pyzbar import decode

# 定义常量
WIDTH, HEIGHT = 300, 100  # 目标图像的宽度和高度
LOWER_YELLOW = np.array([20, 100, 100])  # 黄色的下界
UPPER_YELLOW = np.array([30, 255, 255])  # 黄色的上界
CHANGE_THRESHOLD = 100  # 变化量阈值，用于判断是否是干扰区域

frame_counter = 0

def get_qr_code_perspective(image, pts):
    """获取透视变换后的条形码区域"""
    dst_pts = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (WIDTH, HEIGHT))
    return warped

def order_points(pts):
    """对四个点进行排序，以便进行透视变换"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# 打开保存的视频文件
video_path = 'barcode.mp4'  # 替换为你保存的视频文件路径
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    
    # 处理每隔一帧
    if frame_counter % 2 == 0:
        image_copy = frame.copy()

        # 转换为HSV颜色空间
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建黄色掩膜
        mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)

        # 进行闭运算（先膨胀后腐蚀）
        kernel = np.ones((13, 13), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 计算闭运算前后的差异
        difference = cv2.absdiff(mask, mask_closed)
        _, thresh = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
        change_amount = cv2.countNonZero(thresh)

        # 查找闭运算后的轮廓
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 处理每个轮廓
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 条形码形状假设为宽度大于高度的长方形，并且变化量大于阈值
            if w > h and change_amount > CHANGE_THRESHOLD:
                # 获取最小面积旋转矩形
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                # 绘制轮廓
                cv2.drawContours(image_copy, [box], 0, (0, 255, 0), 2)
                
                # 对四个点进行排序，以便用于透视变换
                rect = order_points(box)
                barcode_region = frame[y-10:y+h+10, x-10:x+w+10]
                barcode = get_qr_code_perspective(frame, rect)
                break

        if 'barcode' in locals():  # 如果检测到了条形码
            # 将条形码转换为灰度图像，并创建掩膜
            barcode_gray = cv2.cvtColor(barcode, cv2.COLOR_BGR2GRAY)
            _, barcode_mask = cv2.threshold(barcode_gray, 50, 255, cv2.THRESH_BINARY)

            # 创建白色背景图像，并将条形码复制到上面
            white_background = np.full_like(barcode, 255)
            barcode_result = cv2.bitwise_or(barcode, white_background, mask=barcode_mask)

            # 解码条形码
            decoded_objects = decode(barcode_result)
            for obj in decoded_objects:
                print('条形码类型:', obj.type)
                print('条形码数据:', obj.data)

            # 使用 OpenCV 显示处理过程中的图像
            cv2.imshow('Original Image with Barcode Detection', image_copy)
            cv2.imshow('Yellow Mask', mask)
            cv2.imshow('Detected Barcode Region', barcode_region)
            cv2.imshow('Decoded Barcode', barcode_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放视频对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
