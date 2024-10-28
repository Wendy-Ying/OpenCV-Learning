import cv2
import time
import os

# 设置总共拍摄的图像数量
total_images = 120

# 创建一个文件夹来保存捕获的图像
save_path = 'captured_images'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 打开默认的摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

img_counter = 0
while img_counter < total_images:
    # 读取摄像头帧
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # 生成图像文件名
    img_name = os.path.join(save_path, f"image_{img_counter}.png")

    # 保存图像
    cv2.imwrite(img_name, frame)
    print(f"{img_name} saved!")

    # 显示当前帧
    cv2.imshow("frame", frame)
    
    # 等待一段时间（1毫秒）
    cv2.waitKey(1)

    # 增加计数器
    img_counter += 1

    # 等待2秒
    time.sleep(0.8)

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
