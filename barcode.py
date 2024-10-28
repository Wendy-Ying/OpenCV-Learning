import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode # type: ignore

# 定义常量
WIDTH, HEIGHT = 300, 100  # 目标图像的宽度和高度
LOWER_YELLOW = np.array([20, 100, 100])  # 黄色的下界
UPPER_YELLOW = np.array([30, 255, 255])  # 黄色的上界
CHANGE_THRESHOLD = 100  # 变化量阈值，用于判断是否是干扰区域

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

# 读取图像
image = cv2.imread('barcode.jpg')
image_copy = image.copy()

# 转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 创建黄色掩膜
mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)

# 复制掩膜图像
mask_copy = mask.copy()

# 进行闭运算（先膨胀后腐蚀）
kernel = np.ones((13, 13), np.uint8)
mask_closed = cv2.morphologyEx(mask_copy, cv2.MORPH_CLOSE, kernel)

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
        barcode_region = image[y-10:y+h+10, x-10:x+w+10]
        barcode = get_qr_code_perspective(image, rect)
        break

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

# 将 BGR 图像转换为 RGB 图像以便 matplotlib 显示
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_copy_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
barcode_region_rgb = cv2.cvtColor(barcode_region, cv2.COLOR_BGR2RGB)

# 使用 matplotlib 显示图像
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Original Image with Barcode Detection")
axs[0, 0].axis('off')

axs[0, 1].imshow(mask, cmap='gray')
axs[0, 1].set_title("Yellow Mask")
axs[0, 1].axis('off')

axs[1, 0].imshow(cv2.cvtColor(barcode_region, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title("Detected Barcode Region")
axs[1, 0].axis('off')

axs[1, 1].imshow(cv2.cvtColor(barcode_result, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title("Decoded Barcode")
axs[1, 1].axis('off')

plt.show()
