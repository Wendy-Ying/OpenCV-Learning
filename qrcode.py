import cv2
import numpy as np
from pyzbar.pyzbar import decode # type: ignore
import matplotlib.pyplot as plt

def get_qr_code_perspective(image, pts):
    # 定义目标图像的尺寸
    width, height = 300, 300

    # 定义目标图像上的点
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst_pts)

    # 进行透视变换
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def decode_qr_code(image_path):
    # 加载图像并转换为灰度图
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用大津法（Otsu's thresholding）进行阈值分割
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 解码二维码
    decoded_objs = decode(binary)

    # 在变换后的图像上绘制边界框并进行透视变换
    warped = None
    for obj in decoded_objs:
        pts = np.array(obj.polygon, dtype=np.float32)
        if len(pts) == 4:
            warped = get_qr_code_perspective(image, pts)
            decoded_objs_warped = decode(warped)
            for obj_warped in decoded_objs_warped:
                print("二维码内容:", obj_warped.data.decode('utf-8'))

            pts = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=5)

    # 显示所有图像
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original Image with Bounding Box")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(gray, cmap='gray')
    axs[0, 1].set_title("Gray Image")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(binary, cmap='gray')
    axs[1, 0].set_title("Binary Image")
    axs[1, 0].axis('off')

    if warped is not None:
        axs[1, 1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title("Warped QR Code")
        axs[1, 1].axis('off')

    plt.show()

    return image

# 执行二维码检测、透视变换和解码
decoded_image = decode_qr_code('qrcode.jpg')
