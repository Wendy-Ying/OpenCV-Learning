import cv2
import numpy as np
import os

# 输入和输出目录
dir = "captured_images"
calibration_result = os.path.join(dir, "calibration_result.txt")

# 获取文件夹中所有图像文件名
image_files = [f for f in os.listdir(dir) if f.endswith(('jpg', 'png', 'jpeg'))]

if not image_files:
    print("没有找到文件")
    exit(-1)

# 初始化变量
image_nums = 0
image_size = None
points_per_row = 11
points_per_col = 8
corner_size = (points_per_row, points_per_col)
points_all_images = []

# 从每张图片中提取角点
print("开始提取角点……")
for image_file_name in image_files:
    image_nums += 1
    image_raw = cv2.imread(os.path.join(dir, image_file_name))
    if image_nums == 1:
        image_size = (image_raw.shape[1], image_raw.shape[0])
        print(f"image_size.width = {image_size[0]}")
        print(f"image_size.height = {image_size[1]}")

    image_gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    success, points_per_image = cv2.findChessboardCorners(image_gray, corner_size)

    if not success:
        print(f"无法找到角点: {image_file_name}")
        continue
    else:
        points_per_image = cv2.cornerSubPix(image_gray, points_per_image, (5, 5), (-1, -1), 
                                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        points_all_images.append(points_per_image)
        cv2.drawChessboardCorners(image_raw, corner_size, points_per_image, success)
        cv2.namedWindow("Camera calibration", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Camera calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Camera calibration", image_raw)

        cv2.waitKey(0)

cv2.destroyAllWindows()

# 输出图像数量
image_sum_nums = len(points_all_images)
print(f"image_sum_nums = {image_sum_nums}")

# 准备标定数据
block_size = (30, 30)
camera_K = np.zeros((3, 3), dtype=np.float32)
distCoeffs = np.zeros((5,), dtype=np.float32)
rotationVecs = []
translationVecs = []

points3D_per_image = [np.array([[j*block_size[0], i*block_size[1], 0] 
                                for j in range(points_per_row) 
                                for i in range(points_per_col)], dtype=np.float32)]
points3D_all_images = points3D_per_image * image_nums

# 相机标定
ret, camera_K, distCoeffs, rotationVecs, translationVecs = cv2.calibrateCamera(points3D_all_images, points_all_images, 
                                                                                image_size, camera_K, distCoeffs)

# 评估标定结果
total_err = 0
print("\n\t每幅图像的标定误差:")
with open(calibration_result, 'w') as fout:
    fout.write("每幅图像的标定误差：\n")
    for i in range(image_nums):
        points_reproject = cv2.projectPoints(points3D_all_images[i], rotationVecs[i], translationVecs[i], 
                                             camera_K, distCoeffs)[0]
        err = cv2.norm(points_all_images[i], points_reproject, cv2.NORM_L2) / len(points_reproject)
        total_err += err
        print(f"第{i+1}幅图像的平均误差为： {err}像素")
        fout.write(f"第{i+1}幅图像的平均误差为： {err}像素\n")
    
    avg_err = total_err / image_nums
    print(f"总体平均误差为： {avg_err}像素")
    fout.write(f"总体平均误差为： {avg_err}像素\n")
    print("评价完成！")

    # 将标定结果写入文件
    fout.write("\n相机内参数矩阵:\n")
    fout.write(f"{camera_K}\n\n")
    fout.write("畸变系数：\n")
    fout.write(f"{distCoeffs}\n\n\n")

    for i in range(image_nums):
        rotationMat = cv2.Rodrigues(rotationVecs[i])[0]
        fout.write(f"第{i+1}幅图像的旋转矩阵为：\n")
        fout.write(f"{rotationMat}\n")
        fout.write(f"第{i+1}幅图像的平移向量为：\n")
        fout.write(f"{translationVecs[i]}\n\n")
