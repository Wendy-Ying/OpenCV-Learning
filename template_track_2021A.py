import cv2
import numpy as np

# 加载模板图像
template = cv2.imread('template_A.jpg', 0)

# 初始化 ORB 检测器，增加特征点数量
orb = cv2.ORB_create(nfeatures=1000)

# 检测并计算模板图像的特征和描述子
keypoints_template, descriptors_template = orb.detectAndCompute(template, None)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# 打开视频文件
cap = cv2.VideoCapture('0705_2021A.mp4')
# cap = cv2.VideoCapture(0)

# 设置跳帧数
frame_n = 1
frame_count = 0

# 用于存储上一帧的匹配结果
prev_top_left = None
prev_bottom_right = None

# 设置匹配数量阈值
match_threshold = 30

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_n == 0:
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

        # 检测并计算当前帧的特征和描述子
        keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_masked_frame, None)

        if descriptors_frame is not None and descriptors_template is not None:
            # 使用KNN匹配
            matches = bf.knnMatch(descriptors_template, descriptors_frame, k=2)

            # 只保留好的匹配点
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # 打印当前帧的匹配数量
            print(f"Frame {frame_count}: Number of good matches: {len(good_matches)}")

            if len(good_matches) > match_threshold:
                # 获取匹配关键点的坐标
                src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # 计算变换矩阵
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # 应用变换矩阵到模板图像的四个角点，获得在当前帧中的位置
                    h, w = template.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    # 计算包围盒
                    prev_top_left = (int(dst[0][0][0]), int(dst[0][0][1]))
                    prev_bottom_right = (int(dst[2][0][0]), int(dst[2][0][1]))
                    center_x = (prev_top_left[0] + prev_bottom_right[0]) // 2
                    center_y = (prev_top_left[1] + prev_bottom_right[1]) // 2
                    print(f"Frame {frame_count}: Center of the box: ({center_x}, {center_y})")
                else:
                    prev_top_left = None
                    prev_bottom_right = None
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
    if frame_count % frame_n == 0:
        cv2.imshow('Masked Frame', white_masked_frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
