import cv2
import numpy as np
import pyrealsense2 as rs

# 配置深度和彩色流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动流
pipeline.start(config)

# 深度图像对齐到彩色图像流
align_to = rs.stream.color
align = rs.align(align_to)

# 初始化RealSense处理块
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
colorizer = rs.colorizer()

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()

        # 对帧进行对齐处理
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        # 获取RGB图像
        color_frame = aligned_color_frame
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('RGB 图像', color_image)

        # 应用滤波器以减少残影误差
        frame = aligned_depth_frame
        frame = decimation.process(frame)
        frame = depth_to_disparity.process(frame)
        frame = spatial.process(frame)
        frame = temporal.process(frame)
        frame = disparity_to_depth.process(frame)
        frame = hole_filling.process(frame)

        # 将原始深度帧转换为颜色映射并调整大小
        depth_image = np.asanyarray(frame.get_data())
        depth_image = cv2.resize(depth_image, (640, 480))  # 调整大小
        depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        cv2.imshow('去残影的深度图像', depth_colormap)

        # 阈值分割
        _, depth_thresh = cv2.threshold(depth_image, 70, 255, cv2.THRESH_BINARY_INV)
        depth_thresh = cv2.morphologyEx(depth_thresh, cv2.MORPH_OPEN, (7,7))

        # 黑色区域检测
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

        # 掩膜
        combined_mask = cv2.bitwise_and(depth_thresh, black_mask)
        cv2.imshow("mask", combined_mask)

        # 查找轮廓
        _, contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制细长杆子的矩形框
        for contour in contours:
            if cv2.contourArea(contour) > 80:  # 忽略小区域
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(h) / w
                if aspect_ratio > 5:  # 检查是否为细长物体
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # 计算轮廓内像素点的深度值均值
                    img = np.asanyarray(aligned_depth_frame.get_data())
                    mask = np.zeros(img.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    mean_depth = cv2.mean(img, mask=mask)[0]
                    # 输出结果
                    img = cv2.convertScaleAbs(img, alpha=0.03)
                    cv2.imshow("img", img)
                    print(f"Contour at ({x}, {y}), width {w}, height {h} - Mean depth: {mean_depth/10:.2f}cm")

        cv2.imshow('检测结果', color_image)

        # 检查按键，按'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止流，关闭窗口
    pipeline.stop()
    cv2.destroyAllWindows()
