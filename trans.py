import numpy as np

# 相机内参参数
fx = 800  # 焦距（x方向）
fy = 800  # 焦距（y方向）
cx = 320  # 主点坐标（x方向）
cy = 240  # 主点坐标（y方向）

# 外参矩阵参数（假设是一个简单的旋转和平移）
r11, r12, r13 = 1, 0, 0
r21, r22, r23 = 0, 1, 0
r31, r32, r33 = 0, 0, 1
tx, ty, tz = 0, 0, 10  # 相机在世界坐标系中的位置

# 像素坐标 (u, v)
u, v = 320, 240

# 深度 Z_c（可以通过深度相机或其他方式获得）
Z_c = 5.0

# 1. 像素坐标到图像坐标
x = (u - cx) / fx
y = (v - cy) / fy

# 2. 图像坐标到相机坐标
X_c = x * Z_c
Y_c = y * Z_c

# 相机坐标系中的点
camera_coords = np.array([[X_c],
                          [Y_c],
                          [Z_c],
                          [1]])

# 创建相机内参矩阵
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# 创建外参矩阵（旋转矩阵 R 和平移向量 t）
R = np.array([[r11, r12, r13],
              [r21, r22, r23],
              [r31, r32, r33]])
t = np.array([[tx],
              [ty],
              [tz]])

# 3. 相机坐标到世界坐标
# 创建4x4的外参矩阵
RT = np.hstack((R, t))
RT = np.vstack((RT, [0, 0, 0, 1]))

# 转换到世界坐标
world_coords = np.dot(RT, camera_coords)

# 提取世界坐标系中的点
X_w, Y_w, Z_w = world_coords[:3, 0]

print(f"World coordinates: X_w={X_w}, Y_w={Y_w}, Z_w={Z_w}")
