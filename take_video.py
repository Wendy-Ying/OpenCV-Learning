import cv2

# 设置视频捕获参数
cap = cv2.VideoCapture(1)  # 打开默认的摄像头（通常是0或者1，取决于你的系统配置）
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# 获取摄像头的分辨率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置视频编解码器和输出文件名
out_file = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编解码器
out = cv2.VideoWriter(out_file, fourcc, 20.0, (frame_width, frame_height))

# 计算录制时长（30秒）
record_duration = 30  # 单位：秒
start_time = cv2.getTickCount()  # 获取起始时间

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # 写入帧到输出视频文件
    out.write(frame)
    
    # 显示当前帧（可选）
    cv2.imshow('Recording', frame)
    
    # 计算已录制的时间
    current_time = cv2.getTickCount()
    elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
    
    # 如果达到了指定的录制时长，就退出循环
    if elapsed_time >= record_duration:
        break
    
    # 检测按键输入，按 'q' 键退出录制
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video recording saved as {out_file}")
