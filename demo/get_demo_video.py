import pyrealsense2 as rs
import cv2
import numpy as np

# 配置 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置颜色流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置深度流

# 启动管道
pipeline.start(config)

# 视频编码设置
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

try:
    print("Recording...")
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 转换图像
        color_image = np.asanyarray(color_frame.get_data())
        
        # 写入视频文件
        out.write(color_image)
        
        # 显示图像
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 停止管道
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped.")
