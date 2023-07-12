import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

try:
    while True:
        # 等待获取一帧数据
        frames = pipeline.wait_for_frames()

        # 获取彩色图像帧和深度图像帧
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 将图像数据转换为可处理的格式
        color_image = np.asarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)
        # 显示彩色图像和深度图像
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_image)

        # 按下'q'键退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止并关闭RealSense相机
    pipeline.stop()
    cv2.destroyAllWindows()

