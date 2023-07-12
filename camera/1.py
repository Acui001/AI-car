import pyrealsense2 as rs
import numpy as np
import cv2
import torch
# 创建Pipeline对象
pipeline = rs.pipeline()

# 配置摄像头的设置
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# 启动摄像头
pipeline.start(config)

aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()
try:
    while True:
        # color_image,depth_image = camera_init(pipeline)
        # 等待摄像头的数据
        frames = pipeline.wait_for_frames()
        frames = aligned_stream.process(frames)
        # 获取彩色图像帧和深度图像帧
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # 将图像帧转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示彩色图像
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image', depth_image)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止并关闭摄像头
    pipeline.stop()
    cv2.destroyAllWindows()

