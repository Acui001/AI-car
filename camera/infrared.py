"""
这里是一个对相机左眼和右眼的使用示例
你能看见一下白点是因为D435i摄像头配备了红外发射器和红外摄像头，用于结构光投射和红外深度感知。
这些斑点实际上是由红外结构光投射器产生的红外点或红外斑点，用于辅助深度图像的计算和深度感知。
"""
import pyrealsense2 as rs
import numpy as np
import cv2

# 创建Pipeline对象
pipeline = rs.pipeline()

# 配置摄像头的设置
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1)  # 左相机黑白图像
config.enable_stream(rs.stream.infrared, 2)  # 右相机黑白图像

# 启动摄像头
pipeline.start(config)

try:
    while True:
        # left_image, right_image=camera_init1(pipeline)
        frames = pipeline.wait_for_frames()

        # 获取黑白图像帧,右眼
        right_frame = frames.get_infrared_frame(1)
        # 图像帧转换为numpy数组,右眼
        right_image = np.asanyarray(right_frame.get_data())

        # 获取黑白图像帧,左眼
        left_frame = frames.get_infrared_frame(2)
        # 左眼
        left_image = np.asanyarray(left_frame.get_data())

        # 显示左相机黑白图像
        cv2.imshow('Left Image', left_image)

        # 显示右相机黑白图像
        cv2.imshow('Right Image', right_image)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止并关闭摄像头
    pipeline.stop()
    cv2.destroyAllWindows()
