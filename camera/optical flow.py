import pyrealsense2 as rs
import numpy as np
import cv2

# 创建Pipeline对象
pipeline = rs.pipeline()

# 配置摄像头的设置
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动摄像头
pipeline.start(config)


def get_distance(depth_frame, x, y):
    # 获取深度图像中指定位置的距离值
    x, y = int(x), int(y)
    depth_value = depth_frame.get_distance(x, y)

    return depth_value


def mark_point(image,distance, x, y):
    # 绘制一个红色的圆圈标记指定的点
    x, y = int(x), int(y)
    marked_image = np.copy(image)
    cv2.circle(marked_image, (x, y), 5, (0, 0, 255), -1)
    if distance !=0:
        cv2.putText(marked_image, str(distance)+"m", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(marked_image, 'out of range', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return marked_image


try:
    while True:

        # 等待摄像头的数据
        frames = pipeline.wait_for_frames()
        # 获取彩色图像帧和深度图像帧
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        print("depth_frame")
        print(type(depth_frame))
        print("color_frame")
        print(type(color_frame))
        # 将图像帧转换为numpy数组(这步只是为了显示的时候用)
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        print("depth_image", type(depth_image))
        # 获取指定位置的距离值
        x = 320
        y = 240
        distance = get_distance(depth_frame, x, y)

        # 在深度图像和彩色图像上标记指定位置
        depth_image_marked = mark_point(depth_image,distance, x, y)
        color_image_marked = mark_point(color_image,distance, x, y)

        # 显示彩色图像和深度图像
        cv2.imshow('Color Image', color_image_marked)
        cv2.imshow('Depth Image', depth_image_marked)

        # 打印距离值
        print("距离:", distance, "米")

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止并关闭摄像头
    pipeline.stop()
    cv2.destroyAllWindows()

