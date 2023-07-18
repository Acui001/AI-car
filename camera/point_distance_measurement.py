import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
# 创建Pipeline对象
pipeline = rs.pipeline()

# 配置摄像头的设置
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动摄像头
pipeline.start(config)

# 创建点云对象
pc = rs.pointcloud()

try:
    while True:
        # 等待摄像头的数据
        frames = pipeline.wait_for_frames()

        # 获取彩色图像帧和深度图像帧
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # 将图像帧转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 生成点云数据
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        point_cloud = pc.calculate(depth_frame)

        # 将点云数据转换为numpy数组
        vertices = np.asarray(point_cloud.get_vertices(), dtype=np.float32)
        vertex_count = vertices.shape[0]

        # 将点云数据转换为Open3D格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.reshape(-1, 3))

        # 可选：将彩色图像应用于点云
        texcoords = np.asarray(point_cloud.get_texture_coordinates(), dtype=np.float32)
        colors = cv2.remap(color_image, texcoords[:, 0] * 640, texcoords[:, 1] * 480, cv2.INTER_LINEAR)
        pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3) / 255.0)

        # 可选：显示点云
        o3d.visualization.draw_geometries([pcd])

finally:
    # 关闭摄像头和窗口
    pipeline.stop()
    cv2.destroyAllWindows()
