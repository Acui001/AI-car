import pyrealsense2 as rs
import numpy as np
import open3d as o3d
def depth_image_to_point_cloud(depth_image, depth_frame):
    # 获取相机内参
    intrinsic = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

    # 创建点云坐标数组
    points = []

    # 遍历深度图像中的每个像素
    for y in range(0, depth_frame.height):
        for x in range(0, depth_frame.width):
            # 获取像素的深度值
            depth = depth_image[y, x]

            # 将像素坐标转换为点云坐标
            point = rs.rs2_deproject_pixel_to_point(intrinsic, [x, y], depth)

            # 添加点云坐标到数组中
            points.append(point)

    return points
# 配置相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# 开始捕获点云数据
pipeline.start(config)

try:
    # 循环捕获数据并转换为点云
    while True:
        # 等待新的帧
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # 将深度图像转换为Numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())

        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(depth_image_to_point_cloud(depth_image, depth_frame))

        # 可视化点云
        o3d.visualization.draw_geometries([pcd])

finally:
    # 停止并关闭相机
    pipeline.stop()


