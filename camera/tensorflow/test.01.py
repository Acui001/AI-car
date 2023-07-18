import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("[INFO] Starting streaming...")
pipeline.start(config)
print("[INFO] Camera ready.")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    print("非数组图像")
    print(type(color_frame))
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    print("数组图像")
    print(type(color_image))
    scaled_size = (color_frame.width, color_frame.height)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Stop streaming...")
pipeline.stop()
cv2.destroyAllWindows()
