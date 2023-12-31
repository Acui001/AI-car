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

# download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv
print("[INFO] Loading model...")
file_path = "frozen_inference_graph.pb"

# Load the Tensorflow model into memory.
# 创建一个新的计算图
detection_graph = tf.Graph()
# # 将detection_graph设置为默认计算图
with detection_graph.as_default():
    # 用于存储计算图的定义
    # 它包含了计算图中所有操作（tf.Operation）的信息，例如操作的类型、名称、输入和输出张量等
    od_graph_def = tf.compat.v1.GraphDef()
    # 打开路径为file_path的文件，并以只读模式进行操作
    with open(file_path, 'rb') as file:
        # 读取文件
        print("读取文件")
        serialized_graph = file.read()
        print("成功读取")
        # 将存储在变量serialized_graph中的字符串内容解析为od_graph_def对象
        print("解析中")
        od_graph_def.ParseFromString(serialized_graph)
        print("解析成功")
        print("导入中")
        # 用于将GraphDef对象或序列化的GraphDef数据导入到当前计算图中
        tf.compat.v1.import_graph_def(od_graph_def, name='')
        print("导入成功")
    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/
print("[INFO] Model loaded.")
colors_hash = {}
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    scaled_size = (color_frame.width, color_frame.height)
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image_expanded = np.expand_dims(color_image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                feed_dict={image_tensor: image_expanded})

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    for idx in range(int(num)):
        class_ = classes[idx]
        score = scores[idx]
        box = boxes[idx]
        
        if class_ not in colors_hash:
            colors_hash[class_] = tuple(np.random.choice(range(256), size=3))
        
        if score > 0.6:
            left = int(box[1] * color_frame.width)
            top = int(box[0] * color_frame.height)
            right = int(box[3] * color_frame.width)
            bottom = int(box[2] * color_frame.height)
            
            p1 = (left, top)
            p2 = (right, bottom)
            # draw box
            r, g, b = colors_hash[class_]
            cv2.rectangle(color_image, p1, p2, (int(r), int(g), int(b)), 2, 1)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_image)
    cv2.waitKey(0)

print("[INFO] stop streaming ...")
pipeline.stop()
