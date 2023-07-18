import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_with_matplotlib(color_img):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, 1)
    plt.imshow(img_RGB)
    plt.title("title")
    plt.axis('off')
    plt.show()
def visualize_image_segments(image, segment_weights):
    """将分割图像区域可视化"""
    height, width, _ = image.shape
    pt1 = []

    # 绘制垂直线条分割图像
    for i in range(6):
        long = int(segment_weights[i] * width)
        if i == 0:
            pt1.append(long)
        else:
            pt1.append(pt1[i - 1] + long)
        cv2.line(image, (pt1[i], 0), (pt1[i], height), (0, 255, 0), 2)
        show_with_matplotlib(image)

# 示例用法
image = np.zeros((400, 600, 3), dtype=np.uint8)  # 创建一个示例图像
anchor_box = [200, 100, 300, 200]  # 示例锚框
image_width = image.shape[1]  # 图像宽度
# segment_weights = [0.25, 0.15, 0.0975, 0.05, 0.0975, 0.15, 0.25]
# segment_weights = [0.22, 0.145, 0.1, 0.07, 0.1, 0.145, 0.22]
segment_weights = [0.3094, 0.0894, 0.069, 0.0644, 0.069, 0.0894, 0.3094]
visualize_image_segments(image,segment_weights)

