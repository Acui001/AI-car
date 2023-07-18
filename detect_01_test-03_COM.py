"""
基于yolo的对识别到人就可以前进
识别不到人就停止的代码
可以运行，但是未更改异步函数部分
加入了转向功能，尚未测试
"""
import argparse
import asyncio
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import torch.backends.cudnn as cudnn
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# ———————————————————————————————————启用摄像头部分———开始——————————————————————————————— #
# 创建Pipeline对象
pipeline = rs.pipeline()
# 配置摄像头的设置
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# 启动摄像头
pipeline.start(config)
depth_image = None
# ———————————————————————————————————启用摄像头部分———结束——————————————————————————————— #

# ————————————————————————————————————启用蓝牙部分————开始—————————————————————————————————— #

# ——————————————————————————————————命令字典————————开始———————————————— #
# 下面是命令字典
send_str = bytearray([0x01])  # 舵机打直
send_left = bytearray([0x08])  # 舵机右转
send_right = bytearray([0x02])  # 舵机左转
speed_up = bytearray([0x58])  # 电机加速100 # 我没有调用，但是你也最好别用
speed_down = bytearray([0x59])  # 电机减速100 # 我没有调用，但是你也最好别用

# ——————速度值——————————
# send_speed_value = bytearray([0x01])  # 给值，但是单独给并不导致车动，需要给舵机打直send_str，
speed_value_zero = bytearray([0x00])  # 固定停车值
# ——————速度方向—————————
send_speed_direction = send_str
# ——————————————————————————————————命令字典————————结束———————————————— #

# ————————————————————————————————————蓝牙相关函数————开始———————————————— #
person_exist = False  # 用于检测是否有人的全局变量

# ————————————————————————————————————蓝牙相关函数————结束———————————————— #

# ————————————————————————————————————启用蓝牙部分————结束—————————————————————————————————— #

# ————————————————————————————————————转向相关函数————开始———————————————— #
def visualize_image_segments(image):
    """将分割图像区域可视化"""
    height, width, _ = image.shape
    segment_width = width // 2

    # 在图像上绘制垂直线条分割三个部分
    cv2.line(image, (segment_width - 20, 0), (segment_width - 20, height), (0, 255, 0), 2)
    cv2.line(image, (segment_width + 20, 0), (segment_width + 20, height), (0, 255, 0), 2)


def detect_anchor_position(anchor_box, image_width):
    scaling_factor = 3
    anchor_center = (anchor_box[0] + anchor_box[2]) / 2  # 计算锚框的中心点 x 坐标
    if anchor_center < (image_width / 2) - 20:  # 锚框在视野的左侧处
        # await asyncio .sleep(1.0)
        deviation_value = ((image_width / 2) - 20 - anchor_center) / scaling_factor  # 放缩操作
        return "Left", deviation_value
    elif anchor_center > (image_width / 2) + 20:  # 锚框在视野的右侧处
        # await asyncio.sleep(1.0)
        deviation_value = ((image_width / 2) + 20 - anchor_center) / scaling_factor
        return "Right", deviation_value
    elif (image_width / 2) - 20 < anchor_center < (image_width / 2) + 20:  # 锚框在视野的中间
        deviation_value = 0
        return "Center", deviation_value


# ————————————————————————————————————转向相关函数————结束———————————————— #

def get_distance(depth_frame, xyxy):
    # xyxy[0]：表示矩形框的左上角的x坐标。
    # xyxy[2]：表示矩形框的右下角的x坐标。
    # xyxy[1]：表示矩形框的左上角的y坐标。
    # xyxy[3]：表示矩形框的右下角的y坐标。
    # 你要用这里的值的话需要用int((xyxy[1].item()))来用不然会数据类型错误
    x = int(((xyxy[0] + xyxy[2]) / 2).item())
    y = int(((xyxy[1] + xyxy[3]) / 2).item())
    depth_value = depth_frame.get_distance(x, y)
    return depth_value, x, y


def mark_point_TxT(image, distance, x, y):
    # 绘制一个红色的圆圈标记指定的点
    x, y = int(x), int(y)
    marked_image = np.copy(image)
    cv2.circle(marked_image, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(marked_image, str(distance) + "m", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return marked_image


def detect(save_img=False):
    global person_exist, send_speed_value_direction
    """
    Args:
        save_img:
    Returns:
    """
    # source（源）：表示要处理的输入源。可能是图像、视频或文件夹的路径
    # weights（权重）：表示要使用的模型权重或参数文件的路径。权重通常是在训练过程中学习到的模型参数
    # view_img（查看图像）：一个布尔值，表示是否显示图像的窗口。如果设置为True，则程序将显示检测结果的图像窗口
    # view_img（查看图像）：一个布尔值，表示是否显示图像的窗口。如果设置为True，则程序将显示检测结果的图像窗口
    # save_txt（保存文本）：一个布尔值，表示是否将检测结果保存为文本文件。如果设置为True，则程序将结果保存为文本文件
    # imgsz（图像大小）：表示输入图像的大小。在图像处理任务中，经常需要将图像调整为相同的大小以进行处理
    # trace（追踪）：一个布尔值，表示是否记录模型的执行跟踪信息。如果设置为True，则程序将记录模型执行的跟踪信息。
    # opt.no_trace是获取命令行参数中是否禁用跟踪的选项的方式，not opt.no_trace的作用是将其反转，以获取真正的跟踪选项。
    # trace参数用于控制是否记录模型的执行跟踪信息。当设置trace=True时，程序将会记录模型执行的跟踪信息，包括网络的前向传播和后向传播过程中的计算步骤、输入输出张量的形状和值等
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    # source.isnumeric() 检查 source 是否是一个纯数字的字符串，用于判断是否使用摄像头作为输入源。如果 source 是一个纯数字的字符串，则返回 True，表示使用摄像头；否则返回 False。
    # source.endswith('.txt') 检查 source 是否以 '.txt' 结尾，用于判断是否使用文本文件作为输入源。如果 source 是以 '.txt' 结尾的字符串，则返回 True，表示使用文本文件；否则返回 False。
    # source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) 检查 source 是否以指定的协议开头，用于判断是否使用网络流作为输入源。将 source 转换为小写字母后，通过 startswith() 方法与给定的协议列表进行匹配。如果 source 以指定的协议之一开头，则返回 True，表示使用网络流；否则返回 False。
    # webcam 的含义是：如果 source 是一个纯数字的字符串（摄像头作为输入源），或者 source 是以 '.txt' 结尾的字符串（文本文件作为输入源），或者 source 是以指定的协议之一开头的字符串（网络流作为输入源），则 webcam 为 True，表示使用摄像头、文本文件或网络流作为输入源；否则 webcam 为 False，表示不使用摄像头、文本文件或网络流作为输入源。
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 目录的设置和创建
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 初始化过程
    set_logging()  # 用于设置日志的配置
    device = select_device(opt.device)  # 选择device
    half = device.type != 'cpu'  # 半精度计算仅在 CUDA 上受支持 如果为cpu则给False，只有不是cpu的时候才是Ture以进行半精度运算

    # 加载model
    model = attempt_load(weights, map_location=device)  # 加载模型load FP32 model 32位浮点数模型
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:  # 如果是半精度计算就加载FP16（半精度浮点数）
        model.half()  # to FP16

    # 第一个分类器模型可能是用于生成候选框或区域的模型，通常称为区域生成网络（Region Proposal Network，RPN）

    # 第二阶段分类器Second-stage classifier
    classify = False  # 默认下不使用第二阶段分类器
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # 初始化并加载resnet101模型
        # 二阶段分类器会涉及到两个分类器模型。因此，在代码中将 n 设置为 2 是为了加载两个分类器模型，以完成二阶段分类器的功能
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    # vid_path, vid_writer = None, None
    # 若为一个摄像头源/文件源
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # 加速使用相同大小图像进行推断set True to speed up constant image size inference
        # 将视频source转化为dataset
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # 若为一个其他源(单个图片)
    else:
        # 将图片source转化为dataset
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 获取名字和颜色Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 对图片进行推断Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # old_img_w和old_img_h的属性从imgsz中来获取
    old_img_w = old_img_h = imgsz
    # 图片批次大小
    old_img_b = 1

    t0 = time.time()
    # `for path, img, im0s, vid_cap in dataset:` 是一个迭代循环，用于遍历数据集中的每个元素。
    # - `path`：表示图像文件的路径或数据的标识符。可以通过`path`来获取图像文件或其他相关信息。
    # - `img`：表示加载的图像数据，通常是一个多维数组或张量。这个变量可以用于后续的图像处理和预测操作。
    # - `im0s`：表示原始的图像数据，即未经过任何处理的原始图像。在某些情况下，会将图像进行副本，并在`im0s`中保存原始图像数据，以便后续比较或其他用途。
    # - `vid_cap`：表示视频捕获对象（如果数据集是视频数据集的话）。这个变量可以用于控制视频的播放、跳帧或其他相关操作。
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup 模型的预热过程
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                # 将输入的图像数据传递给模型进行前向传播(输入数据在层中计算)
                model(img, augment=opt.augment)[0]  # 将图片作为输入并设置好 # 这句不能删啊，这里是预热过程，也也就是让网络随便跑跑

        # 在预热完成后
        # 开始推断Inference
        t1 = time_synchronized()
        with torch.no_grad():  # 计算梯度可能导致GPU内存泄漏Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS(极大非抑制)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier(分类器)
        if classify:
            # pred：YOLO的输出结果，通常是检测边界框和对应的类别概率
            # modelc：第二阶段分类器模型,这里用的是resnet101,用于对YOLO的输出进行进一步的分类
            # img：输入图像
            # im0s：原始输入图像
            # 我们对输入图像进行各种处理，然后我们标框是在原图上标框
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections（处理检测结果）
        for i, det in enumerate(pred):  # 对每帧图片进行检测detections per image
            # 若为一个摄像头源/文件源
            if webcam:  # batch_size >= 1
                # 获取属性
                # p：路径（path）
                # s：前缀字符串（prefix string）在日志或输出中添加前缀这里% i的方法可以为其排上序列
                # im0：原始图像（image 0）
                # frame帧数（frame number）
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            # 若为一个其他源(单个图片)
            else:
                # 获取属性
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            # 将路径变量变成了一个可以读取的真实路径
            p = Path(p)  # 去到这个路径to Path
            save_path = str(save_dir / p.name)  # 图片路径img.jpg
            # 存储图片路径的文本可以在txt里面放很多图片的路径这样他也能读取
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # torch.tensor(im0.shape)获取原图的(height, width, channels)并转化为张量
            # torch.tensor(im0.shape)[[1, 0, 1, 0]]=原始图片的[高度，宽度，高度，宽度]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 增益归一化normalization gain whwh
            visualize_image_segments(im0)  # 可视化分割区域 # 没有也不怕，这段只是为了明显可视化的，你可以将他注释掉
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将边界框从img_size缩放到im0的尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # 打印结果Print results
                # det[:, -1] 是从检测结果中提取出的最后一列，该列包含类别标签信息
                # det[:, -1].unique()返回检测结果中唯一的类别标签
                for c in det[:, -1].unique():  # 对每个唯一的类别标签进行迭代

                    n = (det[:, -1] == c).sum()  # 当前类别 c 的检测结果数量detections per class
                    # 将当前类别的检测结果数量和类别名称添加到字符串 s 中。如果检测结果数量大于1，则在类别名称后面加上 's'，用于表示复数形式
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 保存结果Write results
                # det 是一个包含检测结果的列表。每个检测结果通常由边界框坐标（xyxy）类型为列表张量，置信度（conf）和类别（cls）组成
                # reversed()表示从最后一个开始励遍
                for *xyxy, conf, cls in reversed(det):  # 用于遍历和处理检测结果
                    # 等待摄像头的数据
                    frames = pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)

                    # 获取深度图像中指定位置的距离值
                    distance, X, Y = get_distance(depth_frame, xyxy)
                    # 在图像上标记指定的点
                    im0 = mark_point_TxT(im0, round(distance, 4), X, Y)
                    print("X:", X)
                    print("Y:", Y)
                    print("Box:", xyxy)
                    if save_txt:  # 写入文件Write to file
                        # xywh 变量的计算涉及将边界框的坐标从 (x1, y1, x2, y2) 的形式转换为 (x, y, w, h) 的形式，并将其归一化
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # 根据条件决定创建一个元组，并将该元组赋值给变量 line。
                        # 元组的内容包括类别 (cls)、归一化边界框坐标和尺寸 (*xywh)，以及可选的置信度 (conf)。
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # 标签类型label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        # ————————————————————————————转向处理部分————————开始———————————————————————————————— #
                        image_height, image_width, _ = im0.shape
                        # print('image_height:', image_height)  # image_height: 480
                        # print('image_width:', image_width)  # image_width: 640
                        position, deviation_value = detect_anchor_position(xyxy, image_width)
                        deviation_value_before = 0
                        if isinstance(deviation_value, torch.Tensor):
                            deviation_value = deviation_value.cpu().numpy()
                            deviation_value_before = deviation_value
                            # tens, ones = divmod(int(deviation_value), 10)
                            # print("偏差：", tens, ones)
                        else:
                            # 处理整数对象的情况，例如给出一个默认值或者抛出异常
                            deviation_value = deviation_value_before  # 或者其他处理方式
                        # ———————————————————————————偏差值处理部分————————开始———————————————————————————— #

                        # ———————————————————————————偏差值处理部分————————结束———————————————————————————— #
                        print("当前位置为:", position)
                        if person_exist:
                            if position == "Left":
                                # 处理 "Left" 的情况
                                # 这里可以放置你想要执行的代码
                                # 调整方向的话给 Turn按键赋值
                                # send_speed_direction已经在detect最前面声明了为全局变量了
                                send_speed_value_direction = send_left
                                print("检测到人在左边，正在左转中(:")
                                print('偏差值为：', deviation_value)
                            elif position == "Right":
                                # 处理 "Right" 的情况
                                # 这里可以放置你想要执行的代码
                                # 调整方向的话给 Turn按键赋值
                                send_speed_value_direction = send_right
                                print("检测到人在右边，正在全力右转中:)")
                                print('偏差值为：', deviation_value)
                            else:
                                # 处理 "Center" 的情况
                                # 这里可以放置你想要执行的代码
                                # 调整方向的话给 Turn按键赋值
                                send_speed_value_direction = send_str
                                print("检测到人在中间，舵机正在摸鱼中")
                        else:
                            send_speed_value_direction = speed_value_zero
                            print("没有人，舵机正在摸鱼中")
                        # ————————————————————————————转向处理部分————————结束———————————————————————————————— #
            # 检测到有人就开始通信
            if "person" in s:
                person_exist = True
                # ——————————————————————小车行为(通信部分)加在这里———开始端———————————————————-#
                # ——————————————————————————蓝牙运行部分——————————————开始————————————————————#
                # 运行通信部分
                print("检测到有人,小车开始跟随任务")
            else:
                person_exist = False
                print("检测到没人，那我可以摸鱼了(搓搓手)")
                # ——————————————————————小车行为(通信部分)加在这里———终止端———————————————————-#
            # Print time (inference + NMS)
            print(f'{s}Done.响应时间为({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # Stream results

            cv2.imshow(str(p), im0)
            # cv2.imshow('Depth Image', depth_image)
            cv2.waitKey(1)  # 1 millisecond
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='1', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.40, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_false', help='do not save images/videos')  # 保存选项
    parser.add_argument('--classes', nargs='+', type=int, default=[0],
                        help='filter by class: --class 0, or --class 0 2 3')  # 选中可以只显示特定的标签
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print("在这里opt:", opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)

        else:
            detect()
