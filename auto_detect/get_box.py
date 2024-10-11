import torch
from pathlib import Path
from auto_detect.models.common import DetectMultiBackend
from auto_detect.utils.general import (check_img_size, non_max_suppression, scale_boxes)
from auto_detect.utils.plots import Annotator
from auto_detect.utils.torch_utils import select_device
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_first_frame(video_path, output_image_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    # 读取第一帧
    ret, frame = cap.read()
    if ret:
        # 保存第一帧图像
        cv2.imwrite(output_image_path, frame)
        print(f"First frame saved as {output_image_path}")
    else:
        print("Error: Could not read frame.")
        return None
    
    # 释放视频捕获对象
    cap.release()




def detect_image(weights, source=None, frames=None, imgsz=640, conf_thres=0.25, iou_thres=0.45, max_det=1000, device=''):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load image
    if source is not None:
        img0 = cv2.imread(source)  # BGR
        assert img0 is not None, f'Image not found: {source}'
    
    if frames is not None:
        img0 = frames

    # Padded resize
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # 确保数组是连续的
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    # Inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred[0], conf_thres, iou_thres, max_det=max_det)

    box = []
    classes = []

    # Process predictions
    for i, det in enumerate(pred):  # per image
        s = ''
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xyxy = [int(coord.item()) for coord in xyxy]  # 转换为int并打印
                box.append(xyxy)
                classes.append(names[int(cls)])
                print(f"Class: {names[int(cls)]}, Box: {xyxy}, Confidence: {conf:.2f}")
    
    return box, classes

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def get_box(video_path='./auto_detect/video/Video_new.mp4', output_image_path='./auto_detect/video/first_frame.jpg'):

    get_first_frame(video_path, output_image_path)

    weights = './auto_detect/checkpoints/yolov9e.pt'  # 模型路径
    box, classes = detect_image(weights=weights, source = output_image_path)
    return box, classes


def get_box2(frame):
    weights = './auto_detect/checkpoints/yolov9e.pt'  # 模型路径
    box, classes = detect_image(weights=weights, frames=frame)
    return box, classes



def get_mask(boxes):
    image_shape = (1280, 720, 3)
    height, width, _ = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        mask[y_min:y_max, x_min:x_max] = 1
    return mask

def get_centrel_point(boxes):
    points = []
    for box in boxes:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        center_x = int(round(center_x))
        center_y = int(round(center_y))
        point = [center_x, center_y]
        points.append(point)
    return points


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    if mask.shape[0] == 1:
        mask = mask[0]
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


if __name__ == "__main__":
    box = get_box()
    points = get_centrel_point(boxes=box)
    print(points)
    points = np.array(points)
    print(points)
    box = np.array(box)
    print(box)

