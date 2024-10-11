import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from sam2.build_sam import build_sam2_video_predictor
import os
import re
from collections import defaultdict

class MaskSaver:
    def __init__(self):
        self.index = 1

    def save(self, mask, all_mask, obj_index, merged, save_dir):
        all_mask_save_name = os.path.join(save_dir, f"allmask_{self.index}.png")
        cv2.imwrite(all_mask_save_name, all_mask)
        for i in range(0, len(obj_index)):
            filename = os.path.join(save_dir, f"mask_{self.index}_{merged[i]['label']}.png")
            cv2.imwrite(filename, mask[i])
            print(f"Saved mask {filename}")
        self.index += 1

class BoxSaver:
    def __init__(self):
        self.index = 1
    def save(self, mask, obj_index, merged, save_dir):
        for i in range(0, len(obj_index)):
            filename = os.path.join(save_dir, f"box_{self.index}_{merged[i]['label']}.txt")
            mask[i] = mask[i] / 255
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            data = [x_min, y_min, x_max, y_max]
            with open(filename, 'w') as file:
                for value in data:
                    file.write(f"{value}\n")
            print(f"Saved box {filename}")
        self.index += 1

def convert_property(property_value):
    return 1 if property_value == 'Positive' else 0

def merge_by_label(points, labels, properties):
    merged_data = defaultdict(lambda: {'points': [], 'properties': set()})

    for point, label, prop in zip(points, labels, properties):
        converted_property = convert_property(prop)
        merged_data[label]['points'].append(point)
        merged_data[label]['properties'].add(converted_property)

    merged_points = []
    for label, data in merged_data.items():
        merged_points.append({
            'label': label,
            'points': data['points'],
            'properties': list(data['properties'])
        })

    return merged_points

def extract_number(filename):
    match = re.search(r'(\d+)', os.path.splitext(filename)[0])
    if match:
        return int(match.group(1))
    return 0

def autolabel_video(video_dir, first_frame, points, obj_labels, obj_properties,
                    save_type, save_dir):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = "./autolabel_tools/checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=extract_number)

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0
    if save_type == "mask":
        saver = MaskSaver()
    elif save_type == "box":
        saver = BoxSaver()
    merged = merge_by_label(points=points, labels=obj_labels, properties=obj_properties)
    i = 0
    for item in merged:
        point = item['points']
        obj_property = item['properties']
        point = np.array(point)
        labels = np.array(obj_property)
        ann_obj_id = i
        i += 1
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=point,
            labels=labels,
        )



    video_segments = {}  
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        print((out_mask_logits > 0.0).any().item())
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }


    height, width, _ = first_frame.shape
    print(first_frame.shape)
    for out_frame_idx in range(0, len(frame_names)):
        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        mask_list = []
        out_obj_ids = []
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask = (out_mask > 0.0).transpose(1, 2, 0).astype(
                np.uint8
            ) * 255
            mask_list.append(out_mask)
            out_obj_ids.append(out_obj_id)
            all_mask = cv2.bitwise_or(all_mask, out_mask)
            print(all_mask)
        saver.save(mask_list, all_mask, out_obj_ids, merged, save_dir)