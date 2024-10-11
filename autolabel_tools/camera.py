import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from sam2.build_sam import build_sam2_camera_predictor
import os
from collections import defaultdict

class MaskSaver:
    def __init__(self):
        self.index = 1

    def save(self, mask, obj_index, merged, save_dir):
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
    merged_data = defaultdict(lambda: {'points': [], 'properties': []})

    for point, label, prop in zip(points, labels, properties):
        converted_property = convert_property(prop)
        merged_data[label]['points'].append(point)
        merged_data[label]['properties'].append(converted_property)

    merged_points = []
    for label, data in merged_data.items():
        merged_points.append({
            'label': label,
            'points': data['points'],
            'properties': data['properties']
        })

    return merged_points

def autolabel_camera(camera, camera_type, first_frame, points, obj_labels, obj_properties,
                     save_type, save_dir):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = "./autolabel_tools/checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

    if_init = False
    if save_type == "mask":
        saver = MaskSaver()
    elif save_type == "box":
        saver = BoxSaver()
    merged = merge_by_label(points=points, labels=obj_labels, properties=obj_properties)
    if camera_type == "realsense":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            while True:
                frames = camera.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                width, height = frame.shape[:2][::-1]
                i = 0
                if not if_init:
                    predictor.load_first_frame(first_frame)
                    ann_frame_idx = 0
                    if_init = True
                    for item in merged:
                        point = item['points']
                        obj_property = item['properties']
                        point = np.array(point)
                        labels = np.array(obj_property)
                        ann_obj_id = i
                        i += 1
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            points=point,
                            labels=labels,
                        )

                else:
                    out_obj_ids, out_mask_logits = predictor.track(frame)
                    all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                    mask_list = []
                    for i in range(0, len(out_obj_ids)):
                        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                            np.uint8
                        ) * 255
                        mask_list.append(out_mask)
                        all_mask = cv2.bitwise_or(all_mask, out_mask)
                    saver.save(mask_list, out_obj_ids, merged, save_dir)
                    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
                    frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

                cv2.imshow("frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        
        camera.stop()
        cv2.destroyAllWindows()