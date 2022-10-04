import cv2
from tqdm import tqdm
import csv
import numpy as np
import os


def get_facial_data_gen():
    with open("dataset/training.txt") as f:
        for line in f:
            line = line.strip().split(" ")
            filename = os.path.normpath(line[0])
            landmarks = [float(num) for num in line[1: 11]]
            print(filename, landmarks)


def parse_wider_txt(data_root: str, split: str):
    """
    refer to: torchvision.dataset.widerface.py
    :param data_root:
    :param split:
    :return:
    """
    assert split in ['train', 'val'], f"split must be in ['train', 'val'], got {split}"

    txt_path = os.path.join(data_root, split, 'label.txt')
    img_root = os.path.join(data_root, f'WIDER_{split}', 'images')
    with open(txt_path, "r") as f:
        lines = f.readlines()
        file_name_line, num_boxes_line, box_annotation_line = True, False, False
        num_boxes, box_counter, idx = 0, 0, 0
        labels = []
        progress_bar = tqdm(lines)
        for line in progress_bar:
            line = line.rstrip()
            if file_name_line:
                img_path = line
                file_name_line = False
                num_boxes_line = True
            elif num_boxes_line:
                num_boxes = int(line)
                num_boxes_line = False
                box_annotation_line = True
            elif box_annotation_line:
                box_counter += 1
                line_split = line.split(" ")
                line_values = [x for x in line_split]
                labels.append(line_values)
                if box_counter >= num_boxes:
                    box_annotation_line = False
                    file_name_line = True

                    if num_boxes == 0:
                        print(f"in {img_path}, no object, skip.")
                    else:
                        # 根据个人意愿，在此加上对应处理方法
                        print(img_path)
                        print(labels)
                        pass
                        # 根据个人意愿，在此加上对应处理方法

                    box_counter = 0
                    labels.clear()
                    idx += 1
                    progress_bar.set_description(f"{idx} images")
            else:
                raise RuntimeError("Error parsing annotation file {}".format(txt_path))


if __name__ == "__main__":
    parse_wider_txt("dataset/wider_face/", "val")
