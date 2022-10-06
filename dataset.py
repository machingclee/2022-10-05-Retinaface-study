import cv2
import csv
import numpy as np
import os
import config
import torch
from pydash import set_
from typing import TypedDict, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, image
from torchvision import transforms as T
from device import device
from tqdm import tqdm
from random import shuffle
from PIL import Image


class LandmarkAnnotation(TypedDict):
    filename: str
    bboxes: Tuple[Tuple[float, float, float, float]]
    landmarks: Tuple[Tuple[float, float, float, float]]
    confs: float


def get_facial_data_arr() -> Tuple[LandmarkAnnotation]:
    annotations = []
    temp_bboxes = []
    temp_landmarks = []
    temp_confs = []
    temp_filename = ""

    with open("dataset/wider_face_annotation/train/label.txt") as f:
        for line in f:
            line = line.strip().split(" ")
            if line[0] == "#":
                # save the previous result
                if temp_filename != "":
                    landmark_annotation = LandmarkAnnotation()
                    landmark_annotation["filename"] = temp_filename
                    landmark_annotation["bboxes"] = temp_bboxes
                    landmark_annotation["landmarks"] = temp_landmarks
                    landmark_annotation["confs"] = temp_confs
                    # use [curr_filename] to avoid "period, ." from creating nested dictionary structure
                    annotations.append(landmark_annotation)
                    temp_bboxes = []
                    temp_landmarks = []
                    temp_confs = []

                temp_filename = line[1]

            else:
                x,y,w,h = [float(value) for value in line[0:4]]
                bbox = [x, y, x + w, y + h]
                landmarks_x = [float(value) for value in line[4:-1:3][0: config.landmarks_length]]
                landmarks_y = [float(value) for value in line[5:-1:3][0: config.landmarks_length]]
                landmarks = []

                for (x, y) in zip(landmarks_x, landmarks_y):
                    landmarks += [x] + [y]
                conf = float(line[-1])
                temp_bboxes.append(bbox)
                temp_landmarks.append(landmarks)
                temp_confs.append(conf)

    return annotations


torch_img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def resize_img(img, width, height):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    if h >= w:
        ratio = height / h
        new_h, new_w = int(h * ratio), int(w * ratio)
    else:
        ratio = width / w
        new_h, new_w = int(h * ratio), int(w * ratio)

    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, (w, h)


def pad_img(img):
    h = img.height
    w = img.width
    img = np.array(img)
    img = np.pad(img, pad_width=((0, config.input_height - h), (0, config.input_width - w), (0, 0)), mode="constant")
    img = Image.fromarray(img)
    assert img.height == config.input_height
    assert img.width == config.input_width
    return img


def resize_and_padding(img, width, height, return_window=False):
    img, (ori_w, ori_h) = resize_img(img, width, height)
    w = img.width
    h = img.height
    padding_window = (w, h)
    img = pad_img(img)

    if not return_window:
        return img
    else:
        return img, padding_window, (ori_w, ori_h)


class FacialLandmarkDataset(Dataset):
    def __init__(self):
        super(FacialLandmarkDataset, self).__init__()
        self.data = get_facial_data_arr()
        shuffle(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        filename = data["filename"]
        bboxes = torch.as_tensor(data["bboxes"])
        landmarks = torch.as_tensor(data["landmarks"])
        confs = torch.as_tensor(data["confs"])
        pil_img = Image.open(os.path.join(
            os.path.normpath(config.img_dir_train),
            os.path.normpath(filename)
        ))
        img = resize_and_padding(pil_img, config.input_width, config.input_height)
        img = torch_img_transform(img)
        return img.to(device), bboxes.to(device), landmarks.to(device), confs

    def __len__(self):
        return len(self.data)


facialLandmarkDataloader = DataLoader(dataset=FacialLandmarkDataset(),
                                      batch_size=config.batch_size,
                                      shuffle=True)


if __name__ == "__main__":
    facialLandmarkDataloader = DataLoader(dataset=FacialLandmarkDataset(),
                                          batch_size=config.batch_size,
                                          shuffle=True)

    for img, bboxes, landmarks, confs in facialLandmarkDataloader:
        print(img, bboxes, landmarks, confs)
