import numpy as np
import os
import config
import torch
import config
import albumentations as A
from pydash import set_
from typing import TypedDict, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from device import device
from tqdm import tqdm
from random import shuffle
from PIL import Image


def pts_scaling(set_of_pts, scale):
    return [[scale * pt for pt in pts] for pts in set_of_pts]

# originally we should have been able to use LongestMaxSize instead, but that fails for landmarks in keypoints


def resize_everything(img, bboxes, keypoints=None):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    if h >= w:
        ratio = config.input_height / h
        new_h, new_w = int(h * ratio), int(w * ratio)
    else:
        ratio = config.input_width / w
        new_h, new_w = int(h * ratio), int(w * ratio)

    img = img.resize((new_w, new_h), Image.BILINEAR)
    bboxes = pts_scaling(bboxes, ratio)
    result = (img, bboxes)

    if keypoints is not None:
        keypoints = pts_scaling(keypoints, ratio)
        result += (keypoints,)

    return result


class LandmarkAnnotation(TypedDict):
    filename: str
    bboxes: Tuple[Tuple[float, float, float, float]]
    landmarks: Tuple[Tuple[float, float, float, float]]
    confs: float


class ValAnnotation(TypedDict):
    filename: str
    bboxes: Tuple[Tuple[float, float, float, float]]


def clip(x: float, lower: float, upper: float):
    return min(max(x, lower), upper)


resize_and_padding_transforms_list = [
    # A.LongestMaxSize(max_size=config.longest_side_length, interpolation=1, p=1),
    A.PadIfNeeded(
        min_height=config.input_height,
        min_width=config.input_height,
        border_mode=0,
        value=(0, 0, 0),
        position="top_left"
    )
]

albumentation_transform_training = A.Compose([
    # A.ShiftScaleRotate(shift_limit=0, rotate_limit=10, p=0.7),
    # A.RandomSizedBBoxSafeCrop(width=config.input_width,height=config.input_height, p=1), # fails for keypoints
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.1),
    # A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.1)
    ], p=0.8),
    *resize_and_padding_transforms_list,
],
    p=1,
    bbox_params=A.BboxParams(format="pascal_voc", min_area=0.1, min_visibility=0.3, label_fields=[]),
    keypoint_params=A.KeypointParams(format="xy", label_fields=[], remove_invisible=False)
)


albumentations_resize_and_pad = A.Compose(
    resize_and_padding_transforms_list,
    p=1,
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]),
    keypoint_params=A.KeypointParams(format="xy", label_fields=[])
)


def get_training_data() -> Tuple[LandmarkAnnotation]:
    annotations = []
    temp_bboxes = []
    temp_landmarks = []
    temp_confs = []
    temp_filename = ""

    with open(config.training_annotation) as f:
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
                x, y, w, h = [float(value) for value in line[0:4]]
                bbox = [x, y, x + w, y + h]
                landmarks_x = [float(value) for value in line[4:-1:3][0: config.n_landmark_coordinates // 2]]
                landmarks_y = [float(value) for value in line[5:-1:3][0: config.n_landmark_coordinates // 2]]

                landmarks = []
                for (x, y) in zip(landmarks_x, landmarks_y):
                    # x, y will be clipped in dataset object's __get__ method
                    landmarks += [x] + [y]

                conf = float(line[-1])
                temp_landmarks.append(landmarks)
                temp_bboxes.append(bbox)
                temp_confs.append(conf)

    return annotations


def get_val_data() -> Tuple[ValAnnotation]:
    annotations = []
    temp_filename = ""
    temp_bboxes = []

    with open(config.validation_annotation) as f:
        for line in f:
            line = line.strip().split(" ")
            if line[0] == "#":
                if temp_filename != "":
                    annotation = ValAnnotation()
                    annotation["filename"] = temp_filename
                    annotation["bboxes"] = temp_bboxes
                    annotations.append(annotation)

                    temp_bboxes = []

                temp_filename = line[1]
            else:
                x, y, w, h = [float(value) for value in line[0:4]]
                bbox = [x, y, x + w, y + h]
                temp_bboxes.append(bbox)

    return annotations


torch_img_normalization = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def torch_img_denormalization(img: torch.Tensor) -> Image.Image:
    mean = torch.as_tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
    std = torch.as_tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
    img = (img * std + mean) * 255
    img = Image.fromarray(img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype("uint8"))
    return img


def preprocess_img_annotation(pil_img, bboxes, landmarks=None, augment_data=False):
    if augment_data:
        transforms = albumentation_transform_training
    else:
        transforms = albumentations_resize_and_pad

    if landmarks is None:
        img, bboxes = resize_everything(pil_img, bboxes)
        img = np.array(img)
        transformed = transforms(image=img, bboxes=bboxes, keypoints=[])
    else:
        img, bboxes, landmarks = resize_everything(pil_img, bboxes, landmarks)
        img = np.array(img)
        transformed = transforms(
            image=img,
            bboxes=bboxes,
            keypoints=landmarks
        )

    img = Image.fromarray(transformed["image"])
    bboxes_ = transformed["bboxes"]
    result = (img, bboxes_)

    if landmarks is not None:
        landmarks_ = transformed["keypoints"]
        # if bbox is removed, then landm_ids must be removed as well
        result = (img, bboxes_, landmarks_)

    return result


class FacialLandmarkValidationDataset(Dataset):
    def __init__(self):
        super(FacialLandmarkValidationDataset, self).__init__()
        self.data = get_val_data()
        shuffle(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        bboxes = data["bboxes"]
        filename = data["filename"]
        pil_img = Image.open(os.path.join(
            os.path.normpath(config.img_dir_val),
            os.path.normpath(filename)
        ))
        H = pil_img.height
        W = pil_img.width
        bboxes = [[clip(pt, 0, W - 1 if i % 2 == 0 else H - 1) for i, pt in enumerate(pts)] for pts in bboxes]
        # img: not normalized np.array,
        img, bboxes = preprocess_img_annotation(pil_img, bboxes, augment_data=False)
        return np.array(img), np.array(bboxes)

    def __len__(self):
        return len(self.data)


class FacialLandmarkTrainingDataset(Dataset):
    def __init__(self):
        super(FacialLandmarkTrainingDataset, self).__init__()
        self.data = get_training_data()
        shuffle(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        filename = data["filename"]
        bboxes = data["bboxes"]
        landmarks = data["landmarks"]

        pil_img = Image.open(os.path.join(
            os.path.normpath(config.img_dir_train),
            os.path.normpath(filename)
        ))
        H = pil_img.height
        W = pil_img.width
        img = pil_img
        bboxes = [[clip(pt, 0, W - 1 if i % 2 == 0 else H - 1) for i, pt in enumerate(pts)] for pts in bboxes]
        landmarks = [[clip(pt, 0, W - 1 if i % 2 == 0 else H - 1) for i, pt in enumerate(pts)] for pts in landmarks]

        confs = torch.as_tensor(data["confs"])

        if len(bboxes) != len(landmarks):
            print("not euqal")

        img, bboxes, landmarks = preprocess_img_annotation(img, bboxes, landmarks, augment_data=True)
        if len(bboxes) != len(landmarks):
            print("not euqal")
        img = torch_img_normalization(img)
        return (
            img.to(device, dtype=torch.float32),
            torch.as_tensor(bboxes).to(device, dtype=torch.float32),
            torch.as_tensor(landmarks).to(device, dtype=torch.float32),
            confs
        )

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    get_training_data()
