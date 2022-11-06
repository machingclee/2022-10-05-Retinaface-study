from torchvision import transforms
from torch.utils import data
from PIL import Image, ImageFilter
from data.data_augment import wflw_preproc
from pydash import get, set_
from src import config
import numpy as np
import os
import random
import torch
from src.device import device
from data.data_augment import torch_imgnet_transform


def collate_fn(batch):
    imgs = []
    batch_annotations = []

    for i in range(len(batch)):
        data = batch[i]
        img, annotation = data
        imgs.append(img.unsqueeze(0))
        batch_annotations.append(annotation)

    return torch.cat(imgs, dim=0).to(device), batch_annotations


class wflm_attribute_enum:
    pose_normal = 0
    pose_large = 1
    expression_normal = 0
    expression_exaggerate = 1
    illumination_normal = 0
    illumination_extreme = 1
    makeup_no = 0
    maekup_yes = 1
    occlusion_no = 0
    occlusion_yes = 1
    blur_no = 0
    blur_yes = 1


def random_blur(img, r=1.2, p=0.4):
    if (random.randint(0, int(1 / p) * 100) < 100):
        img = img.filter(ImageFilter.GaussianBlur(radius=random.randint(40, r * 100) / 100))
    return img


def random_occlude(img, r=0.3, p=0.2):
    if (random.randint(1, int(1 / p)) == 1):
        y, x, _ = img.shape
        yr = int(r * y)
        xr = int(r * x)
        ys = random.randint(0, y - yr)
        xs = random.randint(0, x - xr)
        for i in range(ys, ys + yr):
            for j in range(xs, xs + xr):
                img[i, j, 0] = random.randint(0, 255)
                img[i, j, 1] = random.randint(0, 255)
                img[i, j, 2] = random.randint(0, 255)
    return img


class WFLWDatasets(data.Dataset):
    data = None

    def __init__(self,
                 file_list=[config.WFLW_TRAIN_LABEL_TXT, config.WFLW_VAL_LABEL_TXT],
                 img_dir=config.WFLW_TRAIN_IMG_DIR,
                 preproc=wflw_preproc(img_dim=config.input_img_size)):
        self.file_list = file_list
        self.path = None
        self.landmarks = None
        self.attribute = None
        self.loc_img_path = None
        self.bbox_xyxy = None
        self.transforms = transforms
        self.preproc = preproc
        self.img_dir = img_dir
        self.data = None

    def get_data(self):
        file_list = self.file_list
        if WFLWDatasets.data is None:
            img_path_to_annotations = {}

            def get_annotation(file_list):
                # mutate img_path_to_annotations
                with open(file_list, 'r') as f:

                    lines = f.readlines()
                    for line in lines:
                        data = line.strip().split()

                        rel_img_path = data[-1]
                        landmarks = [float(v) for v in data[0: 196]]
                        bbox = [float(v) for v in data[196:200]]

                        anno = landmarks + bbox + [1]

                        annotation = get(img_path_to_annotations, [rel_img_path])

                        if annotation is None:
                            set_(img_path_to_annotations, [rel_img_path], [anno])
                        else:
                            annotation.append(anno)

            if isinstance(file_list, list):
                for file in file_list:
                    get_annotation(file)
            else:
                get_annotation(file_list)

            WFLWDatasets.data = [(path, anno) for path, anno in img_path_to_annotations.items()]

        return WFLWDatasets.data

    def __getitem__(self, index):
        data = self.get_data()
        curr_data = data[index]
        rel_img_path, annotations = curr_data
        self.img = np.array(Image.open(os.path.join(self.img_dir, os.path.normpath(rel_img_path))))
        target = np.array(annotations)

        img, target = self.preproc(self.img, target)
        return img, target

    def __len__(self):
        data = self.get_data()
        return len(data)


if __name__ == "__main__":
    dataset = WFLWDatasets()
    data = next(iter(dataset))
    print(data)
