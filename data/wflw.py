from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image, ImageFilter
from data_augment import wflw_preproc
import numpy as np
import os
import config
import random


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
    def __init__(self, file_list, img_dir):
        self.line = None
        self.path = None
        self.landmarks = None
        self.attribute = None
        self.loc_img_path = None
        self.bbox_xyxy = None
        self.transforms = transforms
        self.preproc = wflw_preproc()

        self.img_dir = img_dir
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.name = self.line[0]
        self.loc_img_path = self.line[-1]
        self.img = np.array(Image.open(os.path.join(self.img_dir, self.loc_img_path)))
        # self.landmark = np.asarray(self.line[0:196], dtype=np.float32)
        # self.attribute = np.asarray(self.line[200:206], dtype=np.float32)
        # self.bbox_xyxy = np.asarray(self.line[197:201], dtype=np.int32)
        target = np.array(self.line[0:-1], dtype=np.float32)

        img, target = self.preproc(img, target)

        return (self.img, self.landmark, self.attribute, self.bbox_xyxy)

    def __len__(self):
        return len(self.lines)


if __name__ == "__main__":
    dataset = WFLWDatasets(
        file_list=config.WFLW_TRAIN_LABEL_TXT,
        img_dir=config.WFLW_TRAIN_IMG_DIR
    )
    data = next(iter(dataset))
    print(data)
