import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from PIL import Image, ImageDraw
from typing import Tuple


def draw_box(pil_img: Image.Image, bboxes):
    draw = ImageDraw.Draw(pil_img)
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(255, 255, 255, 150), width=2)


def draw_dots(pil_img: Image.Image, points: Tuple[float], r=4):
    draw = ImageDraw.Draw(pil_img)
    for x, y in np.array_split(points, 2):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))


def visualize(model, pil_img: Image.Image, bboxes, landmarks):
    draw_box(pil_img, bboxes)
    draw_dots(pil_img, landmarks)
