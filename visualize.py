import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import os
import config
from torchsummary import summary
from PIL import Image, ImageDraw
from typing import Tuple
from dataset import FacialLandmarkTrainingDataset, FacialLandmarkValidationDataset, torch_img_normalization, torch_img_denormalization
from torch.utils.data import Dataset, DataLoader
from retina_face import RetinaFace
from device import device


def draw_box(pil_img: Image.Image, bboxes, confs=None, color=(255, 255, 255, 150)):
    draw = ImageDraw.Draw(pil_img)
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
        if confs is not None:
            conf = confs[i]
            draw.text(
                (xmin, max(ymin - 10, 4)),
                "{:.2f}".format(conf.item()),
                color
            )


def draw_dots(pil_img: Image.Image, landmarks: Tuple[float], r=2):
    draw = ImageDraw.Draw(pil_img)
    for points in landmarks:
        for x, y in np.array_split(points, 5):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))


def visualize_training_data(n_images: int):
    training_data_loader = DataLoader(dataset=FacialLandmarkTrainingDataset(),
                                      batch_size=1,
                                      shuffle=True)
    train_iter = iter(training_data_loader)
    for i in range(n_images):
        img, bboxes, landmarks, _ = next(train_iter)
        bboxes = bboxes.squeeze(0)
        landmarks = landmarks.squeeze(0)
        pil_img = torch_img_denormalization(img)
        draw_box(pil_img, bboxes, color=(0, 0, 255, 150))
        draw_dots(pil_img, landmarks)
        pil_img.save("dataset_check/{}.jpg".format(str(i).zfill(3)))


def visualize(model: nn.Module, epoch=0, batch_id=0):
    model.eval()
    val_data_loader = DataLoader(dataset=FacialLandmarkValidationDataset(),
                                 batch_size=1,
                                 shuffle=True)
    img, target_bboxes = next(iter(val_data_loader))
    target_bboxes = target_bboxes.to(device).squeeze(0)
    img = img.squeeze(0)
    pil_img = Image.fromarray(img.cpu().numpy())
    img_for_predict = torch_img_normalization(pil_img).to(device).unsqueeze(0)
    confs, pred_boxes, landmarks = model(img_for_predict)

    draw_box(pil_img, pred_boxes, color=(0, 0, 255, 150), confs=confs)
    draw_box(pil_img, target_bboxes, color=(0, 255, 0, 150))
    draw_dots(pil_img, landmarks)

    save_img_path = os.path.join(config.model_visualization_dir, "epoch_{}_batch_{}.jpg".format(
        str(epoch).zfill(3),
        str(batch_id).zfill(5)
    ))
    save_img_path_latest = os.path.join(config.model_visualization_dir, "latest.jpg")
    pil_img.save(save_img_path)
    pil_img.save(save_img_path_latest)
    model.train()


if __name__ == "__main__":
    # model = RetinaFace()
    visualize_training_data(n_images=100)
