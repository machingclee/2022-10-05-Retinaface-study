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
from data.wider_face import WiderFaceDetection
from torch.utils.data import Dataset, DataLoader
from device import device
from detect import detect, load_model
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace


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


def draw_dots(pil_img: Image.Image, pred_boxes, pred_landmarks: Tuple[float], r=2, constrain_pts=False):
    draw = ImageDraw.Draw(pil_img)
    for bbox, landmark in zip(pred_boxes, pred_landmarks):
        xmin, ymin, xmax, ymax = bbox
        for x, y in np.array_split(landmark, 5):
            if not constrain_pts:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
            else:
                if xmin <= x and x <= xmax and ymin <= y and y <= ymax:
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
        draw_dots(pil_img, bboxes, landmarks)
        pil_img.save("dataset_check/{}.jpg".format(str(i).zfill(3)))


def visualize_model_on_validation_data(model: nn.Module, epoch=0, batch_id=0):
    model.eval()
    val_data_loader = DataLoader(dataset=WiderFaceDetection(config.WIDER_VAL_LABEL_TXT, config.WIDER_VAL_IMG_DIR, mode="val"),
                                 batch_size=1,
                                 shuffle=True)
    img, target_bboxes = next(iter(val_data_loader))
    pred_img = img.clone().squeeze(0)
    target_bboxes = target_bboxes.squeeze(0)[:, 0:4]
    confs, pred_boxes, pred_landmarks = detect(model, pred_img)
    pil_img = Image.fromarray(img.squeeze(0).clone().cpu().numpy())

    draw_box(pil_img, pred_boxes, color=(0, 0, 255, 150), confs=confs)
    draw_box(pil_img, target_bboxes, color=(0, 255, 0, 150))
    draw_dots(pil_img, pred_boxes, pred_landmarks, constrain_pts=config.constrain_landmarks_prediction_into_bbox)

    save_img_path = os.path.join(config.model_visualization_dir, "epoch_{}_batch_{}.jpg".format(
        str(epoch).zfill(3),
        str(batch_id).zfill(5)
    ))
    save_img_path_latest = os.path.join(config.model_visualization_dir, "latest.jpg")
    pil_img.save(save_img_path)
    pil_img.save(save_img_path_latest)
    model.train()


if __name__ == "__main__":
    net = RetinaFace(cfg=cfg_re50, phase='test').to(device)
    visualize_model_on_validation_data(net)
