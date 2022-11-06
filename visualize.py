import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import os
import src.config as config
from torchsummary import summary
from PIL import Image, ImageDraw
from typing import Tuple
from data.wider_face import WiderFaceDetection
from data.wflw import WFLWDatasets
from torch.utils.data import Dataset, DataLoader
from src.device import device
from detect import detect, load_model
from data import cfg_mnet, cfg_re50
from src.models.retinaface import RetinaFace
from data.data_augment import torch_imgnet_denormalization_to_pil


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


def draw_dots(pil_img: Image.Image, pred_boxes, pred_landmarks: Tuple[float], r=config.landm_dot_radius, constrain_pts=False):
    draw = ImageDraw.Draw(pil_img)
    for bbox, landmark in zip(pred_boxes, pred_landmarks):
        xmin, ymin, xmax, ymax = bbox
        for x, y in np.array_split(landmark, config.n_landmarks):
            if not constrain_pts:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
            else:
                if xmin <= x and x <= xmax and ymin <= y and y <= ymax:
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))


def visualize_training_data(n_images: int):
    train_data_loader = DataLoader(dataset=WFLWDatasets(file_list=[config.WFLW_TRAIN_LABEL_TXT, config.WFLW_VAL_LABEL_TXT],
                                                        img_dir=config.WFLW_TRAIN_IMG_DIR),
                                   batch_size=1,
                                   shuffle=True)
    train_iter = iter(train_data_loader)
    for i in range(n_images):
        img, targets = next(iter(train_iter))
        _, _, im_height, im_width = img.shape
        scale_bbox = torch.Tensor([im_width, im_height] * 2).to(device)
        scale_landm = torch.Tensor([im_width, im_height] * 98).to(device)
        target_bboxes = targets.squeeze(0)[:, 196:-1] * scale_bbox[None]
        target_landm = targets.squeeze(0)[:, 0:196] * scale_landm[None]
        pil_img = torch_imgnet_denormalization_to_pil(img)
        draw_box(pil_img, target_bboxes, color=(0, 0, 255, 150))
        draw_dots(pil_img, target_bboxes, target_landm)
        pil_img.save("dataset_check/{}.jpg".format(str(i).zfill(3)))


def visualize_model_on_validation_data(model: nn.Module, epoch=0, batch_id=0):
    model.eval()
    # val_data_loader = DataLoader(dataset=WiderFaceDetection(config.WIDER_VAL_LABEL_TXT, config.WIDER_VAL_IMG_DIR, mode="val"),
    #                              batch_size=1,
    #                              shuffle=True)
    val_data_loader = DataLoader(dataset=WFLWDatasets(file_list=[config.WFLW_TRAIN_LABEL_TXT, config.WFLW_VAL_LABEL_TXT],
                                                      img_dir=config.WFLW_TRAIN_IMG_DIR),
                                 batch_size=1,
                                 shuffle=True)
    img, targets = next(iter(val_data_loader))
    _, _, im_height, im_width = img.shape
    scale = torch.Tensor([im_width, im_height] * 2).to(device)
    pred_img = img.clone()
    target_bboxes = targets.squeeze(0)[:, 196:-1] * scale[None]
    confs, pred_boxes, pred_landmarks = detect(model, pred_img)
    pil_img = torch_imgnet_denormalization_to_pil(img)

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
    # net = RetinaFace(cfg=cfg_re50, phase='test').to(device)
    # visualize_model_on_validation_data(net)
    visualize_training_data(100)
