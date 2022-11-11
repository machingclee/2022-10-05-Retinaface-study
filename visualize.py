import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import os
import src.config as config
from torchvision.ops import nms
from torchsummary import summary
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
from data.wider_face import WiderFaceDetection
from data.wflw import WFLWDatasets
from torch.utils.data import Dataset, DataLoader
from src.device import device
from detect import detect, load_model
from data import cfg_mnet, cfg_re50
from src.models.retinaface import RetinaFace
from data.data_augment import torch_imgnet_denormalization_to_pil
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from src.regression_parser import RegressionParser


def draw_box(pil_img: Image.Image, bboxes, scores=None, color=(255, 255, 255, 150)):
    draw = ImageDraw.Draw(pil_img)
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
        if scores is not None:
            conf = scores[i]
            draw.text(
                (xmin, max(ymin - 10, 4)),
                "{:.2f}".format(conf.item()),
                color
            )


def draw_dots(pil_img: Image.Image, pred_boxes, pred_landmarks: Tuple[float], r=config.landm_dot_radius, constrain_pts=False):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(config.font_path, config.landm_numbering_font_size)

    def draw_landm_num(x, y, i):
        draw.text((x - 4, y - 10), str(i), color=color, font=font)

    color = (255, 0, 0)
    for bbox, landmark in zip(pred_boxes, pred_landmarks):
        xmin, ymin, xmax, ymax = bbox
        for i, (x, y) in enumerate(np.array_split(landmark, config.n_landmarks)):
            if not constrain_pts:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
                draw_landm_num(x, y, i)
            else:
                if xmin <= x and x <= xmax and ymin <= y and y <= ymax:
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
                    draw_landm_num(x, y, i)


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


def decode_retina_output(bbox_regressions, scores, ldm_regressions, im_width, im_height, priorbox, score_threshold):
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    scale_landm = torch.Tensor([im_width, im_height] * config.n_landmarks).to(device)
    boxes = decode(bbox_regressions.squeeze(0), priorbox, cfg_re50['variance'])
    boxes = boxes * scale[None]
    landms = decode_landm(ldm_regressions.squeeze(0), priorbox, cfg_re50['variance']) * scale_landm[None]
    keep_ = nms(boxes, scores, config.final_nms_iou)[0: config.rpn_n_sample]
    keep = keep_[torch.where(scores[keep_] > score_threshold)[0]]

    boxes = boxes[keep]
    scores = scores[keep]
    landms = landms[keep]
    return boxes, scores, landms


def visualize_model_on_validation_data(model: nn.Module, epoch=0, batch_id=0, prefix=None):
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
    pil_img = torch_imgnet_denormalization_to_pil(img)

    target_bboxes = targets.squeeze(0)[:, 196:-1] * scale[None]
    bbox_regressions, scores, ldm_regressions = model(pred_img)
    regression_parser = RegressionParser(image_size=[im_height, im_width])
    pred_boxes, scores, pred_landmarks = regression_parser(bbox_regressions, scores, ldm_regressions)

    draw_box(pil_img, pred_boxes, color=(0, 0, 255, 150), scores=scores)
    draw_box(pil_img, target_bboxes, color=(0, 255, 0, 150))
    draw_dots(pil_img, pred_boxes, pred_landmarks, constrain_pts=config.constrain_landmarks_prediction_into_bbox)

    save_img_path = os.path.join(config.model_visualization_dir, "{}epoch_{}_batch_{}.jpg".format(
        prefix if prefix is not None else "",
        str(epoch).zfill(3),
        str(batch_id).zfill(5)
    ))

    save_img_path_latest = os.path.join(config.model_visualization_dir, "latest.jpg")
    pil_img.save(save_img_path)
    pil_img.save(save_img_path_latest)
    model.train()


if __name__ == "__main__":
    model = RetinaFace(cfg=cfg_re50).to(device)
    state_dict = torch.load("weights/Resnet50_076.pth")
    model.load_state_dict(state_dict, strict=False)
    visualize_model_on_validation_data(model)
