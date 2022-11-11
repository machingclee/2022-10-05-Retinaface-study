import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from layers.functions.prior_box import PriorBox
from src import config
from src.device import device
from utils.box_utils import decode, decode_landm
from data import cfg_mnet, cfg_re50
from torchvision.ops import nms

cfg = cfg_re50


class RegressionParser(nn.Module):
    prior_boxes = None

    def __init__(self, image_size=(config.input_img_size, config.input_img_size)):
        super(RegressionParser, self).__init__()
        self.image_size = image_size

    def get_prior_boxes(self):
        if RegressionParser.prior_boxes is None:
            RegressionParser.prior_boxes = PriorBox(cfg, image_size=self.image_size).forward().to(device)

        return RegressionParser.prior_boxes

    def forward(self, bboxes, scores, landms, pred_thres=torch.as_tensor([config.pred_thres])):
        resize_shape = [config.cam_width, config.cam_height] if config.onnx_ongoing else self.image_size
        priors = self.get_prior_boxes()
        bboxes = bboxes.squeeze(0)
        scores = scores.squeeze(0)
        landms = landms.squeeze(0)
        pred_thres = pred_thres.squeeze(0)
        scale = torch.as_tensor([resize_shape * 2] * config.n_priors, dtype=torch.float32).to(device)
        bboxes = decode(bboxes, priors, cfg['variance'])
        bboxes = bboxes * scale
        landms = decode_landm(landms, priors, cfg['variance'])
        scale_landm = torch.as_tensor([resize_shape * config.n_landmarks] *
                                      config.n_priors, dtype=torch.float32).to(device).float()
        landms = landms * scale_landm
        keep_ = nms(bboxes, scores, config.final_nms_iou)[0: config.rpn_n_sample]
        keep = keep_[torch.where(scores[keep_] > pred_thres)[0]]
        bboxes = bboxes[keep]
        scores = scores[keep]
        landms = landms[keep]
        return bboxes, scores, landms


class SimplifiedRegressionParser(nn.Module):
    prior_boxes = None

    def __init__(self, image_size=(config.input_img_size, config.input_img_size)):
        super(SimplifiedRegressionParser, self).__init__()
        self.image_size = image_size

    def get_prior_boxes(self):
        if RegressionParser.prior_boxes is None:
            RegressionParser.prior_boxes = PriorBox(cfg, image_size=self.image_size).forward().to(device)

        return RegressionParser.prior_boxes

    def forward(self, bboxes, scores, landms):
        resize_shape = [config.cam_width, config.cam_height] if config.onnx_ongoing else self.image_size
        priors = self.get_prior_boxes()
        bboxes = bboxes.squeeze(0)
        scores = scores.squeeze(0)
        landms = landms.squeeze(0)

        index = torch.argmax(scores)

        scale = torch.as_tensor([resize_shape * 2], dtype=torch.float32).to(device)
        bboxes = decode(bboxes[index][None], priors[index][None], cfg['variance'])
        bboxes = bboxes * scale
        landms = decode_landm(landms[index][None], priors[index][None], cfg['variance'])
        scale_landm = torch.as_tensor([resize_shape * config.n_landmarks], dtype=torch.float32).to(device).float()
        landms = landms * scale_landm

        return bboxes, scores[index][None], landms
