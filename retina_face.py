import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models._utils as _utils
import torch.nn.functional as F
import config
from einops import rearrange, reduce, repeat
from feature_extractor import Resnet50FPNFeactureExtractor
from ssh import SSH
from anchor_generator import AnchorGenerator
from box_utils import assign_targets_to_anchors_or_proposals, decode_deltas_to_boxes, clip_boxes_to_image, encode_boxes_to_deltas, encode_landm
from device import device
from utils import smooth_l1_loss

cce_loss = nn.CrossEntropyLoss()


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.backbone = Resnet50FPNFeactureExtractor()
        out_channels = self.backbone.fpn_out_channels
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channels)

        anchor_gen = AnchorGenerator()
        self.multi_scale_anchors = anchor_gen.get_multi_scale_anchors()
        self.flattened_multi_scale_anchors = anchor_gen.get_flattened_multi_scale_anchors()

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=config.num_anchors):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=config.num_anchors):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=config.num_anchors):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def get_multibox_loss(
        self,
        target_boxes,
        target_landmarks,
        flattened_pred_deltas,
        flattened_pred_fg_bg_logit,
        flattened_landm_predictions
    ):
        flattend_labels, flattended_distributed_targets, flattened_multi_scale_distributed_landmarks, _ = \
            assign_targets_to_anchors_or_proposals(
                target_boxes,
                target_landmarks,
                self.multi_scale_anchors,
                n_sample=config.rpn_n_sample,
                pos_sample_ratio=config.rpn_pos_ratio,
                pos_iou_thresh=config.target_pos_iou_thres,
                neg_iou_thresh=config.target_neg_iou_thres,
                target_cls_indexes=None
            )
        flattend_labels = flattend_labels.to(device)
        pos_mask = flattend_labels == 1
        keep_mask = torch.abs(flattend_labels) == 1

        target_deltas = encode_boxes_to_deltas(flattended_distributed_targets, self.flattened_multi_scale_anchors)
        target_landmarks = encode_landm(flattened_multi_scale_distributed_landmarks, self.flattened_multi_scale_anchors)
        objectness_label = torch.zeros_like(flattend_labels, device=device, dtype=torch.long)
        objectness_label[flattend_labels == 1] = 1.0

        if torch.sum(pos_mask) > 0:
            rpn_reg_loss = smooth_l1_loss(flattened_pred_deltas[pos_mask], target_deltas[pos_mask])
            rpn_landm_reg_loss = smooth_l1_loss(flattened_landm_predictions[pos_mask], target_landmarks[pos_mask])
        else:
            rpn_reg_loss = torch.sum(flattened_pred_deltas) * 0

        rpn_cls_loss = cce_loss(flattened_pred_fg_bg_logit.squeeze(0)[keep_mask], objectness_label[keep_mask])

        return rpn_cls_loss, rpn_reg_loss, rpn_landm_reg_loss

    def get_cls_losses(self):
        pass

    def get_landm_losses(self):
        pass

    def forward(self, inputs, target_boxes=None, target_landmarks=None):
        # FPN
        fpn = self.backbone(inputs)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = []
        classification_logits = []
        landdm_regressions = []

        for i, feature in enumerate(features):
            bboxes = self.BboxHead[i](feature)
            classes = self.ClassHead[i](feature)
            ldmarks = self.LandmarkHead[i](feature)
            bbox_regressions.append(bboxes)
            classification_logits.append(classes)
            landdm_regressions.append(ldmarks)

        if self.training:
            assert target_boxes is not None, "target_boxes should not be none in training"

            if target_boxes is not None:
                target_boxes = target_boxes.to(device)

            pred_fg_bg_logits = []
            pred_deltas = []
            pred_landms = []

            for deltas, logits, landms in zip(bbox_regressions, classification_logits, landdm_regressions):
                logits = logits.squeeze(0)
                deltas = deltas.squeeze(0)
                landms = landms.squeeze(0)

                pred_fg_bg_logits.append(logits)
                pred_deltas.append(deltas)
                pred_landms.append(landms)

            flattened_pred_fg_bg_logits = torch.cat(pred_fg_bg_logits, dim=0).to(device)
            flattened_pred_deltas = torch.cat(pred_deltas, dim=0).to(device)
            flattened_landm_predictions = torch.cat(pred_landms, dim=0).to(device)

            rois = decode_deltas_to_boxes(
                flattened_pred_deltas.detach().clone(),
                self.flattened_multi_scale_anchors
            )
            rois = clip_boxes_to_image(rois)
            rois = rois.squeeze(0)
            rpn_cls_loss, rpn_reg_loss, rpn_landm_reg_loss = self.get_multibox_loss(
                target_boxes,
                target_landmarks,
                flattened_pred_deltas,
                flattened_pred_fg_bg_logits,
                flattened_landm_predictions
            )
            return rpn_cls_loss, rpn_reg_loss, rpn_landm_reg_loss

        else:
            # bbox: [1, 64512, 4], where 64512 = (128*128 + 64*64 + 32*32) * (n_anchors=3),
            # similar holds for
            # classifications: [1, 64512, 2]
            # ldm_regressions: [1, 64512, 10] for 5 landmarks

            bbox_regressions = torch.cat(bbox_regressions, dim=1)
            classification_logits = torch.cat(classification_logits, dim=1)
            landdm_regressions = torch.cat(landdm_regressions, dim=1)
            return bbox_regressions, F.softmax(classification_logits, dim=-1), landdm_regressions


if __name__ == "__main__":
    retina_face = RetinaFace()
    retina_face.train()
    img = torch.randn((1, 3, 1024, 1024))
    target_boxes = torch.randn((1, 4))
    target_landmarks = torch.randn((1, 10))

    out = retina_face(img, target_boxes, target_landmarks)
    print(out)
