import cv2
import numpy as np
import random
import torch
from utils.box_utils import matrix_iof
from torchvision import transforms
from PIL import Image
from src import config
from src.device import device

torch_imgnet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def torch_imgnet_denormalization_to_pil(img: torch.Tensor) -> Image.Image:
    mean = torch.as_tensor([0.485, 0.456, 0.406])[None, :, None, None].to(device)
    std = torch.as_tensor([0.229, 0.224, 0.225])[None, :, None, None].to(device)
    img = (img * std + mean) * 255
    img = Image.fromarray(img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype("uint8"))
    return img


def _crop(image, target_boxes, labels, landm, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi_xyxy = np.array((l, t, l + w, t + h))

        value = matrix_iof(target_boxes, roi_xyxy[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        target_centers = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi_xyxy[:2] < target_centers, target_centers < roi_xyxy[2:]).all(axis=1)
        boxes_t = target_boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        landms_t = landm[mask_a].copy()
        landms_t = landms_t.reshape([-1, config.n_landmarks, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi_xyxy[:2])
        boxes_t[:, :2] -= roi_xyxy[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi_xyxy[2:])
        boxes_t[:, 2:] -= roi_xyxy[:2]

        # landm
        landms_t[:, :, :2] = landms_t[:, :, :2] - roi_xyxy[:2]
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi_xyxy[2:] - roi_xyxy[:2])
        landms_t = landms_t.reshape([-1, config.n_landmarks * 2])

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, landms_t, pad_image_flag
    return image, target_boxes, labels, landm, pad_image_flag


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, config.n_landmarks, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, config.n_landmarks * 2])

    return image, boxes, landms


def _pad_to_square(image, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.zeros((long_side, long_side, 3), dtype=image.dtype)
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    return image


class preproc(object):

    def __init__(self, img_dim):
        self.img_dim = img_dim

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        target_boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 4:-1].copy()

        # this random crop can also change the landmarks, which is problematic in
        # albumentation (keyponits argument cause some incorrect augmented
        # annotation)
        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, target_boxes, labels, landm, self.img_dim)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t, self.rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape

        # change channel dimension ahead of height, width as well:
        # also pad to square:
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        # normalize bboxes and landmarks:
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))

        return image_t, targets_t


class wflw_preproc(object):

    def __init__(self, img_dim, rgb_means=(0, 0, 0)):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        landm_xy = targets[:, 0:196].copy()
        target_boxes_xyxy = targets[:, 196:200].copy()
        labels = targets[:, -1].copy()

        # this random crop can also change the landmarks, which is problematic in
        # albumentation (keyponits argument cause some incorrect augmented
        # annotation)
        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, target_boxes_xyxy, labels, landm_xy, self.img_dim)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t, pad_image_flag)
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape

        # change channel dimension ahead of height, width as well:
        image_t = _resize_subtract_mean(image_t, self.img_dim)
        image_t = torch_imgnet_transform(Image.fromarray(image_t.astype("uint8"))).to(device)

        # normalize bboxes and landmarks:
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((landm_t, boxes_t, labels_t))
        targets_t = torch.as_tensor(targets_t).to(device)

        return image_t, targets_t
