from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from src.models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from src.regression_parser import RegressionParser, SimplifiedRegressionParser
from repo.pytorch2keras.pytorch2keras.pytorch2keras import pytorch_to_keras
from src import config
from onnx import load
from onnx2keras import onnx_to_keras


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-m', '--checkpoint', default='weights/mobilenet0.25_044.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--long_side', default=840, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

    args = parser.parse_args()

    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    # net and model
    retina_face = RetinaFace(cfg=cfg)
    retina_face = load_model(retina_face, args.checkpoint, args.cpu)
    retina_face.eval()

    regression_parser = SimplifiedRegressionParser()
    regression_parser.eval()

    device = torch.device("cpu" if args.cpu else "cuda")
    retina_face = retina_face.to(device)

    # ------------------------ export -----------------------------
    dummy_inputs_retina_face = torch.randn(
        1, 3, config.input_img_size, config.input_img_size
    )
    dummy_inputs_regression_parser = (
        torch.randn(1, config.n_priors, 4),
        torch.randn(1, config.n_priors),
        torch.randn(1, config.n_priors, 196)
    )

    # x = torch.randn(1, 3, 224, 224, requires_grad=False)
    # k_model = pytorch_to_keras(net, x, [(3, None, None,)], verbose=True, names='short')
    # k_model.save('keras.h5')

    # k_model = pytorch_to_keras(retina_face,
    #                            dummy_inputs_retina_face,
    #                            [(3, config.input_img_size, config.input_img_size)],
    #                            name_policy='renumerate',
    #                            verbose=True)
    # k_model.save("RetinaFace.h5")

    # torch.onnx.export(
    #     retina_face,
    #     dummy_inputs_retina_face,
    #     'FaceDetector.onnx',
    #     export_params=True,
    #     verbose=False,
    #     input_names=["inputImage"],
    #     output_names=["bbox_regressions", "scores", "ldm_regressions"],
    #     opset_version=11)

    torch.onnx.export(
        regression_parser,
        args=dummy_inputs_regression_parser,
        f='RegressionParser.onnx',
        export_params=True,
        verbose=False,
        input_names=["bbox_regressions", "scores", "landm_regressions"],
        output_names=["boxes", "scores", "landms"],
        opset_version=11)
