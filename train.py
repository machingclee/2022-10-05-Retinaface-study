import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import time
import math
import src.config as config
from arg_parser import get_args_parser
from data.wflw import WFLWDatasets
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from tqdm import tqdm
from src.models.retinaface import RetinaFace
from visualize import visualize_model_on_validation_data
from data.wflw import collate_fn
from pathlib import Path
from data import cfg_mnet, cfg_re50
from arg_parser import get_args


def train(args):
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    num_classes = 2
    img_dim = cfg['image_size']
    num_gpu = cfg['ngpu']
    batch_size = args.batch_size
    max_epoch = cfg['epoch']
    gpu_train = cfg['gpu_train']

    momentum = args.momentum
    weight_decay = args.weight_decay
    initial_lr = args.lr
    gamma = args.gamma
    save_folder = args.save_folder

    model = RetinaFace(cfg=cfg)
    print("Printing net...")
    print(model)

    if args.checkpoint is not None:
        print('Loading resume network...')
        state_dict = torch.load(args.checkpoint)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    if num_gpu > 1 and gpu_train:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    cudnn.benchmark = True

    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))

    priors = priorbox.forward()
    priors = priors.cuda()

    model.train()
    print('Loading Dataset...')

    dataset = WFLWDatasets()

    epoch_size = math.ceil(len(dataset) / batch_size)

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    for epoch in range(args.start_epoch, max_epoch):
        # batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
        batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn))
        load_t0 = time.time()
        if epoch in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(initial_lr, optimizer, gamma, epoch, step_index, epoch, epoch_size)

        # load train data
        for batch_id, (images, targets) in enumerate(tqdm(batch_iterator)):
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

            # forward
            out = model(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()
            load_t1 = time.time()
            batch_time = load_t1 - load_t0

            if batch_id % config.visualize_result_per_batch == 0 and batch_id > 0:
                with torch.no_grad():
                    if batch_id % config.visualize_result_per_batch == 0 and batch_id > 0:
                        visualize_model_on_validation_data(model, epoch, batch_id)

        torch.save(model.state_dict(), save_folder + cfg['name'] + "_{}.pth".format(str(epoch).zfill(3)))


def adjust_learning_rate(initial_lr, optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    args = get_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train(args)
