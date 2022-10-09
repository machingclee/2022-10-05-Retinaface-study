import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import config
import torch.optim as optim
import os
from tqdm import tqdm
from dataset import FacialLandmarkTrainingDataset
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from retina_face import RetinaFace
from console_log import ConsoleLog
from visualize import visualize
console_log = ConsoleLog(lines_up_on_end=1)


def train(model: Optional[RetinaFace] = None,
          lr=1e-6,
          start_epoch=1,
          epoches=20,
          save_weight_epoch_interval=1):
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    data_loader = DataLoader(dataset=FacialLandmarkTrainingDataset(),
                             batch_size=config.batch_size,
                             shuffle=True)

    for epoch in range(epoches):
        epoch = epoch + start_epoch
        for batch_id, (img, bboxes, landmarks, _) in enumerate(tqdm(data_loader,
                                                                    desc="Epoch {}".format(epoch),
                                                                    bar_format=config.bar_format)):
            try:
                bboxes = bboxes.squeeze(0)
                landmarks = landmarks.squeeze(0)

                rpn_cls_loss, rpn_reg_loss, rpn_landm_reg_loss = model(img, bboxes, landmarks)
                rpn_cls_loss *= 10
                rpn_reg_loss *= 10
                total_loss = rpn_cls_loss + rpn_reg_loss + rpn_landm_reg_loss

                if torch.isnan(total_loss):
                    print("nan loss, skip this image")
                    continue

                opt.zero_grad()
                total_loss.backward()
                opt.step()

                with torch.no_grad():
                    console_log.print([
                        ("total_loss", total_loss.item()),
                        ("-rpn_cls_loss", rpn_cls_loss.item()),
                        ("-rpn_reg_loss", rpn_reg_loss.item()),
                        ("-rpn_landm_reg_loss", rpn_landm_reg_loss.item())
                    ])

                if batch_id % config.save_per_batches == 0 and batch_id > 0:
                    visualize(model, epoch, batch_id)
            except Exception as err:
                print(f"{err}")

        if epoch % save_weight_epoch_interval == 0:
            state_dict = model.state_dict()
            model_path = "pths/{}.pth".format(str(epoch).zfill(3))
            torch.save(state_dict, model_path)
            model_path = "pths/{}.pth".format(str(epoch).zfill(3))
            model.load_state_dict(torch.load(model_path))


if __name__ == "__main__":
    model_path = None
    model = RetinaFace()

    if model_path is not None:
        print(f"Loading model's weight from {model_path}")
        model.load_state_dict(torch.load(model_path))

    model.train()
    train(model, start_epoch=1, lr=1e-6)
