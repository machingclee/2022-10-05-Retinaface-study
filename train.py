import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import config
from tqdm import tqdm
from dataset import FacialLandmarkDataset
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from retina_face import RetinaFace
from console_log import ConsoleLog
console_log = ConsoleLog(lines_up_on_end=1)


def train(model: Optional[RetinaFace] = None,
          lr=1e-5, start_epoch=1,
          epoches=20,
          save_weight_interval=1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = DataLoader(dataset=FacialLandmarkDataset(),
                             batch_size=config.batch_size,
                             shuffle=True)

    for epoch in range(epoches):
        for batch_id, (img, bboxes, landmarks, _) in enumerate(tqdm(data_loader,
                                                                    desc="Epoch {}".format(epoch),
                                                                    bar_format=config.bar_format)):
            bboxes = bboxes.squeeze(0)
            landmarks = landmarks.squeeze(0)

            rpn_cls_loss, rpn_reg_loss, rpn_landm_reg_loss = model(img, bboxes, landmarks)
            total_loss = rpn_cls_loss + rpn_reg_loss + rpn_landm_reg_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            with torch.no_grad():
                console_log.print([
                    ("total_loss", total_loss.item()),
                    ("-rpn_cls_loss", rpn_cls_loss.item()),
                    ("-rpn_reg_loss", rpn_reg_loss.item()),
                    ("-roi_cls_loss", rpn_landm_reg_loss.item())
                ])


if __name__ == "__main__":
    model = RetinaFace()
    train(model)
