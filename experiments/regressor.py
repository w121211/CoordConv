import os
import itertools

import numpy as np
from PIL import Image, ImageDraw
import skimage.draw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import torch.utils.data as utils
import torch.nn.functional as F
from torchsummary import summary

from coordconv import AddCoords
from hrnet import HighResolutionNet
from config import get_cfg_defaults
from data_factory import generate_layer_data, load_data


class Regressor(nn.Module):
    def __init__(self, in_channel, width, latent_size=256, n_class=3, bbox_size=4):
        super(Regressor, self).__init__()
        self.add_coords = AddCoords(rank=2)
        self.conv_class = nn.Sequential(
            nn.Conv2d(in_channel + 2, latent_size, 1, stride=1, padding=0),
            nn.BatchNorm2d(latent_size),
            nn.ReLU(),
            nn.Conv2d(latent_size, latent_size, 1, stride=1, padding=0),
            nn.BatchNorm2d(latent_size),
            nn.ReLU(),
            # nn.Conv2d(latent_size, n_class, 1, stride=1, padding=0),
            # nn.Linear(latent_size, n_class, 1, stride=1, padding=0),
        )
        self.conv_bbox = nn.Sequential(
            nn.Conv2d(in_channel + 2, latent_size, 1, stride=1, padding=0),
            nn.BatchNorm2d(latent_size),
            nn.ReLU(),
            nn.Conv2d(latent_size, latent_size, 1, stride=1, padding=0),
            nn.BatchNorm2d(latent_size),
            nn.ReLU(),
            # nn.Conv2d(latent_size, bbox_dim, 1, stride=1, padding=0),
        )
        self.fc_class = nn.Linear((width ** 2) * latent_size, n_class)
        self.fc_bbox = nn.Linear((width ** 2) * latent_size, bbox_size)

    def forward(self, x):
        N = x.shape[0]
        x = self.add_coords(x)
        x_class = self.conv_class(x)
        x_class = x_class.view(N, -1)
        x_class = self.fc_class(x_class)
        x_bbox = self.conv_bbox(x)
        x_bbox = x_bbox.view(N, -1)
        x_bbox = self.fc_bbox(x_bbox)
        return x_class, x_bbox


class HRNet(nn.Module):
    def __init__(self, width):
        super(HRNet, self).__init__()
        cfg = get_cfg_defaults()
        cfg.merge_from_file("./experiments/hrnet.yaml")
        self.hr0 = HighResolutionNet(cfg)
        self.hr1 = HighResolutionNet(cfg)
        self.head = Regressor(in_channel=540, width=int(width / 4), n_class=5)
        # self.head = Regressor(in_channel=270, width=int(width / 4))

    def forward(self, x):
        im_current = x[:, :3, :, :]
        im_target = x[:, -3:, :, :]
        x0 = self.hr0(im_current)
        x1 = self.hr1(im_target)
        x = torch.cat((x0, x1), dim=1)
        # x = self.hr0(x)
        x = self.head(x)
        return x


def train(epoch, net, train_dataloader, optimizer, loss_fn, device):
    net.train()
    iters = 0
    for batch_idx, (x, y_target_class, y_target_bbox) in enumerate(train_dataloader):
        # print('-------')
        print(x.shape)
        print(y_target_class.shape)
        print(y_target_bbox.shape)
        x, y_target_class, y_target_bbox = (
            x.to(device),
            y_target_class.to(device),
            y_target_bbox.to(device),
        )
        optimizer.zero_grad()
        y = net(x)

        loss = loss_fn(y, y_target_class, y_target_bbox)
        loss.backward()
        optimizer.step()
        iters += len(x)
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                epoch,
                iters,
                len(train_dataloader.dataset),
                100.0 * (batch_idx + 1) / len(train_dataloader),
                loss.data.item(),
            ),
            end="\r",
            flush=True,
        )
    # print(y)
    # print(y_target[0])
    print("")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    width, height = 64, 64
    generate_layer_data(n_sample=100)
    train_label, train_bbox, train_im, test_label, test_bbox, test_im = load_data()
    train_dataset = utils.TensorDataset(
        torch.from_numpy(train_im),
        torch.from_numpy(train_label),
        torch.from_numpy(train_bbox),
    )
    train_dataloader = utils.DataLoader(train_dataset, batch_size=8, shuffle=True)
    # test_dataset = utils.TensorDataset(test_tensor_y, test_tensor_x)
    # test_dataloader = utils.DataLoader(test_dataset, batch_size=32, shuffle=False)

    def _loss(y, y_target_class, y_target_bbox):
        cls_score, bbox = y
        # print(cls_score)
        # print(bbox)
        loss_cls = F.cross_entropy(cls_score, y_target_class)
        loss_bbox = F.smooth_l1_loss(bbox, y_target_bbox)
        loss_sum = loss_cls + loss_bbox
        return loss_sum

    # model = SimpleNet(width).to(device)
    model = HRNet(width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = _loss
    epochs = 1000

    for epoch in range(1, epochs + 1):
        train(epoch, model, train_dataloader, optimizer, loss_fn, device)
