import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from faker import Faker

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_epochs", type=int, default=200, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
)
parser.add_argument(
    "--img_size", type=int, default=28, help="size of each image dimension"
)
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(
    "--n_critic",
    type=int,
    default=50,
    help="number of training steps for discriminator per iter",
)
parser.add_argument(
    "--clip_value",
    type=float,
    default=0.01,
    help="lower and upper clip value for disc. weights",
)
parser.add_argument(
    "--sample_interval", type=int, default=400, help="interval betwen image samples"
)
opt = parser.parse_args()
print(opt)

# ------ Configure data loader --------


def sampler():
    x, y = [], []
    for i in range(opt.img_size):
        x.append(torch.tensor(i))
        y.append(F.one_hot(torch.tensor(i), opt.img_size))
    x = torch.stack(x).view(-1, 1).float()
    y = torch.stack(y).float()
    return x, y


def sampler_linear():
    x, y = [], []
    for i in range(opt.img_size):
        x.append(torch.tensor([i]))
        y.append(torch.tensor([i - 1, i + 1]))
    x = torch.stack(x).float()
    y = torch.stack(y).float()
    return x, y


# x, y = sampler_linear()
x, y = sampler()
dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x, y), batch_size=opt.batch_size, shuffle=True
)


# ------ Models --------


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(1, 2))

    def forward(self, x):
        return self.model(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid())
        self.criterion = torch.nn.MSELoss()

    def loss(self, y, x1, x2, gt):
        loss_restore = self.criterion(y, gt)
        loss_coord = torch.mean(F.relu(-(x2 - x1 - 1.0)))
        loss = loss_restore + loss_coord
        # print(loss_restore.data, loss_coord.data)
        return loss

    def forward(self, x):
        l = opt.img_size
        N = x.shape[0]
        x = self.model(x)
        x1 = x[:, 0].view(-1, 1) * l
        x2 = x[:, 1].view(-1, 1) * l
        # x1 = x - 1.0
        # x2 = x + 1.
        _x1 = F.relu6((torch.arange(l).expand(N, -1).float() - x1) * 6.0)
        _x2 = F.relu6((x2 - torch.arange(l).expand(N, -1).float()) * 6.0)
        y = _x1 * _x2
        y = y / 36.0  # normalize again after relu6 (multiply by 6.)
        return y, x1, x2


# ------ Train --------


def train():
    net = Net()
    # net = LinearNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    if cuda:
        net.cuda()

    for epoch in range(opt.n_epochs):
        for i, (x, gt) in enumerate(dataloader):
            if cuda:
                gt = gt.cuda()
                x = x.cuda()
            net.train()
            y, x1, x2 = net(x)
            # y = net(x)
            optimizer.zero_grad()
            # loss = criterion(y, gt)
            loss = net.loss(y, x1, x2, gt)
            loss.backward()
            optimizer.step()

        # if epoch % 10 == 0 and epoch > 0:
        if epoch % 100 == 0:
            print(epoch, i, loss.item())
            # print(x[0])
            # print(x1)
            # print(x2)
            # print(gt[0])
            # print(y[0])
            # torch.save(net.state_dict(), "saved_models/layout/renderer.pth")
            # save_image(y.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)


train()
