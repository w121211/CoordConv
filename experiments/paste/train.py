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

# os.makedirs("images", exist_ok=True)
# os.makedirs("data/layout/", exist_ok=True)
# os.makedirs("saved_models/layout", exist_ok=True)
# img_shape = (opt.channels, opt.img_size, opt.img_size)


def sampler():
    x, y = [], []
    for i in range(opt.img_size):
        x.append(torch.tensor(i))
        y.append(F.one_hot(torch.tensor(i), opt.img_size))
    x = torch.stack(x).view(-1, 1).float()
    y = torch.stack(y).float()
    return x, y


x, y = sampler()
dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x, y), batch_size=opt.batch_size, shuffle=True
)


# ------ Models --------


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            # nn.Linear(opt.img_size * 2, 32),
            nn.Linear(1, 16),
            # *block(opt.latent_dim, 128, normalize=False),
            *block(16, 16, normalize=True),
            *block(16, 16, normalize=True),
            *block(16, 16, normalize=True),
            *block(16, 16, normalize=True),
            *block(16, 16, normalize=True),
            nn.Linear(16, opt.img_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # _n = x.shape[0]
        # x = x.view(-1, 1).expand(_n, opt.img_size)
        # coord = torch.arange(opt.img_size).float().expand(_n, opt.img_size)
        # x = torch.cat((x, coord), 1)
        # print(x)
        x = self.model(x)
        return x.view(-1, opt.img_size)


# ------ Train --------


def train():
    net = Net()
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
            y = net(x)
            optimizer.zero_grad()
            loss = criterion(y, gt)
            loss.backward()
            optimizer.step()

        # if epoch % 10 == 0 and epoch > 0:
        if epoch % 100 == 0:
            print(epoch, i, loss.item())
            # print(x[0])
            # print(gt[0])
            # print(y[0])
            # torch.save(net.state_dict(), "saved_models/layout/renderer.pth")
            # save_image(y.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)


train()
