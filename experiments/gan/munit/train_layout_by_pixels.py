# %%writefile /content/CoordConv/experiments/gan/munit/train_layout.py
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
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import Discriminator, compute_gradient_penalty, CoordConvPainter, FCN
from datasets import generate_real_samples, ImageDataset
from strokes import sampler, draw_rect

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
    "--latent_dim", type=int, default=10, help="dimensionality of the latent space"
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
parser.add_argument("--data_path", type=str, default="data/layout")
parser.add_argument("--model_path", type=str, default="saved_models/95000.pt")

opt = parser.parse_args()
print(opt)

# -------------------------------
# Define models
# -------------------------------

class Paste2d(nn.Module):
    def __init__(self, im_size):
        super(Paste2d, self).__init__()
        self.model = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid())
        self.criterion = torch.nn.MSELoss()
        self.im_size = im_size

    def loss(self, y, x1, x2, gt):
        loss_restore = self.criterion(y, gt)
        loss_coord = torch.mean(F.relu(-(x2 - x1 - 1.0)))
        loss = loss_restore + loss_coord
        return loss

    def forward(self, x):
        l = self.im_size
        N = x.shape[0]

        x0 = x[:, 0].view(-1, 1) * l - 1.0
        y0 = x[:, 1].view(-1, 1) * l - 1.0
        x1 = x[:, 2].view(-1, 1) * l + 1.0
        y1 = x[:, 3].view(-1, 1) * l + 1.0
        
        coord = torch.arange(l).expand(N, -1).float()
        if cuda:
          coord = coord.cuda()
          
        _x0 = F.relu6((coord - x0) * 6.0)
        _x1 = F.relu6((x1 - coord) * 6.0)
        x_mask = (_x0 * _x1) / 36  # normalize again after relu6 (multiply by 6.)
        x_mask = x_mask.view(N, 1, l)
        
        _y0 = F.relu6((coord - y0) * 6.0)
        _y1 = F.relu6((y1 - coord) * 6.0)
        y_mask = (_y0 * _y1) / 36  # normalize again after relu6 (multiply by 6.)
        y_mask = y_mask.view(N, l, 1)  # align to y-axis
        
        mask = torch.ones(N,l,l)
        if cuda:
          mask = mask.cuda()
        mask = mask * x_mask * y_mask
        return mask.view(-1, 1, l, l)


class LayoutGenerator(nn.Module):
    def __init__(self, in_dim=10):
        super(LayoutGenerator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # *block(in_dim, 128, normalize=False),
            nn.Linear(in_dim, 64),
            *block(64, 64, normalize=False),
            *block(64, 64, normalize=False),
            *block(64, 4, normalize=False),
            nn.Linear(4, 4),
            nn.Sigmoid(),
        )

    def loss_coord(self, coord):
        x0 = coord[:, 0] * opt.img_size
        y0 = coord[:, 1]* opt.img_size
        x1 = coord[:, 2]* opt.img_size
        y1 = coord[:, 3]* opt.img_size
        loss = torch.mean(F.relu(-(x1 - x0 - 1.0))) + torch.mean(F.relu(-(y1 - y0 - 1.0)))
        return loss

    def forward(self, z):
        coord = self.model(z)
        # print(coord[0])
        return painter(coord), coord

# -------------------------------
# Dataset sampling
# -------------------------------

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

os.makedirs("images", exist_ok=True)
os.makedirs(opt.data_path, exist_ok=True)
# os.makedirs(opt.model_path, exist_ok=True)
img_shape = (opt.channels, opt.img_size, opt.img_size)

# painter = Generator(in_dim=4)
# painter.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
painter = Paste2d(opt.img_size)
if cuda:
  painter.cuda()
painter.eval()
for param in painter.parameters():
    param.requires_grad = False  # freeze weight


def sample():
    transform = transforms.ToPILImage()
    box_w, box_h = 20, 20
    w, h = opt.img_size, opt.img_size
    xs = [0, (w - box_w) / 2, w - box_w]
    ys = [0, (h - box_h) / 2, h - box_h]
    i = 0
    for x in xs:
        for y in ys:
            print(x, y)
            x0, y0 = x / w, y / h
            x1, y1 = (x + box_w) / w, (y + box_h) / h
            y = painter(torch.tensor([[x0, y0, x1, y1]]).float())
            im = transform(y[0])
            im.save("%s/%d.png" % (opt.data_path, i), "PNG")
            i += 1


def sample_center():
    transform = transforms.ToPILImage()
    box_range = [5, 30]
    w, h = opt.img_size, opt.img_size

    for i, box_w in enumerate(range(*box_range)):
        x0, y0 = (w - box_w) / 2, (h - box_w) / 2
        x1, y1 = x0 + box_w, y0 + box_w
        x = torch.tensor([[x0 / w, y0 / h, x1 / w, y1 / h]]).float()
        if cuda:
          x = x.cuda()
        y = painter(x)
        im = transform(y[0].cpu())
        im.save("%s/%d.png" % (opt.data_path, i), "PNG")


sample_center()
dataloader = torch.utils.data.DataLoader(
    ImageDataset(
        opt.data_path,
        transforms_=[
            # transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ],
        has_x=False,
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# -------------------------------
# Training GAN
# -------------------------------

def train_wgan():
    lambda_gp = 10

    generator = LayoutGenerator(opt.latent_dim)
    discriminator = Discriminator()
    loss_restore = torch.nn.MSELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    batches_done = 0
    for epoch in range(opt.n_epochs):
        # for i, (imgs, xs) in enumerate(dataloader):
        for i, (imgs, _) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(Tensor))

            #  Train Discriminator
            optimizer_D.zero_grad()

            z = Variable(
                Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
            )
            # z = Variable(Tensor(xs))
            # z = xs
            # if cuda:
            #     z = z.cuda()
            #     imgs = imgs.cuda()
            fake_imgs, coords = generator(z)
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs.data, fake_imgs.data
            )

            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + lambda_gp * gradient_penalty
            )
            d_loss.backward()
            optimizer_D.step()

            #  Train Generator
            optimizer_G.zero_grad()

            if i % opt.n_critic == 0:
                fake_imgs, coords = generator(z)
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity) + generator.loss_coord(coords)
                g_loss.backward()
                optimizer_G.step()

                if batches_done % opt.sample_interval == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (
                            epoch,
                            opt.n_epochs,
                            i,
                            len(dataloader),
                            d_loss.item(),
                            g_loss.item(),
                        )
                    )
                    save_image(
                        fake_imgs.data[:25],
                        "images/%d.png" % batches_done,
                        nrow=5,
                        normalize=True,
                    )
                    # torch.save(generator.state_dict(), opt.save_path)

                batches_done += opt.n_critic


train_wgan()
