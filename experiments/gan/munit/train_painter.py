# %%writefile /content/CoordConv/experiments/gan/munit/train_painter.py
import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import (
    Discriminator,
    compute_gradient_penalty,
    CoordConvPainter,
    FCN,
)
from datasets import generate_real_samples, ImageDataset
from strokes import sampler, draw_rect

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
parser.add_argument("--data_path", type=str, default="data/layout/")
parser.add_argument("--save_path", type=str, default="saved_models/")

opt = parser.parse_args()
print(opt)

os.makedirs("images", exist_ok=True)
os.makedirs(opt.data_path, exist_ok=True)
os.makedirs(opt.save_path, exist_ok=True)
img_shape = (opt.channels, opt.img_size, opt.img_size)

sampler(draw_rect, n_samples=100, save_path=opt.data_path)
dataloader = torch.utils.data.DataLoader(
    ImageDataset(
        opt.data_path,
        transforms_=[
            # transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ],
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


class Generator(nn.Module):
    def __init__(self, in_dim=100):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # *block(in_dim, 128, normalize=False),
            nn.Linear(in_dim, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            *block(128, 128, normalize=False),
            *block(128, 256, normalize=False),
            *block(256, 512, normalize=False),
            *block(512, 1024, normalize=False),
            nn.Linear(1024, int(np.prod(img_shape))),
            # nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

def train_renderer():
    # net = CoordConvPainter(in_dim=4)
    net = FCN(in_dim=opt.latent_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    if cuda:
        net.cuda()

    for epoch in range(opt.n_epochs):
        for i, (gt, x) in enumerate(dataloader):
            if cuda:
                gt = gt.cuda()
                x = x.cuda()
            net.train()
            y = net(x)
            optimizer.zero_grad()
            loss = criterion(y, gt)
            loss.backward()
            optimizer.step()

            if i % 1 == 0:
                print(epoch, i, loss.item())
                print(x[0])
                # print(gt[0])
                # print(y[0])
                # save_image(gt.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)
                # save_image(y.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)

        # if epoch % 10 == 0 and epoch > 0:
        if epoch % 1 == 0:
            # torch.save(net.state_dict(), "saved_models/layout/renderer.pth")
            save_image(y.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)


def train_wgan():
    # Loss weight for gradient penalty
    lambda_gp = 10

    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim)
    discriminator = Discriminator()
    loss_restore = torch.nn.MSELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    # Training
    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, xs) in enumerate(dataloader):
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            # z = Variable(
            #     Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
            # )
            # z = Variable(Tensor(xs))
            z = xs
            if cuda:
                z = z.cuda()
                imgs = imgs.cuda()

            # Generate a batch of images
            fake_imgs = generator(z)

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

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            if i % opt.n_critic == 0:
                fake_imgs = generator(z)
                fake_validity = discriminator(fake_imgs)
#                 y = fake_imgs * imgs  # use ground truth image as filter
#                 g_loss = -torch.mean(fake_validity) + 10 * loss_restore(y, imgs)
                g_loss = -torch.mean(fake_validity) + 10 * loss_restore(fake_imgs, imgs)
                # g_loss = -torch.mean(fake_validity)
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
#                     torch.save(
#                         generator.state_dict(),
#                         os.path.join(opt.save_path, "%d.pt" % batches_done),
#                     )

                batches_done += opt.n_critic


train_wgan()
# train_renderer()
