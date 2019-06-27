import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import Discriminator, Generator, FCN, compute_gradient_penalty
from datasets import generate_real_samples, ImageDataset

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

os.makedirs("images", exist_ok=True)
img_shape = (opt.channels, opt.img_size, opt.img_size)
# img_shape = (1, 64, 64)

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# generate_real_samples(save_path="data/layout/")
dataloader = torch.utils.data.DataLoader(
    ImageDataset(
        "data/layout/",
        transforms_=[
            # transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ],
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [
#                 transforms.Resize(opt.img_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5], [0.5]),
#             ]
#             # [transforms.Resize(opt.img_size), transforms.ToTensor()]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )


def train_renderer():
    net = FCN()

    if cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-6)
    criterion = torch.nn.MSELoss()

    for step in range(10000):
        net.train()
        x = []
        gt = []

        # generate ground truth data
        for i in range(opt.batch_size):
            # _x = np.random.uniform(0, 1, action_dim)
            _x = np.array([0, 0, 0.5, 0.5])
            x.append(_x)
            gt.append(draw_fn(_x))

        x = torch.tensor(x).float()
        gt = torch.tensor(gt).float()
        if use_cuda:
            net = net.cuda()
            x = x.cuda()
            gt = gt.cuda()

        y = net(x)
        # print(gt)
        # print(y)
        optimizer.zero_grad()
        loss = criterion(y, gt)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(step, loss.item())

        # if step < 200000:
        #     lr = 1e-4
        # elif step < 400000:
        #     lr = 1e-5
        # else:
        #     lr = 1e-6
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr
        writer.add_scalar("train/loss", loss.item(), step)

        if step % 1000 == 0:
            save_model(net, output, use_cuda)

            net.eval()
            y = net(x)
            loss = criterion(y, gt)
            writer.add_scalar("val/loss", loss.item(), step)
            for i in range(32):
                G = y[i].cpu().data.numpy()
                GT = gt[i].cpu().data.numpy()
                writer.add_image("train/gen{}.png".format(i), G, step)
                writer.add_image("train/ground_truth{}.png".format(i), GT, step)


def train_wgan():
    # Loss weight for gradient penalty
    lambda_gp = 10

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

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
        # for i, (imgs) in enumerate(dataloader):
        for i, (imgs, _) in enumerate(dataloader):
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(
                Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))
            )

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs.data, fake_imgs.data
            )
            # Adversarial loss
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + lambda_gp * gradient_penalty
            )

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

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

                if batches_done % opt.sample_interval == 0:
                    save_image(
                        fake_imgs.data[:25],
                        "images/%d.png" % batches_done,
                        nrow=5,
                        normalize=True,
                    )

                batches_done += opt.n_critic


train_wgan()
