# %%writefile /content/CoordConv/experiments/gan/munit/train_layout_by_params.py
import argparse
import os
import numpy as np
import math
import sys
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from PIL import Image, ImageDraw, ImageFont
from faker import Faker

# from .train_gan_char import Generator as CharGenerator
from models import PasteLine, Discriminator, compute_gradient_penalty

# -------------------------------
# Toy experiment:
#   1. (z, token_status = (token_width, token_height)) -> G -> Line_coord (x0, y0)
#   2. (x0, y0) -> PasteLine -> Image
#   3. Image -> D -> y
#
# Dataset:
#   Real: token places at center of images
# -------------------------------


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

os.makedirs("images", exist_ok=True)
os.makedirs(opt.data_path, exist_ok=True)
# os.makedirs(opt.model_path, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

img_shape = (opt.channels, opt.img_size, opt.img_size)

# -------------------------------
# Define models
# -------------------------------


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
            *block(64, 2, normalize=False),
            nn.Linear(2, 2),
            nn.Sigmoid(),
        )
        # painter = Generator(in_dim=4)
        # painter.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
        painter = PasteLine(opt.img_size, max_chars=10)
        if cuda:
            painter.cuda()
        painter.eval()
        for param in painter.parameters():
            param.requires_grad = False  # freeze weight
        self.painter = painter

    def loss_coord(self, coord):
        x0 = coord[:, 0] * opt.img_size
        y0 = coord[:, 1] * opt.img_size
        x1 = coord[:, 2] * opt.img_size
        y1 = coord[:, 3] * opt.img_size
        loss = torch.mean(F.relu(-(x1 - x0 - 1.0))) + torch.mean(
            F.relu(-(y1 - y0 - 1.0))
        )
        return loss

    def forward(self, z, text_status, chars, char_sizes):
        x = torch.cat([z, text_status], dim=1)
        coord = self.model(x)
        return self.painter(coord, chars, char_sizes), coord


# -------------------------------
# Dataset sampling & init models
# -------------------------------


def text_to_char_images(
    text,
    font="/notebooks/post-generator/asset/fonts_en/Roboto/Roboto-Regular.ttf",
    font_size=14,
    out_size=14,
):
    font = ImageFont.truetype(font, font_size)
    transform = transforms.ToTensor()
    chars, sizes = [], []
    for c in text:
        size = font.getsize(c)
        im = Image.new("L", (out_size, out_size))
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, font=font, fill=255)
        chars.append(transform(im).unsqueeze(0))
        sizes.append(torch.Tensor(size).view(1, 2))
    return torch.cat(chars), torch.cat(sizes)


class MyDataset(Dataset):
    def __init__(self, img_size=opt.img_size, num_samples=100):
        self.transform = transforms.Compose(
            [
                # transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.font = "/notebooks/post-generator/asset/fonts_en/Roboto/Roboto-Regular.ttf"
        self.font_size = 14
        self.out_size = 14
        self.max_chars = 10
        self.img_size = img_size
        self.samples = self._sample(num_samples)

    def __getitem__(self, index):
        im, text_status, text = self.samples[index]
        chars, sizes = text_to_char_images(
            text[:self.max_chars], self.font, self.font_size, self.out_size
        )
        padding = self.max_chars - len(text)
        if padding > 0:
            chars = torch.cat(
                [chars, torch.zeros((padding, 1, self.out_size, self.out_size))]
            )
            sizes = torch.cat([sizes, torch.zeros((padding, 2))])
        return self.transform(im), torch.tensor(text_status).float(), chars, sizes
        # return self.transform(im)

    def __len__(self):
        return len(self.samples)

    def _sample(self, num_samples):
        fake = Faker()
        font = ImageFont.truetype(self.font, self.font_size)
        samples = []
        for _ in range(num_samples):
            tk = fake.word()
            tk_w, tk_h = font.getsize(tk)
            im = Image.new("L", (self.img_size, self.img_size))
            draw = ImageDraw.Draw(im)
            x0, y0 = (self.img_size - tk_w) / 2, (self.img_size - tk_h) / 2
            draw.text((x0, y0), tk, font=font, fill=255)
            # im.save("%s/%d.png" % (self.root, i), "PNG")
            samples.append((im, np.array([tk_w / self.img_size, tk_h / self.img_size]), tk))
        return samples


# -------------------------------
# Training GAN
# -------------------------------


def train_wgan():
    dataloader = torch.utils.data.DataLoader(
        MyDataset(num_samples=100), batch_size=opt.batch_size, shuffle=True
    )
    lambda_gp = 10

    generator = LayoutGenerator(opt.latent_dim + 2)
    discriminator = Discriminator(img_shape)
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
        for i, (real_imgs, text_status, chars, char_sizes) in enumerate(dataloader):
        # for i, real_imgs in enumerate(dataloader):
            #  Train Discriminator
            optimizer_D.zero_grad()

            z = torch.tensor(
                np.random.normal(0, 1, (real_imgs.shape[0], opt.latent_dim))
            ).float()
            if cuda:
                z = z.cuda()
                real_imgs = real_imgs.cuda()
                text_status = text_status.cuda()
                chars = chars.cuda()
                sizes = sizes.cuda()
            fake_imgs, coords = generator(z, text_status, chars, char_sizes)
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
                fake_imgs, coords = generator(z, text_status, chars, char_sizes)
                fake_validity = discriminator(fake_imgs)
                # g_loss = -torch.mean(fake_validity) + generator.loss_coord(coords)
                g_loss = -torch.mean(fake_validity)
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


if __name__ == "__main__":
    train_wgan()
