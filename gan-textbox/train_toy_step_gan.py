# %%writefile /content/CoordConv/gan-textbox/train_toy_param_comnist_vae.py
import os
import glob
import random
import time
import datetime
from collections import OrderedDict

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

# from models.craft import CRAFTGenerator
# from models.wgan import Generator, Discriminator, compute_gradient_penalty
from models.dcgan import DCDiscriminator
from config import get_parameters
from utils import tensor2var, denorm


# -------------------------------
# Dataset:
#   Real: token of ["A", "B", ..., "Z"] (ie only certain content code is valid) places at center
#   Fake: any content code & style code generated
#
# Toy experiment:
#   1. (z, token_status = (token_width, token_height)) -> G -> Line_coord (x0, y0)
#   2. (x0, y0) -> PasteLine -> Image
#   3. Image -> D -> y
# -------------------------------


class MyDataset(Dataset):
    def __init__(self, opt):
        self.imsize = opt.imsize
        self.trans_real = transforms.Compose(
            [
                transforms.Resize((opt.imsize, opt.imsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.trans_input = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.trans_mask = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.post = "../data/pin"
        # self.photo = "../data/facebook"
        self.photo = "../data/flickr"
        self.photos = sorted(glob.glob(self.photo + "/*.jpg"))[:1000]
        self.posts = sorted(glob.glob(self.post + "/*.jpg"))

    def __getitem__(self, index):
        """
        Returns:
            real_image
            input_image: (N, 1, H, W)
            mask: (N, 1, H, W)
        """
        real_img = Image.open(random.choice(self.posts))
        real_img = real_img.convert('RGB')

        im = Image.new("RGB", (self.imsize, self.imsize))
        p = Image.open(self.photos[index])

        if p.width > self.imsize or p.height > self.imsize:
            w = random.randint(int(self.imsize * 2 / 3), self.imsize - 5)
            p.thumbnail((w, w))

        x, y = int((self.imsize - p.width) / 2), int((self.imsize - p.height) / 2)
        im.paste(p, (x, y))

        mask = Image.new("L", (self.imsize, self.imsize))
        draw = ImageDraw.Draw(mask)
        draw.rectangle((x, y, x + p.width, y + p.height), fill=255)
        mask = PIL.ImageOps.invert(mask)

        try:
            return self.trans_real(real_img), self.trans_input(im), self.trans_mask(mask)
        except:
            print(real_img.mode)
            raise Exception()

    def __len__(self):
        return len(self.photos)


class VAEEncoder(nn.Module):
    def __init__(self, opt):
        super(VAEEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3 + 1, 16, 4, 2, 1),  # 16,32,32
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),  # 32,16,16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32,16,16 -> 64,8,8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64,8,8 -> 128,4,4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.AvgPool2d(kernel_size=2),
        )

        out_size = opt.imsize // 2 ** 4
        self.fc = nn.Linear(128 * out_size ** 2, opt.z_dim)

    def forward(self, x, mask):
        N = x.size(0)
        x = torch.cat([x, mask], dim=1)
        x = self.conv(x).view(N, -1)
        x = self.fc(x)
        # mu = self.fc1(x)
        # # logvar = self.fc2(x)
        # # std = torch.exp(0.5 * logvar)
        # # eps = torch.randn_like(std)
        # z = mu + eps * std
        #         return z, mu, logvar
        return x


class VAEDecoder(nn.Module):
    def __init__(self, opt):
        super(VAEDecoder, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(opt.z_dim, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 4096),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(16, 1, kernel_size=3, padding=1),
        #     nn.Sigmoid(),
        # )
        self.init_size = opt.imsize // 4
        self.l1 = nn.Sequential(nn.Linear(opt.z_dim, 128 * self.init_size ** 2))

        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.im_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        N = z.shape[0]
        x = self.l1(z).view(N, 128, self.init_size, self.init_size)
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # self.opt = opt
        self.imsize = opt.imsize
        self.device = opt.device

        self.model = nn.Sequential(
            nn.Linear(opt.z_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, opt.z_dim),
            nn.Linear(opt.z_dim, 3),  # to RGB
            nn.Tanh(),
        )

    def forward(self, features, x_img, mask):
        N = features.shape[0]
        rgb = self.model(features).view(N, 3, 1, 1)
        bg = rgb * torch.ones(N, 1, self.imsize, self.imsize).to(self.device)
        mask = mask.expand(-1, 3, -1, -1)
        im = mask * bg + (1.0 - mask) * x_img
        return im


def compute_gradient_penalty(D, real_samples, fake_samples, opt):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if len(real_samples.shape) == 4:
        alpha = (
            torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
            .float()
            .to(opt.device)
        )
    elif len(real_samples.shape) == 2:
        alpha = (
            torch.tensor(np.random.random((real_samples.size(0), 1)))
            .float()
            .to(opt.device)
        )
    else:
        raise Exception()

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = torch.ones((real_samples.shape[0], 1)).requires_grad_(False).to(opt.device)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# -------------------------------
# Training
# -------------------------------


class GANTrainer(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.dataloader = dataloader

        self.Enc = VAEEncoder(opt).to(opt.device)
        self.G = Generator(opt).to(opt.device)
        self.D = DCDiscriminator(opt).to(opt.device)
        # self.D_style = StyleDiscriminator(opt).to(opt.device)

        # Enc = VAEEncoder().to(opt.device)
        # Enc.load_state_dict(
        #     torch.load("./out/models/000120_vae_enc.pth", map_location="cpu")
        # )
        # for param in Enc.parameters():
        #     param.requires_grad = False
        # Enc.eval()
        # self.Enc = Enc

        self.enc_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.Enc.parameters()),
            opt.d_lr,
            [opt.beta1, opt.beta2],
        )
        self.g_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.G.parameters()),
            opt.g_lr,
            [opt.beta1, opt.beta2],
        )
        self.d_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D.parameters()),
            opt.d_lr,
            [opt.beta1, opt.beta2],
        )
        # self.d_style_optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.D_style.parameters()),
        #     opt.d_lr,
        #     [opt.beta1, opt.beta2],
        # )

        # if opt.pretrained_model is not None:
        #     self.load_pretrained_model()

    def train(self):
        lambda_gp = 10
        batches_done = 0
        # style_loss_decay = 0.95
        for epoch in range(opt.n_epochs):
            for i, (real_img, x_img, mask) in enumerate(dataloader):
                real_img = real_img.to(opt.device)
                x_img = x_img.to(opt.device)
                mask = mask.to(opt.device)
                # z = (
                #     torch.tensor(np.random.normal(0, 1, (real_img.shape[0], opt.z_dim)))
                #     .float()
                #     .to(opt.device)
                # )

                #  Train Discriminator
                features = self.Enc(x_img, mask)
                fake_img = self.G(features, x_img, mask)

                real_validity = self.D(real_img)
                fake_validity = self.D(fake_img)

                gp = compute_gradient_penalty(
                    self.D, real_img.detach(), fake_img.detach(), self.opt
                )

                d_loss = (
                    -torch.mean(real_validity)
                    + torch.mean(fake_validity)
                    + lambda_gp * gp
                )

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                #  Train Style-Discriminator
                # real_style, _, _ = self.Enc(real_img)

                # real_validity = self.D_style(real_style.detach())
                # fake_validity = self.D_style(fake_style.detach())

                # gp = compute_gradient_penalty(
                #     self.D_style,
                #     real_style.detach(),
                #     fake_style.detach(),
                #     self.opt,
                # )

                # d_style_loss = (
                #     -torch.mean(real_validity)
                #     + torch.mean(fake_validity)
                #     + lambda_gp * gp
                # )

                # self.d_style_optimizer.zero_grad()
                # d_style_loss.backward()
                # self.d_style_optimizer.step()

                # Train G every n_critic steps
                if i % opt.n_critic == 0:
                    # fake_img, fake_style = self.G(z)
                    features = self.Enc(x_img, mask)
                    fake_img = self.G(features, x_img, mask)
                    fake_validity = self.D(fake_img)
                    # fake_validity_style = self.D_style(fake_style)
                    # g_loss = -(
                    #     torch.mean(fake_validity) + torch.mean(fake_validity_style)
                    # )
                    g_loss = -torch.mean(fake_validity)

                    self.g_optimizer.zero_grad()
                    self.enc_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    self.enc_optimizer.step()

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
                            fake_img.data[:25],
                            os.path.join(
                                self.opt.sample_path,
                                "{:06d}_fake.png".format(batches_done),
                            ),
                            nrow=5,
                            normalize=True,
                        )
                        save_image(
                            real_img.data[:25],
                            os.path.join(
                                self.opt.sample_path,
                                "{:06d}_real.png".format(batches_done),
                            ),
                            nrow=5,
                            normalize=True,
                        )

                    batches_done += opt.n_critic


if __name__ == "__main__":
    opt = get_parameters()
    opt.cuda = torch.cuda.is_available()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.img_shape = (opt.im_channels, opt.imsize, opt.imsize)

    os.makedirs(opt.model_save_path, exist_ok=True)
    os.makedirs(opt.sample_path, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.attn_path, exist_ok=True)

    if opt.cuda:
        torch.backends.cudnn.benchmark = True

    dataloader = torch.utils.data.DataLoader(
        MyDataset(opt), batch_size=opt.batch_size, shuffle=True
    )

    if opt.train:
        trainer = GANTrainer(dataloader, opt)
        trainer.train()
