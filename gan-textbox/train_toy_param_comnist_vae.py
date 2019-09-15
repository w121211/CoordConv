# %%writefile /content/CoordConv/gan-textbox/train_toy_param_mnist_vae.py
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


class VAEEncoder(nn.Module):
    def __init__(self, z_dim=20):
        super(VAEEncoder, self).__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 16,32,32
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # 32,16,16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)    #32,16,16 -> 32,16,16
            # self.bn3 = nn.BatchNorm2d(32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 32,16,16 -> 64,8,8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 64,8,8 -> 128,4,4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # 256,2,2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(256, z_dim)
        self.fc2 = nn.Linear(256, z_dim)

    def forward(self, x):
        N = x.size(0)
        x = self.enc(x).view(N, -1)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, z_dim=20):
        super(VAEDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), 16, 16, 16)
        x = self.conv(x)
        return x


class VAE(nn.Module):
    def __init__(self, z_dim=20):
        super(VAE, self).__init__()
        self.enc = VAEEncoder()
        self.dec = VAEDecoder()

    def forward(self, x):
        z, mu, logvar = self.enc(x)
        return self.dec(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt

        dec = VAEDecoder()
        dec.load_state_dict(
            torch.load("./out/models/000120_vae_dec.pth", map_location="cpu")
        )
        for param in dec.parameters():
            param.requires_grad = False  # freeze weight
        self.dec = dec

        self.model = nn.Sequential(
            nn.Linear(opt.z_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 20),
            nn.Linear(20, 20),
        )

    def forward(self, z):
        style = self.model(z)
        img = self.dec(style)
        img = img.view(img.shape[0], *self.opt.img_shape)
        return img, style


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(int(np.prod(opt.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2),  # 32,16,16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),  # 32,16,16
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 32,16,16 -> 64,8,8
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(
            #     64, 128, kernel_size=3, padding=1, stride=2
            # ),  # 32,16,16 -> 64,8,8
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 64,8,8 -> 128,4,4
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),  # 256,2,2
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=16),  # 256,1,1
            #             nn.Conv2d(256, 64, kernel_size=1),  # 256,1,1
            #             nn.Conv2d(64, 16, kernel_size=1),  # 64,1,1
            #             nn.Conv2d(16, 1, kernel_size=1),  # 16,1,1
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        #         validity = self.model(img).view(img.shape[0], -1)
        # validity = self.model(img).mean(dim=1).view(img.shape[0], -1)
        return validity


class StyleDiscriminator(nn.Module):
    def __init__(self, opt):
        super(StyleDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, style):
        return self.model(style)


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


class VAETrainer(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.dataloader = dataloader
        self.model = VAE().to(opt.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        self.model.train()
        train_loss = 0
        for epoch in range(self.opt.n_epochs):
            for i, (real_img, _) in enumerate(self.dataloader):
                real_img = real_img.to(self.opt.device)
                recon_batch, mu, logvar = self.model(real_img)
                loss = vae_loss(recon_batch, real_img, mu, logvar)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % self.opt.log_step == 0:
                print(
                    "Train Epoch: {} Loss: {:.6f}".format(
                        epoch, loss.item() / len(real_img)
                    )
                )
                save_image(
                    recon_batch[:25],
                    os.path.join(self.opt.sample_path, "{:06d}_vae.png".format(epoch)),
                    nrow=5,
                )
                # save_image(
                #     real_img[:25],
                #     os.path.join(self.opt.sample_path, "{:06d}_real.png".format(epoch)),
                #     nrow=5,
                # )
                torch.save(
                    self.model.enc.state_dict(),
                    os.path.join(
                        self.opt.model_save_path, "{:06d}_vae_enc.pth".format(epoch)
                    ),
                )
                torch.save(
                    self.model.dec.state_dict(),
                    os.path.join(
                        self.opt.model_save_path, "{:06d}_vae_dec.pth".format(epoch)
                    ),
                )


class GANTrainer(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.dataloader = dataloader

        self.G = Generator(opt).to(opt.device)
        self.D = Discriminator(opt).to(opt.device)
        # self.D_style = StyleDiscriminator(opt).to(opt.device)

        # Enc = VAEEncoder().to(opt.device)
        # Enc.load_state_dict(
        #     torch.load("./out/models/000120_vae_enc.pth", map_location="cpu")
        # )
        # for param in Enc.parameters():
        #     param.requires_grad = False
        # Enc.eval()
        # self.Enc = Enc

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
            for i, (real_img, _) in enumerate(dataloader):
                real_img = real_img.to(opt.device)
                z = (
                    torch.tensor(np.random.normal(0, 1, (real_img.shape[0], opt.z_dim)))
                    .float()
                    .to(opt.device)
                )

                #  Train Discriminator
                fake_img, fake_style = self.G(z)

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
                real_style, _, _ = self.Enc(real_img)

                real_validity = self.D_style(real_style.detach())
                fake_validity = self.D_style(fake_style.detach())

                gp = compute_gradient_penalty(
                    self.D_style,
                    real_style.detach(),
                    fake_style.detach(),
                    self.opt,
                )

                d_style_loss = (
                    -torch.mean(real_validity)
                    + torch.mean(fake_validity)
                    + lambda_gp * gp
                )

                self.d_style_optimizer.zero_grad()
                d_style_loss.backward()
                self.d_style_optimizer.step()

                # Train G every n_critic steps
                if i % opt.n_critic == 0:
                    fake_img, fake_style = self.G(z)
                    fake_validity = self.D(fake_img)
                    fake_validity_style = self.D_style(fake_style)
                    # g_loss = -(
                    #     torch.mean(fake_validity) + torch.mean(fake_validity_style)
                    # )
                    g_loss = -torch.mean(fake_validity)

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    if batches_done % opt.sample_interval == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D_style loss: %f] [G loss: %f]"
                            % (
                                epoch,
                                opt.n_epochs,
                                i,
                                len(dataloader),
                                d_loss.item(),
                                d_style_loss.item(),
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
                            # normalize=True,
                        )

                    batches_done += opt.n_critic

                # if i % opt.save_interval == 0:
                #     torch.save(
                #         self.G.state_dict(),
                #         os.path.join(
                #             self.opt.model_save_path,
                #             "{:06d}_G.pth".format(batches_done),
                #         ),
                #     )
                #     torch.save(
                #         self.D.state_dict(),
                #         os.path.join(
                #             self.opt.model_save_path,
                #             "{:06d}_D.pth".format(batches_done),
                #         ),
                #     )
                #     torch.save(
                #         self.G.dec.state_dict(),
                #         os.path.join(
                #             self.opt.model_save_path,
                #             "{:06d}_G_dec.pth".format(batches_done),
                #         ),
                #     )


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

    def loader(path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.split()[-1]

    # dataloader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(
    #         "../data/comnist",
    #         transform=transforms.Compose(
    #             [
    #                 transforms.Resize(opt.imsize),
    #                 transforms.ToTensor(),
    #                 # transforms.Normalize([0.5], [0.5]),
    #             ]
    #         ),
    #         loader=loader,
    #     ),
    #     batch_size=opt.batch_size,
    #     shuffle=True,
    # )

    dataset = datasets.MNIST(
            "../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.imsize),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5], [0.5]),
                ]
            ),
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        # sampler = torch.utils.data.sampler.SubsetRandomSampler(get_indices(dataset, 1))
    )

    if opt.train:
        trainer = VAETrainer(dataloader, opt)
        # trainer = GANTrainer(dataloader, opt)
        trainer.train()
