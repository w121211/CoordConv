# %%writefile /content/CoordConv/gan-textbox/train_toy_param_comnist_vae.py
import os
import glob
import random
import time
import datetime
from collections import OrderedDict

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageColor

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
from models.layers import GaussianSmoothing
from config import get_parameters
from utils import tensor2var, denorm


def denormalize(x, mean, std):
    dtype = x.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=x.device)
    std = torch.as_tensor(std, dtype=dtype, device=x.device)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    x.sub_(mean_inv[None, :, None, None]).div_(std_inv[None, :, None, None])
    return x


class MyDataset(Dataset):
    def __init__(self, opt):
        self.imsize = opt.imsize
        self.trans_real = transforms.Compose(
            [
                # transforms.Resize((opt.imsize, opt.imsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.trans_x = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.trans_mask = transforms.Compose([transforms.ToTensor()])

        # self.blur = GaussianSmoothing(3, 5, 5)
        # self.blur.eval()
        self.samples = [self._sample(opt.n_masks) for _ in range(opt.n_samples)]

    def _sample(self, n_dots):
        imsize = (self.imsize, self.imsize)

        hsb = np.random.uniform(0, 0.6, 2)
        palette = [
            ImageColor.getrgb(
                "hsb({}, {}%, {}%)".format(
                    int(hsb[0] * 360),
                    int(hsb[1] * 100),
                    int((0.4 + 0.5 * i / (n_dots + 1)) * 100),
                )
            )
            for i in range(n_dots + 1)
        ]

        im = Image.new("RGB", imsize)
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, *imsize], fill=palette[0])
        bg = im.copy()

        masks = []
        for i in range(n_dots):
            xy = (np.random.uniform(0, 0.8, 2) * np.array(imsize)).astype(int)
            wh = (np.random.normal(0.3, 0.2, 2) * np.array(imsize)).astype(int)
            draw.ellipse(list(xy) + list(xy + wh), fill=palette[i + 1])

            m = Image.new("L", imsize)
            draw_mask = ImageDraw.Draw(m)
            draw_mask.ellipse(list(xy) + list(xy + wh), fill=255)
            masks.append(m)

        return im, bg, masks

    def __getitem__(self, index):
        """
        Returns:
            real_image
            input_image: (N, 1, H, W)
            mask: (N, 1, H, W)
        """
        im, bg, masks = self.samples[index]
        masks = [self.trans_mask(m) for m in masks]
        x = torch.cat([*masks, self.trans_x(bg)])
        return (self.trans_real(im), x)

    def __len__(self):
        return len(self.samples)


class VAEEncoder(nn.Module):
    def __init__(self, opt):
        super(VAEEncoder, self).__init__()

        self.n_masks = opt.n_masks
        self.conv = nn.Sequential(
            nn.Conv2d(3 + opt.n_masks, 16, 4, 2, 1),  # 16,32,32
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
        self.fc = nn.Linear(128 * out_size ** 2, opt.n_masks * opt.z_dim)

    def forward(self, x):
        N = x.size(0)
        x = self.conv(x).view(N, -1)
        x = self.fc(x).view(N, self.n_masks, -1)
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
        self.n_masks = opt.n_masks

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
        # self.blur = GaussianSmoothing(3, 5, 5)

    def forward(self, features, x_img):
        """
        Args:
            features: (N, n_masks, z_dim)
            x_img: (N, n_masks+3, imsize, imsize)
        """
        N = features.shape[0]
        x = self.model(features)  # (N, n_masks, 3)
        rgb = x.view(N, self.n_masks, 3, 1, 1)

        bg = x_img[:, -3:, :, :]
        for i in range(self.n_masks):
            layer = rgb[:, i, :, :, :] * torch.ones(N, 1, self.imsize, self.imsize).to(
                self.device
            )
            mask = x_img[:, i : i + 1, :, :]
            bg = mask * layer + (1.0 - mask) * bg

        return bg



# -------------------------------
# Training
# -------------------------------


class Trainer(object):
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
            for i, (x, y) in enumerate(dataloader):
                x = x.to(opt.device)
                y = y.to(opt.device)

                y_ = self.net(x)
                loss = self.criterion(y, y_)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
                        denormalize(fake_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).data[
                            :25
                        ],
                        os.path.join(
                            self.opt.sample_path, "{:06d}_fake.png".format(batches_done)
                        ),
                        nrow=5,
                        # normalize=True,
                    )
                    save_image(
                        denormalize(
                            x_img[:, -3:, :, :], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                        ).data[:25],
                        os.path.join(
                            self.opt.sample_path, "{:06d}_x.png".format(batches_done)
                        ),
                        nrow=5,
                        # normalize=True,
                    )
                    save_image(
                        denormalize(real_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).data[
                            :25
                        ],
                        os.path.join(
                            self.opt.sample_path, "{:06d}_real.png".format(batches_done)
                        ),
                        nrow=5,
                        # normalize=True,
                    )


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
        trainer = Trainer(dataloader, opt)
        trainer.train()
