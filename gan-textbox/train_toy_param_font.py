# %%writefile /content/CoordConv/experiments/gan/munit/train_toy_char_layout.py
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

# from models.craft import CRAFTGenerator
# from models.wgan import Generator, Discriminator, compute_gradient_penalty
from models.sagan import Generator, Discriminator
from models.munit import Encoder, Decoder
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

class ParamGenerator(nn.Module):
    def __init__(self, opt):
        super(ParamGenerator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # *block(in_dim, 128, normalize=False),
            # nn.Linear(opt.z_dim, 64),
            nn.Linear(256*32*32 + opt.z_dim, 512),  # c_code + z_dim -> 64
            *block(512, 256, normalize=True),
            *block(256, 128, normalize=True),
            *block(128, 64, normalize=True),
            *block(64, 32, normalize=True),
            *block(32, 16, normalize=True),
            *block(16, 8, normalize=True),
            nn.Linear(8, 8),
            nn.Linear(8, 8),
            nn.Linear(8, 8),
            # nn.Tanh(),
            # nn.Sigmoid(),
        )
        dec = Decoder(out_channels=1, dim=64, n_residual=3, n_upsample=2, style_dim=8)
        dec.load_state_dict(torch.load("../data/model_font_transfer/Dec1_0.pth", map_location="cpu"))
        dec.eval()
        for param in dec.parameters():
            param.requires_grad = False  # freeze weight
        self.dec = dec
    
    def forward(self, z, content_code):
        # print(content_code.shape)
        x = torch.cat([z, torch.flatten(content_code, start_dim=1)], dim=1)
        style_code = self.model(x).view(-1, 8, 1, 1)
        # style_code = self.model(z).view(-1, 8, 1, 1)
        x = self.dec(content_code, style_code)
        return x


class MyDataset(Dataset):
    def __init__(self, img_size):
        self.img_size = img_size
        self.root = "/tf/CoordConv/data/fontimg"
        self.src_dir = "/tf/CoordConv/data/fontimg/Roboto-Regular"
        # self.src_font = "/tf/CoordConv/data/fontimg/Roboto-Regular"
        # self.src_font = "/notebooks/CoordConv-pytorch/data/fontimg/Roboto-Regular"

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

        self.srcs, self.samples = self._sample()

    def __getitem__(self, index):
        _, c, s = random.choice(self.srcs)
        im = Image.open(self.samples[index])
        im = PIL.ImageOps.invert(im)        
        return self.transform(im), c, s

    def __len__(self):
        return len(self.samples)
    
    def _sample(self):
        enc = Encoder(in_channels=1, dim=64, n_downsample=2, n_residual=3, style_dim=8)
        enc.load_state_dict(torch.load("../data/model_font_transfer/Enc1_0.pth", map_location="cpu"))
        enc.eval()

        srcs = []
        for f in sorted(glob.glob(self.src_dir + "/*.png")):
            im = Image.open(f)
            im = PIL.ImageOps.invert(im)
            x = self.transform(im)
            c, s = enc(x.unsqueeze(0))
            srcs.append((x, c.squeeze(0).detach(), s.squeeze(0).detach()))

        samples = []
        for dst_folder in sorted(glob.glob(self.root + "/*/")):
            for f in sorted(glob.glob(dst_folder + "/*.png")):
                samples.append(f)

        return srcs, samples


# -------------------------------
# Training GAN
# -------------------------------


class Trainer(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.dataloader = dataloader

        self.G = ParamGenerator(opt)
        self.D = Discriminator(
            opt.batch_size, opt.imsize, opt.d_conv_dim, opt.im_channels
        )

        if opt.cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()

        if opt.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

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

        self.c_loss = torch.nn.CrossEntropyLoss()

        # print(self.G)
        # print(self.D)

        if opt.use_tensorboard:
            from logger import Logger

            self.logger = Logger(opt.log_path)

        if opt.pretrained_model is not None:
            self.load_pretrained_model()

    def train(self):
        data_iter = iter(self.dataloader)
        step_per_epoch = len(self.dataloader)
        model_save_step = int(self.opt.model_save_step * step_per_epoch)

        if self.opt.pretrained_model is not None:
            start = self.opt.pretrained_model + 1
        else:
            start = 0

        start_time = time.time()
        for step in range(start, self.opt.total_step):
            self.D.train()
            self.G.train()

            try:
                real_images, c_code, s_code = next(data_iter)
            except:
                data_iter = iter(self.dataloader)
                real_images, c_code, s_code = next(data_iter)
            z = torch.randn((real_images.size(0), self.opt.z_dim))
            # print(real_images.size(0))

            if self.opt.cuda:
                real_images = real_images.cuda()
                c_code = c_code.cuda()
                s_code = s_code.cuda()
                z = z.cuda()

            d_out_real, _, _ = self.D(real_images)
            if self.opt.adv_loss == "wgan-gp":
                d_loss_real = -torch.mean(d_out_real)
            elif self.opt.adv_loss == "hinge":
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images = self.G(z, c_code)
            d_out_fake, _, _ = self.D(fake_images)

            if self.opt.adv_loss == "wgan-gp":
                d_loss_fake = d_out_fake.mean()
            elif self.opt.adv_loss == "hinge":
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            d_loss = d_loss_real + d_loss_fake
            self.d_optimizer.zero_grad()
            self.g_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            if self.opt.adv_loss == "wgan-gp":
                alpha = torch.rand(real_images.size(0), 1, 1, 1).expand_as(real_images)
                ones = torch.ones(real_images.size(0))
                if self.opt.cuda:
                    alpha = alpha.cuda()
                    ones = ones.cuda()
                
                interpolated = alpha * real_images.data + (1 - alpha) * fake_images.data
                interpolated.requires_grad_()
                out, _, _ = self.D(interpolated)

                grad = torch.autograd.grad(
                    outputs=out,
                    inputs=interpolated,
                    grad_outputs=ones,
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                )[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.opt.lambda_gp * d_loss_gp

                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and gumbel ================== #

            try:
                real_images, c_code, s_code = next(data_iter)
            except:
                data_iter = iter(self.dataloader)
                real_images, c_code, s_code = next(data_iter)
            z = torch.randn((real_images.size(0), self.opt.z_dim))
            if self.opt.cuda:
                real_images = real_images.cuda()
                c_code = c_code.cuda()
                s_code = s_code.cuda()
                z = z.cuda()
            fake_images = self.G(z, c_code)

            g_out_fake, _, _ = self.D(fake_images)  # batch x n
            if self.opt.adv_loss == "wgan-gp":
                g_loss_fake = -g_out_fake.mean()
            elif self.opt.adv_loss == "hinge":
                g_loss_fake = -g_out_fake.mean()

            self.d_optimizer.zero_grad()
            self.g_optimizer.zero_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            if (step + 1) % self.opt.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(
                    "Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_loss_real: {:.4f} g_loss_fake: {:.4f},".format(
                        elapsed,
                        step + 1,
                        self.opt.total_step,
                        (step + 1),
                        self.opt.total_step,
                        d_loss_real.item(),
                        g_loss_fake.item(),
                    )
                )

            if (step + 1) % self.opt.sample_step == 0:
                save_image(
                    # denorm(fake_images.data),
                    fake_images[:25],
                    os.path.join(
                        self.opt.sample_path, "{:06d}_fake.png".format(step + 1)
                    ),
                    nrow=5,
                )
                save_image(
                    # denorm(real_images),
                    real_images[:25],
                    os.path.join(
                        self.opt.sample_path, "{:06d}_real.png".format(step + 1)
                    ),
                    nrow=5,
                )

            # if (step + 1) % model_save_step == 0:
            #     torch.save(
            #         self.G.state_dict(),
            #         os.path.join(self.opt.model_save_path, "{}_G.pth".format(step + 1)),
            #     )
            #     torch.save(
            #         self.D.state_dict(),
            #         os.path.join(self.opt.model_save_path, "{}_D.pth".format(step + 1)),
            #     )

    def load_pretrained_model(self):
        self.G.load_state_dict(
            torch.load(
                os.path.join(
                    self.opt.model_save_path,
                    "{}_G.pth".format(self.opt.pretrained_model),
                )
            )
        )
        self.D.load_state_dict(
            torch.load(
                os.path.join(
                    self.opt.model_save_path,
                    "{}_D.pth".format(self.opt.pretrained_model),
                )
            )
        )
        print("loaded trained models (step: {})..!".format(self.opt.pretrained_model))


if __name__ == "__main__":
    opt = get_parameters()
    opt.cuda = torch.cuda.is_available()

    # train_wgan()
    if opt.cuda:
        torch.backends.cudnn.benchmark = True

    dataloader = torch.utils.data.DataLoader(
        MyDataset(opt.imsize), batch_size=opt.batch_size, shuffle=True
    )
    os.makedirs(opt.model_save_path, exist_ok=True)
    os.makedirs(opt.sample_path, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.attn_path, exist_ok=True)

    if opt.train:
        if opt.model == "sagan":
            trainer = Trainer(dataloader, opt)
        # elif opt.model == "qgan":
        #     trainer = qgan_trainer(dataloader, opt)
        trainer.train()
    # else:
    #     tester = Tester(data_loader.loader(), config)
    #     tester.test()
