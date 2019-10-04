# %%writefile /content/CoordConv/experiments/gan/munit/train_toy_char_layout.py
import os
import glob
import random
import time
import datetime
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.backends import cudnn

# from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models.craft import CRAFTGenerator
from models.sagan import Generator, Discriminator
from config import get_parameters
from utils import tensor2var, denorm

# ----------------fn;---------------
# Toy experiment:
#   1. (z, token_status = (token_width, token_height)) -> G -> Line_coord (x0, y0)
#   2. (x0, y0) -> PasteLine -> Image
#   3. Image -> D -> y
#
# Dataset:
#   Real: token places at center of images
# -------------------------------


# ---------------------------------------------------------------
# Models GAN: G -> produce char masks, real char masks from posters
# ---------------------------------------------------------------


class LayoutGenerator(nn.Module):
    def __init__(self, opt):
        super(LayoutGenerator, self).__init__()
        self.craft = CRAFTGenerator(
            output_class=1, z_dim=opt.z_dim, img_size=opt.imsize
        )

        # def block(in_feat, out_feat, normalize=True):
        #     layers = [nn.Linear(in_feat, out_feat)]
        #     if normalize:
        #         layers.append(nn.BatchNorm1d(out_feat, 0.8))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     return layers

        # self.model = nn.Sequential(
        #     # *block(in_dim, 128, normalize=False),
        #     nn.Linear(in_dim, 64),
        #     *block(64, 64, normalize=False),
        #     *block(64, 64, normalize=False),
        #     *block(64, 2, normalize=False),
        #     nn.Linear(2, 2),
        #     nn.Sigmoid(),
        # )
        # painter = Generator(in_dim=4)
        # painter.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
        # painter = PasteLine(opt.img_size, max_chars=10)
        # if cuda:
        #     painter.cuda()
        # painter.eval()
        # for param in painter.parameters():
        #     param.requires_grad = False  # freeze weight
        # self.painter = painter

    def forward(self, z, img, img2x):
        """
        Args:
            z: (N, z_dim)
            img (N, C, H, W)
            img2x: (N, C, H*2, W*2), 2x size of `img`
        Return:
            mask: (N, C=1, H, W)
            heat: (N, C=3, H, W)
        """
        mask = self.craft(z, img2x)  # mask: (N, 1, H, W)

        # convert mask to heatmap & apply to `img_small`
        heat = mask.repeat([1, 3, 1, 1])  # (N, 1, H, W) to (N, 3, H, W)
        heat = torch.mul(
            heat, torch.tensor([1.0, -1.0, 1.0]).view(1, 3, 1, 1).to(z.device)
        )  # map color
        _mask = (mask > -0.95).float()  # convert to binary
        heat = (1.0 - _mask) * img + _mask * heat

        return mask, heat


# ---------------------------------------------------------------
# Dataset: Convert real poster to char masks
# ---------------------------------------------------------------


class MyDataset(Dataset):
    def __init__(self, img_size):
        # self.post_folder = "/notebooks/CRAFT-pytorch/data"
        # self.mask_folder = "/notebooks/CRAFT-pytorch/result"
        # self.photo_folder = "/notebooks/CoordConv-pytorch/data/facebook"
        self.post_folder = "/tf/CRAFT-pytorch/data"
        self.mask_folder = "/tf/CRAFT-pytorch/result"
        self.photo_folder = "/tf/CoordConv/data/flickr"

        norm_mean = np.array([0.5, 0.5, 0.5])
        norm_sigma = np.array([0.5, 0.5, 0.5])
        self.trans_rgb = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_sigma)]
        )
        self.trans_l = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.trans_pil = transforms.Compose(
            [
                transforms.Normalize(
                    -(norm_mean / norm_sigma), 1 / norm_sigma
                ),  # inverse-normalize
                transforms.ToPILImage(),
            ]
        )

        self.font = "/notebooks/post-generator/asset/fonts_en/Roboto/Roboto-Regular.ttf"
        self.font_size = 14
        self.out_size = 14
        self.max_chars = 10

        self.img_size = img_size
        self.posts = self._sample(self.post_folder + "/*.jpg")
        self.photos = self._sample(self.photo_folder + "/*.jpg")

    def __getitem__(self, index):
        f = self.posts[index]
        post = Image.open(f)
        post = self._resize(post, self.img_size)
        mask = Image.open(
            os.path.join(
                self.mask_folder, os.path.splitext(os.path.basename(f))[0] + "_mask.png"
            )
        )
        mask = self._resize(mask, self.img_size)
        heat = self._heatmap(post, mask)

        # random photo for input
        photo = Image.open(random.choice(self.photos))
        photo = photo.convert("RGB")
        photo2x = self._resize(photo, self.img_size * 2)
        photo = self._resize(photo, self.img_size)

        return self.trans_l(mask), heat, self.trans_rgb(photo), self.trans_rgb(photo2x)

    def __len__(self):
        return len(self.posts)

    def _sample(self, root):
        samples = []
        for f in glob.glob(root):
            img = Image.open(f)
            if img.mode == "RGB":
                samples.append(f)
        return samples

    def _heatmap(self, img, mask):
        """
        Args:
            img: PIL.Image, RGB
            mask: PIL.Image, L
        Returns:
            tensor (C, H, W)
        """
        img = self.trans_rgb(img)
        mask = self.trans_l(mask)

        heat = mask.repeat([3, 1, 1])  # (1, H, W) to (3, H, W)
        heat = heat.permute(1, 2, 0)  # (H, W, C)
        heat = torch.mul(heat, torch.tensor([[1.0, -1.0, 1.0]]))  # map color
        heat = heat.permute(2, 0, 1)  # (C, H, W)
        mask = (mask > -0.95).float()  # convert to binary
        heat = (1.0 - mask) * img + mask * heat

        return heat

    def _heatmap_pil(self, img, mask):
        """Deprecated"""
        heat = np.array(mask)
        heat = np.expand_dims(heat, axis=-1)
        heat = np.repeat(heat, 3, axis=-1)
        heat[:, :, 1] = 1 - heat[:, :, 1]  # color transform

        img.putalpha(255)
        heat = Image.fromarray(heat)
        heat.putalpha(mask.point(lambda x: 0 if x < 30 else 255))
        return Image.alpha_composite(img, heat)

    def _resize(self, img, out_size):
        size = img.size  # old_size[0] is in (width, height) format
        ratio = float(out_size) / max(size)
        new_size = tuple(int(x * ratio) for x in size)
        img = img.resize(new_size, Image.ANTIALIAS)
        res = Image.new(img.mode, (out_size, out_size))
        res.paste(img, ((out_size - new_size[0]) // 2, (out_size - new_size[1]) // 2))
        return res


# -------------------------------
# Training GAN
# -------------------------------


class Trainer(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.dataloader = dataloader

        # self.G = Generator(
        #     opt.batch_size, opt.imsize, opt.z_dim, opt.g_conv_dim, opt.im_channels
        # )
        self.G = LayoutGenerator(opt)
        self.D_mask = Discriminator(
            opt.batch_size, opt.imsize, opt.d_conv_dim, in_channels=1
        )
        self.D_heat = Discriminator(
            opt.batch_size, opt.imsize, opt.d_conv_dim, in_channels=3
        )

        if opt.cuda:
            self.G = self.G.cuda()
            self.D_mask = self.D_mask.cuda()
            self.D_heat = self.D_heat.cuda()

        # if opt.parallel:
        #     self.G = nn.DataParallel(self.G)
        #     self.D = nn.DataParallel(self.D)

        self.g_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.G.parameters()),
            opt.g_lr,
            [opt.beta1, opt.beta2],
        )
        self.d_mask_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D_mask.parameters()),
            opt.d_lr,
            [opt.beta1, opt.beta2],
        )
        self.d_heat_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D_heat.parameters()),
            opt.d_lr,
            [opt.beta1, opt.beta2],
        )

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
            self.D_mask.train()
            self.D_heat.train()
            self.G.train()

            try:
                real_masks, real_heats, x_photos, x_photos2x = next(data_iter)
            except:
                data_iter = iter(self.dataloader)
                real_masks, real_heats, x_photos, x_photos2x = next(data_iter)
            z = torch.randn((real_masks.size(0), self.opt.z_dim))

            if self.opt.cuda:
                real_masks = real_masks.cuda()
                real_heats = real_heats.cuda()
                x_photos = x_photos.cuda()
                x_photos2x = x_photos2x.cuda()
                z = z.cuda()

            d_mask_out_real, _, _ = self.D_mask(real_masks)
            d_heat_out_real, _, _ = self.D_heat(real_heats)
            if self.opt.adv_loss == "wgan-gp":
                d_mask_loss_real = -torch.mean(d_mask_out_real)
                d_heat_loss_real = -torch.mean(d_heat_out_real)
            elif self.opt.adv_loss == "hinge":
                d_mask_loss_real = torch.nn.ReLU()(1.0 - d_mask_out_real).mean()
                d_heat_loss_real = torch.nn.ReLU()(1.0 - d_heat_out_real).mean()

            # fake_images, gf1, gf2 = self.G(z)
            fake_masks, fake_heats = self.G(z, x_photos, x_photos2x)
            d_mask_out_fake, _, _ = self.D_mask(fake_masks)
            d_heat_out_fake, _, _ = self.D_heat(fake_heats)
            if self.opt.adv_loss == "wgan-gp":
                d_mask_loss_fake = d_mask_out_fake.mean()
                d_heat_loss_fake = d_heat_out_fake.mean()
            elif self.opt.adv_loss == "hinge":
                d_mask_loss_fake = torch.nn.ReLU()(1.0 + d_mask_out_fake).mean()
                d_heat_loss_fake = torch.nn.ReLU()(1.0 + d_heat_out_fake).mean()

            d_mask_loss = d_mask_loss_real + d_mask_loss_fake
            d_heat_loss = d_heat_loss_real + d_heat_loss_fake
            # self.g_optimizer.zero_grad()
            self.d_mask_optimizer.zero_grad()
            self.d_heat_optimizer.zero_grad()
            d_mask_loss.backward(retain_graph=True)
            d_heat_loss.backward()
            self.d_mask_optimizer.step()
            self.d_heat_optimizer.step()

            # if self.opt.adv_loss == "wgan-gp":
            #     alpha = torch.rand(real_images.size(0), 1, 1, 1).expand_as(real_images)
            #     ones = torch.ones(real_images.size(0))
            #     if self.opt.cuda:
            #         alpha = alpha.cuda()
            #         ones = ones.cuda()

            #     interpolated = alpha * real_images.data + (1 - alpha) * fake_images.data
            #     interpolated.requires_grad_()
            #     out, _, _ = self.D(interpolated)

            #     grad = torch.autograd.grad(
            #         outputs=out,
            #         inputs=interpolated,
            #         grad_outputs=ones,
            #         retain_graph=True,
            #         create_graph=True,
            #         only_inputs=True,
            #     )[0]

            #     grad = grad.view(grad.size(0), -1)
            #     grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            #     d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            #     # Backward + Optimize
            #     d_loss = self.opt.lambda_gp * d_loss_gp

            #     self.d_optimizer.zero_grad()
            #     self.g_optimizer.zero_grad()
            #     d_loss.backward()
            #     self.d_optimizer.step()

            # ================== Train G and gumbel ================== #

            z = torch.randn((real_masks.size(0), self.opt.z_dim))
            if self.opt.cuda:
                z = z.cuda()
            fake_masks, fake_heats = self.G(z, x_photos, x_photos2x)

            g_out_fake_mask, _, _ = self.D_mask(fake_masks)  # batch x n
            g_out_fake_heat, _, _ = self.D_heat(fake_heats)  # batch x n
            if self.opt.adv_loss == "wgan-gp":
                g_loss_fake = -(g_out_fake_mask.mean() + g_out_fake_heat.mean())
            elif self.opt.adv_loss == "hinge":
                g_loss_fake = -(g_out_fake_mask.mean() + g_out_fake_heat.mean())

            # self.d_mask_optimizer.zero_grad()
            # self.d_heat_optimizer.zero_grad()
            self.g_optimizer.zero_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            if (step + 1) % self.opt.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(
                    "Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_mask_loss_real: {:.4f}, d_heat_loss_real: {:.4f}, g_loss_fake: {:.4f}"
                    " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".format(
                        elapsed,
                        step + 1,
                        self.opt.total_step,
                        (step + 1),
                        self.opt.total_step,
                        d_mask_loss_real.item(),
                        d_heat_loss_real.item(),
                        g_loss_fake.item(),
                        0,
                        0,
                        # self.G.attn1.gamma.mean().item(),
                        # self.G.attn2.gamma.mean().item(),
                    )
                )

            if (step + 1) % self.opt.sample_step == 0:
                # fixed_z = tensor2var(torch.randn(self.opt.batch_size, self.opt.z_dim))
                # fake_images, _, _ = self.G(fixed_z)
                save_image(
                    denorm(fake_masks[:25]),
                    # fake_masks[:25],
                    os.path.join(
                        self.opt.sample_path, "{:06d}_fake_mask.png".format(step + 1)
                    ),
                    nrow=5,
                )
                save_image(
                    denorm(fake_heats[:25]),
                    # fake_masks[:25],
                    os.path.join(
                        self.opt.sample_path, "{:06d}_fake_heat.png".format(step + 1)
                    ),
                    nrow=5,
                )
                save_image(
                    denorm(real_masks[:25]),
                    # real_masks[:25],
                    os.path.join(
                        self.opt.sample_path, "{:06d}_real_mask.png".format(step + 1)
                    ),
                    nrow=5,
                )
                save_image(
                    denorm(real_heats[:25]),
                    # real_masks[:25],
                    os.path.join(
                        self.opt.sample_path, "{:06d}_real_heat.png".format(step + 1)
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
        cudnn.benchmark = True

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
