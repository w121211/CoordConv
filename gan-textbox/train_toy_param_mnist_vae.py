%%writefile /content/CoordConv/gan-textbox/train_toy_param_mnist_vae.py
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

class VAEDecoder(nn.Module):
    def __init__(self):
        super(VAEDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 784),
            nn.Sigmoid()
            )

    def forward(self, z):
        return self.model(z)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.dec = VAEDecoder()

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt

        dec = VAEDecoder()
        dec.load_state_dict(
            torch.load("./out/models_pretrain/000070_vae_dec.pth", map_location="cpu")
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
        x = self.model(z)
        img = self.dec(x)
        img = img.view(img.shape[0], *self.opt.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(opt.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


def compute_gradient_penalty(D, real_samples, fake_samples, opt):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1))).float().to(opt.device)

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
                        epoch,
                        loss.item() / len(real_img),
                    )
                )
                save_image(
                    recon_batch.view(real_img.shape[0], *self.opt.img_shape)[:25],
                    os.path.join(
                        self.opt.sample_path, "{:06d}_vae.png".format(epoch)
                    ),
                    nrow=5,
                    normalize=True,
                )
                torch.save(
                        self.model.dec.state_dict(),
                        os.path.join(
                            self.opt.model_save_path,
                            "{:06d}_vae_dec.pth".format(epoch),
                        ),
                    )
                


class GANTrainer(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.dataloader = dataloader

        self.G = Generator(opt)
        self.D = Discriminator(opt)

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

        # if opt.pretrained_model is not None:
        #     self.load_pretrained_model()

    def train(self):
        lambda_gp = 10
        batches_done = 0
        for epoch in range(opt.n_epochs):
            for i, (real_img, _) in enumerate(dataloader):
                z = torch.tensor(
                    np.random.normal(0, 1, (real_img.shape[0], opt.z_dim))
                ).float()

                if self.opt.cuda:
                    z = z.cuda()
                    real_img = real_img.cuda()

                #  Train DisDcriminator
                fake_img = self.G(z)
                real_validity = self.D(real_img)
                fake_validity = self.D(fake_img)

                gradient_penalty = compute_gradient_penalty(
                    self.D, real_img.detach(), fake_img.detach(), self.opt
                )

                d_loss = (
                    -torch.mean(real_validity)
                    + torch.mean(fake_validity)
                    + lambda_gp * gradient_penalty
                )

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                self.g_optimizer.zero_grad()

                # Train G every n_critic steps
                if i % opt.n_critic == 0:

                    fake_img = self.G(z)
                    fake_validity = self.D(fake_img)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.g_optimizer.step()

                    if batches_done % opt.sample_interval == 0:
                        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
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
                            # normalize=True,
                        )
                        # save_image(
                        #     real_img.data[:25],
                        #     os.path.join(
                        #         self.opt.sample_path,
                        #         "{:06d}_real.png".format(batches_done),
                        #     ),
                        #     nrow=5,
                        #     # normalize=True,
                        # )

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

    if opt.cuda:
        torch.backends.cudnn.benchmark = True

    def get_indices(dataset,class_name):
        indices =  []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == class_name:
                indices.append(i)
        return indices

    dataset = datasets.MNIST(
            "../../data/mnist",
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
        # shuffle=True,
        sampler = torch.utils.data.sampler.SubsetRandomSampler(get_indices(dataset, 1))
    )

    os.makedirs(opt.model_save_path, exist_ok=True)
    os.makedirs(opt.sample_path, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.attn_path, exist_ok=True)

    if opt.train:
        # trainer = VAETrainer(dataloader, opt)
        trainer = GANTrainer(dataloader, opt)
        trainer.train()
    # else:
    #     tester = Tester(data_loader.loader(), config)
    #     tester.test()
