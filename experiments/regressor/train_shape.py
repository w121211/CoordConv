# %%writefile /content/CoordConv/experiments/regressor/train_shape.py
import os
import glob
import argparse
import itertools
import random
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import torch.utils.data as utils
import torch.nn.functional as F
import torchvision.transforms as transforms
from faker import Faker

from ..models.coordconv import AddCoords
from ..models.hrnet import HighResolutionNet
from ..models.config import get_cfg_defaults

# from models.CornerNet_Squeeze import corner_pool
# from models.py_utils import TopPool, LeftPool

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
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
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
parser.add_argument("--n_strokes", type=int, default=1)
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--data_path", type=str, default="data/regressor")
parser.add_argument("--model_path", type=str, default="saved_models/95000.pt")

opt = parser.parse_args()
print(opt)


class SimpleNet(nn.Module):
    def __init__(self, width):
        super(SimpleNet, self).__init__()
        self.width = width
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        # self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        # self.pool = corner_pool(32, TopPool, LeftPool)
        self.add_coords = AddCoords(rank=2)
        self.conv5 = nn.Conv2d(19, 19, 3, padding=1)
        self.conv6 = nn.Conv2d(19, 19, 3, padding=1)
        self.conv7 = nn.Conv2d(19, 4, 1)
        self.pool2 = nn.MaxPool2d(width, stride=width)

        # regressor

    def forward(self, x):
        """
        x: (N, C, H, W)
        """
        # heatmap
        # x0 = self.add_coords(x)
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        x1 = self.pool(x1)
        x1 = F.interpolate(x1, scale_factor=2)
        # print(x.shape)
        # print(x1.shape)
        x = torch.cat((x, x1), dim=1)

        # regression
        x = self.add_coords(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x = self.pool2(x)
        x = x.view(-1, 4)
        return x


class Regressor(nn.Module):
    def __init__(self, in_channel, width, out_size=4, latent_size=128):
        super(Regressor, self).__init__()
        self.add_coords = AddCoords(rank=2)
        #         self.conv0 = nn.Conv2d(in_channel + 2, latent_size, 3, padding=1)
        #         self.bn0 = BatchNorm2d(latent_size)
        #         self.conv1 = nn.Conv2d(latent_size, latent_size, 3, padding=1)
        #         self.bn1 = BatchNorm2d(latent_size)
        # self.conv2 = nn.Conv2d(latent_size, out_size, 1)
        self.conv0 = nn.Conv2d(in_channel + 2, latent_size, 1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(latent_size)
        self.conv2 = nn.Conv2d(latent_size, out_size, 1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(width, stride=width)

    def forward(self, x):
        x = self.add_coords(x)
        x = self.conv0(x)
        x = F.relu(self.bn0(x))
        #         x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(-1, 4)
        return x


class HRNet(nn.Module):
    def __init__(self, width):
        super(HRNet, self).__init__()
        cfg = get_cfg_defaults()
        cfg.merge_from_file("./experiments/models/hrnet.yaml")
        self.hr0 = HighResolutionNet(cfg)
        self.hr1 = HighResolutionNet(cfg)
        self.rg = Regressor(in_channel=540, width=int(width / 4))

    def forward(self, im_current, im_target):
        x0 = self.hr0(im_current)
        x1 = self.hr1(im_target)
        x = torch.cat((x0, x1), dim=1)
        x = self.rg(x)
        return x


class ImageDataset(utils.Dataset):
    def __init__(self, root, n_strokes, transforms_=None, has_x=True):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(root + "/*.png"))
        self.xs = (
            np.load(os.path.join(root, "xs.npy")).astype(np.float32) if has_x else None
        )
        self.n_strokes = n_strokes

    def __getitem__(self, index):
        """
        eg, let n_strokes = 3, then,
            step_0: im_-1 + x_0 = im_0
            step_1: im_0 + x_1 = im_1
            step_2: im_1 + x_2 = im_2
        for training data,
            index = 0: return (im_-1, im_2, x_0)
            index = 1: return (im_0, im_2, x_1)
            index = 2: return (im_1, im_2, x_2)
        """
        im_target = Image.open(
            self.files[(index // self.n_strokes + 1) * self.n_strokes - 1]
        )
        if index % self.n_strokes == 0:
            im_current = Image.new("RGB", (im_target.width, im_target.height))
        else:
            im_current = Image.open(self.files[index - 1])

        im_target = self.transform(im_target)
        im_current = self.transform(im_current)

        if self.xs is not None:
            x = self.xs[index]
            return im_current, im_target, x
        else:
            return im_current, im_target, 0

    def __len__(self):
        return len(self.files)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs("images", exist_ok=True)
os.makedirs(opt.data_path, exist_ok=True)
# os.makedirs(opt.model_path, exist_ok=True)
img_shape = (opt.channels, opt.img_size, opt.img_size)


def sample(n_samples=100, n_strokes=3):
    fake = Faker()

    def _sample_x():
        w = fake.pyint(5, 20)
        h = fake.pyint(5, 20)
        x0 = fake.pyint(0, img_shape[1] - w)
        y0 = fake.pyint(0, img_shape[2] - h)
        rgb = [int(x) for x in fake.rgb_color().split(",")]
        return x0, y0, x0 + w, y0 + h, rgb[0], rgb[1], rgb[2]

    xs = []
    for i in range(n_samples):
        im = Image.new("RGB", (img_shape[1], img_shape[2]))
        draw = ImageDraw.Draw(im)

        _x = sorted([_sample_x() for _ in range(n_strokes)], key=lambda x: (x[0], x[1]))
        for j, x in enumerate(_x):
            x0, y0, x1, y1, r, g, b = x
            draw.rectangle((x0, y0, x1, y1), fill=(r, g, b))
            im.copy().save("%s/%d.png" % (opt.data_path, i * n_strokes + j), "PNG")
            _xs = (
                x0 / img_shape[1],
                y0 / img_shape[2],
                x1 / img_shape[1],
                y1 / img_shape[2],
                # r / 255,
                # g / 255,
                # b / 255,
            )
            xs.append(np.array(_xs, dtype=float))
    np.save("%s/x.npy" % (opt.data_path), np.stack(xs))

def sample_draw(n_samples=100):
    fake = Faker()
    # Brush = Enum('Brush', 'RECT PHOTO TEXT SPRITE')
    Brush = Enum('Brush', 'RECT')

    def _sample_x(frame_xywh, xywh=None, brush=None, rgb=None,min_wh=(5, 5)):
        if xywh is None:
            w = fake.pyint(min_wh[0], frame_xywh[2])
            h = fake.pyint(min_wh[1], frame_xywh[3])
            x = fake.pyint(0, frame_xywh[2] - w) + frame_xywh[0]
            y = fake.pyint(0, frame_xywh[3] - h)+ frame_xywh[1]
            xywh = x, y, w, h
        if brush is None:
            brush = random.choice(list(Brush))
        if rgb is None:
            rgb = [int(x) for x in fake.rgb_color().split(",")]
        return (*xywh, *rgb, brush.value)
        
    def _cut(xywh, ratio=random.choice([1/6, 1/5, 1/4, 1/3, 1/2])):
        x, y, w, h = xywh
        r0 = random.choice([ratio, 1-ratio])
        w0 = int(w * ratio)
        w1 = w - w0
        xywh0 = (x, y, w0, h)
        xywh1 = (x+w0, y, w1, h)
        return xywh0, xywh1

    # draw image
    xs = []
    n_strokes = 4  # a hack
    for i in range(n_samples):
        im = Image.new("RGB", (img_shape[1], img_shape[2]))
        draw = ImageDraw.Draw(im)
        _xs = []
        for f in _cut((0, 0, img_shape[1], img_shape[2])):
            _xs += [_sample_x(f, xywh=f, brush=Brush.RECT), _sample_x(f)]  # bg, item
        
        for j, x in enumerate(_xs):
            x, y, w, h, r, g, b, brush = x
            x0, y0, x1, y1 = x, y, x+w, y+h

            if Brush(brush) == Brush.RECT:
                draw.rectangle((x0, y0, x1, y1), fill=(r, g, b))
            #     elif brush == Brush.PHOTO:
            #         pass
            #     elif brush == Brush.TEXT:
            #         pass
            #     elif brush == Brush.SPRITE:
            #         pass

            im.copy().save("%s/%d.png" % (opt.data_path, i * n_strokes + j), "PNG")
            _x = (
                    x0 / img_shape[1],
                    y0 / img_shape[2],
                    x1 / img_shape[1],
                    y1 / img_shape[2],
                    # r / 255,
                    # g / 255,
                    # b / 255,
                )
            xs.append(np.array(_x, dtype=float))
    np.save("%s/xs.npy" % (opt.data_path), np.stack(xs))

def train(dataloader):
    model = HRNet(img_shape[1]).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (im_current, im_target, gt_x) in enumerate(dataloader):
            im_current, im_target, gt_x = (
                im_current.to(device),
                im_target.to(device),
                gt_x.to(device),
            )
            optimizer.zero_grad()
            x = model(im_current, im_target)
            # print(x)
            # print(gt_x)
            loss = loss_fn(x, gt_x)
            loss.backward()
            optimizer.step()

            if batches_done % opt.sample_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), loss.item())
                )
                # save_image(
                #     fake_imgs.data[:25],
                #     "images/%d.png" % batches_done,
                #     nrow=5,
                #     normalize=True,
                # )
                # torch.save(generator.state_dict(), opt.save_path)
            batches_done += 1


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    sample_draw(n_samples=opt.n_samples)
    dataloader = utils.DataLoader(
        ImageDataset(
            opt.data_path,
            opt.n_strokes,
            transforms_=[
                # transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ],
            has_x=True,
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    train(dataloader)
