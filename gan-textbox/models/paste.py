# %%writefile /content/CoordConv/gan-textbox/models/paste.py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary
import torch.autograd as autograd
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Paste2dMask(nn.Module):
    def __init__(self, im_size):
        super(Paste2dMask, self).__init__()
        self.model = nn.Sequential(nn.Linear(1, 2), nn.Sigmoid())
        self.criterion = torch.nn.MSELoss()
        self.im_size = im_size

    def loss(self, y, x1, x2, gt):
        loss_restore = self.criterion(y, gt)
        loss_coord = torch.mean(F.relu(-(x2 - x1 - 1.0)))
        loss = loss_restore + loss_coord
        return loss

    def forward(self, x):
        l = self.im_size
        N = x.shape[0]

        x0 = x[:, 0].view(-1, 1) * l - 1.0
        y0 = x[:, 1].view(-1, 1) * l - 1.0
        x1 = x[:, 2].view(-1, 1) * l + 1.0
        y1 = x[:, 3].view(-1, 1) * l + 1.0

        coord = torch.arange(l).expand(N, -1).float()
        if cuda:
            coord = coord.cuda()

        _x0 = F.relu6((coord - x0) * 6.0)
        _x1 = F.relu6((x1 - coord) * 6.0)
        x_mask = (_x0 * _x1) / 36  # normalize again after relu6 (multiply by 6.)
        x_mask = x_mask.view(N, 1, l)

        _y0 = F.relu6((coord - y0) * 6.0)
        _y1 = F.relu6((y1 - coord) * 6.0)
        y_mask = (_y0 * _y1) / 36  # normalize again after relu6 (multiply by 6.)
        y_mask = y_mask.view(N, l, 1)  # align to y-axis

        mask = torch.ones(N, l, l)
        if cuda:
            mask = mask.cuda()
        mask = mask * x_mask * y_mask
        return mask.view(-1, 1, l, l)


class Paste2dMulti(nn.Module):
    def __init__(self, im_size, num_images):
        super(Paste2dMulti, self).__init__()
        self.im_size = im_size
        self.paste2dMask = Paste2dMask(im_size)
        self.num_images = num_images

    def _paste(self, coord):
        w = torch.Tensor([[self.im_size]]).float().expand(coord.shape[0], 4)
        coord = torch.round(w * coord)
        ims = []
        for i, _coord in enumerate(coord):
            x0, y0, x1, y1 = _coord[0:4]
            ims.append(F.pad(ims[i], (x0, y0, 3, 4), "constant", 0))
        return F.pad(input, (1, 2, 3, 4), "constant", 0)

    def _pad_image(self, coord, image):
        """
        Arguments:
            coord: (N, 4=(x0, y0, x1, y1))
            image: (N, C, H, W)
        Return:
            padded image: (N, C, self.im_size, self.im_size)
        """
        N, C, H, W = image.shape
        y = []
        for _coord, _im in zip(coord, image):
            f = lambda i: int(_coord[i].item())
            x0, y0, _, _ = f(0), f(1), f(2), f(3)  # lost gradient
            x1, y1 = x0 + W, y0 + H
            if any(
                [
                    x0 >= x1,
                    y0 >= y1,
                    x0 > self.im_size,
                    x1 < 1,
                    y0 > self.im_size,
                    y1 < 1,
                ]
            ):
                # null mask
                y.append(torch.zeros(1, C, self.im_size, self.im_size).type(_im.type()))
            else:
                y.append(
                    F.pad(
                        _im,
                        (x0, self.im_size - x1, y0, self.im_size - y1),
                        "constant",
                        0,
                    ).unsqueeze(0)
                )
        return torch.cat(y, dim=0)

    def forward(self, coords, images):
        """
        Arguments:
            coords: (N, num_images, 4=(x0, y0, x1, y1))
            images: (N, num_images, C, H, W)
        """
        y = []
        for _coord, _im in zip(coords.split(1, dim=1), images.split(1, dim=1)):
            _coord = _coord.squeeze(1)  # (N, 4)
            _im = _im.squeeze(1)  # (N, C, H, W)
            _im = self._pad_image(_coord, _im)  # (N, C, self.im_size, self.im_size)
            mask = self.paste2dMask(
                _coord / self.im_size
            )  # (N, C, self.im_size, self.im_size)
            _im = mask * _im
            y.append(_im)
        y = torch.cat(y, 1).sum(1, True)  # TODO: change to alpha blending
        ones = torch.ones(*y.shape).cuda() if cuda else torch.ones(*y.shape)
        y = torch.min(ones, y)
        return y


class PasteLine(nn.Module):
    def __init__(self, im_size, max_chars):
        super(PasteLine, self).__init__()
        self.im_size = im_size
        self.max_chars = max_chars
        self.paste2dMulti = Paste2dMulti(im_size, max_chars)

    def forward(self, coord, chars, char_sizes):
        """
        Args:
            coord: (N, 2=(x0, y0)), coordinate for the line
            chars: (N, num_chars, C, H, W)
            char_sizes: (N, num_chars, 2=(char_w, char_h))
        """
        coord = coord.unsqueeze(1)
        coords = []  # (N, num_chars, 4=(x0, y0, x1, y1))
        for size in char_sizes.split(1, dim=1):
            coords.append(torch.cat((coord, coord + size), dim=2))
            mask = torch.tensor([[[1, 0]]]).float()
            mask = mask.cuda() if cuda else mask
            coord = coord + size * mask  # (x0 += w, y0)
        coords = torch.cat(coords, dim=1)
        return self.paste2dMulti(coords, chars)


class PaintLine(nn.Module):
    def __init__(self, im_size, max_chars):
        super(PaintLine, self).__init__()
        self.im_size = im_size
        self.max_chars = max_chars
        self.paste_line = PasteLine(im_size, max_chars)
        self.char_generator = CharGenerator()

    def forward(self, coord, font, text):
        """
        Args:
            coord: (N, 2=(x0, y0)), coordinate for the line
            font: (N, font_vec_dim)
            text: a list contains N different stirngs
        """
        chars, char_sizes = [], []
        for c in text:
            char, char_size = self.char_generator(font, c)
            chars.append(char)
            char_sizes.append(char_sizes)
        chars = torch.stack(chars).repeat(dim=0)  # (N, max_chars, C, H, W)
        char_sizes = torch.stack(char_sizes).repeat(
            dim=0
        )  # (N, max_chars, 4=(x0, y0, x1, y1))

        return self.paste_line(coord, chars, char_sizes)


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

    def forward(self, z, text_state):
        """
        Args:
            z: (N, z_dim)
            text_state:
                1. (N, text_state=(n_chars))
                2. (N, text_embeded_dim)
        """
        x = self.rnn(char)
        x = self.model(z, x)
        coord, style, font_size, color = x.split(2, dim=1)
        return self.painter(coord, style, font_size, color)
