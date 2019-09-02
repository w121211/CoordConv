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


# -------------------------------
#            WGAN
# -------------------------------

def conv_norm_relu_module(norm_type, norm_layer, input_nc, ngf, kernel_size, padding, stride=1, relu='relu'):

    model = [nn.Conv2d(input_nc, ngf, kernel_size=kernel_size, padding=padding,stride=stride)]
    if norm_layer:
        model += [norm_layer(ngf)]

    if relu=='relu':
        model += [nn.ReLU(True)]
    elif relu=='Lrelu':
        model += [nn.LeakyReLU(0.2, True)]


    return model

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, norm_type='batch'):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, norm_type)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, norm_type):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm

        conv_block += conv_norm_relu_module(norm_type, norm_layer, dim,dim, 3, p)
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0.0)]
        

        if norm_type=='batch' or norm_type=='instance':
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                        norm_layer(dim)]
        else:
            assert("norm not defined")

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, norm_type='batch', gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids


        model = conv_norm_relu_module(norm_type, norm_layer, input_nc, ngf, 7, 3)
                 
        n_downsampling = 2
        for i in range(n_downsampling):
            factor_ch = 3 #2**i : 3**i is a more complicated filter
            mult = factor_ch**i 
            model += conv_norm_relu_module(norm_type,norm_layer, ngf * mult, ngf * mult * factor_ch, 3,1, stride=2)

        mult = factor_ch**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout, norm_type=norm_type)]

        for i in range(n_downsampling):
            mult = factor_ch**(n_downsampling - i)

            model += convTranspose_norm_relu_module(norm_type,norm_layer, ngf * mult, int(ngf * mult / factor_ch), 3, 1,
                                        stride=2, output_padding=1)

        if norm_type=='batch' or norm_type=='instance':
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            assert('norm not defined')

        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, encoder=False):  
    
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class Generator(nn.Module):
    def __init__(self, img_shape, in_dim=100):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # *block(in_dim, 128, normalize=False),
            nn.Linear(in_dim, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            *block(128, 128, normalize=False),
            *block(128, 256, normalize=False),
            *block(256, 512, normalize=False),
            *block(512, 1024, normalize=False),
            nn.Linear(1024, int(np.prod(img_shape))),
            # nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [
#                 nn.Conv2d(in_filters, out_filters, 3, 2, 1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Dropout2d(0.25),
#             ]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = nn.Sequential(
#             *discriminator_block(img_shape[0], 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         ds_size = img_shape[1] // 2 ** 4
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         validity = self.adv_layer(out)
#         # print(validity.shape)
#         return validity


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
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
            Tensor (N, C, self.im_size, self.im_size), padded image
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
                y.append(torch.zeros(N, self.im_size, self.im_size).type(_im.type()))
            else:
                y.append(
                    F.pad(
                        _im,
                        (x0, self.im_size - x1, y0, self.im_size - y1),
                        "constant",
                        0,
                    )
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
        y = torch.min(torch.ones(*y.shape), y)
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
            coord = coord + size * torch.tensor([[[1, 0]]]).float()  # (x0 += w, y0)
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
