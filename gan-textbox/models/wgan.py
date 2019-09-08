import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.autograd as autograd
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# -------------------------------
#            WGAN
# -------------------------------


def conv_norm_relu_module(
    norm_type, norm_layer, input_nc, ngf, kernel_size, padding, stride=1, relu="relu"
):

    model = [
        nn.Conv2d(
            input_nc, ngf, kernel_size=kernel_size, padding=padding, stride=stride
        )
    ]
    if norm_layer:
        model += [norm_layer(ngf)]

    if relu == "relu":
        model += [nn.ReLU(True)]
    elif relu == "Lrelu":
        model += [nn.LeakyReLU(0.2, True)]

    return model


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, norm_type="batch"):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, norm_type
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, norm_type):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert padding_type == "zero"
        p = 1

        # TODO: InstanceNorm

        conv_block += conv_norm_relu_module(norm_type, norm_layer, dim, dim, 3, p)
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        else:
            conv_block += [nn.Dropout(0.0)]

        if norm_type == "batch" or norm_type == "instance":
            conv_block += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                norm_layer(dim),
            ]
        else:
            assert "norm not defined"

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.InstanceNorm2d,
        use_dropout=False,
        n_blocks=6,
        norm_type="batch",
        gpu_ids=[],
    ):
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = conv_norm_relu_module(norm_type, norm_layer, input_nc, ngf, 7, 3)

        n_downsampling = 2
        for i in range(n_downsampling):
            factor_ch = 3  # 2**i : 3**i is a more complicated filter
            mult = factor_ch ** i
            model += conv_norm_relu_module(
                norm_type,
                norm_layer,
                ngf * mult,
                ngf * mult * factor_ch,
                3,
                1,
                stride=2,
            )

        mult = factor_ch ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    "zero",
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    norm_type=norm_type,
                )
            ]

        for i in range(n_downsampling):
            mult = factor_ch ** (n_downsampling - i)

            model += convTranspose_norm_relu_module(
                norm_type,
                norm_layer,
                ngf * mult,
                int(ngf * mult / factor_ch),
                3,
                1,
                stride=2,
                output_padding=1,
            )

        if norm_type == "batch" or norm_type == "instance":
            model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        else:
            assert "norm not defined"

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
        self.img_shape = img_shape

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
        img = img.view(z.shape[0], *self.img_shape)
        return img


# class Discriminator(nn.Module):
#     def __init__(self, img_shape):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(img_shape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#         )

#     def forward(self, img):
#         img_flat = img.view(img.shape[0], -1)
#         validity = self.model(img_flat)
#         return validity


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        # print(validity.shape)
        return validity


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

