import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.z_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class DCDiscriminator(nn.Module):
    def __init__(self, opt):
        super(DCDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # self.model = nn.Sequential(
        #     *discriminator_block(opt.im_channels, 16, bn=False),
        #     *discriminator_block(16, 32),
        #     *discriminator_block(32, 64),
        #     *discriminator_block(64, 128),
        # )
        self.model = nn.Sequential(
            *discriminator_block(opt.im_channels, 4, bn=False),
            *discriminator_block(4, 8),
            *discriminator_block(8, 16),
            *discriminator_block(16, 32),
        )

        # The height and width of downsampled image
        ds_size = opt.imsize // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(32 * ds_size ** 2, 1), 
            # nn.Tanh(),
            # nn.Sigmoid()
            )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class LinearDiscriminator(nn.Module):
    def __init__(self, opt):
        super(LinearDiscriminator, self).__init__()
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
