import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg16bn import VGG16BN, init_weights


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False, output_class=2):
        super(CRAFT, self).__init__()

        # Base network
        self.basenet = VGG16BN(pretrained, freeze)

        # U network
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)

        # final conv classifier
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, output_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(
            y, size=sources[2].size()[2:], mode="bilinear", align_corners=False
        )

        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(
            y, size=sources[3].size()[2:], mode="bilinear", align_corners=False
        )

        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)
        y = F.interpolate(
            y, size=sources[4].size()[2:], mode="bilinear", align_corners=False
        )

        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        # ToDo - Remove the interpolation and make changes in the dataloader to make target width, height //2
        y = F.interpolate(y, size=(768, 768), mode="bilinear", align_corners=False)

        return y


class CRAFTGenerator(nn.Module):
    def __init__(
        self, pretrained=False, freeze=False, output_class=1, z_dim=128, img_size=128
    ):
        super(CRAFTGenerator, self).__init__()

        # Base network
        self.vgg = VGG16BN(pretrained, freeze)

        # U network
        self.upconv1 = DoubleConv(1024 + 1, 512, 256)
        self.upconv2 = DoubleConv(512 + 1, 256, 128)
        self.upconv3 = DoubleConv(256 + 1, 128, 64)
        self.upconv4 = DoubleConv(128 + 1, 64, 32)

        # z to latent
        self.fc1 = nn.Linear(z_dim, int((img_size * 2 / 16) ** 2))
        self.fc2 = nn.Linear(z_dim, int((img_size * 2 / 8) ** 2))
        self.fc3 = nn.Linear(z_dim, int((img_size * 2 / 4) ** 2))
        self.fc4 = nn.Linear(z_dim, int((img_size * 2 / 2) ** 2))

        # final conv classifier
        self.conv_classifier = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, output_class, kernel_size=1),
            nn.Tanh(),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_classifier.modules())

    def forward(self, z, img):
        """
        Args:
            img: (N, C, H*2, W*2)
        Return:
            mask: (N, 1, H, W)
        """
        # Downsampling
        sources = self.vgg(img)

        # Upsampling
        N, _, H, W = sources[0].shape
        x = torch.cat([sources[0], sources[1], self.fc1(z).view(N, 1, H, W)], dim=1)
        x = self.upconv1(x)
        x = F.interpolate(
            x, size=sources[2].size()[2:], mode="bilinear", align_corners=False
        )

        N, _, H, W = sources[2].shape
        x = torch.cat([x, sources[2], self.fc2(z).view(N, 1, H, W)], dim=1)
        x = self.upconv2(x)
        x = F.interpolate(
            x, size=sources[3].size()[2:], mode="bilinear", align_corners=False
        )

        N, _, H, W = sources[3].shape
        x = torch.cat([x, sources[3], self.fc3(z).view(N, 1, H, W)], dim=1)
        x = self.upconv3(x)
        x = F.interpolate(
            x, size=sources[4].size()[2:], mode="bilinear", align_corners=False
        )

        N, _, H, W = sources[4].shape
        x = torch.cat([x, sources[4], self.fc4(z).view(N, 1, H, W)], dim=1)
        feature = self.upconv4(x)

        x = self.conv_classifier(feature)

        # ToDo - Remove the interpolation and make changes in the dataloader to make target width, height //2
        # y = F.interpolate(y, size=(768, 768), mode="bilinear", align_corners=False)

        return x
