import os
import itertools

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import torch.utils.data as utils
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from coordconv import AddCoords
from hrnet import HighResolutionNet
from config import get_cfg_defaults
from models.CornerNet_Squeeze import corner_pool
from models.py_utils import TopPool, LeftPool


def norm(x, width):
    return (int)(x * (width - 1) + 0.5)


def draw_rect_pil(xy, width=64):
    x0, y0, x1, y1 = xy
    x1 -= 0.5
    y1 -= 0.5
    im = Image.new("F", (width, width))
    draw = ImageDraw.Draw(im)
    draw.rectangle([x0, y0, x1, y1], fill=1)
    im = np.array(im)  # (H, W)
    return im


def rand_draw(draw_fn=draw_rect_pil, n_strokes=2, width=64):
    canvas = np.zeros((width, width, 3), dtype=np.int8)
    im = [canvas.copy()]
    x = []
    for _ in range(n_strokes):
        x0, y0 = np.random.randint(width, size=2)
        x1 = x0 + np.random.randint(1, width - x0 + 1)
        y1 = y0 + np.random.randint(1, width - y0 + 1)
        _x = np.array((x0, y0, x1, y1))

        color = np.random.randint(255, size=(3))  # (3)
        stroke = draw_fn(_x, width)  # (H, W)
        stroke = np.expand_dims(stroke, axis=2)  # (H, W, 1)
        canvas = canvas * (1 - stroke) + stroke * color  # (H, W, 3)

        x.append(_x)
        im.append(canvas.copy())

    x = np.stack(x) / width  # (n_strokes, action_dim+3)
    im = np.stack(im)  # (n_strokes+1, H, W, 3)
    return x, im


def generate_data(width=128, n_sample=1000, n_strokes=2):
    print("Generating datasets...")
    if not os.path.exists("data-multi-rect/"):
        os.makedirs("data-multi-rect/")

    x, im = [], []
    for _ in range(n_sample):
        _x, _im = rand_draw(n_strokes=n_strokes, width=width)
        _im_inter = _im[:-1]
        _im_target = _im[-1]
        for i in range(n_strokes):
            im.append(np.concatenate((_im_inter[i], _im_target), axis=2))
            x.append(_x[i])

    x = np.stack(x)  # (N*n_stroke, 4=(x0, y0, x1, y1))
    im = np.stack(im).transpose(
        0, 3, 1, 2
    )  # (N*n_stroke, H, W, C) -> (N*n_stroke, C, H, W)

    indices = np.arange(0, len(x), dtype="int32")
    train, test = train_test_split(indices, test_size=0.2, random_state=0)

    np.save("data-multi-rect/train_x.npy", x[train])
    np.save("data-multi-rect/train_images.npy", im[train])
    np.save("data-multi-rect/test_x.npy", x[test])
    np.save("data-multi-rect/test_images.npy", im[test])


def load_data():
    print("loading data...")
    train_x = np.load("data-multi-rect/train_x.npy").astype(np.float32)
    train_im = np.load("data-multi-rect/train_images.npy").astype(np.float32)
    test_x = np.load("data-multi-rect/test_x.npy").astype(np.float32)
    test_im = np.load("data-multi-rect/test_images.npy").astype(np.float32)

    # print("Train set : ", train_set.shape, train_set.max(), train_set.min())
    # print("Test set : ", test_set.shape, test_set.max(), test_set.min())

    # Visualize the datasets
    # plt.imshow(np.sum(train_onehot, axis=0)[0, :, :], cmap="gray")
    # plt.title("Train One-hot dataset")
    # plt.show()
    # plt.imshow(np.sum(test_onehot, axis=0)[0, :, :], cmap="gray")
    # plt.title("Test One-hot dataset")
    # plt.show()

    return train_x, train_im, test_x, test_im


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
        cfg.merge_from_file("./experiments/hrnet.yaml")
        self.hr0 = HighResolutionNet(cfg)
        self.hr1 = HighResolutionNet(cfg)
        self.rg = Regressor(in_channel=540, width=int(width / 4))

    def forward(self, x):
        im_current = x[:, :3, :, :]
        im_target = x[:, -3:, :, :]
        x0 = self.hr0(im_current)
        x1 = self.hr1(im_target)
        x = torch.cat((x0, x1), dim=1)
        x = self.rg(x)
        return x


def train(epoch, net, train_dataloader, optimizer, loss_fn, device):
    net.train()
    iters = 0
    for batch_idx, (x, y_target) in enumerate(train_dataloader):
        x, y_target = x.to(device), y_target.to(device)
        optimizer.zero_grad()
        y = net(x)
        loss = loss_fn(y, y_target)
        # print('-------')
        # print(x.shape)
        # print(y.shape)
        # print(y_target)
        loss.backward()
        optimizer.step()
        iters += len(x)
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                epoch,
                iters,
                len(train_dataloader.dataset),
                100.0 * (batch_idx + 1) / len(train_dataloader),
                loss.data.item(),
            ),
            end="\r",
            flush=True,
        )
    # print(y[0])
    # print(y_target[0])
    print("")


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    width = 64
    # generate_data(width, n_sample=100)
    train_x, train_im, test_x, test_im = load_data()
    train_tensor_x = torch.from_numpy(train_x)
    train_tensor_im = torch.from_numpy(train_im)
    train_dataset = utils.TensorDataset(train_tensor_im, train_tensor_x)
    train_dataloader = utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
    # test_tensor_x = torch.stack([torch.Tensor(i) for i in test_set])
    # test_tensor_y = torch.stack([torch.LongTensor(i) for i in test_onehot])
    # test_dataset = utils.TensorDataset(test_tensor_y, test_tensor_x)
    # test_dataloader = utils.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # model = SimpleNet(width).to(device)
    model = HRNet(width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    epochs = 1000

    for epoch in range(1, epochs + 1):
        train(epoch, model, train_dataloader, optimizer, loss_fn, device)
