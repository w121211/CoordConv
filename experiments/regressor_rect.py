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

# from models.CornerNet_Squeeze import corner_pool
# from models.py_utils import TopPool, LeftPool


def norm(x, width):
    return (int)(x * (width - 1) + 0.5)


def one_hot(x, y, width=64):
    z = np.zeros((width, width), dtype=int)
    z[y][x] = 1
    z = z.transpose(1, 0).reshape(width * width)  # (W, H) -> (H, W) -> (H*W)
    return z


def draw_rect_pil(xy, width=64):
    x0, y0, x1, y1 = xy
    x1 -= 0.5
    y1 -= 0.5
    im = Image.new("F", (width, width))
    draw = ImageDraw.Draw(im)
    draw.rectangle([x0, y0, x1, y1], fill=1)
    im = np.array(im)  # (H, W)
    return im


def draw_rect_np(xy, width=3):
    x0, y0, x1, y1 = xy
    im = np.zeros((width, width))
    for i, j in itertools.product(range(x0, x1), range(y0, y1)):
        im[i][j] = 1.0
    return im


def rand_draw(draw_fn=draw_rect_pil, n_strokes=1, width=128, action_dim=4):
    canvas = np.zeros((width, width, 3), dtype=int)
    x = []

    for _ in range(n_strokes):
        _x = np.random.rand(action_dim)
        color = np.random.randint(255, size=(3))  # (3)
        x.append(np.concatenate((_x, color / 255.0)))

        stroke = draw_fn(_x, width)  # (w, w)
        stroke = np.expand_dims(stroke, axis=2)  # (w, w, 1)
        canvas = canvas * (1 - stroke) + stroke * color  # (w, h, 3)

    x = np.stack(x)  # (n_strokes, action_dim+3)
    return canvas.astype(int), x


def draw_l2_distance(x, y, width=64):
    im = np.zeros((width, width), dtype=float)
    for (i, j), _ in np.ndenumerate(im):
        im[i][j] = np.linalg.norm(np.array([x, y]) - np.array([i, j])) / width
    return im


def generate_data(width=64, n_sample=1000):
    print("Generating datasets...")
    if not os.path.exists("data-rect/"):
        os.makedirs("data-rect/")

    _xs, x, im, dist = [], [], [], []
    if width < 10:
        for x0, y0 in itertools.product(range(width), range(width)):
            for _w, _h in itertools.product(
                range(1, width - x0 + 1), range(1, width - y0 + 1)
            ):
                x1 = x0 + _w
                y1 = y0 + _h
                _xs.append(np.array((x0, y0, x1, y1), dtype=int))
    else:
        for _ in range(n_sample):
            x0, y0 = np.random.randint(width, size=2)
            x1 = x0 + np.random.randint(1, width - x0 + 1)
            y1 = y0 + np.random.randint(1, width - y0 + 1)
            _xs.append(np.array((x0, y0, x1, y1), dtype=int))
    for _x in _xs:
        # _im = draw_rect_np(_x, width)
        _im = draw_rect_pil(_x, width)
        #         _dist = draw_l2_distance(x0, y0, width)
        im.append(_im)
        _x = _x.astype(float) / width
        x.append(_x)
    #         dist.append(_dist)

    x = np.stack(x)
    im = np.stack(im)  # (N, W, H)
    im = np.expand_dims(im, axis=-1)
    # im = im.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)  only when PIL.Image
    im = im.transpose(0, 3, 2, 1)  # (N, W, H, C) -> (N, C, H, W)
    #     dist = np.stack(dist)
    #     dist = np.expand_dims(dist, axis=-1)
    #     dist = dist.transpose(0, 3, 2, 1)  # (N, H, W, C) -> (N, C, H, W)

    indices = np.arange(0, len(x), dtype="int32")
    train, test = train_test_split(indices, test_size=0.2, random_state=0)

    np.save("data-rect/train_x.npy", x[train])
    np.save("data-rect/train_images.npy", im[train])
    #     np.save("data-rect/train_dist.npy", dist[train])
    np.save("data-rect/test_x.npy", x[test])
    np.save("data-rect/test_images.npy", im[test])


#     np.save("data-rect/test_dist.npy", dist[test])


def load_data():
    print("loading data...")
    train_x = np.load("data-rect/train_x.npy").astype("float32")
    train_im = np.load("data-rect/train_images.npy").astype("float32")
    #     train_dist = np.load("data-rect/train_dist.npy").astype("float32")
    train_dist = None
    test_x = np.load("data-rect/test_x.npy").astype("float32")
    test_im = np.load("data-rect/test_images.npy").astype("float32")
    #     test_dist = np.load("data-rect/test_dist.npy").astype("float32")
    test_dist = None

    # print("Train set : ", train_set.shape, train_set.max(), train_set.min())
    # print("Test set : ", test_set.shape, test_set.max(), test_set.min())

    # Visualize the datasets
    # plt.imshow(np.sum(train_onehot, axis=0)[0, :, :], cmap="gray")
    # plt.title("Train One-hot dataset")
    # plt.show()
    # plt.imshow(np.sum(test_onehot, axis=0)[0, :, :], cmap="gray")
    # plt.title("Test One-hot dataset")
    # plt.show()

    return train_x, train_im, train_dist, test_x, test_im, test_dist


class OnehotNet(nn.Module):
    def __init__(self, width=64):
        super(OnehotNet, self).__init__()
        self.width = width
        # self.coordconv = CoordConv2d(2, 32, 1, with_r=True)
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 1, 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(1)
        # self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
        self.conv4 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        # x = self.coordconv(x)
        # x = F.relu(self.conv1(x))
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = self.conv4(x)
        x = x.view(-1, self.width ** 2)
        return x


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


class HRNet(nn.Module):
    def __init__(self, width):
        super(HRNet, self).__init__()
        cfg = get_cfg_defaults()
        cfg.merge_from_file("./experiments/exp.yaml")
        self.hr = HighResolutionNet(cfg)
        self.add_coords = AddCoords(rank=2)
        self.conv5 = nn.Conv2d(7, 7, 3, padding=1)
        self.conv6 = nn.Conv2d(7, 7, 3, padding=1)
        self.conv7 = nn.Conv2d(7, 4, 1)
        self.pool = nn.MaxPool2d(width, stride=width)

    def forward(self, x):
        x1 = self.hr(x)
        x1 = F.interpolate(x1, scale_factor=4)
        x = torch.cat((x, x1), dim=1)
        x = self.add_coords(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        x = self.pool(x)
        x = x.view(-1, 4)
        return x


def train(epoch, net, train_dataloader, optimizer, loss_fn, device):
    net.train()
    iters = 0
    for batch_idx, (x, y_target) in enumerate(train_dataloader):
        x, y_target = x.to(device), y_target.to(device)
        optimizer.zero_grad()
        y = net(x)
        # print('-------')
        #         print(x.shape)
        #         print(y.shape)
        #         print(y_target)
        loss = loss_fn(y, y_target)
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
    width = 512
    generate_data(width)
    train_x, train_im, train_dist, test_x, test_im, test_dist = load_data()
    train_tensor_x = torch.from_numpy(train_x)
    train_tensor_im = torch.from_numpy(train_im)
    # train_tensor_dist = torch.from_numpy(train_dist)

    train_dataset = utils.TensorDataset(train_tensor_im, train_tensor_x)
    train_dataloader = utils.DataLoader(train_dataset, batch_size=16, shuffle=True)

    # test_tensor_x = torch.stack([torch.Tensor(i) for i in test_set])
    # test_tensor_y = torch.stack([torch.LongTensor(i) for i in test_onehot])
    # test_dataset = utils.TensorDataset(test_tensor_y, test_tensor_x)
    # test_dataloader = utils.DataLoader(test_dataset, batch_size=32, shuffle=False)

    def cross_entropy_one_hot(input, target):
        _, labels = target.max(dim=1)
        return nn.CrossEntropyLoss()(input, labels)

    # model = SimpleNet(width).to(device)
    model = HRNet(width).to(device)
    # summary(model, input_size=(1, 64, 64))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # loss_fn = cross_entropy_one_hot
    epochs = 1000

    for epoch in range(1, epochs + 1):
        train(epoch, model, train_dataloader, optimizer, loss_fn, device)
