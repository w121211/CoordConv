import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import torch.utils.data as utils
import torch.nn.functional as F

from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torchsummary import summary

from coordconv import AddCoords


def generate_data(datatype="uniform"):
    print('Generating datasets...')
    assert datatype in ["uniform", "quadrant"]

    if not os.path.exists("data-uniform/"):
        os.makedirs("data-uniform/")

    if not os.path.exists("data-quadrant/"):
        os.makedirs("data-quadrant/")

    onehots = np.pad(
        np.eye(3136, dtype="float32").reshape((3136, 56, 56, 1)),
        ((0, 0), (4, 4), (4, 4), (0, 0)),
        mode="constant",
    )
    onehots = onehots.transpose(0, 3, 1, 2)  # (N, C, H, W)
    
    onehots_tensor = torch.from_numpy(onehots)
    conv_layer = nn.Conv2d(
        in_channels=1, out_channels=1, kernel_size=(9, 9), padding=4, stride=1
    )
    w = torch.ones(1, 1, 9, 9)
    conv_layer.weight.data = w
    images_tensor = conv_layer(onehots_tensor)
    images = images_tensor.detach().numpy()

    if datatype == "uniform":
        # Create the uniform datasets
        indices = np.arange(0, len(onehots), dtype="int32")
        train, test = train_test_split(indices, test_size=0.2, random_state=0)

        train_onehot = onehots[train]
        train_images = images[train]
        test_onehot = onehots[test]
        test_images = images[test]

        np.save("data-uniform/train_onehot.npy", train_onehot)
        np.save("data-uniform/train_images.npy", train_images)
        np.save("data-uniform/test_onehot.npy", test_onehot)
        np.save("data-uniform/test_images.npy", test_images)
    else:
        pos_quadrant = np.where(onehots == 1.0)
        X = pos_quadrant[2]
        Y = pos_quadrant[3]

        train_set = []
        test_set = []

        train_ids = []
        test_ids = []

        for i, (x, y) in enumerate(zip(X, Y)):
            if x > 32 and y > 32:  # 4th quadrant
                test_ids.append(i)
                test_set.append([x, y])
            else:
                train_ids.append(i)
                train_set.append([x, y])

        train_set = np.array(train_set)
        test_set = np.array(test_set)

        train_set = train_set[:, :, None, None]
        test_set = test_set[:, :, None, None]

        print(train_set.shape)
        print(test_set.shape)

        train_onehot = onehots[train_ids]
        test_onehot = onehots[test_ids]

        train_images = images[train_ids]
        test_images = images[test_ids]

        print(train_onehot.shape, test_onehot.shape)
        print(train_images.shape, test_images.shape)

        np.save("data-quadrant/train_set.npy", train_set)
        np.save("data-quadrant/test_set.npy", test_set)
        np.save("data-quadrant/train_onehot.npy", train_onehot)
        np.save("data-quadrant/train_images.npy", train_images)
        np.save("data-quadrant/test_onehot.npy", test_onehot)
        np.save("data-quadrant/test_images.npy", test_images)


def load_data(datatype="uniform"):
    if datatype == "uniform":
        # Load the one hot datasets
        train_onehot = np.load("data-uniform/train_onehot.npy").astype("float32")
        test_onehot = np.load("data-uniform/test_onehot.npy").astype("float32")

        # (N, C, H, W) <=== 数据格式
        # make the train and test datasets
        # train
        pos_train = np.where(train_onehot == 1.0)
        X_train = pos_train[2]
        Y_train = pos_train[3]
        train_set = np.zeros((len(X_train), 2), dtype="float32")
        for i, (x, y) in enumerate(zip(X_train, Y_train)):
            train_set[i, 0] = x
            train_set[i, 1] = y

        # test
        pos_test = np.where(test_onehot == 1.0)
        X_test = pos_test[2]
        Y_test = pos_test[3]
        test_set = np.zeros((len(X_test), 2), dtype="float32")
        for i, (x, y) in enumerate(zip(X_test, Y_test)):
            test_set[i, 0] = x
            test_set[i, 1] = y

        # Normalize the datasets
        train_set /= 64.0 - 1.0  # 64x64 grid, 0-based index
        test_set /= 64.0 - 1.0  # 64x64 grid, 0-based index

        print("Train set : ", train_set.shape, train_set.max(), train_set.min())
        print("Test set : ", test_set.shape, test_set.max(), test_set.min())

        # Visualize the datasets
        plt.imshow(np.sum(train_onehot, axis=0)[0, :, :], cmap="gray")
        plt.title("Train One-hot dataset")
        plt.show()
        plt.imshow(np.sum(test_onehot, axis=0)[0, :, :], cmap="gray")
        plt.title("Test One-hot dataset")
        plt.show()
    else:
        # Load the one hot datasets and the train / test set
        train_set = np.load("data-quadrant/train_set.npy").astype("float32")
        test_set = np.load("data-quadrant/test_set.npy").astype("float32")

        train_onehot = np.load("data-quadrant/train_onehot.npy").astype("float32")
        test_onehot = np.load("data-quadrant/test_onehot.npy").astype("float32")

        train_set = np.tile(train_set, [1, 1, 64, 64])
        test_set = np.tile(test_set, [1, 1, 64, 64])

        # Normalize datasets
        train_set /= train_set.max()
        test_set /= test_set.max()

        print("Train set : ", train_set.shape, train_set.max(), train_set.min())
        print("Test set : ", test_set.shape, test_set.max(), test_set.min())

        # Visualize the datasets

        plt.imshow(np.sum(train_onehot, axis=0)[0, :, :], cmap="gray")
        plt.title("Train One-hot dataset")
        plt.show()
        plt.imshow(np.sum(test_onehot, axis=0)[0, :, :], cmap="gray")
        plt.title("Test One-hot dataset")
        plt.show()
    return train_set, test_set, train_onehot, test_onehot


def train(epoch, net, train_dataloader, optimizer, loss_fn, device):
    net.train()
    iters = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        # data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        iters += len(data)
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
    print("")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.add_coords = AddCoords(rank=2)
        self.conv1 = nn.Conv2d(1, 8, 1)
        self.conv2 = nn.Conv2d(8, 8, 1)
        # self.conv3 = nn.Conv2d(8, 8, 1)
        # self.conv4 = nn.Conv2d(8, 8, 1)
        # self.conv5 = nn.Conv2d(8, 8, 3)
        self.conv6 = nn.Conv2d(8, 2, 3)
        self.pool = nn.MaxPool2d(64, stride=64)

    def forward(self, x):
        # x: (N, C_in, H, W)
        # x = self.add_coords(x)
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = F.relu(self.conv6(x))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv6(x)
        x = self.pool(x)
        x = x.view(-1, 2)
        return x


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    datatype = "uniform"
    # generate_data(datatype)
    train_set, test_set, train_onehot, test_onehot = load_data(datatype)

    # train_tensor_x = torch.stack([torch.Tensor(i) for i in train_set])
    # train_tensor_y = torch.stack([torch.Tensor(i) for i in train_onehot])
    train_tensor_x = torch.from_numpy(train_set)  # coordinates
    train_tensor_y = torch.from_numpy(train_onehot)  # onehot-images
    print(train_tensor_x[0])
    train_dataset = utils.TensorDataset(train_tensor_y, train_tensor_x)
    train_dataloader = utils.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # test_tensor_x = torch.stack([torch.Tensor(i) for i in test_set])
    # test_tensor_y = torch.stack([torch.LongTensor(i) for i in test_onehot])
    # test_dataset = utils.TensorDataset(test_tensor_y, test_tensor_x)
    # test_dataloader = utils.DataLoader(test_dataset, batch_size=32, shuffle=False)

    net = Net().to(device)
    # summary(net, input_size=(1, 64, 64))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    epochs = 100

    for epoch in range(1, epochs + 1):
        train(epoch, net, train_dataloader, optimizer, loss_fn, device)

