import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from faker import Faker


class MultiImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(root + "/*.*"))
        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):
        # im = Image.open(self.files[index % len(self.files)])
        im = Image.open(self.files[index % len(self.files)])
        im = self.transform(im)
        target = 0
        return im, target

    def __len__(self):
        return len(self.files)


def generate_real_samples(n_sample=1000, save_path="data/layout/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    width, height = 28, 28
    fake = Faker()

    def _sample(save_path):
        n_circles = fake.pyint(min=1, max=1)
        radius = fake.pyint(min=5, max=10)
        space = fake.pyint(min=2, max=4)
        x0 = fake.pyint(min=0, max=width - 1 - (n_circles * (radius + space)))
        y0 = fake.pyint(min=0, max=height - 1 - radius)

        im = Image.new("L", (width, height))
        draw = ImageDraw.Draw(im)

        for _ in range(n_circles):
            draw.rectangle((x0, y0, x0 + radius, y0 + radius), fill=255)
            x0 += radius + space
        # plt.imshow(np.array(im))
        im.save(save_path, "PNG")

        return np.array(im)

    for i in range(n_sample):
        _sample(os.path.join(save_path, "%d.png" % (i)))


def load_data(data_path="data-layout"):
    print("Loading datasets...")
    real_im = np.load(os.path.join(data_path, "real_images.npy")).astype(np.float32)
    return real_im

