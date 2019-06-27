import os
import itertools

import numpy as np
from PIL import Image, ImageDraw
import skimage.draw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from faker import Faker
from random import shuffle


LABEL_CLASS = {"rectangle": 0, "circle": 1, "triangle": 2, "photo": 3, "text": 4}


def rand_draw(n_strokes=2, width=64):
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


def load_data(data_folder="data-layer"):
    print("Loading datasets...")
    train_im = np.load(os.path.join(data_folder, "train_images.npy")).astype(np.float32)
    train_label = np.load(os.path.join(data_folder, "train_label.npy")).astype(np.long)
    train_bbox = np.load(os.path.join(data_folder, "train_bbox.npy")).astype(np.float32)
    test_im = np.load(os.path.join(data_folder, "test_images.npy")).astype(np.float32)
    test_label = np.load(os.path.join(data_folder, "test_label.npy")).astype(np.long)
    test_bbox = np.load(os.path.join(data_folder, "test_bbox.npy")).astype(np.float32)
    return train_label, train_bbox, train_im, test_label, test_bbox, test_im


def generate_text_data(width=64, n_sample=1000, n_strokes=1):
    print("Generating datasets...")

    fake = Faker()

    width = 256
    space_x, space_y = 4, 0  # in pixel
    im = Image.new("RGB", (width, width))
    draw = ImageDraw.Draw(im)

    x, y, w, h = 30, 50, 0, 0
    _y = y
    for i in range(fake.pyint(min=1, max=5)):
        dx = 0
        for j in range(fake.pyint(min=1, max=5)):
            word = fake.word()
            draw.text((x + dx, _y), word, fill=(255, 255, 255))
            _w, _h = draw.textsize(word)  # size of token
            dx += _w + space_x
        w = dx - space_x if dx - space_x > w else w
        _y += _h + space_y
    h = _y - y

    draw.rectangle([x, y, x + w, y + h])

    tokens = fake.sentence().split()

    label, bbox, im = [], [], []
    for _ in range(n_sample):
        _im, _labels = skimage.draw.random_shapes(
            (64, 64), min_shapes=1, max_shapes=n_strokes, min_size=10
        )
        _label, ((r0, r1), (c0, c1)) = _labels[0]
        _class = LABEL_CLASS[_label]
        if r0 < r1:
            y0, y1 = r0, r1
            x0, x1 = c0, c1
        else:
            y0, y1 = r1, r0
            x0, x1 = c1, c0

        if x0 > x1 or y0 > y1:
            print((r0, r1), (c0, c1))

        label.append(np.array((_class), dtype="uint8"))
        bbox.append(np.array((x0, y0, x1, y1), dtype="uint8"))
        im.append(_im)
    label = np.stack(label)
    bbox = np.stack(bbox)  # (N, 5=(class, x0, y0, x1, y1))
    im = np.stack(im).transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    indices = np.arange(0, len(label), dtype="int32")
    train, test = train_test_split(indices, test_size=0.2, random_state=0)

    if not os.path.exists("data-multi-shape/"):
        os.makedirs("data-multi-shape/")

    np.save("data-multi-shape/train_label.npy", label[train])
    np.save("data-multi-shape/train_bbox.npy", bbox[train])
    np.save("data-multi-shape/train_images.npy", im[train])
    np.save("data-multi-shape/test_label.npy", label[train])
    np.save("data-multi-shape/test_bbox.npy", bbox[train])
    np.save("data-multi-shape/test_images.npy", im[test])


def generate_shape_data(width=64, n_sample=1000, n_strokes=1):
    print("Generating datasets...")
    if not os.path.exists("data-multi-shape/"):
        os.makedirs("data-multi-shape/")

    label, bbox, im = [], [], []
    for _ in range(n_sample):
        _im, _labels = skimage.draw.random_shapes(
            (64, 64), min_shapes=1, max_shapes=n_strokes, min_size=10
        )
        _label, ((r0, r1), (c0, c1)) = _labels[0]
        _class = LABEL_CLASS[_label]
        if r0 < r1:
            y0, y1 = r0, r1
            x0, x1 = c0, c1
        else:
            y0, y1 = r1, r0
            x0, x1 = c1, c0

        if x0 > x1 or y0 > y1:
            print((r0, r1), (c0, c1))

        label.append(np.array((_class), dtype="uint8"))
        bbox.append(np.array((x0, y0, x1, y1), dtype="uint8"))
        im.append(_im)
    label = np.stack(label)
    bbox = np.stack(bbox)  # (N, 5=(class, x0, y0, x1, y1))
    im = np.stack(im).transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    indices = np.arange(0, len(label), dtype="int32")
    train, test = train_test_split(indices, test_size=0.2, random_state=0)

    np.save("data-multi-shape/train_label.npy", label[train])
    np.save("data-multi-shape/train_bbox.npy", bbox[train])
    np.save("data-multi-shape/train_images.npy", im[train])
    np.save("data-multi-shape/test_label.npy", label[train])
    np.save("data-multi-shape/test_bbox.npy", bbox[train])
    np.save("data-multi-shape/test_images.npy", im[test])


def generate_layer_data(
    width=64,
    height=64,
    n_sample=1000,
    n_strokes=1,
    photo_folder="/notebooks/CoordConv-pytorch/data-images",
    data_folder="data-layer",
):
    print("Generating datasets...")

    files = os.listdir(photo_folder)

    def _sample():
        x = np.random.randint(width, size=4).tolist()
        y = np.random.randint(height, size=4).tolist()
        x.sort()
        y.sort()

        # sample intersected bboxes
        fake = Faker()
        if fake.pybool():
            if fake.pybool():
                ax0, bx0, ax1, bx1 = x
            else:
                bx0, ax0, bx1, ax1 = x
            if fake.pybool():
                ay0, by0, ay1, by1 = y
            else:
                by0, ay0, by1, ay1 = y
        else:
            ax0, bx0, bx1, ax1 = x
            ay0, by0, by1, ay1 = y
        bboxes = [(ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1)]

        # sample layers
        layers = [0, 3]
        shuffle(layers)
        im = Image.new("RGB", (width, width))
        draw = ImageDraw.Draw(im)
        im_t = [np.array(im)]
        for layer, bbox in zip(layers, bboxes):
            x0, y0, x1, y1 = bbox
            if layer == 4:
                draw.text((x0, y0), fake.sentence(), fill=fake.hex_color())
            elif layer == 3:
                f = os.path.join(
                    photo_folder, files[fake.pyint(min=0, max=len(files) - 1)]
                )
                _im = Image.open(f).resize((x1 - x0, y1 - y0))
                im.paste(_im, box=(x0, y0))
            elif layer == 0:
                draw.rectangle((x0, y0, x1, y1), fill=fake.hex_color())
            im_t.append(np.array(im))

        # sample final layer: text
        x0 = np.random.randint(width / 2)
        y0 = np.random.randint(height / 2)
        text = fake.sentence()
        draw.text((x0, y0), text, fill=fake.hex_color())
        w, h = draw.textsize(text)
        layers.append(4)
        bboxes.append((x0, y0, w, h))
        ims = np.stack([np.concatenate([x, np.array(im)], axis=2) for x in im_t])

        return ims, np.array(layers), np.array(bboxes)

    im, label, bbox = [], [], []
    for _ in range(n_sample):
        _im, _layers, _bboxes = _sample()
        im.append(_im)
        label.append(_layers)
        bbox.append(_bboxes)
    im = np.concatenate(im).transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    label = np.concatenate(label)
    bbox = np.concatenate(bbox)

    indices = np.arange(0, len(label), dtype="int32")
    train, test = train_test_split(indices, test_size=0.2, random_state=0)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    np.save(os.path.join(data_folder, "train_images.npy"), im[train])
    np.save(os.path.join(data_folder, "train_label.npy"), label[train])
    np.save(os.path.join(data_folder, "train_bbox.npy"), bbox[train])
    np.save(os.path.join(data_folder, "test_images.npy"), im[test])
    np.save(os.path.join(data_folder, "test_label.npy"), label[test])
    np.save(os.path.join(data_folder, "test_bbox.npy"), bbox[test])


def generate_gan_font_data(width=64, n_sample=1000, n_strokes=1):
    print("Generating datasets...")

    label, bbox, im = [], [], []
    for _ in range(n_sample):
        _im, _labels = skimage.draw.random_shapes(
            (64, 64), min_shapes=1, max_shapes=n_strokes, min_size=10
        )
        _label, ((r0, r1), (c0, c1)) = _labels[0]
        _class = LABEL_CLASS[_label]
        if r0 < r1:
            y0, y1 = r0, r1
            x0, x1 = c0, c1
        else:
            y0, y1 = r1, r0
            x0, x1 = c1, c0

        if x0 > x1 or y0 > y1:
            print((r0, r1), (c0, c1))

        label.append(np.array((_class), dtype="uint8"))
        bbox.append(np.array((x0, y0, x1, y1), dtype="uint8"))
        im.append(_im)
    label = np.stack(label)
    bbox = np.stack(bbox)  # (N, 5=(class, x0, y0, x1, y1))
    im = np.stack(im).transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    indices = np.arange(0, len(label), dtype="int32")
    train, test = train_test_split(indices, test_size=0.2, random_state=0)

    if not os.path.exists("data-multi-shape/"):
        os.makedirs("data-multi-shape/")

    np.save("data-multi-shape/train_label.npy", label[train])
    np.save("data-multi-shape/train_bbox.npy", bbox[train])
    np.save("data-multi-shape/train_images.npy", im[train])
    np.save("data-multi-shape/test_label.npy", label[train])
    np.save("data-multi-shape/test_bbox.npy", bbox[train])
    np.save("data-multi-shape/test_images.npy", im[test])
