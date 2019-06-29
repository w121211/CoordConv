import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from faker import Faker

fake = Faker()


def normal(x, width):
    return (int)(x * (width - 1) + 0.5)


def draw(f, width=128):
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype("float32")
    tmp = 1.0 / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
        y = (int)((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
        z = (int)((1 - t) * z0 + t * z2)
        w = (1 - t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))
    # return cv2.resize(canvas, dsize=(width, width))


def draw_text(x, width=64):
    x0, y0, font_size = x
    im = Image.new("RGB", (width, width))
    draw = ImageDraw.Draw(im)
    # font = ImageFont.truetype("arial", 40)
    # draw.text((0, 0), "hello", fill=(255, 255, 255), font=font)
    draw.text((0, 0), "hello", fill=(255, 255, 255))
    return np.array(im)


def draw_textbox(x, width=64):
    """
    1. random position?
    2. fixed position?
    3. along with line?

    anchor (x0, y0, x1, y1) [ char0, char1, char2, ... ]
    """
    x0, y0, x1, y1, n_tokens, n_lines = x
    pass


# def rand_draw(draw_fn=draw_rect, n_strokes=1, width=128, action_dim=4):
#     canvas = np.zeros((width, width, 3), dtype=int)
#     x = []

#     for _ in range(n_strokes):
#         _x = np.random.rand(action_dim)
#         color = np.random.randint(255, size=(3))  # (3)
#         x.append(np.concatenate((_x, color / 255.0)))

#         stroke = draw_fn(_x, width)  # (w, w)
#         stroke = np.expand_dims(stroke, axis=2)  # (w, w, 1)
#         canvas = canvas * (1 - stroke) + stroke * color  # (w, h, 3)

#     x = np.stack(x)  # (n_strokes, action_dim+3)
#     return canvas.astype(int), x


def draw_rect(xy=None, width=64, height=64):
    if xy is None:
        # n_circles = fake.pyint(min=1, max=1)
        # space = fake.pyint(min=2, max=4)
        n_circles = 1
        space = 0
        radius = fake.pyint(min=3, max=10)
        x0 = fake.pyint(min=0, max=width - 1 - (n_circles * (radius + space)))
        y0 = fake.pyint(min=0, max=height - 1 - radius)
        x1 = x0 + radius
        y1 = y0 + radius
        # x0 = 0
        # y0 = 0
        # x1 = width / 2
        # y1 = height / 2
        xy = (x0, y0, x1, y1)
        _xy = np.array(
            [x0 / width, y0 / height, x1 / width, y1 / height], dtype="float"
        )
    im = Image.new("L", (width, width))
    draw = ImageDraw.Draw(im)
    draw.rectangle(xy, fill=255)
    xy = _xy
    return xy, im


def sampler(draw_fn, n_samples=10000, save_path="data/layout/"):
    x = []
    for i in range(n_samples):
        _x, _im = draw_fn()
        x.append(_x)
        _im.save(os.path.join(save_path, "%d.png" % (i)), "PNG")
    np.save(os.path.join(save_path, "x.npy"), np.stack(x))
