import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from ML_Pipeline.network import UNetPP
from argparse import ArgumentParser
from albumentations.augmentations import transforms
from albumentations import Resize
from albumentations.core.composition import Compose


val_transform = Compose([
    Resize(256, 256),
    transforms.Normalize(),
])


def image_loader(image_name):
    img = imread(image_name)
    img = val_transform(image=img)["image"]
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)

    return img

