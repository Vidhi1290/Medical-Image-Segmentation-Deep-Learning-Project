
import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from ML_Pipeline.utils import AverageMeter,iou_score
from albumentations import Resize
from albumentations.augmentations import transforms
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import RandomRotate90
from ML_Pipeline.network import UNetPP, VGGBlock
from ML_Pipeline.dataset import DataSet
from ML_Pipeline.train import train
from ML_Pipeline.validate import validate
from ML_Pipeline.predict import val_transform
from ML_Pipeline.predict import image_loader
import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


# training the model

with open("config.yaml") as f:
    config = yaml.load(f)

extn = config["extn"]
epochs = config["epochs"]
log_path = config["log_path"]
mask_path = config["mask_path"]
image_path = config["image_path"]
model_path = config["model_path"]

log = OrderedDict([
    ('epoch', []),
    ('loss', []),
    ('iou', []),
    ('val_loss', []),
    ('val_iou', []),
])

best_iou = 0
trigger = 0

extn_ = f"*{extn}"
img_ids = glob(os.path.join(image_path, extn_))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2)

train_transform = Compose([
    RandomRotate90(),
    transforms.Flip(),
    OneOf([
        transforms.HueSaturationValue(),
        transforms.RandomBrightness(),
        transforms.RandomContrast(),
    ], p=1),
    Resize(256, 256),
    transforms.Normalize(),
])

val_transform = Compose([
    Resize(256, 256),
    transforms.Normalize(),
])

train_dataset = DataSet(
    img_ids=train_img_ids,
    img_dir=image_path,
    mask_dir=mask_path,
    img_ext=extn,
    mask_ext=extn,
    transform=train_transform)

val_dataset = DataSet(
    img_ids=val_img_ids,
    img_dir=image_path,
    mask_dir=mask_path,
    img_ext=extn,
    mask_ext=extn,
    transform=val_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=False)

model = UNetPP(1, 3, True)
if torch.cuda.is_available():
    model.cuda()
criterion = nn.BCEWithLogitsLoss()
params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-4)

for epoch in range(epochs):
    print(f'Epoch [{epoch}/{epochs}]')

    # train for one epoch
    train_log = train(True, train_loader, model, criterion, optimizer)
    # evaluate on validation set
    val_log = validate(True, val_loader, model, criterion)

    print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

    log['epoch'].append(epoch)
    log['loss'].append(train_log['loss'])
    log['iou'].append(train_log['iou'])
    log['val_loss'].append(val_log['loss'])
    log['val_iou'].append(val_log['iou'])

    pd.DataFrame(log).to_csv(log_path, index=False)

    trigger += 1

    if val_log['iou'] > best_iou:
        torch.save(model.state_dict(), model_path)
        best_iou = val_log['iou']
        print("=> saved best model")
        trigger = 0

# prediction

parser = ArgumentParser()
parser.add_argument("--test_img", default="../input/PNG/Original/50.png", help="path to test image")

opt = parser.parse_args()
with open("config.yaml") as f:
    config = yaml.load(f)

im_width = config["im_width"]
im_height = config["im_height"]
model_path = config["model_path"]
output_path = config["output_path"]

model = UNetPP(1, 3, True)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
if torch.cuda.is_available():
    model.cuda()
model.eval()

image = image_loader(opt.test_img)
image = np.expand_dims(image,0)
image = torch.from_numpy(image)

if torch.cuda.is_available():
    image = image.to(device="cuda")
mask = model(image)
mask = mask[-1]
mask = mask.detach().cpu().numpy()
mask = np.squeeze(np.squeeze(mask, axis=0), axis=0)
mask1 = mask.copy()
mask1[mask1 > -2.5] = 255
mask1[mask1 <= -2.5] = 0
mask1 = cv2.resize(mask1, (im_width, im_height))
plt.imsave(output_path, mask1, cmap="gray")