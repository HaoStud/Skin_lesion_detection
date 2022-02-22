#!/bin/env pipenv-shebang

import cv2
import os
import albumentations as A
import numpy as np
from tqdm import tqdm

from PIL import Image
from os import walk
from skimage import io
from torchvision import transforms as tr
from torchvision.transforms import Compose

# Data directories
input_orga  = '../input/orga'
input_aug   = '../input/aug'

# Augmentation
transform_augmetation = A.Compose([
    A.Rotate(limit=180),
    A.HorizontalFlip(p=0.5),
    A.CLAHE(p=0.3),
    A.FancyPCA(p=0.1),
    A.RandomGamma(p=0.2),
    A.ImageCompression(quality_lower=60, p=0.4),
    A.RandomBrightnessContrast(p=0.3),
    A.Resize(224, 280),
    A.CenterCrop(224, 224),
    #A.Normalize(mean=(0.77883167, 0.53756908, 0.55911462), std=(0.229, 0.224, 0.225))
])

transform_preprocess = A.Compose([
    A.Resize(224, 280),
    A.CenterCrop(224, 224),
])

# Get all files from dataset
neoplasms = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
datasets = ['train', 'val', 'test']

# Create dir if not exists
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

create_dir(input_aug)
for set in datasets:
    create_dir(f'{input_aug}/{set}')
    for neoplasm in neoplasms: create_dir(f'{input_aug}/{set}/{neoplasm}')

for set in datasets:
    print(f'Preprocessing {set} dataset!')
    for neoplasm in tqdm(neoplasms):
        filenames = next(walk(f'{input_orga}/{set}/{neoplasm}'), (None, None, []))[2]
        imgs = []

        # Preprocess normal images
        for name in filenames:
            img = cv2.imread(f'{input_orga}/{set}/{neoplasm}/{name}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

            img = transform_preprocess(image=img)['image']
            io.imsave(f'{input_aug}/{set}/{neoplasm}/{name}', img, quality=100)

        # Create augmented images for training
        i = 0
        num = 0
        class_num = len(filenames)
        while set == 'train' and class_num + num <= 2200:
            if i >= class_num - 1:
                i = 0

            # Augment images
            img = transform_augmetation(image=imgs[i])['image']
            io.imsave(f'{input_aug}/{set}/{neoplasm}/aug_{num}.jpg', img, quality=100)

            i += 1
            num += 1
print('Finished preprocessing!')