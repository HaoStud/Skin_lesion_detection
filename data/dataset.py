#!/bin/env pipenv-shebang

import os
import pandas as pd
from skimage import io
from tqdm import tqdm

# Data directories
input_orig = '../input/orig'
input_orga  = '../input/orga'

neoplasms = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
datasets = ['train', 'val', 'test']

# Read train, val and test dataset
for set in datasets:
    exec(f'{set} = pd.read_csv(f"../metadata/{{set}}.csv")')

# Create dir if not exists
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# Create directories
create_dir(input_orga)
for set in datasets:
    create_dir(f'{input_orga}/{set}')
    for neoplasm in neoplasms: create_dir(f'{input_orga}/{set}/{neoplasm}')

# Returns image if it does exist in either part 1 or 2
def get_image_if_exists(img_name):
    possible_paths = [f'{input_orig}/HAM10000_images_part_{i}/{img_name}.jpg' for i in [1,2]]
    for path in possible_paths:
        if os.path.exists(path):
            return io.imread(path)

    print(f'Image not found: {img_name}')
    return None

# Create dataset
for set in datasets:
    print(f'Creating {set} dataset!')
    df = eval(set)
    for neoplasm in tqdm(neoplasms):
        image_ids = df[df['dx'] == neoplasm]['image_id'].tolist()
        for img_name in tqdm(image_ids):
            img = get_image_if_exists(img_name)
            if img is not None:
                io.imsave(f'{input_orga}/{set}/{neoplasm}/{img_name}.jpg', img, quality=100)
print('Finished!')
