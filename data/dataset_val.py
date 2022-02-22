import os
import pandas as pd
from skimage import io

# Data directories
input_orig = "input/val-orig"
input_orga  = "input/val-orga"

neoplasms = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']


# Read dataset and rename neoplasms to lowercase
df = pd.read_csv(f'{input_orig}/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv')
cols = {col: col.lower() for col in df.columns}
df.rename(columns=cols, inplace=True)


# Create dir if not exists
def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

create_dir(input_orga)
for neoplasm in neoplasms: create_dir(f'{input_orga}/{neoplasm}')


# Returns image if it does exist in either part 1 or 2
def get_image_if_exists(img_name):
    path = f'{input_orig}/ISIC2018_Task3_Validation_Input/{img_name}.jpg'
    if os.path.exists(path):
        return io.imread(path)

    print(f'Image not found: {img_name}')
    return None


# Create dataset
print('Creating dataset!')
for neoplasm in neoplasms:
    image_ids = df[df[neoplasm] == 1.0]['image'].tolist()
    for img_name in image_ids:
        img = get_image_if_exists(img_name)
        if img is not None:
            io.imsave(f'{input_orga}/{neoplasm}/{img_name}.jpg', img, quality=100)
print('Finished dataset!')
