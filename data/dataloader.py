import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.dataframe import dataframe_from_dir

# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, input_dir, transform=None):
        self.df = dataframe_from_dir(input_dir)
        if transform == None:
            transform = transforms.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        X = self.transform(X)
        y = torch.tensor(int(self.df['y'][index]))

        return X, y
