from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OrchidDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform: transforms):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        f = self.df["filename"].iloc[index]
        img_path = Path(self.img_dir) / f
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img
