import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class VertebraDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, self.mask_files[idx])).convert("L")

        img = np.array(img) / 255.0
        mask = np.array(mask) / 255.0

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask