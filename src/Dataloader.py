import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self):
        # Construct path relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_dir = os.path.join(project_root, "dataset/images128x128")

        # Load image file names
        self.image_files = [f for f in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = read_image(image_path)
        image = TF.resize(image[:3].float(), [64, 64]) / 127.5 - 1.0
        return image