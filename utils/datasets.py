# utils/datasets.py

import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

LABELS = [
    "D.Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS",
    "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"
]

class RetinaDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = self.df["ID"].tolist()
        self.labels = self.df[LABELS].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_names[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.png")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f'Imagem não encontrada ou corrompida: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = torch.tensor(self.labels[idx])

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label
