# visualize_predictions.py

import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.query2label import Query2Label
from utils.datasets import RetinaDataset
from preprocessing.augmentations import get_train_augmentations

# Config
CSV_PATH = "data/RFMiD/Training_Labels.csv"
IMAGE_DIR = "data/RFMiD/train_preprocessed"
MODEL_PATH = "outputs/query2label_final.pth"
LABELS = [
    "D.Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS",
    "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"
]
NUM_CLASSES = len(LABELS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

def show_image(image_tensor, true_labels, pred_labels, image_id):
    image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np)
    ax.axis('off')
    ax.set_title(f"📷 {image_id}\n\n✅ Verdadeiros: {true_labels}\n🔮 Preditos: {pred_labels}", fontsize=9)
    plt.tight_layout()
    plt.show()

def main():
    dataset = RetinaDataset(CSV_PATH, IMAGE_DIR, transform=get_train_augmentations())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Query2Label(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("🔍 Mostrando algumas predições...")
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            image_id = dataset.image_names[count]
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()[0]

            true = [LABELS[i] for i, v in enumerate(labels[0]) if v == 1]
            pred = [LABELS[i] for i, p in enumerate(preds) if p > 0.5]

            show_image(images.cpu(), true, pred, image_id)

            count += 1
            if count >= 5:  # Mostrar só 5 exemplos
                break

if __name__ == "__main__":
    main()
