import torch
import pandas as pd
from torch.utils.data import DataLoader

from models.query2label import Query2Label
from utils.datasets import RetinaDataset
from utils.metrics import multilabel_accuracy
from preprocessing.augmentations import get_train_augmentations

# Config
CSV_PATH = "data/RFMiD/Training_Labels.csv"
IMAGE_DIR = "data/RFMiD/train_preprocessed"
MODEL_PATH = "outputs/query2label_final.pth"
NUM_CLASSES = 29
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LABELS = [
    "D.Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS",
    "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"
]

def main():
    print("📦 Carregando dataset completo para avaliação...")
    dataset = RetinaDataset(CSV_PATH, IMAGE_DIR, transform=get_train_augmentations())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Query2Label(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    print("🔎 Avaliando modelo...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            all_preds.append(outputs.cpu().sigmoid().numpy())
            all_labels.append(labels.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    print("📊 Calculando métricas por classe:")
    accs = multilabel_accuracy(y_true, y_pred)

    for i in range(NUM_CLASSES):
        print(f"{LABELS[i]:<8}: {accs[i]*100:.1f}%")

if __name__ == "__main__":
    main()
