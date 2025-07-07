import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from models.query2label import Query2Label
from utils.datasets import RetinaDataset
from preprocessing.augmentations import get_train_augmentations

# Config
CSV_PATH = "data/RFMiD/Training_Labels.csv"
IMAGE_DIR = "data/RFMiD/train_preprocessed"
BATCH_SIZE = 16
EPOCHS = 10
VAL_SPLIT = 0.15
NUM_CLASSES = 29
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("🔄 Carregando dataset...")
    full_dataset = RetinaDataset(CSV_PATH, IMAGE_DIR, transform=get_train_augmentations())

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print("⚙️ Inicializando modelo Query2Label...")
    model = Query2Label(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    print("🚀 Iniciando treinamento...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"📊 Loss de treino: {avg_train_loss:.4f}")

        # Validação simples
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"✅ Val loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "outputs/query2label_final.pth")
    print("💾 Modelo salvo em outputs/query2label_final.pth")

if __name__ == "__main__":
    main()
