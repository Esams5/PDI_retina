# preprocessing/run_preprocessing.py

import os
import pandas as pd
import cv2
from crop_center import crop_and_resize
from upsampling import upsample_images
from enhancement import illumination_correction  # 🚨 ADICIONADO

# Caminhos
RAW_IMAGE_DIR = "data/RFMiD/images/Training"  # 🚨 Corrigido para 'Training'
LABEL_CSV = "data/RFMiD/Training_Labels.csv"
PROCESSED_DIR = "data/RFMiD/train_preprocessed"

# Parâmetros
TARGET_SIZE = (384, 384)
MIN_SAMPLES = 100  # mínimo por classe após upsampling

def salvar_imagem_saida(img, nome_saida):
    out_path = os.path.join(PROCESSED_DIR, nome_saida)
    cv2.imwrite(out_path, img)

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    print("🧠 Lendo CSV de rótulos...")
    df = pd.read_csv(LABEL_CSV)

    print("✂️  Cortando e redimensionando imagens...")
    for idx, row in df.iterrows():
        img_name = str(row["ID"]) + ".png"  # 🚨 Corrigido para .png
        img_path = os.path.join(RAW_IMAGE_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"⚠️  Imagem não encontrada: {img_name}")
            continue

        img_out = crop_and_resize(img_path, TARGET_SIZE)
        img_out = illumination_correction(img_out)  # 🚨 Aplicação da correção

        salvar_imagem_saida(img_out, img_name)

        if idx % 100 == 0:
            print(f"✅ {idx}/{len(df)} imagens processadas")

    print("📈 Fazendo upsampling das classes minoritárias...")
    upsample_images(df, PROCESSED_DIR, PROCESSED_DIR, min_samples=MIN_SAMPLES)

    print("✅ Pré-processamento finalizado. Imagens salvas em:", PROCESSED_DIR)

if __name__ == "__main__":
    main()
