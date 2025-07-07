# 🔍 Retina Diagnosis with Query2Label (RFMiD)

Este projeto implementa o modelo **Query2Label** para diagnóstico automático de **29 patologias oculares** com o dataset **RFMiD**. A arquitetura é baseada em **Transformers** e permite **multi-rótulo** em imagens de retina.

---

## 📦 Estrutura do Projeto

```
retina-query2label/
├── data/
│   └── RFMiD/
│       ├── images/
│       │   ├── Training/
│       │   ├── Validation/
│       │   └── Test/
│       ├── Training_Labels.csv
│       ├── RFMiD_Validation_Labels.csv
│       └── RFMiD_Testing_Labels.csv
├── models/
│   └── query2label.py
├── preprocessing/
│   ├── run_preprocessing.py
│   ├── crop_center.py
│   ├── upsampling.py
│   └── augmentations.py
├── utils/
│   ├── datasets.py
│   └── metrics.py
├── train.py
├── evaluate.py
├── compare_results.py
├── visualize_predictions.py
├── requirements.txt
└── README.md
```

---

## 🚀 Reproduzindo o Artigo

### ✅ 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/retina-query2label.git
cd retina-query2label
```

### ✅ 2. Criar ambiente virtual e instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### ✅ 3. Organizar os dados do RFMiD

Baixe os dados do RFMiD e organize assim:

```
data/RFMiD/
├── images/
│   ├── Training/
│   ├── Validation/
│   └── Test/
├── Training_Labels.csv
├── RFMiD_Validation_Labels.csv
└── RFMiD_Testing_Labels.csv
```

> Os arquivos de imagem devem estar no formato **.png**, com nomes idênticos aos do CSV.

---

### ✅ 4. Pré-processamento (corte central, resize e upsampling)

```bash
python preprocessing/run_preprocessing.py
```

Gera:
- Imagens cortadas e redimensionadas (`384x384`)
- Aumenta as classes minoritárias
- Salva em: `data/RFMiD/train_preprocessed`

---

### ✅ 5. Treinamento do modelo Query2Label

```bash
python train.py
```

- Treina por 10 épocas
- Salva o modelo em `outputs/query2label_final.pth`

---

### ✅ 6. Avaliação

```bash
python evaluate.py
```

- Mede a acurácia por classe
- Exibe resultados em formato tabular

---

### ✅ 7. Comparação com Estado da Arte

```bash
python compare_results.py
```

- Compara Query2Label com:
  - ResNet101
  - CvT-W24
  - Estado da arte (Rodriguez et al.)

---

### ✅ 8. Visualização das predições

```bash
python visualize_predictions.py
```

- Exibe 5 imagens com:
  - ✅ Labels reais
  - 🔮 Labels preditos pelo modelo

---

## 🧠 Créditos

- Arquitetura: [Query2Label: A Query-based End-to-End Framework for Multi-Label Image Recognition](https://arxiv.org/abs/2107.10821)
- Dados: [RFMiD - Retinal Fundus Multi-Disease Image Dataset](https://www.kaggle.com/datasets/andrewmvd/retinal-fundus-images-dataset)

---

## 🧪 Requisitos

- Python 3.8+
- PyTorch
- OpenCV
- Albumentations
- scikit-learn