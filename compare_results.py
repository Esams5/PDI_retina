# compare_results.py

import numpy as np
from utils.metrics import multilabel_accuracy
import pandas as pd

# Precisão do artigo (convertido de % para 0-1)
labels = [
    "D.Risk", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "LS",
    "CSR", "ODC", "CRVO", "ODP", "ODE", "RS", "CRS", "OTHER"
]

resnet101 = [
    0.987, 0.608, 0.860, 0.460, 0.873, 0.577, 0.649, 0.539,
    0.281, 0.552, 0.733, 0.370, 0.729, 0.825, 0.271, 0.175
]

cvt_w24 = [
    0.998, 0.671, 0.862, 0.292, 0.832, 0.653, 0.752, 0.659,
    0.654, 0.358, 0.824, 0.207, 0.728, 0.859, 0.297, 0.207
]

estado_arte = [
    0.859, 0.800, 0.875, 0.708, 0.810, 0.929, 0.800, 0.500,
    0.444, 0.661, 0.600, 0.000, 0.833, 1.000, 0.400, 0.587
]

def main():
    df = pd.DataFrame({
        "Patologia": labels,
        "Q2L-ResNet101": resnet101,
        "Q2L-CVT W24": cvt_w24,
        "Rodriguez et al. (2022)": estado_arte
    })

    df["Melhor Modelo"] = df[["Q2L-ResNet101", "Q2L-CVT W24", "Rodriguez et al. (2022)"]].idxmax(axis=1)
    df = df.sort_values("Patologia")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
