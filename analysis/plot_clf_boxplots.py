#!/usr/bin/env python3
"""
Genera la figura de boxplots de CLASIFICACION en INGLES para el paper, como un
grid 3x4 (filas = datasets iris/wine/bc, columnas = combos encoder-decoder
ss-fs/ss-v/t-fs/t-v). Cada celda muestra las 8 neuronas IZ-A..H con la
distribucion de sus 11 corridas (accuracy).

Lee de:  results/metrics/runs_classification.csv  (dataset,encoder,decoder,model,run,fitness)
Salida:  results/figures/Clasificacion/clf_boxplots_en.png
Uso: python plot_clf_boxplots.py

El resumen numerico (mean/std/median por neurona) vive en
results/metrics/resumen_clasificacion.csv (generado por src/graficar_boxplots.py).
"""
import os

import matplotlib
matplotlib.use("Agg")  # backend sin display
import matplotlib.pyplot as plt
import pandas as pd

# Fuentes mas grandes (la figura se reduce a \textwidth en el paper -> letras
# chicas con el default). Feedback del profesor sobre legibilidad.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],  # Times-like, combina con IEEEtran
    "mathtext.fontset": "stix",                          # math serif (Times-like)
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

HERE    = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.normpath(os.path.join(HERE, "..", "results", "metrics"))
OUTDIR  = os.path.normpath(os.path.join(HERE, "..", "results", "figures", "Clasificacion"))
os.makedirs(OUTDIR, exist_ok=True)

OUTPNG = os.path.join(OUTDIR, "clf_boxplots_en.png")

MODELS  = ["IZ-A", "IZ-B", "IZ-C", "IZ-D", "IZ-E", "IZ-F", "IZ-G", "IZ-H"]
LETTERS = [m.split("-")[1] for m in MODELS]  # A..H (etiquetas compactas)
DATASETS = ["iris", "wine", "bc"]
DATASET_TITLES = {"iris": "Iris", "wine": "Wine", "bc": "Breast Cancer"}
COMBOS = [("ss", "fs"), ("ss", "v"), ("t", "fs"), ("t", "v")]
COMBO_TITLES = {("ss", "fs"): "ss--fs", ("ss", "v"): "ss--v",
                ("t", "fs"): "t--fs", ("t", "v"): "t--v"}

df = pd.read_csv(os.path.join(METRICS, "runs_classification.csv"))

fig, axes = plt.subplots(len(DATASETS), len(COMBOS), figsize=(13, 8), sharey=True)

for i, ds in enumerate(DATASETS):
    for j, (enc, dec) in enumerate(COMBOS):
        ax = axes[i][j]
        data = []
        for m in MODELS:
            v = df[(df.dataset == ds) & (df.encoder == enc) &
                   (df.decoder == dec) & (df.model == m)]["fitness"].tolist()
            data.append(v)
        bp = ax.boxplot([d if d else [0] for d in data], patch_artist=True,
                        showmeans=True, meanline=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#9bc3f0")
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(MODELS) + 1))
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.set_title(COMBO_TITLES[(enc, dec)])
        if i == len(DATASETS) - 1:
            ax.set_xticklabels(LETTERS)
            ax.set_xlabel("Neuron model (IZ-)")
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(f"{DATASET_TITLES[ds]}\nAccuracy")

fig.tight_layout()
fig.savefig(OUTPNG, dpi=200)
plt.close(fig)
print(f"OK figura -> {OUTPNG}")
