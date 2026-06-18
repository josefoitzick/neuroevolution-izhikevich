#!/usr/bin/env python3
"""
Genera la figura de boxplots de RL en INGLES para el paper, como una sola
imagen con tres paneles en fila (Acrobot, CartPole, Mountain Car), a partir de
results/metrics/runs_rl.csv (11 corridas por modelo y problema).

Una unica imagen a ancho completo evita que cada panel se reduzca por separado
y deje las letras ilegibles (mismo tratamiento que la figura de clasificacion).

Salida: results/figures/RL/rl_boxplots_en.png
Uso: python plot_rl_boxplots.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Fuentes grandes + serif Times, mismo estilo que plot_clf_boxplots.py para
# combinar con IEEEtran y quedar legibles a \textwidth.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

HERE    = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.normpath(os.path.join(HERE, "..", "results", "metrics"))
OUTDIR  = os.path.normpath(os.path.join(HERE, "..", "results", "figures", "RL"))
os.makedirs(OUTDIR, exist_ok=True)
OUTPNG  = os.path.join(OUTDIR, "rl_boxplots_en.png")

MODELS   = ["IZ-A", "IZ-B", "IZ-C", "IZ-D", "IZ-E", "IZ-F", "IZ-G", "IZ-H"]
LETTERS  = [m.split("-")[1] for m in MODELS]
PROBLEMS = ["acrobot", "cartpole", "mountain"]
TITLES   = {"acrobot": "Acrobot", "cartpole": "CartPole", "mountain": "Mountain Car"}

df = pd.read_csv(os.path.join(METRICS, "runs_rl.csv"))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
for ax, problem in zip(axes, PROBLEMS):
    sub = df[df["problem"] == problem]
    data = [sub[sub["model"] == m]["fitness"].values for m in MODELS]
    bp = ax.boxplot(data, tick_labels=LETTERS, patch_artist=True,
                    showmeans=True, meanline=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9bc3f0")
        patch.set_alpha(0.6)
    ax.set_title(TITLES[problem])
    ax.set_xlabel("Neuron model (IZ-)")
    ax.set_ylabel("Fitness (mean return)")
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPNG, dpi=200)
plt.close(fig)
print(f"OK figura -> {OUTPNG}")
