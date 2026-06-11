#!/usr/bin/env python3
"""
Genera los boxplots de RL en ingles para el paper, a partir de
results/metrics/runs_rl.csv (11 corridas por modelo y problema).

Salida: results/figures/RL/{problema}_en.png  (una caja por modelo IZ-A..H).
Uso: python plot_rl_boxplots.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE    = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.normpath(os.path.join(HERE, "..", "results", "metrics"))
OUTDIR  = os.path.normpath(os.path.join(HERE, "..", "results", "figures", "RL"))
os.makedirs(OUTDIR, exist_ok=True)

MODELS = ["IZ-A", "IZ-B", "IZ-C", "IZ-D", "IZ-E", "IZ-F", "IZ-G", "IZ-H"]
TITLES = {"acrobot": "Acrobot", "cartpole": "CartPole", "mountain": "Mountain Car"}

df = pd.read_csv(os.path.join(METRICS, "runs_rl.csv"))

for problem in ["acrobot", "cartpole", "mountain"]:
    sub = df[df["problem"] == problem]
    data = [sub[sub["model"] == m]["fitness"].values for m in MODELS]

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, tick_labels=MODELS, patch_artist=True,
                    showmeans=True, meanline=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9bc3f0")
        patch.set_alpha(0.6)

    ax.set_title(TITLES[problem])
    ax.set_xlabel("Neuron model")
    ax.set_ylabel("Fitness (mean return)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUTDIR, f"{problem}_en.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"OK  {out}")
