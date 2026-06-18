#!/usr/bin/env python3
"""
Genera la figura de boxplots de RL en INGLES para el paper, como una sola
imagen con tres paneles en fila (Acrobot, CartPole, Mountain Car), a partir de
results/metrics/runs_rl.csv (11 corridas por modelo y problema).

Sobre cada panel se dibujan corchetes con asteriscos para los pares de neuronas
con diferencia significativa segun el post-hoc de Dunn (correccion de Holm),
leidos de results/stats/rl_{problema}_dunn.csv:
    *  p < 0.05    **  p < 0.01    ***  p < 0.001

Salida: results/figures/RL/rl_boxplots_en.png
Uso: python plot_rl_boxplots.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
STATS   = os.path.normpath(os.path.join(HERE, "..", "results", "stats"))
OUTDIR  = os.path.normpath(os.path.join(HERE, "..", "results", "figures", "RL"))
os.makedirs(OUTDIR, exist_ok=True)
OUTPNG  = os.path.join(OUTDIR, "rl_boxplots_en.png")

MODELS   = ["IZ-A", "IZ-B", "IZ-C", "IZ-D", "IZ-E", "IZ-F", "IZ-G", "IZ-H"]
LETTERS  = [m.split("-")[1] for m in MODELS]
PROBLEMS = ["acrobot", "cartpole", "mountain"]
TITLES   = {"acrobot": "Acrobot", "cartpole": "CartPole", "mountain": "Mountain Car"}
ALPHA    = 0.05


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < ALPHA: return "*"
    return None


def significant_pairs(problem):
    """Pares (i, j, stars) con i<j (posiciones 0..7) significativos en Dunn-Holm."""
    path = os.path.join(STATS, f"rl_{problem}_dunn.csv")
    if not os.path.isfile(path):
        return []
    d = pd.read_csv(path, index_col=0).reindex(index=MODELS, columns=MODELS)
    out = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            s = stars(float(d.iloc[i, j]))
            if s:
                out.append((i, j, s))
    return out


def assign_levels(pairs):
    """Apila los corchetes en niveles para que no se solapen horizontalmente."""
    pairs = sorted(pairs, key=lambda p: (p[1] - p[0], p[0]))
    levels = []           # cada nivel: lista de (lo, hi) ocupados
    assigned = []
    for (i, j, s) in pairs:
        placed = False
        for lvl, occ in enumerate(levels):
            if all(j < lo or i > hi for (lo, hi) in occ):
                occ.append((i, j)); assigned.append((i, j, s, lvl)); placed = True; break
        if not placed:
            levels.append([(i, j)]); assigned.append((i, j, s, len(levels) - 1))
    return assigned


df = pd.read_csv(os.path.join(METRICS, "runs_rl.csv"))
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))

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

    all_vals = np.concatenate(data)
    pairs = significant_pairs(problem)

    # Acrobot: acotar el eje para que el outlier -500 de IZ-G no aplaste las cajas
    if problem == "acrobot":
        ybot = -175.0
        ytop_data = all_vals[all_vals > -400].max()
        ax.annotate(r"IZ-G: one run at $-500$ (off-scale)",
                    xy=(0.5, 0.025), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=11, color="dimgray")
    else:
        ybot = ax.get_ylim()[0]
        ytop_data = all_vals.max()

    if pairs:
        rng  = ytop_data - ybot
        y0   = ytop_data + 0.05 * rng
        step = 0.085 * rng
        drop = 0.015 * rng
        ytop = y0
        for (i, j, s, lvl) in assign_levels(pairs):
            y = y0 + lvl * step
            x1, x2 = i + 1, j + 1
            ax.plot([x1, x1, x2, x2], [y - drop, y, y, y - drop], lw=1.2, color="black")
            ax.text((x1 + x2) / 2.0, y, s, ha="center", va="bottom", fontsize=14)
            ytop = max(ytop, y + 0.55 * step)
        ax.set_ylim(ybot, ytop)
    elif problem == "acrobot":
        ax.set_ylim(ybot, ytop_data + 0.05 * (ytop_data - ybot))

fig.tight_layout()
fig.savefig(OUTPNG, dpi=200)
plt.close(fig)
print(f"OK figura -> {OUTPNG}")
