#!/usr/bin/env python3
"""
Analisis estadistico de la comparacion entre los 8 modelos de neurona Izhikevich
(IZ-A..H). LIF queda fuera: la comparacion es entre tipos de neurona.

Lee los datos crudos (11 corridas por combo) desde results/metrics/ y produce:

  Clasificacion (12 escenarios = 3 datasets x 4 combos enc-dec)
    - Test de Friedman sobre la matriz escenario x modelo (mediana por combo).
    - Post-hoc de Nemenyi + diagrama de diferencias criticas (CD).
    - Ranking promedio de los modelos.

  RL (3 problemas: acrobot, cartpole, mountain)
    - Pocos escenarios para un ranking global, asi que se analiza por problema:
      Kruskal-Wallis sobre los 8 modelos (11 corridas c/u) + post-hoc de Dunn (Holm).

Convencion: mayor fitness = mejor (accuracy en clasificacion, recompensa en RL),
por lo que el rank 1 corresponde al mejor modelo.

Salidas en results/stats/ (tablas .csv + diagrama .png). Uso: python stats.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp

# Fuentes grandes + serif Times para la figura (diagrama CD), consistente con
# las demas figuras del paper (plot_*_boxplots.py). Feedback del profesor.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

ALPHA = 0.05

HERE     = os.path.dirname(os.path.abspath(__file__))
METRICS  = os.path.normpath(os.path.join(HERE, "..", "results", "metrics"))
OUTDIR   = os.path.normpath(os.path.join(HERE, "..", "results", "stats"))
os.makedirs(OUTDIR, exist_ok=True)


def avg_ranks(matrix):
    """Ranking promedio por modelo (rank 1 = mejor = mayor fitness)."""
    # rank descendente dentro de cada escenario (fila), luego promedio por columna
    ranks = matrix.rank(axis=1, ascending=False)
    return ranks.mean(axis=0).sort_values()


# ===================== CLASIFICACION =====================

def analizar_clasificacion():
    print("\n" + "=" * 64)
    print("  CLASIFICACION  —  Friedman + Nemenyi (12 escenarios)")
    print("=" * 64)

    df = pd.read_csv(os.path.join(METRICS, "runs_classification.csv"))
    df["scenario"] = df["dataset"] + "-" + df["encoder"] + "-" + df["decoder"]

    # Matriz escenario x modelo con la MEDIANA de las 11 corridas (input estilo Demsar)
    matrix = (df.groupby(["scenario", "model"])["fitness"]
                .median().unstack("model").sort_index())

    models = sorted(matrix.columns)
    matrix = matrix[models]

    # --- Friedman ---
    stat, p = ss.friedmanchisquare(*[matrix[m].values for m in models])
    print(f"\nFriedman: chi2 = {stat:.4f}   p = {p:.6f}   (alpha = {ALPHA})")
    if p < ALPHA:
        print("-> Hay diferencias significativas entre los modelos.")
    else:
        print("-> No se detectan diferencias significativas entre los modelos.")

    # --- Ranking promedio ---
    ranks = avg_ranks(matrix)
    print("\nRanking promedio (1 = mejor):")
    for m, r in ranks.items():
        print(f"   {m}: {r:.3f}")
    ranks.rename("rank_promedio").to_csv(os.path.join(OUTDIR, "clasificacion_ranking.csv"))

    # --- Nemenyi post-hoc ---
    nemenyi = sp.posthoc_nemenyi_friedman(matrix.values)
    nemenyi.index = models
    nemenyi.columns = models
    nemenyi.to_csv(os.path.join(OUTDIR, "clasificacion_nemenyi.csv"))

    # --- Diagrama de diferencias criticas (CD) ---
    plt.figure(figsize=(11, 3.2))
    sp.critical_difference_diagram(ranks, nemenyi)
    plt.title("Critical difference diagram (classification)")
    plt.tight_layout()
    cd_path = os.path.join(OUTDIR, "cd_diagram_clasificacion.png")
    plt.savefig(cd_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nDiagrama CD -> {os.path.relpath(cd_path, HERE)}")

    # Guardar tambien Friedman en texto
    with open(os.path.join(OUTDIR, "clasificacion_friedman.txt"), "w", encoding="utf-8") as f:
        f.write(f"Friedman chi2={stat:.6f} p={p:.6e} alpha={ALPHA} n_escenarios={len(matrix)}\n")
        f.write("significativo\n" if p < ALPHA else "no significativo\n")


# ===================== RL =====================

def analizar_rl():
    print("\n" + "=" * 64)
    print("  RL  —  Kruskal-Wallis + Dunn por problema")
    print("=" * 64)

    df = pd.read_csv(os.path.join(METRICS, "runs_rl.csv"))
    resumen = []

    for problem in sorted(df["problem"].unique()):
        sub = df[df["problem"] == problem]
        models = sorted(sub["model"].unique())
        grupos = [sub[sub["model"] == m]["fitness"].values for m in models]

        stat, p = ss.kruskal(*grupos)
        n_total = sum(len(g) for g in grupos)
        eps2 = stat / (n_total - 1)            # epsilon-cuadrado (tamano de efecto, 0-1)
        sig = "significativo" if p < ALPHA else "no significativo"
        print(f"\n[{problem}]  Kruskal-Wallis: H = {stat:.4f}  p = {p:.6f}  "
              f"eps2 = {eps2:.3f}  -> {sig}")
        resumen.append({"problema": problem, "H": stat, "p_value": p,
                        "epsilon2": eps2, "resultado": sig})

        # Dunn post-hoc (correccion Holm) — siempre se guarda como referencia
        dunn = sp.posthoc_dunn(sub, val_col="fitness", group_col="model", p_adjust="holm")
        dunn = dunn.reindex(index=models, columns=models)
        dunn.to_csv(os.path.join(OUTDIR, f"rl_{problem}_dunn.csv"))

        # Tamano de efecto pareado: correlacion rank-biserial  r = 2U/(n1 n2) - 1
        # (r > 0 indica que el modelo de la fila supera al de la columna)
        rb = pd.DataFrame(index=models, columns=models, dtype=float)
        for a in models:
            xa = sub[sub["model"] == a]["fitness"].values
            for b in models:
                xb = sub[sub["model"] == b]["fitness"].values
                U = ss.mannwhitneyu(xa, xb, alternative="two-sided").statistic
                rb.loc[a, b] = 2.0 * U / (len(xa) * len(xb)) - 1.0
        rb.to_csv(os.path.join(OUTDIR, f"rl_{problem}_rankbiserial.csv"))

        # Mejor modelo por mediana (mayor = mejor)
        med = sub.groupby("model")["fitness"].median().sort_values(ascending=False)
        print(f"   Mejor por mediana: {med.index[0]} ({med.iloc[0]:.2f})")

    pd.DataFrame(resumen).to_csv(os.path.join(OUTDIR, "rl_kruskal.csv"), index=False)


if __name__ == "__main__":
    analizar_clasificacion()
    analizar_rl()
    print("\n" + "=" * 64)
    print(f"  Resultados guardados en: results/stats/")
    print("=" * 64)
