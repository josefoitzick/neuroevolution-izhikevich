#!/usr/bin/env python3
"""Analisis factorial de la clasificacion, complementario a stats.py.

stats.py resume cada escenario por la mediana de sus 11 corridas y compara los ocho
tipos de neurona con Friedman sobre esos 12 valores (marco de Demsar). Ese resumen
descarta la variacion dentro de cada escenario. Aca se analiza el diseno completo:

    8 tipos de neurona  x  3 datasets  x  4 combos encoder-decoder  x  11 corridas
    = 1056 observaciones, con 11 replicas en cada una de las 96 celdas (balanceado).

Al estar balanceado, las sumas de cuadrados tipo I/II/III coinciden y el ANOVA
factorial se calcula de forma exacta con aritmetica directa (numpy), sin necesidad
de una libreria de modelos.

Se reportan dos analisis:

  1. ANOVA factorial de tres vias sobre datos con transformacion de rangos alineados
     (aligned rank transform, ART), que no asume normalidad. Para cada termino se
     alinea la respuesta restando las medias de celda y sumando el efecto puro de ese
     termino, se rankea y se corre el ANOVA completo, leyendo solo el F del termino.
     Se informa eta^2 parcial como tamano de efecto.

  2. Test de permutacion del efecto del tipo de neurona restringido DENTRO de cada
     escenario (respeta la estructura anidada y no asume nada sobre la distribucion).
     El estadistico es la suma de los H de Kruskal-Wallis de los 12 escenarios.

Salida: results/stats/clasificacion_factorial.csv

Uso: python analysis/factorial.py
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RUNS = os.path.join(ROOT, "results", "metrics", "runs_classification.csv")
OUT = os.path.join(ROOT, "results", "stats", "clasificacion_factorial.csv")

FACTORS = ["model", "dataset", "encdec"]
LABEL = {"model": "neuron type", "dataset": "dataset", "encdec": "encoder-decoder"}
N_PERM = 10000
SEED = 0


def load():
    df = pd.read_csv(RUNS)
    df["encdec"] = df["encoder"] + "-" + df["decoder"]
    df["scenario"] = df["dataset"] + "-" + df["encdec"]
    return df


def to_cube(df):
    """(a, b, c, n) con las replicas de cada celda; falla si el diseno no es balanceado."""
    levels = {f: sorted(df[f].unique()) for f in FACTORS}
    a, b, c = (len(levels[f]) for f in FACTORS)
    sizes = set(df.groupby(FACTORS).size())
    if len(sizes) != 1:
        raise SystemExit(f"El diseno no esta balanceado: tamanos de celda {sorted(sizes)}")
    n = sizes.pop()
    idx = {f: df[f].map({v: i for i, v in enumerate(levels[f])}).to_numpy() for f in FACTORS}
    cube = np.full((a, b, c, n), np.nan)
    ctr = np.zeros((a, b, c), int)
    for i, j, k, y in zip(idx["model"], idx["dataset"], idx["encdec"],
                          df["fitness"].to_numpy(float)):
        cube[i, j, k, ctr[i, j, k]] = y
        ctr[i, j, k] += 1
    assert not np.isnan(cube).any()
    return cube, levels, idx


def decompose(cube):
    """Efectos puros del modelo factorial completo, difundidos a la grilla (a, b, c)."""
    a, b, c, _ = cube.shape
    cell = cube.mean(axis=3)
    g = cell.mean()
    mi = cell.mean(axis=(1, 2))[:, None, None]
    mj = cell.mean(axis=(0, 2))[None, :, None]
    mk = cell.mean(axis=(0, 1))[None, None, :]
    mij = cell.mean(axis=2)[:, :, None]
    mik = cell.mean(axis=1)[:, None, :]
    mjk = cell.mean(axis=0)[None, :, :]
    eff = {
        ("A",): mi - g,
        ("B",): mj - g,
        ("C",): mk - g,
        ("A", "B"): mij - mi - mj + g,
        ("A", "C"): mik - mi - mk + g,
        ("B", "C"): mjk - mj - mk + g,
        ("A", "B", "C"): cell - mij - mik - mjk + mi + mj + mk - g,
    }
    return {t: np.broadcast_to(v, (a, b, c)) for t, v in eff.items()}, cell


TERMS = [("A",), ("B",), ("C",), ("A", "B"), ("A", "C"), ("B", "C"), ("A", "B", "C")]
TERM_FACTOR = {"A": "model", "B": "dataset", "C": "encdec"}


def anova(cube):
    """{termino: (F, df_efecto, df_error, p, eta2_parcial)} del factorial balanceado."""
    a, b, c, n = cube.shape
    dfn = {"A": a - 1, "B": b - 1, "C": c - 1}
    eff, cell = decompose(cube)
    ss_err = ((cube - cell[..., None]) ** 2).sum()
    df_err = a * b * c * (n - 1)
    out = {}
    for t in TERMS:
        ss = n * (eff[t] ** 2).sum()
        d = int(np.prod([dfn[f] for f in t]))
        f_stat = (ss / d) / (ss_err / df_err)
        out[t] = (f_stat, d, df_err, stats.f.sf(f_stat, d, df_err), ss / (ss + ss_err))
    return out


def art_anova(cube):
    """ANOVA factorial sobre rangos alineados: un alineamiento por termino."""
    out = {}
    eff, cell = decompose(cube)
    resid = cube - cell[..., None]
    for t in TERMS:
        aligned = resid + eff[t][..., None]
        ranks = stats.rankdata(aligned.ravel()).reshape(cube.shape)
        out[t] = anova(ranks)[t]
    return out


def permutation_neuron(df, idx, n_models):
    """Permuta las etiquetas de neurona dentro de cada escenario; suma de H por escenario."""
    y = df["fitness"].to_numpy(float)
    scen = df["scenario"].to_numpy()
    masks = {s: scen == s for s in np.unique(scen)}

    def statistic(labels):
        return sum(stats.kruskal(*[y[m][labels[m] == g] for g in range(n_models)]).statistic
                   for m in masks.values())

    labels = idx["model"]
    observed = statistic(labels)
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for it in range(N_PERM):
        perm = labels.copy()
        for m in masks.values():
            perm[m] = rng.permutation(perm[m])
        null[it] = statistic(perm)
    p = (1 + int((null >= observed).sum())) / (1 + N_PERM)
    return observed, null.mean(), p


def term_name(t):
    return " x ".join(LABEL[TERM_FACTOR[f]] for f in t)


def main():
    df = load()
    cube, levels, idx = to_cube(df)
    a, b, c, n = cube.shape
    print(f"Diseno {a} x {b} x {c}, n={n} -> {len(df)} corridas (balanceado)")

    art = art_anova(cube)
    rows = []
    print("\nART ANOVA de tres vias")
    print(f"{'efecto':34s} {'F':>9s} {'df':>13s} {'p':>11s} {'eta2_p':>8s}")
    for t in TERMS:
        f_stat, d, d_err, p, eta = art[t]
        rows.append({"analysis": "art_anova", "term": term_name(t), "statistic": f_stat,
                     "df1": d, "df2": d_err, "p_value": p, "partial_eta2": eta,
                     "significant": p < 0.05})
        print(f"{term_name(t):34s} {f_stat:9.2f} {d:5d},{d_err:6d} {p:11.3g} {eta:8.3f}"
              f"   {'' if p < 0.05 else 'n.s.'}")

    obs, null_mean, p_perm = permutation_neuron(df, idx, a)
    rows.append({"analysis": "permutation", "term": LABEL["model"], "statistic": obs,
                 "df1": "", "df2": N_PERM, "p_value": p_perm, "partial_eta2": "",
                 "significant": p_perm < 0.05})
    print(f"\nPermutacion dentro de escenario ({N_PERM} iter): suma de H = {obs:.2f} "
          f"(nula: {null_mean:.2f}), p = {p_perm:.4f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nEscrito {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
