#!/usr/bin/env python3
"""
Analisis de complejidad topologica de las redes ganadoras por tipo de neurona
Izhikevich (IZ-A..H). Complementa al analisis de performance (stats.py): permite
discutir si algun tipo de neurona alcanza performance similar con redes mas chicas.

Lee los datos de complejidad (un genoma campeon por corrida) desde results/metrics/
y produce:
  - una tabla resumen con la media +/- desviacion estandar de nodos y conexiones
    por modelo, separada en clasificacion y RL (complejidad_resumen.csv);
  - tests de si el tamano de red difiere entre tipos de neurona, con la misma
    maquinaria que el analisis de performance (stats.py): Friedman sobre los 12
    escenarios para clasificacion y Kruskal-Wallis por problema para RL
    (complejidad_tests.csv).

Nota sobre los nodos en clasificacion: el conteo de nodos esta dominado por las
entradas fijas (varian por dataset/encoder), de modo que la senal de complejidad
evolucionada esta sobre todo en las conexiones y en los nodos ocultos (hidden).
El bloque de clasificacion comparte el caracter exploratorio del de performance
(los 12 escenarios estan anidados por dataset; ver stats.py).

Salidas en results/stats/. Uso: python complejidad.py
"""

import os
import pandas as pd
import scipy.stats as ss

ALPHA = 0.05

HERE    = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.normpath(os.path.join(HERE, "..", "results", "metrics"))
OUTDIR  = os.path.normpath(os.path.join(HERE, "..", "results", "stats"))
os.makedirs(OUTDIR, exist_ok=True)

METRICAS = ["nodes", "connections", "hidden"]


def resumir(df, task):
    """Media y desviacion estandar por modelo para una familia de tareas."""
    g = df.groupby("model")
    out = pd.DataFrame({"task": task, "n": g.size()})
    for col in METRICAS:
        out[f"{col}_mean"] = g[col].mean()
        out[f"{col}_sd"]   = g[col].std()
    return out.reset_index().sort_values("model")


def imprimir(tabla, titulo):
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)
    print(f"{'IZ':5} {'nodos (m+/-sd)':>18} {'conexiones (m+/-sd)':>22} "
          f"{'ocultos':>12}   n")
    for _, r in tabla.iterrows():
        print(f"{r['model']:5} "
              f"{r['nodes_mean']:7.1f} +/- {r['nodes_sd']:5.1f}   "
              f"{r['connections_mean']:8.1f} +/- {r['connections_sd']:6.1f}   "
              f"{r['hidden_mean']:5.2f} +/- {r['hidden_sd']:4.2f}   "
              f"{int(r['n'])}")


def test_clasificacion(clf, models):
    """Friedman por metrica sobre los 12 escenarios (mediana de las 11 corridas),
    mismo input estilo Demsar que el test de accuracy en stats.py."""
    clf = clf.copy()
    clf["scenario"] = clf["dataset"] + "-" + clf["encoder"] + "-" + clf["decoder"]
    rows = []
    for col in ["connections", "nodes"]:
        matrix = (clf.groupby(["scenario", "model"])[col]
                     .median().unstack("model")[models])
        chi2, p = ss.friedmanchisquare(*[matrix[m].values for m in models])
        rows.append({"task": "classification", "metric": col, "test": "friedman",
                     "stat": chi2, "p_value": p, "n_scenarios": len(matrix),
                     "significant": p < ALPHA})
    return rows


def test_rl(rl, models):
    """Kruskal-Wallis por problema y metrica (8 modelos, 11 corridas c/u),
    con epsilon-cuadrado como tamano de efecto, igual que stats.py."""
    rows = []
    for problem in sorted(rl["problem"].unique()):
        sub = rl[rl["problem"] == problem]
        for col in ["nodes", "connections"]:
            grupos = [sub[sub["model"] == m][col].values for m in models]
            H, p = ss.kruskal(*grupos)
            n = sum(len(g) for g in grupos)
            rows.append({"task": "rl", "metric": col, "test": "kruskal",
                         "problem": problem, "stat": H, "p_value": p,
                         "epsilon2": H / (n - 1), "significant": p < ALPHA})
    return rows


def imprimir_tests(tests):
    print("\n" + "=" * 70)
    print("  TESTS DE TAMANO DE RED (alpha = 0.05)")
    print("=" * 70)
    for t in tests:
        loc = t.get("problem", t["task"])
        extra = f" eps2={t['epsilon2']:.2f}" if "epsilon2" in t else ""
        sig = "SIG" if t["significant"] else "ns"
        print(f"  [{loc:13} {t['metric']:11}] {t['test']:8} "
              f"stat={t['stat']:6.2f} p={t['p_value']:.4f}{extra}  {sig}")


def main():
    clf = pd.read_csv(os.path.join(METRICS, "runs_complexity_classification.csv"))
    rl  = pd.read_csv(os.path.join(METRICS, "runs_complexity_rl.csv"))
    models = sorted(clf["model"].unique())

    t_clf = resumir(clf, "classification")
    t_rl  = resumir(rl,  "rl")

    imprimir(t_clf, "CLASIFICACION  -  complejidad de red por tipo de neurona")
    imprimir(t_rl,  "RL  -  complejidad de red por tipo de neurona")

    tabla = pd.concat([t_clf, t_rl], ignore_index=True)
    out = os.path.join(OUTDIR, "complejidad_resumen.csv")
    tabla.to_csv(out, index=False)
    print(f"\nResumen -> {os.path.relpath(out, HERE)}")

    tests = test_clasificacion(clf, models) + test_rl(rl, models)
    imprimir_tests(tests)
    out_t = os.path.join(OUTDIR, "complejidad_tests.csv")
    pd.DataFrame(tests).to_csv(out_t, index=False)
    print(f"\nTests -> {os.path.relpath(out_t, HERE)}")


if __name__ == "__main__":
    main()
