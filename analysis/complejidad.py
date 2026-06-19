#!/usr/bin/env python3
"""
Analisis de complejidad topologica de las redes ganadoras por tipo de neurona
Izhikevich (IZ-A..H). Complementa al analisis de performance (stats.py): permite
discutir si algun tipo de neurona alcanza performance similar con redes mas chicas.

Lee los datos de complejidad (un genoma campeon por corrida) desde results/metrics/
y produce una tabla resumen con la media +/- desviacion estandar de nodos y
conexiones por modelo, separada en clasificacion y RL.

Nota sobre los nodos en clasificacion: el conteo de nodos esta dominado por las
entradas fijas (varian por dataset/encoder), de modo que la senal de complejidad
evolucionada esta sobre todo en las conexiones y en los nodos ocultos (hidden).

Salida en results/stats/complejidad_resumen.csv. Uso: python complejidad.py
"""

import os
import pandas as pd

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


def main():
    clf = pd.read_csv(os.path.join(METRICS, "runs_complexity_classification.csv"))
    rl  = pd.read_csv(os.path.join(METRICS, "runs_complexity_rl.csv"))

    t_clf = resumir(clf, "classification")
    t_rl  = resumir(rl,  "rl")

    imprimir(t_clf, "CLASIFICACION  -  complejidad de red por tipo de neurona")
    imprimir(t_rl,  "RL  -  complejidad de red por tipo de neurona")

    tabla = pd.concat([t_clf, t_rl], ignore_index=True)
    out = os.path.join(OUTDIR, "complejidad_resumen.csv")
    tabla.to_csv(out, index=False)
    print(f"\nResumen -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
