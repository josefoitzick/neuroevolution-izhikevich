#!/usr/bin/env python3
"""
Construye los datos crudos del analisis estadistico a partir de las salidas de
los experimentos (un fitnesses.csv por combinacion, con las 11 corridas).

Genera dos CSV tidy en results/metrics/:
  - runs_classification.csv : dataset, encoder, decoder, model, run, fitness
  - runs_rl.csv             : problem, model, run, fitness

Estos dos archivos ya vienen incluidos en el repositorio; este script se conserva
solo como trazabilidad de como se generaron. Para regenerarlos hay que apuntar
SOURCE al arbol de salidas crudas de los experimentos (Resultados_Corridas/ para
clasificacion y Resultados_RL/ para RL), que no se versiona por su tamano.

Uso: editar SOURCE y ejecutar  python consolidar_runs.py
"""

import os
import csv

# Arbol con las salidas crudas de los experimentos (fitnesses.csv por combinacion).
SOURCE = os.environ.get("RUNS_SOURCE", "")

HERE   = os.path.dirname(os.path.abspath(__file__))
OUT    = os.path.normpath(os.path.join(HERE, "..", "results", "metrics"))


def read_runs(path):
    runs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            runs.append((int(row["corrida"]), float(row["fitness"])))
    return runs


def build_classification(source):
    """Resultados_Corridas/{dataset}/{enc}-{dec}/{model}/fitnesses.csv"""
    root = os.path.join(source, "Classification", "Resultados_Corridas")
    rows = []
    for dataset in sorted(os.listdir(root)):
        dpath = os.path.join(root, dataset)
        if not os.path.isdir(dpath):
            continue
        for combo in sorted(os.listdir(dpath)):
            cpath = os.path.join(dpath, combo)
            if not os.path.isdir(cpath):
                continue
            enc, dec = combo.split("-", 1)
            for model in sorted(os.listdir(cpath)):
                fpath = os.path.join(cpath, model, "fitnesses.csv")
                if not os.path.isfile(fpath):
                    continue
                for run, fit in read_runs(fpath):
                    rows.append((dataset, enc, dec, model, run, fit))
    return rows


def build_rl(source):
    """Resultados_RL/{problem}/{model}/fitnesses.csv"""
    root = os.path.join(source, "NEAT", "Resultados_RL")
    rows = []
    for problem in sorted(os.listdir(root)):
        ppath = os.path.join(root, problem)
        if not os.path.isdir(ppath):
            continue
        for model in sorted(os.listdir(ppath)):
            fpath = os.path.join(ppath, model, "fitnesses.csv")
            if not os.path.isfile(fpath):
                continue
            for run, fit in read_runs(fpath):
                rows.append((problem, model, run, fit))
    return rows


def main():
    if not SOURCE or not os.path.isdir(SOURCE):
        raise SystemExit(
            "Define la variable de entorno RUNS_SOURCE (o edita SOURCE) apuntando "
            "al arbol de salidas crudas de los experimentos."
        )

    cl = build_classification(SOURCE)
    with open(os.path.join(OUT, "runs_classification.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "encoder", "decoder", "model", "run", "fitness"])
        w.writerows(cl)

    rl = build_rl(SOURCE)
    with open(os.path.join(OUT, "runs_rl.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "model", "run", "fitness"])
        w.writerows(rl)

    print(f"Clasificacion: {len(cl)} filas ({len(cl)//11} combos)")
    print(f"RL: {len(rl)} filas ({len(rl)//11} combos)")


if __name__ == "__main__":
    main()
