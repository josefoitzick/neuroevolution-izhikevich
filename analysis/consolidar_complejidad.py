#!/usr/bin/env python3
"""
Construye los datos de complejidad topologica de las redes ganadoras a partir de
las salidas crudas de los experimentos. Cada corrida guarda el genoma campeon en
best0.txt como lista de aristas "origen;destino;peso" (una por linea), y el numero
de entradas/salidas en config.cfg (numberInputs / numberOutputs).

Para cada genoma se calcula:
  - nodes        : nodos distintos que aparecen en las aristas
  - connections  : numero de aristas (conexiones activas)
  - hidden       : nodos ocultos = ids > numberInputs + numberOutputs
                   (entradas 1..NI, salidas NI+1..NI+NO, ocultos por encima)

Genera dos CSV tidy en results/metrics/:
  - runs_complexity_classification.csv : dataset, encoder, decoder, model, run,
                                         nodes, connections, hidden
  - runs_complexity_rl.csv             : problem, model, run, nodes, connections, hidden

Estos dos archivos ya vienen incluidos en el repositorio; este script se conserva
solo como trazabilidad. Para regenerarlos hay que apuntar RUNS_SOURCE al arbol de
salidas crudas (Classification/Resultados_Corridas/ y NEAT/Resultados_RL/), que no
se versiona por su tamano.

Uso: RUNS_SOURCE=/ruta/al/arbol  python consolidar_complejidad.py
"""

import os
import re
import csv

# Arbol con las salidas crudas de los experimentos.
SOURCE = os.environ.get("RUNS_SOURCE", "")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.normpath(os.path.join(HERE, "..", "results", "metrics"))

_RUN_RE = re.compile(r"corrida-(\d+)$")


def genome_complexity(best_path):
    """nodos distintos y numero de conexiones a partir de un best0.txt."""
    nodes = set()
    connections = 0
    with open(best_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                src, dst = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            nodes.add(src)
            nodes.add(dst)
            connections += 1
    return nodes, connections


def read_io(cfg_path):
    """numberInputs / numberOutputs desde un config.cfg."""
    ni = no = None
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            for line in f:
                m = re.match(r"numberInputs=(\d+)", line)
                if m:
                    ni = int(m.group(1))
                m = re.match(r"numberOutputs=(\d+)", line)
                if m:
                    no = int(m.group(1))
    return ni, no


def measure_run(run_dir):
    """(run, nodes, connections, hidden) para un directorio corrida-N, o None."""
    m = _RUN_RE.search(os.path.basename(run_dir))
    best = os.path.join(run_dir, "best0.txt")
    if m is None or not os.path.isfile(best):
        return None
    run = int(m.group(1))
    nodes, connections = genome_complexity(best)
    ni, no = read_io(os.path.join(run_dir, "config.cfg"))
    hidden = sum(1 for n in nodes if ni is not None and no is not None and n > ni + no)
    return run, len(nodes), connections, hidden


def build_classification(source):
    """Resultados_Corridas/{dataset}/{enc}-{dec}/{model}/corrida-N/"""
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
                mpath = os.path.join(cpath, model)
                if not os.path.isdir(mpath):
                    continue
                for run_dir in sorted(os.listdir(mpath)):
                    res = measure_run(os.path.join(mpath, run_dir))
                    if res is None:
                        continue
                    run, nodes, conns, hidden = res
                    rows.append((dataset, enc, dec, model, run, nodes, conns, hidden))
    return rows


def build_rl(source):
    """Resultados_RL/{problem}/{model}/corrida-N/"""
    root = os.path.join(source, "NEAT", "Resultados_RL")
    rows = []
    for problem in sorted(os.listdir(root)):
        ppath = os.path.join(root, problem)
        if not os.path.isdir(ppath):
            continue
        for model in sorted(os.listdir(ppath)):
            mpath = os.path.join(ppath, model)
            if not os.path.isdir(mpath):
                continue
            for run_dir in sorted(os.listdir(mpath)):
                res = measure_run(os.path.join(mpath, run_dir))
                if res is None:
                    continue
                run, nodes, conns, hidden = res
                rows.append((problem, model, run, nodes, conns, hidden))
    return rows


def main():
    if not SOURCE or not os.path.isdir(SOURCE):
        raise SystemExit(
            "Define la variable de entorno RUNS_SOURCE (o edita SOURCE) apuntando "
            "al arbol de salidas crudas de los experimentos."
        )

    cl = build_classification(SOURCE)
    with open(os.path.join(OUT, "runs_complexity_classification.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "encoder", "decoder", "model", "run",
                    "nodes", "connections", "hidden"])
        w.writerows(cl)

    rl = build_rl(SOURCE)
    with open(os.path.join(OUT, "runs_complexity_rl.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "model", "run", "nodes", "connections", "hidden"])
        w.writerows(rl)

    print(f"Clasificacion: {len(cl)} genomas ({len(cl)//11} combos)")
    print(f"RL: {len(rl)} genomas ({len(rl)//11} combos)")


if __name__ == "__main__":
    main()
