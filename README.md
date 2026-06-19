# neuroevolution-izhikevich

Reference code and results for a research project (USACH) on **neuroevolution of spiking
neural networks** with diverse **Izhikevich** neuron models, built on a NEAT engine and
[ANNarchy](https://annarchy.github.io/) (`<5`).

A single C++ NEAT core (`genome`, `population`, `species`, `innovation`, …) evolves the
networks. The two experiment tracks share that engine and differ only in how each network's
fitness is evaluated.

## Neuron models

Eight Izhikevich dynamic regimes (Izhikevich, 2003) are used, labelled **IZ-A…H**. Each is
defined by the parameters `a, b, c, d` (set in the per-model `neuron.cfg`):

| Model | a | b | c | d | Dynamics |
|-------|------|-------|-----|------|----------------------------|
| IZ-A | 0.02 | 0.2 | -65 | 6 | Tonic spiking |
| IZ-B | 0.02 | 0.25 | -65 | 6 | Phasic spiking |
| IZ-C | 0.02 | 0.2 | -50 | 2 | Tonic bursting |
| IZ-D | 0.02 | 0.25 | -55 | 0.05 | Phasic bursting |
| IZ-E | 0.02 | 0.2 | -55 | 4 | Mixed mode |
| IZ-F | 0.01 | 0.2 | -65 | 8 | Spike frequency adaptation |
| IZ-G | 0.02 | -0.1 | -55 | 6 | Class 1 excitable |
| IZ-H | 0.2 | 0.26 | -65 | 0 | Class 2 excitable |

## Experiments

### Classification — `src/Classification/`
Supervised classification on tabular datasets.

- **Datasets:** iris, wine, breast cancer (`bc`)
- **Encoders × decoders:** `ss`/`t` × `fs`/`v` (4 combinations)
- **Models:** 8 Izhikevich variants (IZ-A…H) plus a **LIF** baseline
- **Fitness:** classification accuracy
- Orchestrated by `run_classification.py` (11 runs per combination)

### Reinforcement Learning — `src/NEAT/`
Control on Gym/Gymnasium environments.

- **Problems:** acrobot, cartpole, mountain
- **Models:** 8 Izhikevich variants (IZ-A…H)
- **Fitness:** cumulative episode reward
- Orchestrated by `run_rl.py` (11 runs per combination)

## Repository layout

```
config/                 Final parameter configurations (.cfg)
analysis/               Statistical analysis (scripts + dependencies)
results/
  metrics/              Summary CSVs + per-run data (runs_*.csv)
  figures/              Final boxplots (Clasificacion/ + RL/)
  stats/                Statistical test outputs (tables + CD diagram)
src/
  Classification/       Classification track (C++ NEAT core + Python orchestration)
  NEAT/                 RL track (same C++ NEAT core + Python orchestration)
  optimization/         Optuna hyperparameter studies (.db) + summaries
```

## Optuna studies

`src/optimization/` holds the Optuna studies (`.db`) and summaries
(`all_studies.csv`, `mejores_parametros.csv`) for both tracks, including the IZ and LIF
classification studies, kept for comparison against the reported results.

## Statistical analysis

The main comparison is **between the eight Izhikevich neuron types** (LIF excluded).
`analysis/stats.py` runs it on the per-run data and writes its outputs to `results/stats/`.

- **Data:** `results/metrics/runs_classification.csv` and `runs_rl.csv` hold the individual
  fitness values (11 runs per combination). `analysis/consolidar_runs.py` documents how these
  were built from the raw experiment outputs.
- **Classification** (12 scenarios = 3 datasets × 4 encoder–decoder combinations):
  Friedman test + Nemenyi post-hoc, summarised by a critical-difference (CD) diagram.
- **Reinforcement Learning** (per problem): Kruskal–Wallis + Dunn post-hoc (Holm correction).
- Significance level `α = 0.05`.

Run with `python analysis/stats.py` (dependencies in `analysis/requirements.txt`).

### Network complexity

`analysis/complejidad.py` summarises the **topology of the winning networks** (nodes and
connections per model), to discuss whether some neuron types reach comparable performance
with smaller networks.

- **Data:** `results/metrics/runs_complexity_classification.csv` and
  `runs_complexity_rl.csv` hold, per winning genome (one per run), its node count, connection
  count and hidden-node count. `analysis/consolidar_complejidad.py` documents how these were
  built from the raw `best0.txt` genomes and `config.cfg` files.
- **Outputs:** `results/stats/complejidad_resumen.csv` (mean ± sd of nodes and connections per
  model, split into classification and RL) and `results/stats/complejidad_tests.csv` (tests of
  whether network size differs between neuron types: Friedman over the 12 scenarios for
  classification, Kruskal–Wallis per problem for RL, mirroring `stats.py`).
- In classification the node count is dominated by the fixed inputs (they vary per
  dataset/encoder), so the evolved-complexity signal lives mainly in the connections and the
  hidden nodes.

Run with `python analysis/complejidad.py`.

## Notes

- The C++ core compiles with `make` (per-track `Makefile`); the `NEAT` binary and ANNarchy
  build directories are generated artifacts and are not tracked.
- Requires ANNarchy `<5` (the 5.x line breaks the legacy API used here).
