#!/usr/bin/env python3
"""
Orquestador para las 11 corridas de los experimentos RL.

Ejecutar desde la carpeta NEAT/.

Uso:
  python run_rl.py                                  # TODO: 3 problemas x 8 modelos x 11 corridas
  python run_rl.py --problem cartpole               # solo cartpole, los 8 modelos, 11 corridas
  python run_rl.py --problem cartpole --model IZ-A  # solo cartpole + IZ-A, 11 corridas
  python run_rl.py --problem cartpole --model IZ-A --runs 1   # PRUEBA RAPIDA: 1 sola corrida
  python run_rl.py --runs 2                         # todo pero solo 2 corridas por combo

Resultados en:  NEAT/Resultados_RL/{problema}/{modelo}/corrida-{N}/
"""

import argparse
import subprocess
import shutil
import os

# ===================== CONFIGURACION =====================

CONFIGS_BASE = "../Configs"
RESULTS_BASE = "Resultados_RL"
N_RUNS       = 11
NEAT_EXEC    = "./NEAT"

PROBLEMS = ['acrobot', 'cartpole', 'mountain']

# Orden de los argumentos que espera el binario NEAT (claves de config.cfg)
NEAT_ARG_ORDER = [
    'keep', 'threshold', 'interSpeciesRate', 'noCrossoverOff',
    'probabilityWeightMutated',
    'probabilityAddNodeSmall', 'probabilityAddLink_small',
    'probabilityAddNodeLarge', 'probabilityAddLink_Large',
    'c1', 'c2', 'c3',
]

# =========================================================


def parse_cfg(path):
    cfg = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip()
    return cfg


def list_models(problem):
    """Devuelve los nombres de modelos disponibles en Configs/{problema}/."""
    pdir = os.path.join(CONFIGS_BASE, problem)
    if not os.path.isdir(pdir):
        return []
    return sorted(d for d in os.listdir(pdir)
                  if d.startswith('IZ-') and os.path.isdir(os.path.join(pdir, d)))


def install_configs(problem, model_key):
    """Copia Configs/{problema}/{modelo}/{config.cfg, neuron.cfg} a config/."""
    src_dir = os.path.join(CONFIGS_BASE, problem, model_key)
    os.makedirs('config', exist_ok=True)
    shutil.copy(os.path.join(src_dir, 'config.cfg'),  'config/config.cfg')
    shutil.copy(os.path.join(src_dir, 'neuron.cfg'),  'config/neuron.cfg')


def run_neat(cfg, trial_number):
    cmd = [NEAT_EXEC] + [str(cfg[k]) for k in NEAT_ARG_ORDER] + [str(trial_number)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, stderr = p.communicate()
    output  = stdout.decode('utf-8', errors='replace').strip()
    err_txt = stderr.decode('utf-8', errors='replace').strip()
    try:
        fitness = float(output.split('\n')[-1])
    except (ValueError, IndexError):
        fitness = float('-inf')
    if fitness == -1.0 or fitness == float('-inf'):
        print(f"\n    [DIAG] fitness={fitness} - posible error. Salida del binario:")
        print("    --- STDOUT (ultimas 40 lineas) ---")
        for line in output.splitlines()[-40:]:
            print(f"    {line}")
        if err_txt:
            print("    --- STDERR (ultimas 20 lineas) ---")
            for line in err_txt.splitlines()[-20:]:
                print(f"    {line}")
    elif err_txt:
        print(f"    STDERR: {err_txt[:300]}")
    return fitness


def save_results(trial_number, problem, model_name, run):
    src  = f"results/trial-{trial_number}"
    dest = f"{RESULTS_BASE}/{problem}/{model_name}/corrida-{run}"
    os.makedirs(os.path.dirname(dest + '/x'), exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    if os.path.exists(src):
        shutil.copytree(src, dest)
    else:
        print(f"    ADVERTENCIA: {src} no encontrado, no se guardaron resultados")


def save_fitnesses_csv(problem, model_name, fitnesses):
    """Guarda los fitnesses finales del combo a CSV (insumo para graficar_boxplots.py)."""
    path = f"{RESULTS_BASE}/{problem}/{model_name}/fitnesses.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("corrida,fitness\n")
        for i, fit in enumerate(fitnesses):
            f.write(f"{i},{fit}\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Orquestador RL — admite filtros para pruebas rapidas.")
    ap.add_argument('--problem', choices=PROBLEMS, help="Correr solo este problema.")
    ap.add_argument('--model',   help="Correr solo este modelo (ej. IZ-A).")
    ap.add_argument('--runs', type=int, default=N_RUNS,
                    help=f"Corridas por combinacion (default {N_RUNS}).")
    ap.add_argument('--process-max', type=int, default=None,
                    help="Sobrescribe process_max del config (util en WSL: usar 2-4).")
    ap.add_argument('--evolutions', type=int, default=None,
                    help="Sobrescribe evolutions del config (util para prueba: usar 2-3).")
    ap.add_argument('--genomes', type=int, default=None,
                    help="Sobrescribe numberGenomes del config (util para prueba: usar 4-10).")
    return ap.parse_args()


def override_config(overrides):
    """Sobrescribe campos en config/config.cfg con los pasados por CLI."""
    if not overrides:
        return
    path = 'config/config.cfg'
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    keys = set(overrides.keys())
    new_lines = []
    seen = set()
    for line in lines:
        k = line.split('=', 1)[0].strip() if '=' in line else None
        if k in keys:
            new_lines.append(f"{k}={overrides[k]}\n")
            seen.add(k)
        else:
            new_lines.append(line)
    for k in keys - seen:
        new_lines.append(f"{k}={overrides[k]}\n")
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(new_lines)


def main():
    args = parse_args()

    if not os.path.isdir(CONFIGS_BASE):
        print(f"ERROR: No existe {CONFIGS_BASE}/. Ejecuta primero: python ../generar_configs.py")
        return

    problems = [args.problem] if args.problem else PROBLEMS
    n_runs = args.runs

    print(f"Configuracion: problemas={problems}  modelo={args.model or 'todos'}  corridas={n_runs}")

    trial_counter = 0

    for problem in problems:
        models = list_models(problem)
        if args.model:
            if args.model not in models:
                print(f"[SKIP] {args.model} no existe en {CONFIGS_BASE}/{problem}/")
                continue
            models = [args.model]
        if not models:
            print(f"[SKIP] No hay modelos en {CONFIGS_BASE}/{problem}/")
            continue

        print(f"\n{'='*60}")
        print(f"  Problema: {problem.upper()} ({len(models)} modelos)")
        print(f"{'='*60}")

        overrides = {}
        if args.process_max is not None: overrides['process_max'] = args.process_max
        if args.evolutions  is not None: overrides['evolutions']  = args.evolutions
        if args.genomes     is not None: overrides['numberGenomes'] = args.genomes

        for model_key in models:
            csv_path = f"{RESULTS_BASE}/{problem}/{model_key}/fitnesses.csv"
            if os.path.exists(csv_path):
                print(f"\n  Modelo {model_key} [SKIP — {csv_path} ya existe]")
                continue

            install_configs(problem, model_key)
            override_config(overrides)
            cfg = parse_cfg('config/config.cfg')
            print(f"\n  Modelo {model_key}")
            if overrides:
                print(f"    overrides: {overrides}")

            fitnesses = []
            for run in range(n_runs):
                trial_counter += 1
                print(f"    Corrida {run+1}/{n_runs} (trial={trial_counter})...", end=' ', flush=True)
                fitness = run_neat(cfg, trial_counter)
                save_results(trial_counter, problem, model_key, run)
                fitnesses.append(fitness)
                print(f"fitness = {fitness:.4f}")

            mean_f = sum(fitnesses) / len(fitnesses)
            std_f  = (sum((x - mean_f)**2 for x in fitnesses) / len(fitnesses)) ** 0.5
            print(f"    -> Media: {mean_f:.4f}  Std: {std_f:.4f}")
            save_fitnesses_csv(problem, model_key, fitnesses)

    print(f"\n{'='*60}")
    print("  Experimentos RL completados")
    print(f"  Resultados en: NEAT/{RESULTS_BASE}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
