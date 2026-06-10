#!/usr/bin/env python3
"""
Re-optimizacion Optuna de un combo/modelo de clasificacion, usando el binario
CANONICO (./NEAT con 12 params NEAT + trial) y el classification.py canonico
(que carga el archivo de datos correcto segun el encoder de config/neuron.cfg).

Replica EXACTAMENTE la busqueda del optuna_script.py del companero (mismos rangos,
mismo prior del paper, TPE + Hyperband, n_trials por defecto 50). La unica
diferencia respecto a la corrida original del companero es que aca los datos se
cargan correctos (wine_data_ss para encoder single_spike), por lo que aisla el
efecto del fix de datos.

Ejecutar desde un directorio aislado (ej. ~/wine_reopt) que tenga: ./NEAT,
classification.py, data/, y config/{config.cfg,neuron.cfg} YA instalados para el
combo/modelo a optimizar (encoder/decoder/dataset/a-b-c-d en neuron.cfg).

Uso:
  python reopt_optuna.py --name wine-ssfs-IZA --trials 50
  python reopt_optuna.py --name wine-ssfs-IZA --trials 50 --no-prior

NOTA: inputWeights es INERTE en clasificacion (el input se fuerza con v=30, no se
inyecta corriente). Se incluye en la busqueda solo por fidelidad al espacio del
companero; reescribir su valor en config.cfg no cambia el fitness.
"""

import optuna
import sys
import logging
import subprocess
import os
import argparse
import numpy as np

CONFIG = 'config/config.cfg'


def set_input_weights(mn, mx):
    """Reescribe la linea inputWeights en config.cfg preservando el resto.
    (Inerte en clasificacion, pero mantiene el config consistente con el trial.)"""
    lines = open(CONFIG, encoding='utf-8').read().splitlines()
    out, found = [], False
    for ln in lines:
        if ln.startswith('inputWeights='):
            out.append(f'inputWeights={mn},{mx}')
            found = True
        else:
            out.append(ln)
    if not found:
        out.append(f'inputWeights={mn},{mx}')
    open(CONFIG, 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')


def objective(trial):
    os.makedirs(f"results/trial-{trial.number}", exist_ok=True)
    # --- mismos rangos que optuna_script.py del companero ---
    keep                     = round(trial.suggest_float('keep', 0.4, 0.6), 3)
    threshold                = round(trial.suggest_float('threshold', 2.0, 4.0), 3)
    interSpeciesRate         = round(trial.suggest_float('interSpeciesRate', 0.0005, 0.0015), 3)
    noCrossoverOff           = round(trial.suggest_float('noCrossoverOff', 0.15, 0.35), 3)
    probabilityWeightMutated = round(trial.suggest_float('probabilityWeightMutated', 0.7, 0.9), 3)
    probabilityAddNodeSmall  = round(trial.suggest_float('probabilityAddNodeSmall', 0.02, 0.04), 3)
    probabilityAddLink_small = round(trial.suggest_float('probabilityAddLink_small', 0.01, 0.05), 3)
    probabilityAddNodeLarge  = round(trial.suggest_float('probabilityAddNodeLarge', 0.02, 0.4), 3)
    probabilityAddLink_Large = round(trial.suggest_float('probabilityAddLink_Large', 0.05, 0.2), 3)
    c1                       = round(trial.suggest_float('c1', 0.5, 1.5), 3)
    c2                       = round(trial.suggest_float('c2', 0.5, 1.5), 3)
    c3                       = round(trial.suggest_float('c3', 0.3, 0.5), 3)
    inputWeights_min         = round(trial.suggest_float('inputWeights_min', 90.0, 130.0), 3)
    inputWeights_max         = round(trial.suggest_float('inputWeights_max', 130.0, 170.0), 3)

    set_input_weights(inputWeights_min, inputWeights_max)

    # binario canonico: 12 params NEAT + trial (lee inputWeights/etc de config.cfg)
    cmd = ['./NEAT',
           str(keep), str(threshold), str(interSpeciesRate), str(noCrossoverOff),
           str(probabilityWeightMutated), str(probabilityAddNodeSmall),
           str(probabilityAddLink_small), str(probabilityAddNodeLarge),
           str(probabilityAddLink_Large), str(c1), str(c2), str(c3),
           str(trial.number)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    stdout, stderr = p.communicate()
    output = stdout.decode('utf-8', errors='replace')
    try:
        value = float(output.strip().split('\n')[-1])
    except (ValueError, IndexError):
        value = -np.inf
    return value


# prior = hiperparametros del paper (Tabla 9) por combo. Default: Single-First-IZH.
PRIORS = {
    'single_first': {
        'keep': 0.552, 'threshold': 2.976, 'interSpeciesRate': 0.001, 'noCrossoverOff': 0.195,
        'probabilityWeightMutated': 0.873, 'probabilityAddNodeSmall': 0.034,
        'probabilityAddLink_small': 0.028, 'probabilityAddNodeLarge': 0.331,
        'probabilityAddLink_Large': 0.195, 'c1': 0.67, 'c2': 1.364, 'c3': 0.377,
        'inputWeights_min': 110, 'inputWeights_max': 150,
    },
    'single_voting': {
        'keep': 0.412, 'threshold': 3.039, 'interSpeciesRate': 0.001, 'noCrossoverOff': 0.244,
        'probabilityWeightMutated': 0.797, 'probabilityAddNodeSmall': 0.022,
        'probabilityAddLink_small': 0.042, 'probabilityAddNodeLarge': 0.182,
        'probabilityAddLink_Large': 0.151, 'c1': 1.289, 'c2': 1.113, 'c3': 0.364,
        'inputWeights_min': 110, 'inputWeights_max': 150,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help="nombre del estudio (y del .db)")
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--prior', choices=list(PRIORS.keys()), default='single_first')
    ap.add_argument('--no-prior', action='store_true', help="no encolar el prior del paper")
    args = ap.parse_args()

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name=args.name,
        storage=f"sqlite:///{args.name}.db",
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        load_if_exists=True,
    )
    if not args.no_prior:
        study.enqueue_trial(PRIORS[args.prior])

    study.optimize(objective, n_trials=args.trials)

    print("=" * 50)
    print(f"BEST VALUE : {study.best_value}")
    print(f"BEST PARAMS: {study.best_params}")
    print("=" * 50)


if __name__ == '__main__':
    main()
