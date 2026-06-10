#!/usr/bin/env python3
"""
Genera los 120 archivos config.cfg + neuron.cfg combinando:
  - ConfigsDefaults/{problema}/.../config.cfg  (parametros estaticos del profesor)
  - ResultadosOpti/optimization/.../mejores_parametros.csv  (hiperparametros optimizados)
  - Tabla de modelos Izhikevich (a, b, c, d)

Salida:
  Configs/{problema}/{modelo}/config.cfg + neuron.cfg                          (RL: 3 x 8 = 24)
  Configs/{dataset}/{dataset}-iz-{enc}-{dec}/{modelo}/config.cfg + neuron.cfg  (Clasif: 3 x 4 x 8 = 96)

Ejecutar desde la raiz del proyecto:
  python generar_configs.py
"""

import csv
import os
import shutil

# ===================== RUTAS =====================

DEFAULTS_DIR     = "ConfigsDefaults"
# Busca primero dentro del proyecto, luego en Downloads como fallback.
OPTUNA_DIR_CANDIDATES = [
    "optimization",
    "ResultadosOpti/optimization",
    os.path.expanduser("~/Downloads/ResultadosOpti/optimization"),
]
OUTPUT_DIR       = "Configs"


def find_optuna_dir():
    for path in OPTUNA_DIR_CANDIDATES:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "No se encontro la carpeta de resultados de Optuna. Buscado en:\n  "
        + "\n  ".join(OPTUNA_DIR_CANDIDATES)
    )


OPTUNA_DIR = None  # se asigna en main()

# ===================== MODELOS IZHIKEVICH =====================

MODELS = {
    'IZ-A': (0.02,  0.2,  -65.0, 6.0),    # tonic spiking
    'IZ-B': (0.02,  0.25, -65.0, 6.0),    # phasic spiking
    'IZ-C': (0.02,  0.2,  -50.0, 2.0),    # tonic bursting
    'IZ-D': (0.02,  0.25, -55.0, 0.05),   # phasic bursting
    'IZ-E': (0.02,  0.2,  -55.0, 4.0),    # mixed mode
    'IZ-F': (0.01,  0.2,  -65.0, 8.0),    # spike frequency adaptation
    'IZ-G': (0.02, -0.1,  -55.0, 6.0),    # Class 1
    'IZ-H': (0.2,   0.26, -65.0, 0.0),    # Class 2
}

# ===================== MAPEOS RL =====================
# (carpeta_en_ConfigsDefaults, carpeta_en_Optuna)
RL_PROBLEMS = [
    ('acrobot',  'acrobot'),
    ('cartpole', 'cartpole'),
    ('mountain', 'mountaincar'),
]

# ===================== MAPEOS CLASIFICACION =====================
DATASETS = ['iris', 'wine', 'bc']
ENC_DEC_COMBOS = [
    ('ss', 'fs', 'single_spike', 'first_spike'),
    ('ss', 'v',  'single_spike', 'vote'),
    ('t',  'fs', 'temporal',     'first_spike'),
    ('t',  'v',  'temporal',     'vote'),
]

# ===================== HIPERPARAMETROS DE OPTUNA =====================
# Estos son los campos del CSV que sobreescriben los del config.cfg base.
OPTUNA_FIELDS = [
    'keep', 'threshold', 'interSpeciesRate', 'noCrossoverOff',
    'probabilityWeightMutated',
    'probabilityAddNodeSmall', 'probabilityAddLink_small',
    'probabilityAddNodeLarge', 'probabilityAddLink_Large',
    'c1', 'c2', 'c3',
]

# ===================================================================


def parse_cfg(path):
    """Lee un archivo key=value y devuelve un dict (preserva orden de lineas)."""
    cfg = {}
    order = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            if k not in cfg:
                order.append(k)
            cfg[k] = v
    return cfg, order


def write_cfg(path, cfg, order):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for k in order:
            f.write(f"{k}={cfg[k]}\n")
        # campos nuevos que no estaban en el original
        for k in cfg:
            if k not in order:
                f.write(f"{k}={cfg[k]}\n")


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def normalize_model(name):
    return name.strip().replace('_', '-').upper()


def merge_config(base_cfg, optuna_row):
    """Sobreescribe campos de Optuna sobre la config base, y arma inputWeights."""
    merged = dict(base_cfg)
    for field in OPTUNA_FIELDS:
        if field in optuna_row and optuna_row[field] != '':
            merged[field] = optuna_row[field]
    # inputWeights viene como dos columnas en el CSV
    iw_min = optuna_row.get('inputWeights_min', '').strip()
    iw_max = optuna_row.get('inputWeights_max', '').strip()
    if iw_min and iw_max:
        merged['inputWeights'] = f"{iw_min},{iw_max}"
    return merged


def write_neuron_cfg(path, model_key, encoder=None, decoder=None, dataset=None):
    a, b, c, d = MODELS[model_key]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"a={a}\n")
        f.write(f"b={b}\n")
        f.write(f"c={c}\n")
        f.write(f"d={d}\n")
        if encoder is not None:
            f.write(f"encoder={encoder}\n")
            f.write(f"decoder={decoder}\n")
            f.write(f"dataset={dataset}\n")


def generar_rl():
    print(f"\n{'='*60}\n  Generando configs RL\n{'='*60}")
    total = 0
    for default_name, optuna_name in RL_PROBLEMS:
        defaults_path = os.path.join(DEFAULTS_DIR, default_name, 'config.cfg')
        csv_path      = os.path.join(OPTUNA_DIR, optuna_name, 'mejores_parametros.csv')

        if not os.path.exists(defaults_path):
            print(f"  [SKIP] No existe: {defaults_path}")
            continue
        if not os.path.exists(csv_path):
            print(f"  [SKIP] No existe: {csv_path}")
            continue

        base_cfg, base_order = parse_cfg(defaults_path)
        rows = read_csv(csv_path)
        print(f"\n  {default_name}: {len(rows)} modelos en CSV")

        for row in rows:
            model_key = normalize_model(row['study_name'])
            if model_key not in MODELS:
                print(f"    [SKIP] Modelo desconocido: {model_key}")
                continue

            merged = merge_config(base_cfg, row)
            out_cfg    = os.path.join(OUTPUT_DIR, default_name, model_key, 'config.cfg')
            out_neuron = os.path.join(OUTPUT_DIR, default_name, model_key, 'neuron.cfg')

            write_cfg(out_cfg, merged, base_order)
            write_neuron_cfg(out_neuron, model_key)
            print(f"    [OK] {out_cfg}")
            total += 1
    print(f"\n  Total RL: {total} archivos config.cfg")
    return total


def generar_clasificacion():
    print(f"\n{'='*60}\n  Generando configs Clasificacion\n{'='*60}")
    total = 0
    for dataset in DATASETS:
        for enc_short, dec_short, encoder_full, decoder_full in ENC_DEC_COMBOS:
            combo = f"{dataset}-iz-{enc_short}-{dec_short}"
            defaults_path = os.path.join(DEFAULTS_DIR, dataset, combo, 'config.cfg')
            csv_path      = os.path.join(OPTUNA_DIR, dataset, combo, 'mejores_parametros.csv')

            if not os.path.exists(defaults_path):
                print(f"  [SKIP] No existe: {defaults_path}")
                continue
            if not os.path.exists(csv_path):
                print(f"  [SKIP] No existe: {csv_path}")
                continue

            base_cfg, base_order = parse_cfg(defaults_path)
            rows = read_csv(csv_path)
            print(f"\n  {combo}: {len(rows)} modelos en CSV")

            for row in rows:
                model_key = normalize_model(row['study_name'])
                if model_key not in MODELS:
                    print(f"    [SKIP] Modelo desconocido: {model_key}")
                    continue

                merged = merge_config(base_cfg, row)
                out_cfg    = os.path.join(OUTPUT_DIR, dataset, combo, model_key, 'config.cfg')
                out_neuron = os.path.join(OUTPUT_DIR, dataset, combo, model_key, 'neuron.cfg')

                write_cfg(out_cfg, merged, base_order)
                write_neuron_cfg(out_neuron, model_key, encoder_full, decoder_full, dataset)
                print(f"    [OK] {out_cfg}")
                total += 1
    print(f"\n  Total Clasificacion: {total} archivos config.cfg")
    return total


def main():
    global OPTUNA_DIR
    OPTUNA_DIR = find_optuna_dir()
    print(f"Usando Optuna: {OPTUNA_DIR}")

    if os.path.exists(OUTPUT_DIR):
        resp = input(f"La carpeta '{OUTPUT_DIR}/' ya existe. Sobrescribir? (s/N): ").strip().lower()
        if resp != 's':
            print("Cancelado.")
            return
        shutil.rmtree(OUTPUT_DIR)

    n_rl = generar_rl()
    n_cl = generar_clasificacion()

    print(f"\n{'='*60}")
    print(f"  Resumen: {n_rl + n_cl} pares (config.cfg + neuron.cfg)")
    print(f"           Esperado: 24 RL + 96 Clasif = 120")
    print(f"  Salida en: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
