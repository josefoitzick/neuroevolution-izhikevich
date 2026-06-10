#!/usr/bin/env python3
"""
Genera boxplots y CSV resumen a partir de los fitnesses.csv producidos por
run_rl.py y run_classification.py.

Estructura esperada de entrada:
  NEAT/Resultados_RL/{problema}/{modelo}/fitnesses.csv
  Classification/Resultados_Corridas/{dataset}/{enc}-{dec}/{modelo}/fitnesses.csv

Cada fitnesses.csv tiene formato:
  corrida,fitness
  0,387.5
  1,402.1
  ...

Salida:
  Graficos/RL/{problema}_boxplot.png             (3 figuras, 1 por problema RL)
  Graficos/Clasificacion/{dataset}_{enc}-{dec}_boxplot.png   (12 figuras)
  Graficos/resumen_rl.csv
  Graficos/resumen_clasificacion.csv

Cada boxplot agrupa los 8 modelos Izhikevich (IZ-A..IZ-H) en el eje X y muestra
la distribucion de las 11 corridas como caja.

Ejecutar desde la raiz del proyecto:
  python graficar_boxplots.py
"""

import csv
import os
import statistics

import matplotlib
matplotlib.use('Agg')  # backend sin display (necesario en cluster/WSL sin X)
import matplotlib.pyplot as plt


# ===================== RUTAS =====================

RL_BASE        = "NEAT/Resultados_RL"
CLASIF_BASE    = "Classification/Resultados_Corridas"
OUTPUT_DIR     = "Graficos"

MODELOS = ['IZ-A', 'IZ-B', 'IZ-C', 'IZ-D', 'IZ-E', 'IZ-F', 'IZ-G', 'IZ-H']
RL_PROBLEMAS = ['acrobot', 'cartpole', 'mountain']
DATASETS = ['iris', 'wine', 'bc']
ENC_DEC_COMBOS = [('ss', 'fs'), ('ss', 'v'), ('t', 'fs'), ('t', 'v')]


def leer_fitnesses(csv_path):
    """Lee un fitnesses.csv y devuelve la lista de fitnesses (float)."""
    if not os.path.exists(csv_path):
        return None
    vals = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vals.append(float(row['fitness']))
            except (ValueError, KeyError):
                continue
    return vals if vals else None


def resumen_stats(vals):
    """Devuelve dict con estadisticas de una lista de fitnesses."""
    n = len(vals)
    return {
        'n': n,
        'min': min(vals),
        'max': max(vals),
        'mean': sum(vals) / n,
        'std': statistics.stdev(vals) if n > 1 else 0.0,
        'median': statistics.median(vals),
    }


def boxplot_combo(datos_por_modelo, titulo, ylabel, out_path):
    """
    datos_por_modelo: dict {modelo_key: [fitnesses]}
    Genera un boxplot con una caja por modelo.
    """
    labels = []
    data   = []
    for m in MODELOS:
        if m in datos_por_modelo:
            labels.append(m)
            data.append(datos_por_modelo[m])

    if not data:
        print(f"  [SKIP] Sin datos para {titulo}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#9bc3f0')
        patch.set_alpha(0.6)

    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel('Modelo Izhikevich')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [OK] {out_path}")


def procesar_rl():
    print(f"\n{'='*60}\n  RL\n{'='*60}")
    resumen_filas = []
    for problema in RL_PROBLEMAS:
        datos = {}
        for modelo in MODELOS:
            csv_path = os.path.join(RL_BASE, problema, modelo, 'fitnesses.csv')
            vals = leer_fitnesses(csv_path)
            if vals is None:
                print(f"  [SKIP] No existe o vacio: {csv_path}")
                continue
            datos[modelo] = vals
            s = resumen_stats(vals)
            resumen_filas.append({
                'problema': problema, 'modelo': modelo,
                **s,
            })

        if datos:
            out_png = os.path.join(OUTPUT_DIR, 'RL', f'{problema}_boxplot.png')
            boxplot_combo(
                datos,
                titulo=f"Fitness final - {problema.upper()} (11 corridas por modelo)",
                ylabel='Fitness',
                out_path=out_png,
            )

    # CSV resumen
    if resumen_filas:
        out_csv = os.path.join(OUTPUT_DIR, 'resumen_rl.csv')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['problema', 'modelo', 'n', 'min',
                                              'max', 'mean', 'std', 'median'])
            w.writeheader()
            w.writerows(resumen_filas)
        print(f"\n  [OK] Resumen RL: {out_csv}")


def procesar_clasificacion():
    print(f"\n{'='*60}\n  CLASIFICACION\n{'='*60}")
    resumen_filas = []
    for dataset in DATASETS:
        for enc, dec in ENC_DEC_COMBOS:
            combo_dir = os.path.join(CLASIF_BASE, dataset, f"{enc}-{dec}")
            datos = {}
            for modelo in MODELOS:
                csv_path = os.path.join(combo_dir, modelo, 'fitnesses.csv')
                vals = leer_fitnesses(csv_path)
                if vals is None:
                    continue
                datos[modelo] = vals
                s = resumen_stats(vals)
                resumen_filas.append({
                    'dataset': dataset, 'encoder': enc, 'decoder': dec,
                    'modelo': modelo, **s,
                })

            if datos:
                out_png = os.path.join(
                    OUTPUT_DIR, 'Clasificacion',
                    f'{dataset}_{enc}-{dec}_boxplot.png'
                )
                boxplot_combo(
                    datos,
                    titulo=(f"Fitness final - {dataset.upper()} "
                            f"({enc}-{dec}) (11 corridas por modelo)"),
                    ylabel='Fitness (accuracy)',
                    out_path=out_png,
                )
            else:
                print(f"  [SKIP] Sin datos para {dataset}/{enc}-{dec}")

    if resumen_filas:
        out_csv = os.path.join(OUTPUT_DIR, 'resumen_clasificacion.csv')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['dataset', 'encoder', 'decoder',
                                              'modelo', 'n', 'min', 'max',
                                              'mean', 'std', 'median'])
            w.writeheader()
            w.writerows(resumen_filas)
        print(f"\n  [OK] Resumen Clasificacion: {out_csv}")


def main():
    if not os.path.exists(RL_BASE) and not os.path.exists(CLASIF_BASE):
        print(f"ERROR: No existen {RL_BASE}/ ni {CLASIF_BASE}/")
        print("Corre primero run_rl.py o run_classification.py")
        return

    if os.path.exists(RL_BASE):
        procesar_rl()
    if os.path.exists(CLASIF_BASE):
        procesar_clasificacion()

    print(f"\n{'='*60}")
    print(f"  Listo. Graficos y resumenes en: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
