#!/bin/bash
#==============================================================================
# Job SLURM para ejecutar las 11 corridas de los 3 datasets de clasificacion
# x 4 combos encoder/decoder x 8 modelos Izhikevich = 96 combinaciones x 11
# corridas = 1056 ejecuciones del binario.
#
# Estrategia: ARRAY JOB con 96 tasks paralelos. Cada task hace las 11 corridas
# de UNA combinacion (dataset x enc-dec x modelo).
#
# Mapeo task_id -> (dataset, enc, dec, modelo):
#   datasets = [iris, wine, bc]              (3)
#   combos   = [(ss,fs), (ss,v), (t,fs), (t,v)]  (4)
#   modelos  = [IZ-A..IZ-H]                  (8)
#   Total: 3 * 4 * 8 = 96
#
# Orden lexicografico: dataset > combo > modelo
#   task  1- 8: iris,  ss-fs, IZ-A..IZ-H
#   task  9-16: iris,  ss-v,  IZ-A..IZ-H
#   task 17-24: iris,  t-fs,  IZ-A..IZ-H
#   task 25-32: iris,  t-v,   IZ-A..IZ-H
#   task 33-40: wine,  ss-fs, ...
#   ...
#   task 89-96: bc,    t-v,   IZ-A..IZ-H
#
# Lanzar desde Classification/ con:    sbatch run_corridas.slurm
#==============================================================================

#SBATCH --job-name=neat-cls
#SBATCH --partition=batch
#SBATCH --array=1-96
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=25                # debe ser >= process_max del config
#SBATCH --time=48:00:00                   # BC es pesado, dar margen
#SBATCH --output=outputs/cls-%A_%a.out
#SBATCH --error=outputs/cls-%A_%a.err

# -----------------------------------------------------------------------------
# AJUSTAR ESTAS RUTAS PARA TU CLUSTER
# -----------------------------------------------------------------------------
PROJECT_DIR="$HOME/ANNarchy-NEAT"
VENV_DIR="$PROJECT_DIR/.venv"

# -----------------------------------------------------------------------------
# (OPCIONAL) Cargar modulos si tu cluster los usa. Comentar si no aplica.
# -----------------------------------------------------------------------------
# module purge
# module load python/3.12
# module load gcc/13

# -----------------------------------------------------------------------------
# Activar entorno virtual
# -----------------------------------------------------------------------------
source "$VENV_DIR/bin/activate"

# -----------------------------------------------------------------------------
# Mapear SLURM_ARRAY_TASK_ID (1..96) a (dataset, enc, dec, modelo)
# -----------------------------------------------------------------------------
DATASETS=(iris wine bc)
ENC_LIST=(ss ss t t)
DEC_LIST=(fs v fs v)
MODELS=(IZ-A IZ-B IZ-C IZ-D IZ-E IZ-F IZ-G IZ-H)

idx=$((SLURM_ARRAY_TASK_ID - 1))           # 0..95
dataset_idx=$((idx / 32))                  # 0..2  (cada dataset = 32 tasks)
within_dataset=$((idx % 32))
combo_idx=$((within_dataset / 8))          # 0..3  (cada combo = 8 modelos)
model_idx=$((within_dataset % 8))          # 0..7

DATASET=${DATASETS[$dataset_idx]}
ENC=${ENC_LIST[$combo_idx]}
DEC=${DEC_LIST[$combo_idx]}
MODEL=${MODELS[$model_idx]}

echo "==================================================="
echo "Task $SLURM_ARRAY_TASK_ID / 96"
echo "Dataset:  $DATASET"
echo "Encoder:  $ENC"
echo "Decoder:  $DEC"
echo "Modelo:   $MODEL"
echo "Nodo:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Inicio:   $(date)"
echo "==================================================="

# -----------------------------------------------------------------------------
# Ir a Classification/ y ejecutar las 11 corridas de esta combinacion
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR/Classification"
mkdir -p outputs

python3 run_classification.py \
    --dataset "$DATASET" \
    --enc "$ENC" \
    --dec "$DEC" \
    --model "$MODEL" \
    --runs 11

echo "==================================================="
echo "Fin:      $(date)"
echo "==================================================="

# -----------------------------------------------------------------------------
# Limpieza opcional
# -----------------------------------------------------------------------------
# rm -rf annarchy
# rm -rf __pycache__
