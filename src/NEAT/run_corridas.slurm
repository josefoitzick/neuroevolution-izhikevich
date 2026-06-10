#!/bin/bash
#==============================================================================
# Job SLURM para ejecutar las 11 corridas de los 3 problemas RL x 8 modelos
# Izhikevich = 24 combinaciones x 11 corridas = 264 ejecuciones del binario.
#
# Estrategia: ARRAY JOB con 24 tasks paralelos. Cada task hace las 11 corridas
# de UNA combinacion (problema x modelo). Si el cluster solo deja correr unos
# pocos a la vez, los demas esperan en cola.
#
# task_id  |  problema   | modelo
# ---------+-------------+-------
#   1- 8   |  acrobot    | IZ-A..IZ-H
#   9-16   |  cartpole   | IZ-A..IZ-H
#  17-24   |  mountain   | IZ-A..IZ-H
#
# Lanzar desde NEAT/ con:    sbatch run_corridas.slurm
#==============================================================================

#SBATCH --job-name=neat-rl
#SBATCH --partition=batch
#SBATCH --array=1-24
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=25                # debe ser >= process_max del config
#SBATCH --time=24:00:00                   # ajustar segun lo que tarde una combinacion
#SBATCH --output=outputs/rl-%A_%a.out
#SBATCH --error=outputs/rl-%A_%a.err

# -----------------------------------------------------------------------------
# AJUSTAR ESTAS RUTAS PARA TU CLUSTER
# -----------------------------------------------------------------------------
PROJECT_DIR="$HOME/ANNarchy-NEAT"          # raiz del proyecto en el cluster
VENV_DIR="$PROJECT_DIR/.venv"              # entorno virtual de Python

# -----------------------------------------------------------------------------
# (OPCIONAL) Cargar modulos si tu cluster los usa. Comentar si no aplica.
# -----------------------------------------------------------------------------
# module purge
# module load python/3.12
# module load gcc/13

# -----------------------------------------------------------------------------
# Activar entorno virtual (debe existir previamente, ver README seccion 8.2)
# -----------------------------------------------------------------------------
source "$VENV_DIR/bin/activate"

# -----------------------------------------------------------------------------
# Mapear SLURM_ARRAY_TASK_ID a (problema, modelo)
# -----------------------------------------------------------------------------
PROBLEMS=(acrobot cartpole mountain)
MODELS=(IZ-A IZ-B IZ-C IZ-D IZ-E IZ-F IZ-G IZ-H)

idx=$((SLURM_ARRAY_TASK_ID - 1))
problem_idx=$((idx / 8))
model_idx=$((idx % 8))

PROBLEM=${PROBLEMS[$problem_idx]}
MODEL=${MODELS[$model_idx]}

echo "==================================================="
echo "Task $SLURM_ARRAY_TASK_ID / 24"
echo "Problema: $PROBLEM"
echo "Modelo:   $MODEL"
echo "Nodo:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Inicio:   $(date)"
echo "==================================================="

# -----------------------------------------------------------------------------
# Ir a NEAT/ y ejecutar las 11 corridas de esta combinacion
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR/NEAT"
mkdir -p outputs

python3 run_rl.py --problem "$PROBLEM" --model "$MODEL" --runs 11

echo "==================================================="
echo "Fin:      $(date)"
echo "==================================================="

# -----------------------------------------------------------------------------
# Limpieza opcional (la carpeta 'annarchy' la genera ANNarchy con codigo Cython
# compilado; se puede borrar tras la corrida si quieres ahorrar espacio).
# -----------------------------------------------------------------------------
# rm -rf annarchy
# rm -rf __pycache__
