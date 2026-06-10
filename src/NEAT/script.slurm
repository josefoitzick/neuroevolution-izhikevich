#!/bin/bash
#SBATCH --job-name=dicyt-t13
#SBATCH --output=/home/XI/bloyola/neat-tests4/tests/NEAT-33/output.txt
#SBATCH --error=/home/XI/bloyola/neat-tests4/tests/NEAT-33/error.txt
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16  # Cada tarea usará 4 CPU

# Ir al directorio que contiene el ejecutable NEAT
cd /home/XI/bloyola/neat-tests4/tests/NEAT-33
mkdir results

# Ejecutar el ejecutable NEAT
python3 optuna_script.py

# Mover output.txt y error.txt a la carpeta results
mv output.txt results/
mv error.txt results/
mv example-study.db results/

# Generar un nombre único para la carpeta results
RESULTS_DIR=~/neat-tests4/runs-results
UNIQUE_RESULTS_DIR="${RESULTS_DIR}/results"
COUNTER=1

while [ -d "${UNIQUE_RESULTS_DIR}" ]; do
    UNIQUE_RESULTS_DIR="${RESULTS_DIR}/results_${COUNTER}"
    COUNTER=$((COUNTER + 1))
done

# Mover la carpeta results a la ubicación deseada con un nombre único
mv results "${UNIQUE_RESULTS_DIR}"

# Comandos para limpiar el directorio
rm -rf __pycache__
rm -rf annarchy

