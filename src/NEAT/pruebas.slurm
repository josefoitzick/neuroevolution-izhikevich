#!/bin/bash
#SBATCH --job-name=A-IZ
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --array=1-11:1
#SBATCH --output=outputs/job-%A_%a.out # job-523_1.out
#SBATCH --error=outputs/job-%A_%a.err # job-523_1.err
#SBATCH --cpus-per-task=25  # Cada tarea usará 25 CPU

# Ejecutar el ejecutable ./NEAT
./NEAT $SLURM_ARRAY_TASK_ID

# Comandos para limpiar el directorio
rm -rf __pycache__
rm -rf annarchy
