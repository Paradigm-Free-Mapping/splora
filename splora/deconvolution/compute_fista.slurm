#!/bin/bash
#SBATCH --partition serial
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time=48:00:00

module purge
module load Python

conda activate /scratch/enekouru/conda_envs/splora/

echo $CONDA_DEFAULT_ENV

echo $CONDA_PREFIX

python -u /scratch/enekouru/splora/splora/deconvolution/compute_fista.py
