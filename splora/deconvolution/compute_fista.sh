#!/bin/bash
#SBATCH --partition serial
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time=20:00:00

module load Python

conda activate /scratch/enekouru/conda_envs/splora/

python -u /scratch/enekouru/splora/splora/deconvolution/compute_fista.py
