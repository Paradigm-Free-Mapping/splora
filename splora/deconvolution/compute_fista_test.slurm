#!/bin/bash
#SBATCH --partition serial
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time=20:00:00

module load Python

conda activate /scratch/enekouru/conda_envs/splora/

echo $CONDA_DEFAULT_ENV

echo $CONDA_PREFIX

echo "This is a test to ensure that the correct file is being run"

python -u /scratch/enekouru/splora/splora/deconvolution/compute_fista_test.py
