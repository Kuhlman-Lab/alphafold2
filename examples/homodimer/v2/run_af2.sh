#!/bin/bash

#SBATCH -J af2_homodimer
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 03-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@email.com

source ~/.bashrc
module load gcc; module load cuda
conda activate af2
python /proj/kuhl_lab/alphafold/run/run_af2.py @flags/flags_longleaf.txt
