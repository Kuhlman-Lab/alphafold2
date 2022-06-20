#!/bin/bash

#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 00-00:30:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --constraint=rhel8
#SBATCH --mail-type=END
#SBATCH --mail-user=user@email.com

source ~/.bashrc
module add cuda/11.2
conda activate af2
python /proj/kuhl_lab/alphafold/run/run_af2.py @flags_longleaf.txt
