#!/bin/bash

#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4g
#SBATCH -t 00-00:30:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

module add anaconda/2020.07 cuda/11.2
python alphafold -s PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK 
