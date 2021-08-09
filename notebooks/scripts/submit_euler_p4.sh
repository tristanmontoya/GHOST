#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=24:00:00
#SBATCH --job-name euler_p4

module load NiaEnv/2019b python/3.8
source ~/.virtualenvs/tristan/bin/activate


cd /scratch/z/zingg/tmontoya/GHOST_private/notebooks

./run_euler_p4.sh
