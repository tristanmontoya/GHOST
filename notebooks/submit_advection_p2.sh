#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --job-name advection_p2

module load NiaEnv/2019b python/3.8
source ~/.virtualenvs/tristan/bin/activate


export OMP_NUM_THREADS=1

cd /scratch/z/zingg/tmontoya/GHOST/notebooks

./run_advection_p2.sh
