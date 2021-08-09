#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:20:00
#SBATCH --job-name advection_p3_debug

module load NiaEnv/2019b python/3.8
source ~/.virtualenvs/tristan/bin/activate


#export OMP_NUM_THREADS=1

cd /scratch/z/zingg/tmontoya/GHOST_private/notebooks

papermill advection_driver.ipynb advection_p3b0c0t1_strong.ipynb -p p 3 -p p_geo 1 -p upwind_parameter 0.0 -p c 'c_dg' -p discretization_type 1 -p form 'strong'


