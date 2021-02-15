#!/bin/bash
#SBATCH -N 1
#SBATCH -J test_shaker
#SBATCH -A hpc-prf-isys
#SBATCH -p batch
#SBATCH --mail-type all
#SBATCH --mail-user mhshaker@mail.upb.de

module add singularity
singularity run --bind /upb/scratch/departments/pc2/groups/hpc-prf-isys/mhshaker/:/upb/scratch/departments/pc2/groups/hpc-prf-isys/mhshaker/ /upb/scratch/departments/pc2/groups/hpc-prf-isys/mhshaker/s_python3.simg python3 Sampling.py $1
echo job_id $1 Done
