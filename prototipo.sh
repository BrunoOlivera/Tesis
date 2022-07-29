#!/bin/bash
#SBATCH --job-name=prototipo
#SBATCH --ntasks=13
#SBATCH --mem=16G
#SBATCH --partition=ute
#SBATCH --qos=ute
#SBATCH --output=output_cluster/%x_%j.txt
#SBATCH --time=2-12

# $1 - pss
# $2 - vss
# $3 - sigma
# $4 - eps
# $5 - niveles_RBF
# $6 - sigma_RBF
singularity exec singularity-prototipo.simg mpiexec -n 13 python run.py $1 $2 1 $3 10000 $4 $5 $6 0
