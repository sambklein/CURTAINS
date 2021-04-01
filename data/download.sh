#!/bin/sh
#SBATCH --job-name=download
#SBATCH --cpus-per-task=1
#SBATCH --time=00-04:00:00
#SBATCH --partition=shared-cpu,private-dpnc-cpu,public-cpu
#SBATCH --output=/srv/beegfs/scratch/groups/dpnc/atlas/BIB/implicitBIBae/jobs/slurm-%A-%x_%a.out
#SBATCH --chdir=/home/users/k/kleins/atlas/BIB/implicitBIBae
#SBATCH --mem=10GB
export XDG_RUNTIME_DIR=""
module load GCCcore/8.2.0 Singularity/3.4.0-Go-1.12

srun singularity exec --nv /srv/beegfs/scratch/groups/dpnc/atlas/BIB/implicitBIBae/container/pytorch.sif\
	python3 /srv/beegfs/scratch/groups/dpnc/atlas/BIB/implicitBIBae/data/data_loaders.py --download 1