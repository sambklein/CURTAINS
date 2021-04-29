#!/bin/sh
#SBATCH --job-name=download
#SBATCH --cpus-per-task=1
#SBATCH --time=00-05:00:00
#SBATCH --partition=shared-cpu,private-dpnc-cpu,public-cpu
#SBATCH --output=/home/users/k/kleins/MLproject/CURTAINS/jobs/slurm-%A-%x_%a.out
#SBATCH --chdir=/home/users/k/kleins/MLproject/CURTAINS
#SBATCH --mem=10GB
export XDG_RUNTIME_DIR=""
module load GCCcore/8.2.0 Singularity/3.4.0-Go-1.12

srun singularity exec --nv -B /srv/beegfs/scratch/groups/dpnc/atlas/AnomalousJets:/srv/beegfs/scratch/groups/dpnc/atlas/AnomalousJets /home/users/k/kleins/MLproject/CURTAINS/container/latest_latest.sif jupyter notebook --no-browser --ip=$SLURMD_NODENAME --notebook-dir=/home/users/k/kleins/MLproject/CURTAINS
