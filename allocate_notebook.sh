#!/bin/sh
#SBATCH --partition=shared-cpu,private-dpnc-cpu
#SBATCH --time=9:00:00
#SBATCH --output=/home/users/k/kleins/MLproject/CURTAINS/jobs/slurm-%A-%x_%a.out
#SBATCH --mem=10GB

# load Anaconda, this will provide Jupyter as well.
module load GCCcore/8.2.0 Singularity/3.4.0-Go-1.12
export XDG_RUNTIME_DIR=""

# specify here the directory containing your notebooks and images
RUNDIR=/home/users/k/kleins/MLproject/CURTAINS
IMGDIR=/home/users/k/kleins/MLproject/CURTAINS/container/latest_latest.sif
DATADIR=/srv/beegfs/scratch/groups/dpnc/atlas/CURTAINS,/srv/beegfs/scratch/groups/rodem/LHCO/

# launch Jupyter notebook
srun singularity exec --nv -B $RUNDIR:/$RUNDIR,$DATADIR $IMGDIR jupyter notebook --no-browser --ip=$SLURMD_NODENAME --notebook-dir=$RUNDIR
 