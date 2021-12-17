#!/bin/bash
#SBATCH --partition=shared-cpu,private-dpnc-cpu
#SBATCH --time=4:00:00
#SBATCH --output=/home/users/k/kleins/MLproject/CURTAINS/jobs/slurm-%A-%x_%a.out
#SBATCH --mem=10GB

_image_location=/home/users/k/kleins/MLproject/funnels/container/tensorflow_1.15.sif

_log_dir=/home/users/k/kleins/MLproject/CURTAINS/logs/OT_16_Dec/

module load GCC/9.3.0 Singularity/3.7.3-Go-1.14
srun singularity exec ${_image_location} tensorboard --logdir ${_log_dir}