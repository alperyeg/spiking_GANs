#!/bin/bash
#SBATCH --job-name genesis
#SBATCH --workdir /home/a.yegenoglu/projects/spiking_GANs/src/
#SBATCH -o /home/a.yegenoglu/qsub/sgans/sgans.%A.%a.out
#SBATCH -e /home/a.yegenoglu/qsub/sgans/sgans.%A.%a.err
# SBATCH --time 20:00:00
#SBATCH -N 5
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
# SBATCH --ntasks=24
#SBATCH --mail-type=END
# SBATCH --exclude=blaustein01,blaustein22
source activate py3
module purge
# module load pystuff_new
module load slurm/default
mpirun python /home/a.yegenoglu/projects/spiking_GANs/src/save_data_mpi.py
