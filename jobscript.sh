#!/bin/bash
#SBATCH --job-name sgangs
#SBATCH --workdir /home/a.yegenoglu/projects/spiking_GANs/src/
#SBATCH -o /home/a.yegenoglu/qsub/sgans/sgans.%A.%a.out
#SBATCH -e /home/a.yegenoglu/qsub/sgans/sgans.%A.%a.err
#SBATCH --time 20:00:00
#SBATCH --array=6,7,8,9,11,12,13,14,15
# SBATCH -N 1
# SBATCH --ntasks-per-node=9
# SBATCH --cpus-per-task=2
# SBATCH --ntasks=1
# SBATCH --exclusive
#SBATCH --mail-type=END
# SBATCH --exclude=blaustein01,blaustein22
source activate py3
module purge
# module load pystuff_new
# module load slurm/default
# mpirun python /home/a.yegenoglu/projects/spiking_GANs/src/save_data_mpi.py
# python dcgan_spikes_torch.py
# python save_data.py
python transfer.py
