#!/bin/bash
#SBATCH --job-name spiking_gans
#SBATCH --workdir /home/a.yegenoglu/projects/spiking_GANs/src/
#SBATCH -o /home/a.yegenoglu/qsub/sgans/sgans.%A.%a.out
#SBATCH -e /home/a.yegenoglu/qsub/sgans/sgans.%A.%a.err
# SBATCH --time 20:00:00
# SBATCH --ntasks-per-node=10
# SBATCH --ntasks=24
#SBATCH --mail-type=END
#SBATCH --exclude=blaustein01,blaustein22
source activate py3
module purge
# module load pystuff_new
python dcgan_spikes_torch.py
