#!/bin/csh
#PBS -j oe
#PBS -N RPCD_normal
#PBS -M s2030415@jaist.ac.jp -m be

source /etc/profile.d/modules.csh
module purge
module load cuda

setenv PATH ${PATH}:${HOME}/.poetry/bin
cd ${PBS_O_WORKDIR}

poetry run python -m src.RPCD.main

