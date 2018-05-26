#!/bin/sh
#PBS -P ee
#PBS -l select=1:ncpus=4:ngpus=2
#PBS -l walltime=20:00:00
#PBS -o out_WD_final
#PBS -e err_WD_final

module load pythonpackages/2.7.13/ucs4/gnu/447/keras/2.0.3/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/scikit-learn/0.18.1/gnu
python /home/ee/mtech/eet162639/majorproject/WD_final.py


