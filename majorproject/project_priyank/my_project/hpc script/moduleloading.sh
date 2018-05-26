#!/bin/sh
#PBS -N extra.py
#PBS -P ee
#PBS -m bea
#PBS -M $eet162639@iitd.ac.in
#PBS -l select=5:ncpus=2
#PBS -l walltime=01:00:00
#PBS -o out.txt
#PBS -e hello_world_job.stderr
#PBS -l software=
cd $PBS_O_WORKDIR


module load apps/tensorflow/1.5.0/gpu
module load pythonpackages/2.7.13/tensorflow_tensorboard/1.5.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/numpy/1.12.1/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/opencv/3.2.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/scikit-learn/0.18.1/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/keras/2.0.3/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/matplotlib/2.0.0/gnu



