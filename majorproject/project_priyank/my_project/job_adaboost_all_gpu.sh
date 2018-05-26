#!/bin/sh
#PBS -P ee
#PBS -l select=2:ncpus=20:ngpus=2
#PBS -o out_adaboost_all_gpu
#PBS -e err_adaboost_all_gpu
cd $PBS_O_WORKDIR


module load pythonpackages/2.7.13/ucs4/gnu/447/keras/2.0.3/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/scikit-learn/0.18.1/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/opencv/3.2.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/pillow/4.1.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/pandas/0.20.0rc1/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/h5py/2.7.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/matplotlib/2.0.0/gnu
module load apps/tensorflow/1.1.0/gpu
python adaboost_hpc_all_users.py
