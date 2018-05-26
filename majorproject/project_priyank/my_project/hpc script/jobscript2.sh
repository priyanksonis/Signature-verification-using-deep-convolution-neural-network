#!/bin/sh
#PBS -N normal_cnn
#PBS -P ee
#PBS -m bea
#PBS -M $eet162639@iitd.ac.in
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -o out1.txt
#PBS -e err1.txt
cd $PBS_O_WORKDIR

module load apps/tensorflow/1.1.0/gpu
module load pythonpackages/2.7.13/ucs4/gnu/447/opencv/3.2.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/keras/2.0.3/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/scikit-learn/0.18.1/gnu
python custom_data_cnn_with_comments.py



