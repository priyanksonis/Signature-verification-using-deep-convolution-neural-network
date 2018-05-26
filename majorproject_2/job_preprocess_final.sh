#!/bin/sh
#PBS -P ee
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -o out_preprocess_final
#PBS -e err_preprocess_final

module load pythonpackages/2.7.13/ucs4/gnu/447/keras/2.0.3/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/opencv/3.2.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/scipy/0.19.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/pillow/4.1.0/gnu
module load pythonpackages/2.7.13/ucs4/gnu/447/matplotlib/2.0.0/gnu
python /home/ee/mtech/eet162639/majorproject/preprocess_dataset_final.py
