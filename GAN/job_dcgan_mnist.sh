#!/bin/sh
#PBS -P ee
#PBS -l select=2:ncpus=10:ngpus=2
#PBS -o out_dcgan_mnist
#PBS -e err_dcgan_mnist

cd $PBS_O_WORKDIR


python dcgan_mnist.py

