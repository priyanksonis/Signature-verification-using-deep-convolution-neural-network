#!/bin/sh
#PBS -N somejob
#PBS -P cc
#PBS -m bea
#PBS -M $eet162639@iitd.ac.in
#PBS -l select=n:ncpus=m
#PBS -l walltime=01:00:00
#PBS -o stdout_file
#PBS -e stderr_file
#PBS -l software=
cd $PBS_O_WORKDIR
time -p mpirun -n executable
