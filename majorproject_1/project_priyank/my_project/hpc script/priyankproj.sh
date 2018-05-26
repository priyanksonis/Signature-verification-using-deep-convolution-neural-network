#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l walltime=00:00:59
cd /home/rcf-proj3/pv/test/
source /usr/usc/sas/default/setup.sh
sas my.sas


