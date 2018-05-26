#!/bin/sh
### Job name
#PBS -N hello_world_job
#PBS -P ee
### Output files
#PBS -o hello_world_job.stdout
#PBS -e hello_world_job.stderr

#PBS -m bea
### Specify email address to use for notification.
#PBS -M $eet162639@iitd.ac.in
####
#PBS -l nodes=1:ppn=2
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=01:00:00


### Queue name
#PBS -q standard
### Number of nodes
#PBS -l nodes=4:compute#shared
# Print the default PBS server
echo PBS default server is $/home/ee/mtech/eet162639/project_priyank/my_project/
# Print the job's working directory and enter it.
echo Working directory is $/home/ee/mtech/eet162639/project_priyank/my_project/
cd $/home/ee/mtech/eet162639/
python keras_signet.py
# Print some other environment information
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo This jobs runs on the following processors:
NODES=`cat $PBS_NODEFILE`
echo $NODES
# Compute the number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes
# Run hello_world
for NODE in $NODES; do
ssh $NODE "hello_world" &
done
# Wait for background jobs to complete.
wait

