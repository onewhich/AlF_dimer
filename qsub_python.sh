#!/bin/bash 
#$ -N Work 
#$ -cwd
#$ -pe mpi 64
#$ -S /bin/bash
#$ -q short.q@compute-0-7
#$ -e $JOB_NAME.e$JOB_ID
#$ -o $JOB_NAME.o$JOB_ID


python run_AlF_dimer.py > output

