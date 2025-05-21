#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=rearrange_tiles_657      #Set the job name to "JobExample1"
#SBATCH --time=10:00:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=48        #Request 8 tasks/cores per node
#SBATCH --mem=16G                     #Request 8GB per node 
#SBATCH --output=/scratch/user/anshulya/hls/github/hls/jobs/log_files/rearrange_tiles_657.%j      #Send stdout/err to "Example1Out.[jobID]"

#First Executable Line
module load Anaconda3          # Load Anaconda module
source activate /scratch/user/anshulya/.conda/envs/hls_env
echo "Environment Activated"

python /scratch/user/anshulya/hls/github/hls/code/rearrange_tiles.py --tiles 13SBS 13SBT 13SCS 13SCT --path /scratch/user/anshulya/hls/github/hls/