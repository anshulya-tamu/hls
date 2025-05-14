#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=cluster_area_1       #Set the job name to "JobExample1"
#SBATCH --time=10:00:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=48
#SBATCH --mem=64G                     #Request 8GB per node 
#SBATCH --output=/scratch/user/anshulya/hls/jobs/log_files/cluster_area_1_northCONUS.%j      #Send stdout/err to "Example1Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=anshulya@tamu.edu    #Send all emails to email_address

#First Executable Line



#First Executable Line
module load Anaconda3          # Load Anaconda module
source activate /scratch/user/anshulya/.conda/envs/hls_env
echo "Environment Activated"
module load WebProxy
pip install geopandas

python code/cluster_reservoir_area.py 01-01-2016 12-31-2017 40
