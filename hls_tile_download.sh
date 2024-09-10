#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=tile_13TDE       #Set the job name to "JobExample1"
#SBATCH --time=1:30:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=64        #Request 8 tasks/cores per node
#SBATCH --mem=32G                     #Request 8GB per node 
#SBATCH --output=/scratch/user/anshulya/hls/jobs/log_files/tile_13TDE.%j      #Send stdout/err to "Example1Out.[jobID]"

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

python code/download_tile.py 13TDE 2022-01-01 2023-12-31 120
