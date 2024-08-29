#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=237_area       #Set the job name to "JobExample1"
#SBATCH --time=3:30:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=48        #Request 8 tasks/cores per node
#SBATCH --mem=32G                     #Request 8GB per node 
#SBATCH --output=log_files/237_area_estimation.%j      #Send stdout/err to "Example1Out.[jobID]"

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
pip install opencv-python
python estimate_area_canny_parallel.py 237
