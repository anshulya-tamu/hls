#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=classify_657      #Set the job name to "JobExample1"
#SBATCH --time=10:00:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=48        #Request 8 tasks/cores per node
#SBATCH --mem=64G                     #Request 8GB per node 
#SBATCH --output=/scratch/user/anshulya/hls/github/hls/jobs/log_files/classify_657.%j      #Send stdout/err to "Example1Out.[jobID]"

#First Executable Line
module load Anaconda3          # Load Anaconda module
source activate /scratch/user/anshulya/.conda/envs/hls_env
echo "Environment Activated"
module load WebProxy
pip install geopandas
pip install rasterio

python /scratch/user/anshulya/hls/github/hls/code/cluster_reservoir_area.py --start_date 01-01-2016 --end_date 12-31-2022 --num_workers 40 --base_dir /scratch/user/anshulya/hls/github/hls/
