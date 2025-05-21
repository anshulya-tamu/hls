#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=bulk_tile_download_657      #Set the job name to "JobExample1"
#SBATCH --time=10:00:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=8        #Request 8 tasks/cores per node
#SBATCH --mem=8G                     #Request 8GB per node 
#SBATCH --output=/scratch/user/anshulya/hls/github/hls/jobs/log_files/bulk_tile_download_657.%j      #Send stdout/err to "Example1Out.[jobID]"

#First Executable Line
module load Anaconda3          # Load Anaconda module
source activate /scratch/user/anshulya/.conda/envs/hls_env
echo "Environment Activated"
module load WebProxy
pip install geopandas
pip install rasterio

bash /scratch/user/anshulya/hls/github/hls/code/bulk_download/getHLS.sh /scratch/user/anshulya/hls/github/hls/code/bulk_download/tiles/tile_id.txt 2016-01-01 2022-12-31 /scratch/user/anshulya/hls/github/hls/data
