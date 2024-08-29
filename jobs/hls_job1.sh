#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=237_serial_clip       #Set the job name to "JobExample3"
#SBATCH --time=03:00:00              #Set the wall clock limit to 1 Day and 12hr
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=48         #Request 8 tasks/cores per node
#SBATCH --mem=32G                  #Request 4096MB (4GB) per node 
#SBATCH --output=log_files/237_serial_clip_2016_17.%j      #Send stdout/err to "Example3Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=anshulya@tamu.edu    #Send all emails to email_address 

#First Executable Line



#First Executable Line
module load Anaconda3          # Load Anaconda module
source activate hls_env
module load WebProxy
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

python clipped_data_creation.py 01-01-2016 01-01-2017 237
