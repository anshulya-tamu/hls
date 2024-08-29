#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=batch      #Set the job name to "JobExample1"
#SBATCH --time=1-23:30:00               #Set the wall clock limit to 6hr and 30min
#SBATCH --nodes=1                    #Request 1 node
#SBATCH --ntasks-per-node=4         #Request 8 tasks/cores per node
#SBATCH --mem=8G                     #Request 8GB per node 
#SBATCH --output=log_files/batch_processing.%j      #Send stdout/err to "Example1Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=anshulya@tamu.edu    #Send all emails to email_address

#First Executable Line

for index in 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808
do
    job_output=$(sbatch script_"${index}".sh)
    
    # Extract the job ID from the captured output
    job_id1=$(echo "{$job_output}" | awk -F'Submitted batch job ' '{print $2}' | tr -d '[:space:]')

    echo "Submitted batch job $job_id1"

    for job_name in $job_id1; do
        while true; do
            job_status=$(squeue -j "${job_name}" --noheader -o "%T")
            if [ -z "${job_status}" ]; then
                echo "Job ${job_name} has completed."
                break
            else
                echo "Job ${job_name} is still running. Waiting..."
                sleep 600  # Adjust the sleep duration as needed
            fi
        done
        
    done
    
done
echo "All reservoir jobs have completed. Ending the script!!"
