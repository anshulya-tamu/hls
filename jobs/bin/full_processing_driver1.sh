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

for index in 505 508 509 511 512 529 538 542 543 545 548 555 557 560 561 562 573 581 606 610 611 612 613 614 615 616 618 619 620 624 626 627 629 644 653 654 659 662 689 699
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
