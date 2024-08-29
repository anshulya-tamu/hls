#!/bin/bash

job_output=$(sbatch script_100_1.sh)

# Extract the job ID from the captured output
job_id1=$(echo "$job_output" | awk -F'Submitted batch job ' '{print $2}' | tr -d '[:space:]')

job_output=$(sbatch script_100_2.sh)

# Extract the job ID from the captured output
job_id2=$(echo "$job_output" | awk -F'Submitted batch job ' '{print $2}' | tr -d '[:space:]')

job_output=$(sbatch script_100_3.sh)

# Extract the job ID from the captured output
job_id3=$(echo "$job_output" | awk -F'Submitted batch job ' '{print $2}' | tr -d '[:space:]')

job_output=$(sbatch script_100_4.sh)

# Extract the job ID from the captured output
job_id4=$(echo "$job_output" | awk -F'Submitted batch job ' '{print $2}' | tr -d '[:space:]')

echo "Submitted batch job $job_id1"
echo "Submitted batch job $job_id2"
echo "Submitted batch job $job_id3"
echo "Submitted batch job $job_id4"

for job_name in $job_id1 $job_id2 $job_id3 $job_id4; do
    while true; do
        job_status=$(squeue -j "${job_name}" --noheader -o "%T")
        if [ -z "${job_status}" ]; then
            echo "Job ${job_name} has completed."
            break
        else
            echo "Job ${job_name} is still running. Waiting..."
            sleep 60  # Adjust the sleep duration as needed
        fi
    done
done

echo "All sbatch jobs have completed. Continuing with the script."