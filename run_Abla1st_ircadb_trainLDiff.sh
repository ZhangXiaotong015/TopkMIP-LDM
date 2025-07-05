#!/bin/bash
# Create an array to store SLURM job IDs
declare -a JOB_IDS
fold_num=5

# Define base configuration files (no commas)
base_config=(
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_IRCADB_F0.yaml"
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_IRCADB_F1.yaml"
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_IRCADB_F2.yaml"
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_IRCADB_F3.yaml"
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_IRCADB_F4.yaml"
)

# Submit jobs in a loop
for ((i=0; i<fold_num; i++)); do
    JOB_NAME="Abla1F${i}"
    
    # Submit the job and capture the output
    job_output=$(sbatch --job-name=$JOB_NAME \
                        --output="/slurm_logs/${JOB_NAME}.out" \
                        --error="/slurm_logs/${JOB_NAME}.err" \
                        --ntasks=1 \
                        --cpus-per-task=6 \
                        --partition=gpu-long \
                        --nodelist=node864,node865,node866,node867,node868,node869,node870,node871,node872,node875,node876 \
                        --gres=gpu:1 \
                        --mem=64GB \
                        --time=7-00:00:00 <<EOF
#!/bin/bash
NODE_NAME=$(hostname)

python "main.py" \
    --name "Abla1_IRCADBExp_plain_leave1out_F${i}" \
    --base "${base_config[$i]}"
EOF
)
    
    # Extract the job ID from sbatch output
    job_id=$(echo $job_output | awk '{print $4}')
    JOB_IDS+=($job_id)

done

# Wait for all jobs to finish
for job_id in "${JOB_IDS[@]}"; do
    scontrol wait $job_id
done



