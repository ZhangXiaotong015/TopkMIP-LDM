#!/bin/bash
# Create an array to store background process IDs
declare -a JOB_IDS
fold_num=5

# Define base configuration files (no commas)

base_config=(
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_inference_IRCADB_F0.yaml"
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_inference_IRCADB_F1.yaml"
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_inference_IRCADB_F2.yaml"
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_inference_IRCADB_F3.yaml"
    "/configs/ldm/vessel_seg_256_256_256/Abla1st_inference_IRCADB_F4.yaml"
)

# Submit jobs in a loop
for ((i=0; i<fold_num; i++)); do
    JOB_NAME="Abl1S${i}"
    sbatch --job-name=$JOB_NAME \
        --output="/slurm_logs/${JOB_NAME}.out" \
        --error="/slurm_logs/${JOB_NAME}.err" \
        --ntasks=1 \
        --cpus-per-task=6 \
        --partition=gpu-long,gpu-medium \
        --gres=gpu:1 \
        --mem=64GB \
        --time=1-00:00:00 <<EOF &
#!/bin/bash
NODE_NAME=$(hostname)

python "main_infer.py" \
    --base "${base_config[$i]}" 
EOF

        # Get the job ID of the last submitted job
        JOB_IDS+=($!)
    done
done

# Wait for all background jobs to finish
wait "${JOB_IDS[@]}"
