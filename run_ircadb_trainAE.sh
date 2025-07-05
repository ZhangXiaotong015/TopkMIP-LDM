#!/bin/bash
# Create an array to store background process IDs
declare -a JOB_IDS
fold_num=5

# Define base configuration files (no commas)
## 5Folds
base_config=(
    "/configs/first_stage_models/en-de-kl-256/config_IRCADB_Leave1Out_en_de_kl_F0.yaml"
    "/configs/first_stage_models/en-de-kl-256/config_IRCADB_Leave1Out_en_de_kl_F1.yaml"
    "/configs/first_stage_models/en-de-kl-256/config_IRCADB_Leave1Out_en_de_kl_F2.yaml"
    "/configs/first_stage_models/en-de-kl-256/config_IRCADB_Leave1Out_en_de_kl_F3.yaml"
    "/configs/first_stage_models/en-de-kl-256/config_IRCADB_Leave1Out_en_de_kl_F4.yaml"
)

# Submit jobs in a loop
for ((i=0; i<fold_num; i++)); do
    JOB_NAME="ircaAEF${i}"
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

python "main_EnDe.py" \
    --name "3DIRCADBExp_AE_leave1out_F${i}" \
    --base "${base_config[$i]}"
EOF

    # Get the job ID of the last submitted job
    JOB_IDS+=($!)
done

# Wait for all background jobs to finish
wait "${JOB_IDS[@]}"
