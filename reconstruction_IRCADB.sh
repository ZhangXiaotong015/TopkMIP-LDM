#!/bin/bash
# Create an array to store background process IDs
declare -a JOB_IDS
N=35 # 7*5=num_of_validation_folds*num_of_test_cases. If testing is performed at only a single iteration, then N=1*5

# Submit jobs in a loop
for ((i=0; i<N; i++)); do
    JOB_NAME="Recon${i}"
    sbatch --job-name=$JOB_NAME \
           --output="/slurm_logs/${JOB_NAME}.out" \
           --error="/slurm_logs/${JOB_NAME}.err" \
           --ntasks=1 \
           --cpus-per-task=4 \
           --partition=gpu-long,gpu-medium \
           --gres=gpu:1 \
           --mem=200GB \
           --time=1-00:00:00 <<EOF &
#!/bin/bash
NODE_NAME=$(hostname)

python "reconstruction_IRCADB.py" \
    --name "Recon${i}" \
    --N ${N} \
    --N_idx ${i} \
    --key_in_pred_folder "Abla1_IRCADBExp_plain_leave1out" \
    --key_in_config "Abla1st_inference_IRCADB"
EOF

    # Get the job ID of the last submitted job
    JOB_IDS+=($!)
done

# Wait for all background jobs to finish
wait "${JOB_IDS[@]}"
