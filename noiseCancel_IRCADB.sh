#!/bin/bash
# Create an array to store background process IDs
declare -a JOB_IDS
N=35

# Submit jobs in a loop
for ((i=0; i<N; i++)); do
    JOB_NAME="noiseC${i}"
    sbatch --job-name=$JOB_NAME \
           --output="/slurm_logs/${JOB_NAME}.out" \
           --error="/slurm_logs/${JOB_NAME}.err" \
           --ntasks=1 \
           --cpus-per-task=4 \
           --partition=gpu-long,gpu-medium,cpu-long,cpu-medium \
           --gres=gpu:1 \
           --mem=32GB \
           --time=1-00:00:00 <<EOF &
#!/bin/bash
NODE_NAME=$(hostname)

python "noiseCancel_IRCADB.py" \
    --name "noiseC${i}" \
    --N ${N} \
    --N_idx ${i} 
EOF

    # Get the job ID of the last submitted job
    JOB_IDS+=($!)
done

# Wait for all background jobs to finish
wait "${JOB_IDS[@]}"
