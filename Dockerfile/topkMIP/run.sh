#!/bin/bash
OUTPUT_DIR="/home/xzhang/inference_project/topkMIP/output/Pre-ablation/Portal"
mkdir -p "$OUTPUT_DIR"
resized_CT_root="/home/xzhang/inference_project/topkMIP/data/resized_CT"
mkdir -p "$resized_CT_root"
topkMIP_CT_root="/home/xzhang/inference_project/topkMIP/data/topkMIP_CT"
mkdir -p "$topkMIP_CT_root"
batch_topkMIP_CT_root="/home/xzhang/inference_project/topkMIP/data/batch_topkMIP_CT"
mkdir -p "$batch_topkMIP_CT_root"
rotMat_root="/home/xzhang/inference_project/topkMIP/data/RotMat"
mkdir -p "$rotMat_root"
systemMatrix_root="/home/xzhang/inference_project/topkMIP/data/system_matrix"
mkdir -p "$systemMatrix_root"


docker run --rm --gpus "device=0" \
           --tmpfs /dev/shm:rw,noexec,nosuid,size=1g \
    --mount type=bind,src="$OUTPUT_DIR",dst=/outputs \
    --mount type=bind,src=/mnt/e/WSL/TestData/LiverVesselSeg/Pre-ablation/Portal,dst=/CT_root,readonly \
    --mount type=bind,src="$resized_CT_root",dst=/resized_CT_root,readonly \
    --mount type=bind,src="$topkMIP_CT_root",dst=/topkMIP_CT_root,readonly \
    --mount type=bind,src="$batch_topkMIP_CT_root",dst=/batch_topkMIP_CT_root,readonly \
    --mount type=bind,src="$rotMat_root",dst=/rotMat_root,readonly \
    --mount type=bind,src="$systemMatrix_root",dst=/systemMatrix_root,readonly,consistency=cached \
    -e base_config=/app/configs/ldm/vessel_seg_256_256_256/Abla1st_inference_IRCADB_F0.yaml \
    topk_mip:latest \
    --base "$base_config" \
    --CT_root /CT_root \
    --resized_CT_root /resized_CT_root \
    --topkMIP_CT_root /topkMIP_CT_root \
    --batch_topkMIP_CT_root /batch_topkMIP_CT_root \
    --rotMat_root /rotMat_root \
    --systemMatrix_root /systemMatrix_root \
    --percentile 98

# ## Interactive
# OUTPUT_DIR="/home/xzhang/inference_project/topkMIP/output/Pre-ablation/Portal"
# mkdir -p "$OUTPUT_DIR"
# resized_CT_root="/home/xzhang/inference_project/topkMIP/data/resized_CT"
# mkdir -p "$resized_CT_root"
# topkMIP_CT_root="/home/xzhang/inference_project/topkMIP/data/topkMIP_CT"
# mkdir -p "$topkMIP_CT_root"
# batch_topkMIP_CT_root="/home/xzhang/inference_project/topkMIP/data/batch_topkMIP_CT"
# mkdir -p "$batch_topkMIP_CT_root"
# rotMat_root="/home/xzhang/inference_project/topkMIP/data/RotMat"
# mkdir -p "$rotMat_root"
# systemMatrix_root="/home/xzhang/inference_project/topkMIP/data/system_matrix"
# mkdir -p "$systemMatrix_root"

# docker run -it --rm --gpus "device=0" \
#            --tmpfs /dev/shm:rw,noexec,nosuid,size=1g \
#     --mount type=bind,src="$OUTPUT_DIR",dst=/outputs \
#     --mount type=bind,src=/mnt/e/WSL/TestData/LiverVesselSeg/Pre-ablation/Portal,dst=/CT_root,readonly \
#     --mount type=bind,src="$resized_CT_root",dst=/resized_CT_root,readonly \
#     --mount type=bind,src="$topkMIP_CT_root",dst=/topkMIP_CT_root,readonly \
#     --mount type=bind,src="$batch_topkMIP_CT_root",dst=/batch_topkMIP_CT_root,readonly \
#     --mount type=bind,src="$rotMat_root",dst=/rotMat_root,readonly \
#     --mount type=bind,src="$systemMatrix_root",dst=/systemMatrix_root,readonly,consistency=cached \
#     -e base_config=/app/configs/ldm/vessel_seg_256_256_256/Abla1st_inference_IRCADB_F0.yaml \
#     topk_mip:latest \
#     bash

##### All mounted paths must exist!!!
##### PowerShell Setting
# notepad $env:USERPROFILE\.wslconfig
# [wsl2]
# memory=96GB
# processors=4
# swap=0
# localhostForwarding=true

