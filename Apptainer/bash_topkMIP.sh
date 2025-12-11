echo "Using GPUs:"
nvidia-smi || echo "No GPU visible"

# --- Ensure module system is available ---
source /etc/profile.d/modules.sh

# --- Load Apptainer module ---
module load container/apptainer/1.4.1

# --- Base directory for all Apptainer data ---
BASE=/exports/lkeb-hpc/xzhang/docker_archive
mkdir -p $BASE/slurm_logs

# --- 1. Set Apptainer tmp & cache to your large persistent directory ---
export APPTAINER_TMPDIR=$BASE/apptainer_tmp
export APPTAINER_CACHEDIR=$BASE/apptainer_cache

mkdir -p "$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR"
chmod 700 "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# --- 2. Where to store your .sif ---
export IMAGEDIR=$BASE/apptainer_images
mkdir -p "$IMAGEDIR"
chmod 700 "$IMAGEDIR"

# --- 3. Path to your Docker archive (.tar) ---
TARFILE=$BASE/topk_mip.tar

# --- 4. Build SIF only if not already present ---
if [ ! -f "$IMAGEDIR/topk_mip.sif" ]; then
    echo "Building SIF from $TARFILE ..."
    apptainer build "$IMAGEDIR/topk_mip.sif" docker-archive://$TARFILE
fi

# --- 5. Prepare output and data directories ---
echo "Running container..."

OUT_BASE=/exports/lkeb-hpc/xzhang/docker_archive/VesselSeg_topkMIP

OUTPUT_DIR="$OUT_BASE/output/Pre-ablation/Portal"
mkdir -p "$OUTPUT_DIR"
resized_CT_root="$OUT_BASE/data/resized_CT"
mkdir -p "$resized_CT_root"
topkMIP_CT_root="$OUT_BASE/data/topkMIP_CT"
mkdir -p "$topkMIP_CT_root"
batch_topkMIP_CT_root="$OUT_BASE/data/batch_topkMIP_CT"
mkdir -p "$batch_topkMIP_CT_root"
rotMat_root="$OUT_BASE/data/RotMat"
mkdir -p "$rotMat_root"
systemMatrix_root="$OUT_BASE/data/system_matrix"
mkdir -p "$systemMatrix_root"

CT_root=/exports/lkeb-hpc/xzhang/docker_archive/data/LiverVesselSeg/Pre-ablation/Portal

# --- 6. Run topkMIP container on GPU ---
apptainer run --nv \
    --bind "$OUTPUT_DIR:/outputs" \
    --bind "$CT_root:/CT_root:ro" \
    --bind "$resized_CT_root:/resized_CT_root" \
    --bind "$topkMIP_CT_root:/topkMIP_CT_root" \
    --bind "$batch_topkMIP_CT_root:/batch_topkMIP_CT_root" \
    --bind "$rotMat_root:/rotMat_root" \
    --bind "$systemMatrix_root:/systemMatrix_root" \
    "$IMAGEDIR/topk_mip.sif" \
        --base /app/configs/ldm/vessel_seg_256_256_256/Abla1st_inference_IRCADB_F0.yaml \
        --CT_root /CT_root \
        --resized_CT_root /resized_CT_root \
        --topkMIP_CT_root /topkMIP_CT_root \
        --batch_topkMIP_CT_root /batch_topkMIP_CT_root \
        --rotMat_root /rotMat_root \
        --systemMatrix_root /systemMatrix_root \
        --percentile 98


