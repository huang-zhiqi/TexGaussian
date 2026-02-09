#!/usr/bin/env bash
set -euo pipefail

# Optional: activate conda env.
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate texgaussian

# =========================
# Runtime
# =========================
GPU_IDS="0,1"
NUM_GPUS=2
ACC_CONFIG="acc_configs/gpu${NUM_GPUS}.yaml"
MAIN_PORT=8878

# =========================
# Experiment
# =========================
EXP_NAME="texverse_longclip_heads"
WORKSPACE="../experiments/${EXP_NAME}"
RESUME_CKPT="./assets/ckpts/PBR_model.safetensors"  # leave empty to train from scratch

# =========================
# Data
# =========================
TRAIN_LIST="../experiments/common_splits/train.txt"
TEST_LIST="../experiments/common_splits/test.txt"
TEXT_DESCRIPTION_CSV="../experiments/common_splits/captions.csv"
IMAGE_DIR="path_to_texverse_image_dir"
POINTCLOUD_DIR="path_to_texverse_pointcloud_dir"

# =========================
# Feature switches
# =========================
USE_TEXT="True"
USE_MATERIAL="True"
USE_LONGCLIP="True"
LONGCLIP_MODEL="third_party/Long-CLIP/checkpoints/longclip-L.pt"
LONGCLIP_CONTEXT_LENGTH=248
USE_NORMAL_HEAD="True"
USE_ROTATION_HEAD="True"

# =========================
# Optimization
# =========================
BATCH_SIZE=8
GRAD_ACC=1
NUM_EPOCHS=300
LR=4e-4
MIXED_PRECISION="bf16"
LAMBDA_GEO_NORMAL=1.0
LAMBDA_TEX_NORMAL=1.0

echo "[INFO] Training config"
echo "  GPUs: ${GPU_IDS} (num_processes=${NUM_GPUS})"
echo "  Accelerate config: ${ACC_CONFIG}"
echo "  Workspace: ${WORKSPACE}"
echo "  Resume: ${RESUME_CKPT}"
echo "  LongCLIP: ${USE_LONGCLIP} (${LONGCLIP_MODEL})"
echo "  Heads: normal=${USE_NORMAL_HEAD}, rotation=${USE_ROTATION_HEAD}"

ARGS=(
  --main_process_port "${MAIN_PORT}"
  --config_file "${ACC_CONFIG}"
  main.py objaverse
  --workspace "${WORKSPACE}"
  --batch_size "${BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --lr "${LR}"
  --mixed_precision "${MIXED_PRECISION}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  --trainlist "${TRAIN_LIST}"
  --testlist "${TEST_LIST}"
  --text_description "${TEXT_DESCRIPTION_CSV}"
  --image_dir "${IMAGE_DIR}"
  --pointcloud_dir "${POINTCLOUD_DIR}"
  --use_text "${USE_TEXT}"
  --use_material "${USE_MATERIAL}"
  --use_longclip "${USE_LONGCLIP}"
  --longclip_model "${LONGCLIP_MODEL}"
  --longclip_context_length "${LONGCLIP_CONTEXT_LENGTH}"
  --use_normal_head "${USE_NORMAL_HEAD}"
  --use_rotation_head "${USE_ROTATION_HEAD}"
  --lambda_geo_normal "${LAMBDA_GEO_NORMAL}"
  --lambda_tex_normal "${LAMBDA_TEX_NORMAL}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  ARGS+=(--resume "${RESUME_CKPT}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch "${ARGS[@]}"

