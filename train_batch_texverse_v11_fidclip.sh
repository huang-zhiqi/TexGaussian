#!/usr/bin/env bash
set -euo pipefail

# =========================
# Conda / CUDA env
# =========================
CONDA_EXE_PATH="${CONDA_EXE:-$(type -P conda || true)}"
if [[ -z "${CONDA_EXE_PATH}" ]]; then
  echo "[ERROR] conda not found in PATH."
  exit 1
fi
CONDA_BASE="$(cd "$(dirname "${CONDA_EXE_PATH}")/.." && pwd)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate scripts are often not nounset-safe under `set -u`.
set +u
CONDA_NO_PLUGINS=true conda activate texgaussian
set -u

export TORCH_CUDA_ARCH_LIST="8.6;8.9"
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
export C_INCLUDE_PATH="$CONDA_PREFIX/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# =========================
# Runtime
# =========================
GPU_IDS="0,1"
NUM_GPUS=2
ACC_CONFIG="acc_configs/gpu${NUM_GPUS}.yaml"
MAIN_PORT=8879

# =========================
# Experiment
# =========================
EXP_NAME="texverse_stage1_fidclip_v11"
WORKSPACE="../experiments/${EXP_NAME}"
RESUME_CKPT="./assets/ckpts/PBR_model.safetensors"

# =========================
# Data
# =========================
SPLIT_DIR="../experiments/common_splits"
TRAIN_LIST="${SPLIT_DIR}/train.tsv"
TEST_LIST="${SPLIT_DIR}/test.tsv"
CAPTION_FIELD="caption_long"
TRAIN_IMAGE_DIR="../datasets/texverse_rendered_train"
TEST_IMAGE_DIR="../datasets/texverse_rendered_test"
POINTCLOUD_DIR="../datasets/texverse_pointcloud_npz"

# =========================
# Feature switches
# =========================
USE_TEXT="True"
USE_MATERIAL="True"
USE_LONGCLIP="True"
LONGCLIP_MODEL="third_party/Long-CLIP/checkpoints/longclip-L.pt"
LONGCLIP_CONTEXT_LENGTH=248
USE_GGCA="True"
USE_TEXT_ADAPTER="True"
FREEZE_BASE="True"
FID_SAFE_MODE="True"
TRAIN_CONV_HEAD="False"

# =========================
# Optimization
# =========================
BATCH_SIZE=1
GRAD_ACC=8
NUM_EPOCHS=2
LR=3e-4
MIXED_PRECISION="bf16"
ADAPT_LR_SCALE=0.03
LAMBDA_LPIPS=0.5

# =========================
# FID / CLIP oriented losses
# =========================
USE_CLIP_SEMANTIC_LOSS="True"
CLIP_LOSS_MODEL="ViT-B/32"
CLIP_LOSS_NUM_VIEWS=2
CLIP_LOSS_RANDOM_VIEWS="True"
CLIP_LOSS_USE_GT_MASK="True"
CLIP_LOSS_IMG_SIZE=224
LAMBDA_CLIP_IMAGE=0.20
LAMBDA_CLIP_TEXT=0.05
LAMBDA_COLOR_STATS=0.05
ALPHA_GT_BLEND=0.25

# =========================
# Checkpoint selection
# =========================
COMPUTE_EVAL_FID="True"
EVAL_FID_USE_GT_MASK="True"
BEST_SELECTION_METRIC="fid"

echo "[INFO] Running ${EXP_NAME}"
echo "  GPUs: ${GPU_IDS}  num_processes=${NUM_GPUS}"
echo "  workspace: ${WORKSPACE}"
echo "  resume: ${RESUME_CKPT}"
echo "  use_text_adapter: ${USE_TEXT_ADAPTER}"
echo "  clip_loss: ${USE_CLIP_SEMANTIC_LOSS} (img=${LAMBDA_CLIP_IMAGE}, text=${LAMBDA_CLIP_TEXT})"

for p in "${TRAIN_LIST}" "${TEST_LIST}" "${ACC_CONFIG}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[ERROR] missing file: ${p}"
    exit 1
  fi
done

for d in "${TRAIN_IMAGE_DIR}" "${TEST_IMAGE_DIR}" "${POINTCLOUD_DIR}"; do
  if [[ ! -d "${d}" ]]; then
    echo "[ERROR] missing directory: ${d}"
    exit 1
  fi
done

if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "[ERROR] RESUME_CKPT not found: ${RESUME_CKPT}"
  exit 1
fi

if [[ "${USE_LONGCLIP}" == "True" && ! -f "${LONGCLIP_MODEL}" ]]; then
  echo "[ERROR] LongCLIP checkpoint not found: ${LONGCLIP_MODEL}"
  exit 1
fi

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
  --caption_field "${CAPTION_FIELD}"
  --image_dir "${TRAIN_IMAGE_DIR}"
  --test_image_dir "${TEST_IMAGE_DIR}"
  --pointcloud_dir "${POINTCLOUD_DIR}"
  --use_text "${USE_TEXT}"
  --use_material "${USE_MATERIAL}"
  --use_longclip "${USE_LONGCLIP}"
  --longclip_model "${LONGCLIP_MODEL}"
  --longclip_context_length "${LONGCLIP_CONTEXT_LENGTH}"
  --use_ggca "${USE_GGCA}"
  --use_text_adapter "${USE_TEXT_ADAPTER}"
  --freeze_base "${FREEZE_BASE}"
  --fid_safe_mode "${FID_SAFE_MODE}"
  --train_conv_head "${TRAIN_CONV_HEAD}"
  --adapt_lr_scale "${ADAPT_LR_SCALE}"
  --lambda_lpips "${LAMBDA_LPIPS}"
  --use_clip_semantic_loss "${USE_CLIP_SEMANTIC_LOSS}"
  --clip_loss_model "${CLIP_LOSS_MODEL}"
  --clip_loss_num_views "${CLIP_LOSS_NUM_VIEWS}"
  --clip_loss_random_views "${CLIP_LOSS_RANDOM_VIEWS}"
  --clip_loss_use_gt_mask "${CLIP_LOSS_USE_GT_MASK}"
  --clip_loss_img_size "${CLIP_LOSS_IMG_SIZE}"
  --lambda_clip_image "${LAMBDA_CLIP_IMAGE}"
  --lambda_clip_text "${LAMBDA_CLIP_TEXT}"
  --lambda_color_stats "${LAMBDA_COLOR_STATS}"
  --alpha_gt_blend "${ALPHA_GT_BLEND}"
  --compute_eval_fid "${COMPUTE_EVAL_FID}"
  --eval_fid_use_gt_mask "${EVAL_FID_USE_GT_MASK}"
  --best_selection_metric "${BEST_SELECTION_METRIC}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  ARGS+=(--resume "${RESUME_CKPT}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch "${ARGS[@]}" \
  2> >(grep -v -E '^libpng warning: eXIf: duplicate$' >&2)
