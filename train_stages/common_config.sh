#!/bin/bash
# =============================================================================
# 共享配置文件 - 所有训练阶段共用
# =============================================================================

# 1. 初始化 Conda 环境
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate texgaussian

# CUDA 环境
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH="$CONDA_PREFIX/include:${CPATH}"
export C_INCLUDE_PATH="$CONDA_PREFIX/include:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

# =========================
# 显存优化配置 (防止OOM)
# =========================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# =========================
# GPU 配置 (根据你的显卡数量调整)
# =========================
# 2卡配置
# GPU_IDS="0,1"
# NUM_GPUS=2

# 4卡配置 (解开注释使用)
GPU_IDS="0,1,2,3"
NUM_GPUS=4

ACC_CONFIG="acc_configs/gpu${NUM_GPUS}.yaml"
MAIN_PORT=8878

# =========================
# 数据路径 (不需要修改)
# =========================
SPLIT_DIR="../experiments/common_splits"
TRAIN_LIST="${SPLIT_DIR}/train.tsv"
TEST_LIST="${SPLIT_DIR}/test.tsv"
CAPTION_FIELD="caption_long"
TRAIN_IMAGE_DIR="../datasets/texverse_rendered_train"
TEST_IMAGE_DIR="../datasets/texverse_rendered_test"
POINTCLOUD_DIR="../datasets/texverse_pointcloud_npz"

# =========================
# 预训练模型
# =========================
BASE_PRETRAINED_CKPT="./assets/ckpts/PBR_model.safetensors"

# =========================
# LongCLIP 配置
# =========================
USE_LONGCLIP="True"
LONGCLIP_MODEL="third_party/Long-CLIP/checkpoints/longclip-L.pt"
LONGCLIP_CONTEXT_LENGTH=248

# =========================
# 材质配置
# =========================
USE_TEXT="True"
USE_MATERIAL="True"
MIXED_PRECISION="bf16"

# =========================
# 验证样本 (用于检查训练效果)
# =========================
VALIDATION_SAMPLES=(
  "e5c658101c2f467ba090aaedb93dd6c0"  # 黄色物体 - 关键测试
  "9e5230175e864a2281ce6cdd1231c04e"
)

# =========================
# 工程根目录
# =========================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_ROOT="../experiments"

# =========================
# 辅助函数
# =========================
validate_paths() {
  local has_error=0
  
  if [[ ! -f "${TRAIN_LIST}" ]]; then
    echo "[ERROR] train split not found: ${TRAIN_LIST}"
    has_error=1
  fi
  if [[ ! -f "${TEST_LIST}" ]]; then
    echo "[ERROR] test split not found: ${TEST_LIST}"
    has_error=1
  fi
  if [[ ! -f "${ACC_CONFIG}" ]]; then
    echo "[ERROR] accelerate config not found: ${ACC_CONFIG}"
    has_error=1
  fi
  if [[ ! -d "${TRAIN_IMAGE_DIR}" ]]; then
    echo "[ERROR] TRAIN_IMAGE_DIR does not exist: ${TRAIN_IMAGE_DIR}"
    has_error=1
  fi
  if [[ ! -d "${TEST_IMAGE_DIR}" ]]; then
    echo "[ERROR] TEST_IMAGE_DIR does not exist: ${TEST_IMAGE_DIR}"
    has_error=1
  fi
  if [[ ! -d "${POINTCLOUD_DIR}" ]]; then
    echo "[ERROR] POINTCLOUD_DIR does not exist: ${POINTCLOUD_DIR}"
    has_error=1
  fi
  if [[ "${USE_LONGCLIP}" == "True" && ! -f "${LONGCLIP_MODEL}" ]]; then
    echo "[ERROR] LongCLIP checkpoint not found: ${LONGCLIP_MODEL}"
    has_error=1
  fi
  
  return $has_error
}

print_stage_info() {
  local stage_name="$1"
  local workspace="$2"
  local resume="$3"
  
  echo "=============================================="
  echo " ${stage_name}"
  echo "=============================================="
  echo "  GPUs: ${GPU_IDS} (${NUM_GPUS} GPUs)"
  echo "  Workspace: ${workspace}"
  echo "  Resume from: ${resume:-'None (fresh start)'}"
  echo "  LongCLIP: ${USE_LONGCLIP}"
  echo "=============================================="
}

run_validation() {
  local ckpt_path="$1"
  local output_dir="$2"
  local sample_id="$3"
  
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[WARN] Checkpoint not found for validation: ${ckpt_path}"
    return 1
  fi
  
  echo "[INFO] Running validation for sample: ${sample_id}"
  
  # 创建临时TSV
  local temp_tsv=$(mktemp /tmp/validate_XXXXXX.tsv)
  
  python - "${TEST_LIST}" "${sample_id}" "${temp_tsv}" <<'PY'
import csv
import sys
test_tsv, sample_id, output_tsv = sys.argv[1:]
with open(test_tsv, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    fieldnames = reader.fieldnames
    rows = [row for row in reader if row.get("obj_id", "").strip() == sample_id]
if not rows:
    sys.exit(1)
with open(output_tsv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
PY
  
  if [[ $? -ne 0 ]]; then
    echo "[WARN] Sample ${sample_id} not found in test set"
    rm -f "${temp_tsv}"
    return 1
  fi
  
  python texture.py \
    --ckpt_path "${ckpt_path}" \
    --output_dir "${output_dir}" \
    --use_longclip "${USE_LONGCLIP}" \
    --longclip_model "${LONGCLIP_MODEL}" \
    --longclip_context_length "${LONGCLIP_CONTEXT_LENGTH}" \
    --use_material "${USE_MATERIAL}" \
    --mesh_path "../datasets/texverse_meshes" \
    --pointcloud_dir "${POINTCLOUD_DIR}" \
    --tsv_path "${temp_tsv}" \
    --caption_field "${CAPTION_FIELD}" \
    --use_ggca "True" \
    --use_text_adapter "True" \
    --num_gpus 1 \
    --gpu_ids "0"
  
  rm -f "${temp_tsv}"
}
