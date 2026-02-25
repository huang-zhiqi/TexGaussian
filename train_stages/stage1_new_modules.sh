#!/bin/bash
# =============================================================================
# 阶段1：训练新增模块（TextAdapter + GGCA）
# =============================================================================
# 目标：冻结预训练 UNet backbone，训练新引入的模块
#   - TextAdapter: 单一embedding流，同时适配 LongCLIP 到 CrossAttention 和 GGCA
#   - GGCA: Geometry-Gated Cross-Attention，放在 conv_out 入口，enriches ALL features
#
# 前置条件：预训练基础 PBR 模型 (PBR_model.safetensors)
# 预期时间：20 epochs，约 2-3 小时（2卡3090）
# 新增模块总参数：~1.48M（TextAdapter 394K + GGCA 1.09M）
#
# v2 优化策略：
#   - 单一embedding流（TextAdapter适配所有路径）
#   - GGCA 在 conv_out 入口（enriches ALL output features）
#   - LPIPS权重 2.0（直接改善FID/KID）
#   - EMA rate 0.9999（小参数量更平滑的指数平均）
#   - gate_bias 0.0（GGCA更快发挥作用）
#   - epoch 20（避免过拟合）
#   - LPIPS loss 改用 pred_alphas 加权（和 MSE 一致，空洞由 mask_loss 处理）
#   - 不输出法线，渲染时直接使用 mesh 法线
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_config.sh"

# =========================
# 显存优化配置
# =========================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# =========================
# 阶段1 特定配置
# =========================
STAGE_NAME="Stage1_New_Modules"
EXP_NAME="texverse_stage1_new_modules_v6"
WORKSPACE="${EXPERIMENTS_ROOT}/${EXP_NAME}"

# 自动查找最新的 checkpoint（支持多个来源）
find_latest_ckpt() {
  local search_dirs=("$@")
  local ckpt=""

  for dir in "${search_dirs[@]}"; do
    if [[ ! -d "${dir}" ]]; then
      continue
    fi

    # 优先找 best_ckpt（按修改时间排序，取最新）
    ckpt=$(find "${dir}" -name "model.safetensors" -path "*/best_ckpt/*" -type f 2>/dev/null | \
           xargs -I {} stat --format='%Y %n' {} 2>/dev/null | \
           sort -rn | head -1 | cut -d' ' -f2-)

    if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
      echo "${ckpt}"
      return 0
    fi

    # 再找任意 model.safetensors（按修改时间排序）
    ckpt=$(find "${dir}" -name "model.safetensors" -type f 2>/dev/null | \
           xargs -I {} stat --format='%Y %n' {} 2>/dev/null | \
           sort -rn | head -1 | cut -d' ' -f2-)

    if [[ -n "${ckpt}" && -f "${ckpt}" ]]; then
      echo "${ckpt}"
      return 0
    fi
  done

  echo ""
  return 1
}

# 根据 checkpoint 路径自动定位 training_state.pt
resolve_training_state_path() {
  local ckpt_path="$1"
  local ckpt_dir=""
  local candidates=()

  if [[ -z "${ckpt_path}" ]]; then
    echo ""
    return 1
  fi

  ckpt_dir="$(dirname "${ckpt_path}")"
  candidates=(
    "${ckpt_dir}/training_state.pt"
    "${ckpt_dir}/best_ckpt/training_state.pt"
  )

  for state_path in "${candidates[@]}"; do
    if [[ -f "${state_path}" ]]; then
      echo "${state_path}"
      return 0
    fi
  done

  echo ""
  return 1
}

# 智能选择 checkpoint：
# 1. 优先检查 stage1 自己的 checkpoint（恢复中断的训练）
# 2. 如果没有，使用预训练基础模型（开始新训练）
STAGE1_CKPT=$(find_latest_ckpt "${WORKSPACE}" || echo "")

if [[ -n "${STAGE1_CKPT}" && -f "${STAGE1_CKPT}" ]]; then
  RESUME_CKPT="${STAGE1_CKPT}"
  RESUME_MODE="stage1"
  echo "[INFO] 发现 Stage1 checkpoint，将恢复中断的训练"
else
  RESUME_CKPT="${BASE_PRETRAINED_CKPT}"
  RESUME_MODE="pretrained"
  echo "[INFO] 从预训练模型开始新的 Stage1 训练"
fi

# 特征开关 - 启用所有新模块，冻结 base
USE_TEXT_ADAPTER="True"    # LongCLIP 特征适配（单一embedding流，同时适配CrossAttention和GGCA）
USE_GGCA="True"            # Geometry-Gated Cross-Attention（conv_out入口，enriches ALL features）
FREEZE_BASE="True"         # 冻结基础模型，只训练 GGCA/TextAdapter

# 优化配置
BATCH_SIZE=1
GRAD_ACC=4                 # 有效batch = 1 * NUM_GPUS * 4
NUM_EPOCHS=20              # 20 epochs足够（避免过拟合）
LR=4e-4                    # 标准学习率
LAMBDA_LPIPS=2.0           # LPIPS感知损失权重（直接提升FID/KID）
EMA_RATE=0.9999            # EMA平滑率（小参数量下需要更平滑的平均）

# =========================
# 验证并打印配置
# =========================
if ! validate_paths; then
  echo "[ERROR] Path validation failed. Exiting."
  exit 1
fi

# 检查预训练 checkpoint
if [[ -z "${RESUME_CKPT}" || ! -f "${RESUME_CKPT}" ]]; then
  echo "=============================================="
  echo " [ERROR] 未找到可用的 checkpoint!"
  echo "=============================================="
  echo ""
  echo " 请确保预训练模型存在: ${BASE_PRETRAINED_CKPT}"
  echo " 或手动指定 checkpoint 路径:"
  echo ""
  echo "   RESUME_CKPT=/path/to/model.safetensors ./train_stages/stage1_new_modules.sh"
  echo ""
  exit 1
fi

print_stage_info "${STAGE_NAME}" "${WORKSPACE}" "${RESUME_CKPT}"

echo "  Resume Mode: ${RESUME_MODE} (stage1=恢复训练, pretrained=新训练)"
echo "  TextAdapter: ${USE_TEXT_ADAPTER} (单一embedding流，同时适配CA+GGCA)"
echo "  GGCA: ${USE_GGCA} (conv_out入口)"
echo "  Freeze Base: ${FREEZE_BASE}"
echo "  lambda_lpips: ${LAMBDA_LPIPS}"
echo "  ema_rate: ${EMA_RATE}"
echo "  Learning Rate: ${LR}"
echo "  Epochs: ${NUM_EPOCHS}"
echo ""

# 确认开始
read -p "开始阶段1训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "已取消"
  exit 0
fi

# =========================
# 数据预检
# =========================
echo "[INFO] Running dataset preflight checks..."
python - "${TRAIN_LIST}" "${TRAIN_IMAGE_DIR}" "${TEST_LIST}" "${TEST_IMAGE_DIR}" "${POINTCLOUD_DIR}" "${CAPTION_FIELD}" "${USE_MATERIAL}" <<'PY'
import csv
import glob
import os
import sys
import numpy as np

train_tsv, train_image_dir, test_tsv, test_image_dir, pointcloud_dir, caption_field, use_material = sys.argv[1:]
use_material = use_material.lower() == "true"

with open(train_tsv, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    cols = set(reader.fieldnames or [])
    missing = sorted({"obj_id"} - cols)
    if missing:
        raise SystemExit(f"[ERROR] train TSV missing required columns: {missing}")
    if caption_field not in cols:
        raise SystemExit(f"[ERROR] caption field '{caption_field}' not in TSV columns")

def has_multiview_channel(image_obj_dir, channel):
    aliases = [channel]
    if channel == "normal":
        aliases = ["normal", "normals"]
    roots = [os.path.join(image_obj_dir, "unlit"), image_obj_dir]
    for root in roots:
        for c in aliases:
            patterns = [
                os.path.join(root, f"*_{c}.png"),
                os.path.join(root, f"{c}_*.png"),
                os.path.join(root, c, "*.png"),
            ]
            for pat in patterns:
                if glob.glob(pat):
                    return True
    return False

def first_uid(tsv_path):
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        row = next(reader, None)
    if row is None:
        raise SystemExit(f"[ERROR] TSV is empty: {tsv_path}")
    uid = (row.get("obj_id") or "").strip()
    return uid

uid = first_uid(train_tsv)
pc_path = os.path.join(pointcloud_dir, uid + ".npz")
if not os.path.isfile(pc_path):
    raise SystemExit(f"[ERROR] missing pointcloud for first sample: {pc_path}")

cam_dir = os.path.join(train_image_dir, uid)

print(f"[INFO] Preflight passed for stage 1")
PY

# =========================
# 开始训练
# =========================
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
  --lambda_lpips "${LAMBDA_LPIPS}"
  --ema_rate "${EMA_RATE}"
  --resume "${RESUME_CKPT}"
)

# 如果是恢复中断的训练，添加 training state 恢复
if [[ "${RESUME_MODE}" == "stage1" ]]; then
  TRAINING_STATE_PATH=$(resolve_training_state_path "${RESUME_CKPT}" || echo "")
  if [[ -n "${TRAINING_STATE_PATH}" && -f "${TRAINING_STATE_PATH}" ]]; then
    ARGS+=(--resume_training_state "${TRAINING_STATE_PATH}")
    echo "[INFO] Will restore training state from: ${TRAINING_STATE_PATH}"
  else
    echo "[WARN] training_state.pt not found near checkpoint: ${RESUME_CKPT}, will train from epoch 0"
  fi
fi

echo "[INFO] Starting Stage 1 Training..."
echo "[INFO] Memory optimization: PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[INFO] Gradient accumulation: ${GRAD_ACC}"
echo "[INFO] Trainable modules: TextAdapter(394K) + GGCA(1.09M) = ~1.48M params"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch "${ARGS[@]}" \
  2> >(grep -v -E '^libpng warning: eXIf: duplicate$' >&2)

TRAIN_EXIT_CODE=$?

if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] Training failed with exit code: ${TRAIN_EXIT_CODE}"
  exit $TRAIN_EXIT_CODE
fi

# =========================
# 训练完成
# =========================
echo ""
echo "=============================================="
echo " 阶段1训练完成！"
echo "=============================================="
echo ""
echo " 检查要点:"
echo "   1. albedo 颜色是否正确"
echo "   2. albedo_loss / material_loss 稳步下降"
echo "   3. PSNR 达到 22-25"
echo ""
echo " 下一步:"
echo "   a) 使用当前模型进行推理"
echo "   b) 运行阶段2进行全模型微调（可选）:"
echo "      ./train_stages/stage2_finetune.sh"
echo "=============================================="
