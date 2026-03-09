#!/bin/bash
# =============================================================================
# 阶段2：全模型端到端微调（可选）
# =============================================================================
# 目标：解冻预训练 UNet backbone，与所有新模块一起端到端微调
# 前置条件：阶段1完成，所有指标正常
# 预期时间：15 epochs，约 4-5 小时（2卡3090），2-3 小时（4卡3090）
#
# ⚠️ 注意：这个阶段是可选的。如果阶段1效果已经满意，可以跳过。
# 全模型微调有过拟合风险，请密切监控指标。
#
# 关键指标：PSNR 在阶段1基础上进一步提升，所有 loss 继续下降（非上升）
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
# 阶段2 特定配置
# =========================
STAGE_NAME="Stage2_Full_Finetune"
EXP_NAME="texverse_stage2_finetune_v11"
WORKSPACE="${EXPERIMENTS_ROOT}/${EXP_NAME}"

# 从阶段1的checkpoint继续
STAGE1_WORKSPACE="${EXPERIMENTS_ROOT}/texverse_stage1_new_modules_v11"

# 自动查找最新的 checkpoint（按修改时间排序）
find_latest_ckpt() {
  local search_dir="$1"
  local ckpt=""
  # 优先找 best_ckpt（按修改时间排序，取最新）
  ckpt=$(find "${search_dir}" -name "model.safetensors" -path "*/best_ckpt/*" -type f 2>/dev/null | \
         xargs -I {} stat --format='%Y %n' {} 2>/dev/null | \
         sort -rn | head -1 | cut -d' ' -f2-)

  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    # 再找任意 model.safetensors（按修改时间排序）
    ckpt=$(find "${search_dir}" -name "model.safetensors" -type f 2>/dev/null | \
           xargs -I {} stat --format='%Y %n' {} 2>/dev/null | \
           sort -rn | head -1 | cut -d' ' -f2-)
  fi
  echo "${ckpt}"
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
# 1. 优先检查 stage2 自己的 checkpoint（恢复中断的训练）
# 2. 如果没有，使用 stage1 的 checkpoint（开始新训练）
STAGE2_CKPT=$(find_latest_ckpt "${WORKSPACE}")
STAGE1_CKPT=$(find_latest_ckpt "${STAGE1_WORKSPACE}")

if [[ -n "${STAGE2_CKPT}" && -f "${STAGE2_CKPT}" ]]; then
  RESUME_CKPT="${STAGE2_CKPT}"
  RESUME_MODE="stage2"
  echo "[INFO] 发现 Stage2 checkpoint，将恢复中断的训练"
elif [[ -n "${STAGE1_CKPT}" && -f "${STAGE1_CKPT}" ]]; then
  RESUME_CKPT="${STAGE1_CKPT}"
  RESUME_MODE="stage1"
  echo "[INFO] 从 Stage1 checkpoint 开始新的 Stage2 训练"
else
  RESUME_CKPT=""
  RESUME_MODE="none"
fi

# 特征开关 - 保持所有模块启用，解冻 base
USE_TEXT_ADAPTER="True"
USE_GGCA="True"
FREEZE_BASE="False"         # ⚠️ 关键：解冻基础模型！

# 解冻开关 - 与 Stage1 保持一致（仍需要三层参数分组）
UNFREEZE_ATTN_KV="True"    # CA K,V 投影
UNFREEZE_ATTN_QO="True"    # CA Q,O 投影
UNFREEZE_NORMS="True"      # GroupNorm/LayerNorm
ADAPT_LR_SCALE=0.1         # Tier 2 相对 Tier 1 的学习率比例

# 优化配置 - 使用较低学习率避免破坏预训练权重
BATCH_SIZE=4               # 3090 24GB 足以支持 BS=2（Stage1 用 BS=4, ~8GB 显存）
GRAD_ACC=1                 # 有效batch = 2 * NUM_GPUS * 2 = 16（与 Stage1 一致）
NUM_EPOCHS=10              # 全模型微调过拟合风险高，10 epoch 较安全
LR=1e-4                    # Tier 1 (新模块) 学习率
                            # Tier 2 (CA K/V/Q/O+norms): LR*0.1 = 1e-5
                            # Tier 3 (backbone): LR*0.01 = 1e-6（main.py 硬编码）
LAMBDA_LPIPS=2.0           # 与 Stage1 保持一致
EMA_RATE=0.9999            # 与 Stage1 保持一致

# =========================
# 验证并打印配置
# =========================
if ! validate_paths; then
  echo "[ERROR] Path validation failed. Exiting."
  exit 1
fi

# 检查 checkpoint
if [[ -z "${RESUME_CKPT}" || ! -f "${RESUME_CKPT}" ]]; then
  echo "=============================================="
  echo " [ERROR] 未找到可用的 checkpoint!"
  echo "=============================================="
  echo ""
  echo " 请确保已完成阶段1训练，或手动指定 checkpoint 路径:"
  echo ""
  echo "   RESUME_CKPT=/path/to/model.safetensors ./train_stages/stage2_finetune.sh"
  echo ""
  echo " 或者先运行阶段1:"
  echo "   ./train_stages/stage1_new_modules.sh"
  echo ""
  exit 1
fi

print_stage_info "${STAGE_NAME}" "${WORKSPACE}" "${RESUME_CKPT}"

echo ""
echo "  ⚠️  警告: 此阶段将解冻基础模型进行全参数微调（~297M 参数）"
echo "  ⚠️  这可能导致过拟合，请密切监控指标"
echo ""
echo "  Resume Mode: ${RESUME_MODE} (stage2=恢复训练, stage1=新训练)"
echo "  TextAdapter: ${USE_TEXT_ADAPTER}"
echo "  GGCA: ${USE_GGCA}"
echo "  Freeze Base: ${FREEZE_BASE}"
echo "  Unfreeze K,V: ${UNFREEZE_ATTN_KV}"
echo "  Unfreeze Q,O: ${UNFREEZE_ATTN_QO}"
echo "  Unfreeze Norms: ${UNFREEZE_NORMS}"
echo "  Adapt LR Scale: ${ADAPT_LR_SCALE}"
echo "  Learning Rate: ${LR} (Tier1=${LR}, Tier2=${LR}*${ADAPT_LR_SCALE}, Tier3=${LR}*0.01)"
echo "  Epochs: ${NUM_EPOCHS}"
echo ""

# 确认开始
read -p "开始阶段2全模型微调? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "已取消"
  exit 0
fi

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
  --unfreeze_attn_kv "${UNFREEZE_ATTN_KV}"
  --unfreeze_attn_qo "${UNFREEZE_ATTN_QO}"
  --unfreeze_norms "${UNFREEZE_NORMS}"
  --adapt_lr_scale "${ADAPT_LR_SCALE}"
  --lambda_lpips "${LAMBDA_LPIPS}"
  --ema_rate "${EMA_RATE}"
  --resume "${RESUME_CKPT}"
)

# 如果是恢复中断的训练（stage2 自己的 checkpoint），添加 training state 恢复
if [[ "${RESUME_MODE}" == "stage2" ]]; then
  TRAINING_STATE_PATH=$(resolve_training_state_path "${RESUME_CKPT}" || echo "")
  if [[ -n "${TRAINING_STATE_PATH}" && -f "${TRAINING_STATE_PATH}" ]]; then
    ARGS+=(--resume_training_state "${TRAINING_STATE_PATH}")
    echo "[INFO] Will restore training state from: ${TRAINING_STATE_PATH}"
  else
    echo "[WARN] training_state.pt not found near checkpoint: ${RESUME_CKPT}, will train from epoch 0"
  fi
fi

echo "[INFO] Starting Stage 2 Training (Full Finetune)..."
echo "[INFO] Memory optimization: PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[INFO] Gradient accumulation: ${GRAD_ACC}"

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
echo " 阶段2训练完成！"
echo "=============================================="
echo ""
echo " 检查要点:"
echo "   1. 与阶段1对比，PSNR 是否有提升"
echo "   2. 所有 loss 是否继续下降（而非过拟合上升）"
echo "   3. 生成效果是否满意"
echo ""
echo " 监控指标:"
echo "   tensorboard --logdir ../experiments"
echo ""
echo " 批量推理:"
echo "   ./inference_batch.sh"
echo "=============================================="
