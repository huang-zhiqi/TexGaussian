#!/bin/bash
# =============================================================================
# 阶段2：FID/CLIP 导向低LR微调（可选）
# =============================================================================
# 目标：在 Stage1 基础上做小步微调，进一步优化 FID/CLIP 指标
# 前置条件：阶段1完成，所有指标正常
# 预期时间：8 epochs，约 2-4 小时（取决于GPU数量）
#
# ⚠️ 注意：该阶段可选。若 Stage1 的 FID/KID/CLIP_FID 已满足目标，可跳过。
# 该阶段默认低学习率 + FID 选 best，避免“PSNR升而FID降”。
#
# 关键指标：FID/KID/CLIP_FID/CMMD 改善，同时 PSNR 不明显崩溃
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
STAGE_NAME="Stage2_FIDCLIP_Lite_Finetune"
EXP_NAME="texverse_stage2_fidclip_lite_v11"
WORKSPACE="${EXPERIMENTS_ROOT}/${EXP_NAME}"

# 从阶段1的checkpoint继续
STAGE1_WORKSPACE="${EXPERIMENTS_ROOT}/texverse_stage1_fidclip_v11"

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

# 特征开关 - 保持所有模块启用，进行低LR微调
USE_TEXT_ADAPTER="True"
USE_GGCA="True"
FREEZE_BASE="False"         # 解冻 base 做小步微调
FID_SAFE_MODE="False"       # 关键：若为 True，会强制 freeze_base=True
TRAIN_CONV_HEAD="False"

# 解冻开关 - 以稳态为主，避免过快漂移
UNFREEZE_ATTN_KV="True"    # CA K,V 投影
UNFREEZE_ATTN_QO="False"   # Stage2 默认继续关闭 Q,O
UNFREEZE_NORMS="True"      # GroupNorm/LayerNorm
ADAPT_LR_SCALE=0.02        # Tier 2 相对 Tier 1 的学习率比例（更保守）

# 优化配置 - 低学习率，防止分布指标反弹
BATCH_SIZE=2
GRAD_ACC=2
NUM_EPOCHS=8
LR=5e-5                    # Tier 1 学习率；Tier 3(base)=LR*0.01（main.py）
LAMBDA_LPIPS=0.5
EMA_RATE=0.9999            # 与 Stage1 保持一致

# Stage2 适当降低语义/分布损失权重，做平滑收敛
LAMBDA_CLIP_IMAGE=0.10
LAMBDA_CLIP_TEXT=0.03
LAMBDA_COLOR_STATS=0.03

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
echo "  ⚠️  警告: 此阶段会解冻 base 进行低LR微调，请优先看 FID/KID/CLIP_FID"
echo ""
echo "  Resume Mode: ${RESUME_MODE} (stage2=恢复训练, stage1=新训练)"
echo "  TextAdapter: ${USE_TEXT_ADAPTER}"
echo "  GGCA: ${USE_GGCA}"
echo "  Freeze Base: ${FREEZE_BASE}"
echo "  FID-safe mode: ${FID_SAFE_MODE}"
echo "  Train conv head: ${TRAIN_CONV_HEAD}"
echo "  Unfreeze K,V: ${UNFREEZE_ATTN_KV}"
echo "  Unfreeze Q,O: ${UNFREEZE_ATTN_QO}"
echo "  Unfreeze Norms: ${UNFREEZE_NORMS}"
echo "  Adapt LR Scale: ${ADAPT_LR_SCALE}"
echo "  Learning Rate: ${LR} (Tier1=${LR}, Tier2=${LR}*${ADAPT_LR_SCALE}, Tier3=${LR}*0.01)"
echo "  clip_loss_model: ${CLIP_LOSS_MODEL}"
echo "  lambda_clip_image: ${LAMBDA_CLIP_IMAGE}"
echo "  lambda_clip_text: ${LAMBDA_CLIP_TEXT}"
echo "  lambda_color_stats: ${LAMBDA_COLOR_STATS}"
echo "  alpha_gt_blend: ${ALPHA_GT_BLEND}"
echo "  best_selection_metric: ${BEST_SELECTION_METRIC}"
echo "  Epochs: ${NUM_EPOCHS}"
echo ""

# 确认开始
read -p "开始阶段2 FID/CLIP 低LR微调? (y/n) " -n 1 -r
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
  --fid_safe_mode "${FID_SAFE_MODE}"
  --train_conv_head "${TRAIN_CONV_HEAD}"
  --unfreeze_attn_kv "${UNFREEZE_ATTN_KV}"
  --unfreeze_attn_qo "${UNFREEZE_ATTN_QO}"
  --unfreeze_norms "${UNFREEZE_NORMS}"
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

echo "[INFO] Starting Stage2 FID/CLIP Lite Finetune..."
echo "[INFO] Memory optimization: PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[INFO] Gradient accumulation: ${GRAD_ACC}"
echo "[INFO] Objective = reconstruction + LPIPS + CLIP(image/text) + color_stats"
echo "[INFO] Checkpoint selection = ${BEST_SELECTION_METRIC}"

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
echo " 阶段2 FID/CLIP 微调完成！"
echo "=============================================="
echo ""
echo " 检查要点:"
echo "   1. FID/KID/CLIP_FID/CMMD 是否继续改善"
echo "   2. PSNR 是否保持稳定（不过度牺牲）"
echo "   3. eval_fid 是否优于 stage1 best"
echo ""
echo " 监控指标:"
echo "   tensorboard --logdir ../experiments"
echo ""
echo " 批量推理:"
echo "   ./inference_batch.sh"
echo "=============================================="
