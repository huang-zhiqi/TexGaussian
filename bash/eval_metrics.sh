#!/bin/bash
# 计算 GT 与生成结果之间的指标，参数与 scripts/eval_metrics.py 保持一致

# ============================================
# 指标说明 (↑ = 越大越好, ↓ = 越小越好)
# ============================================
#
# === Unlit 通道指标 (逐像素比较) ===
# Albedo_PSNR_Masked      ↑  峰值信噪比，衡量颜色重建质量
# Albedo_SSIM_Masked      ↑  结构相似性，衡量感知质量
# Albedo_LPIPS_Masked     ↓  感知损失，越小越相似
# Roughness_L1            ↓  粗糙度通道的L1误差
# Metallic_L1             ↓  金属度通道的L1误差
# Normal_MeanAngularError ↓  法线角度误差（度），越小越准确
#
# === Lit 分布指标 (整体分布比较) ===
# FID                     ↓  Fréchet Inception Distance，越小分布越接近
# KID_mean                ↓  Kernel Inception Distance，越小越好
# KID_std                 -  KID的标准差，仅供参考
#
# === Lit 语义指标 (CLIP相似度) ===
# CLIP_Image_Similarity   ↑  图像级CLIP相似度 (Gen vs GT)
# CLIP_Text_Similarity    ↑  图像-文本CLIP相似度 (Gen vs caption_short)
# LongCLIP_Text_Similarity↑  长文本CLIP相似度 (Gen vs caption_long)
#
# === 多视角一致性指标 (检测Janus问题) ===
# CrossView_LPIPS         ↓  相邻视角感知一致性，越小越一致
# CrossView_L1            ↓  相邻视角像素一致性
# Normal_Consistency      ↑  法线跨视角一致性，越大越好
# Normal_Distribution_Div ↓  法线分布散度，越小说明无多面问题
# Reproj_L1               ↓  重投影误差（需深度图）
# Reproj_LPIPS            ↓  重投影感知误差（需深度图）
#
# ============================================

# 不使用 set -e，防止脚本自动退出
set -uo pipefail

# 初始化 conda（非交互 shell 必须手动做）
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-metric}"
set +u
conda activate "$ENV_NAME"
set -u

# 默认参数，可通过环境变量或位置参数覆盖
# 位置参数: $1=EXPERIMENT_NAME, $2=METRICS(可选，优先级最高)
EXPERIMENT_NAME="${1:-texgaussian_baseline_mini}"
BASE_GT_DIR="${BASE_GT_DIR:-"../datasets/texverse_rendered_test"}"
BASE_GEN_DIR="${BASE_GEN_DIR:-"../experiments/${EXPERIMENT_NAME}/texverse_gen_renders"}"
# If LIT_SUBDIR contains HDRI subfolders, eval_metrics.py will combine all images
# from all HDRIs to compute FID/KID (more statistically stable).
# Lit metrics are recorded directly (e.g., FID, KID, CLIP_Image_Score).
LIT_SUBDIR="${LIT_SUBDIR:-"lit"}"
UNLIT_SUBDIR="${UNLIT_SUBDIR:-"unlit"}"
# 降低默认批量大小以避免 OOM (从 8 改为 4)
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"
# KID subset size: 100 is recommended for stable estimation
KID_SUBSET_SIZE="${KID_SUBSET_SIZE:-100}"
CLIP_MODEL="${CLIP_MODEL:-"ViT-B/32"}"
LONGCLIP_MODEL="${LONGCLIP_MODEL:-"../third_party/Long-CLIP/checkpoints/longclip-L.pt"}"
LONGCLIP_ROOT="${LONGCLIP_ROOT:-"../third_party/Long-CLIP"}"
LONGCLIP_CONTEXT_LENGTH="${LONGCLIP_CONTEXT_LENGTH:-248}"
OUTPUT="${OUTPUT:-"../experiments/${EXPERIMENT_NAME}/metrics.json"}"
PROMPTS_FILE="${PROMPTS_FILE:-"../experiments/${EXPERIMENT_NAME}/generated_manifest.tsv"}"

# 多视角一致性指标参数 (CVPR 2025 / ECCV 2024-2025 标准)
# 用于检测 Janus (多面怪) 问题和纹理闪烁
CONSISTENCY_PAIRS="${CONSISTENCY_PAIRS:-10}"           # 每个物体采样的相邻视角对数量
CONSISTENCY_CHANNEL="${CONSISTENCY_CHANNEL:-albedo}" # 一致性评估使用的通道: albedo/lit/normal

# 启用 CUDA 同步调用（更稳定但更慢）
export CUDA_LAUNCH_BLOCKING=1

# 指标选择:
# - 预设: all | pixel (psnr,ssim,lpips) | dist (fid,kid) | semantic (clip) | consistency (多视角)
# - 自定义逗号列表: 例如 psnr,ssim,clip,consistency
# - 一致性指标包括: CrossView_LPIPS, CrossView_L1, Normal_Consistency, Reproj_L1 (需深度图)
METRICS="${METRICS:-all}"

echo "=============================================="
echo "Evaluate Metrics"
echo "Experiment:         $EXPERIMENT_NAME"
echo "GT Dir:             $BASE_GT_DIR"
echo "Gen Dir:            $BASE_GEN_DIR"
echo "Lit Subdir:         $LIT_SUBDIR"
echo "Unlit Subdir:       $UNLIT_SUBDIR"
echo "Batch size:         $BATCH_SIZE"
echo "Device:             $DEVICE"
echo "KID subset:         $KID_SUBSET_SIZE"
echo "CLIP model:         $CLIP_MODEL"
echo "LongCLIP ckpt:      $LONGCLIP_MODEL"
echo "LongCLIP root:      $LONGCLIP_ROOT"
echo "LongCLIP ctx:       $LONGCLIP_CONTEXT_LENGTH"
echo "Prompts file:       $PROMPTS_FILE"
echo "Metrics:            $METRICS"
echo "Consistency pairs:  $CONSISTENCY_PAIRS"
echo "Consistency chan:   $CONSISTENCY_CHANNEL"
echo "Output:             $OUTPUT"
echo "Conda env:          $ENV_NAME"
echo "=============================================="

# 创建日志文件
LOG_DIR="../experiments/${EXPERIMENT_NAME}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/eval_metrics_$(date +%Y%m%d_%H%M%S).log"
echo "Log file:      $LOG_FILE"
echo "=============================================="

# 设置 PyTorch 内存管理选项以提高稳定性
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# 禁用 tokenizers 并行以防止死锁
export TOKENIZERS_PARALLELISM="false"
# 可选：如果仍然崩溃，取消下面这行的注释以启用同步 CUDA 调用（会更慢但更稳定）
# export CUDA_LAUNCH_BLOCKING=1

# 运行评估脚本，同时输出到终端和日志文件
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_metrics.py \
  --experiment_name "$EXPERIMENT_NAME" \
  --base_gt_dir "$BASE_GT_DIR" \
  --base_gen_dir "$BASE_GEN_DIR" \
  --lit_subdir "$LIT_SUBDIR" \
  --unlit_subdir "$UNLIT_SUBDIR" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --kid_subset_size "$KID_SUBSET_SIZE" \
  --clip_model "$CLIP_MODEL" \
  --longclip_model "$LONGCLIP_MODEL" \
  --longclip_root "$LONGCLIP_ROOT" \
  --longclip_context_length "$LONGCLIP_CONTEXT_LENGTH" \
  --prompts_file "$PROMPTS_FILE" \
  --metrics "$METRICS" \
  --consistency_pairs "$CONSISTENCY_PAIRS" \
  --consistency_channel "$CONSISTENCY_CHANNEL" \
  --output "$OUTPUT" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation complete!"
else
    echo "Evaluation FAILED with exit code $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
fi
echo "=============================================="
