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
# CLIP_FID                ↓  基于CLIP特征的FID (clean-fid)，比Inception FID更贴近感知质量
# CMMD                    ↓  CLIP Maximum Mean Discrepancy，不假设高斯分布，小样本下更稳定
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

print_sep() {
  echo "=============================================="
}

# 初始化 conda（非交互 shell 必须手动做）
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found. Please install Conda or initialize your shell first."
  exit 127
fi

if ! CONDA_BASE="$(CONDA_NO_PLUGINS=true conda info --base 2>/dev/null)"; then
  echo "[ERROR] Failed to query Conda base path. Try: CONDA_NO_PLUGINS=true conda info --base"
  exit 1
fi

if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  echo "[ERROR] Cannot find conda init script: $CONDA_BASE/etc/profile.d/conda.sh"
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="${ENV_NAME:-metric}"
set +u
if ! conda activate "$ENV_NAME"; then
  echo "[ERROR] Failed to activate conda env: $ENV_NAME"
  exit 1
fi
set -u

# 默认参数，可通过环境变量或位置参数覆盖
# 位置参数: $1=EXPERIMENT_NAME, $2=METRICS(可选，优先级最高), GPU_ID环境变量选择GPU
EXPERIMENT_NAME="${1:-texgaussian_mini}"
BASE_GT_DIR="${BASE_GT_DIR:-"../datasets/texverse_rendered_test"}"
BASE_GEN_DIR="${BASE_GEN_DIR:-"../experiments/${EXPERIMENT_NAME}/texverse_gen_renders"}"
# If LIT_SUBDIR contains HDRI subfolders, eval_metrics.py will combine all images
# from all HDRIs to compute FID/KID (more statistically stable).
# Lit metrics are recorded directly (e.g., FID, KID, CLIP_Image_Score).
LIT_SUBDIR="${LIT_SUBDIR:-"lit"}"
UNLIT_SUBDIR="${UNLIT_SUBDIR:-"unlit"}"
# 降低默认批量大小以避免 OOM (从 8 改为 4)
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
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

# FID/KID 白色背景: 将 RGBA 图像合成到白色背景上再计算 FID/KID
# 默认黑色背景下 ~80% 背景像素会主导 Inception 特征，开启后可降低背景噪声
FID_WHITE_BG="${FID_WHITE_BG:-true}"

# 启用 CUDA 同步调用（更稳定但更慢）
export CUDA_LAUNCH_BLOCKING=1

# 指标选择:
# - 预设: all | pixel (psnr,ssim,lpips) | dist (fid,kid,clip_fid,cmmd) | semantic (clip) | consistency (多视角)
# - 自定义逗号列表: 例如 psnr,ssim,clip,clip_fid,cmmd,consistency
# - 新增分布指标:
#   CLIP_FID: 使用 CLIP 特征替代 Inception 的 FID (clean-fid, pip install clean-fid)
#   CMMD: CLIP Maximum Mean Discrepancy (clip-mmd, pip install clip-mmd)
# - 一致性指标包括: CrossView_LPIPS, CrossView_L1, Normal_Consistency, Reproj_L1 (需深度图)
METRICS="${METRICS:-all}"

print_sep
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
echo "FID white bg:       $FID_WHITE_BG"
echo "Output:             $OUTPUT"
echo "GPU ID:             $GPU_ID"
echo "Conda env:          $ENV_NAME"
print_sep

# 创建日志文件
LOG_DIR="../experiments/${EXPERIMENT_NAME}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/eval_metrics_$(date +%Y%m%d_%H%M%S).log"
echo "Log file:      $LOG_FILE"
print_sep

# 设置 PyTorch 内存管理选项以提高稳定性
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# 禁用 tokenizers 并行以防止死锁
export TOKENIZERS_PARALLELISM="false"
# 可选：如果仍然崩溃，取消下面这行的注释以启用同步 CUDA 调用（会更慢但更稳定）
# export CUDA_LAUNCH_BLOCKING=1

# 运行评估脚本，同时输出到终端和日志文件
# 使用数组构造命令，避免续行符误改导致 shell 把下一行当成新命令执行。
EVAL_CMD=(
  python -u scripts/eval_metrics.py
  --gpu "$GPU_ID"
  --experiment_name "$EXPERIMENT_NAME"
  --base_gt_dir "$BASE_GT_DIR"
  --base_gen_dir "$BASE_GEN_DIR"
  --lit_subdir "$LIT_SUBDIR"
  --unlit_subdir "$UNLIT_SUBDIR"
  --batch_size "$BATCH_SIZE"
  --device "$DEVICE"
  --kid_subset_size "$KID_SUBSET_SIZE"
  --clip_model "$CLIP_MODEL"
  --longclip_model "$LONGCLIP_MODEL"
  --longclip_root "$LONGCLIP_ROOT"
  --longclip_context_length "$LONGCLIP_CONTEXT_LENGTH"
  --prompts_file "$PROMPTS_FILE"
  --metrics "$METRICS"
  --consistency_pairs "$CONSISTENCY_PAIRS"
  --consistency_channel "$CONSISTENCY_CHANNEL"
  --output "$OUTPUT"
)

# 条件追加 --fid_white_bg 标志
if [[ "$FID_WHITE_BG" == "true" || "$FID_WHITE_BG" == "True" || "$FID_WHITE_BG" == "1" ]]; then
  EVAL_CMD+=(--fid_white_bg)
fi

"${EVAL_CMD[@]}" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

print_sep
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation complete!"
else
    echo "Evaluation FAILED with exit code $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
fi
print_sep

exit "$EXIT_CODE"
