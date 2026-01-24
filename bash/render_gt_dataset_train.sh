#!/bin/bash

# ========================================================
# 脚本名称: render_gt_dataset_train.sh
# 功能: 使用 render_gt_dataset.py 渲染 Ground Truth (Lit + Unlit)
# 说明: 渲染范围由 MANIFEST_PATH 决定（可用于 train/test/val）
# ========================================================

# 1. 初始化 Conda 环境
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活安装了 bpy 的环境
conda activate blender

# 2. 设置路径变量

# 输入: 划分好的数据清单
MANIFEST_PATH="../experiments/common_splits/train.tsv"

# 输出: 数据集根目录 (脚本会自动创建 {obj_id}/lit 与 {obj_id}/unlit)
OUT_ROOT="../datasets/texverse_rendered"

# 并行配置: GPU 数量
NUM_GPUS=2
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))
WORKERS_PER_GPU="auto"          # 每GPU进程数：'auto' 自动计算，或指定数字

# 资源: HDRI 环境贴图 (lit 渲染必需，可配置多个)
#HDRI_PATHS=(
#  "../datasets/hdri/rogland_sunset_4k.exr"
#)

HDRI_ARGS=()
for hdri in "${HDRI_PATHS[@]}"; do
  HDRI_ARGS+=(--hdri "$hdri")
done

echo "=========================================="
echo "Start Rendering GT (Lit+Unlit)"
echo "Manifest: $MANIFEST_PATH"
echo "Output:   $OUT_ROOT/{obj_id}/(lit|unlit)"
echo "HDRIs:    ${HDRI_PATHS[*]}"
echo "GPUs:     $GPU_IDS (num=$NUM_GPUS)"
echo "Workers:  $WORKERS_PER_GPU per GPU"
echo "=========================================="

# 3. 执行 Python 渲染命令
python ./scripts/render_gt_dataset.py \
  --manifest "$MANIFEST_PATH" \
  --out-root "$OUT_ROOT" \
  --resolution 512 \
  --views 32 \
  --samples 1 \
  --seed 42 \
  --save-blend \
  --background transparent \
  --unlit-only \
  --gpu-ids "$GPU_IDS" \
  --num-gpus "$NUM_GPUS" \
  --workers-per-gpu "$WORKERS_PER_GPU"

echo "Done."

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate texgaussian
