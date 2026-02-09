#!/bin/bash

# ========================================================
# 脚本名称: render_gen_aligned.sh
# 功能: 读取 GT 渲染时保存的 transforms.json，相同视角渲染生成的贴图/模型
# 模式: 统一输出 lit（PBR+HDRI）和 unlit（各贴图 emission）
# 需要: manifest.tsv 至少包含 obj_id、mesh、albedo，可选 rough/metal/normal/transforms
# 支持: 多GPU多进程并行渲染
# ========================================================

# 1. 初始化 Conda 环境
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活安装了 bpy 的环境
conda activate blender

# 2. 路径配置（按需修改）

# 生成结果的 manifest（路径支持相对路径）
MANIFEST_PATH="../experiments/normal_mini/generated_manifest.tsv"

# GT 渲染结果根目录（内部包含 {obj_id}/transforms.json）
GT_ROOT="../datasets/texverse_rendered_test"
TRANSFORMS_SUBDIR=""   # 可选：如果 transforms 在子目录内，设置该值

# 输出目录
OUT_ROOT="../experiments/normal_mini/texverse_gen_renders"

# HDRI（lit 渲染必需，可配置多个）
HDRI_PATHS=(
  "../datasets/hdri/studio_small_09_2k.exr",
  "../datasets/hdri/shanghai_bund_2k.exr",
  "../datasets/hdri/sunflowers_puresky_2k.exr"
)

HDRI_ARGS=()
for hdri in "${HDRI_PATHS[@]}"; do
  HDRI_ARGS+=(--hdri "$hdri")
done

# 3. 多GPU配置
GPU_IDS="0,1"                # 使用的GPU ID列表，逗号分隔
NUM_GPUS=2                 # 使用的GPU数量
WORKERS_PER_GPU=1          # 每GPU进程数：'auto' 自动计算，或指定数字

echo "=============================================="
echo "Render Generated Assets with GT Cameras"
echo "Manifest:   $MANIFEST_PATH"
echo "GT Root:    $GT_ROOT"
echo "Output:     $OUT_ROOT/{obj_id}/(lit|unlit)"
echo "HDRIs:      ${HDRI_PATHS[*]}"
echo "GPUs:       $GPU_IDS (num=$NUM_GPUS)"
echo "Workers:    $WORKERS_PER_GPU per GPU"
echo "=============================================="

# 4. 执行渲染
python ./scripts/render_gen_aligned.py \
  --manifest "$MANIFEST_PATH" \
  --gt-root "$GT_ROOT" \
  --transforms-subdir "$TRANSFORMS_SUBDIR" \
  --out-root "$OUT_ROOT" \
  "${HDRI_ARGS[@]}" \
  --samples 64 \
  --hdri-strength 1.0 \
  --save-blend \
  --background transparent \
  --gpu-ids "$GPU_IDS" \
  --num-gpus "$NUM_GPUS" \
  --workers-per-gpu "$WORKERS_PER_GPU"

echo "Done."
