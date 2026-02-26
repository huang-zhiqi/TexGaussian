# 1. 初始化 Conda 环境
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate texgaussian

export TORCH_CUDA_ARCH_LIST="8.6;8.9"
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH="$CONDA_PREFIX/include:${CPATH}"
export C_INCLUDE_PATH="$CONDA_PREFIX/include:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

# ================= 配置区 =================

# 实验名称
EXP_NAME="texverse_stage2_finetune"

# TSV 路径 (批量模式使用)
BATCH_TSV="../experiments/common_splits/test.tsv"

# 输出根目录
OUTPUT_ROOT="../experiments/${EXP_NAME}/attention_maps"

# 文本字段（caption_short 或 caption_long）
CAPTION_FIELD="caption_long"

# 最大处理样本数 (-1 表示处理所有样本)
MAX_SAMPLES=2

# 预训练权重路径
CKPT_PATH="../experiments/texverse_stage2_finetune/2026.02.25-23:15:19_lr_5e-05_num_views_8/best_ckpt/model.safetensors"

# 点云目录
POINTCLOUD_DIR="../datasets/texverse_pointcloud_npz"

# 图像 DPI
DPI=200

# 3D 关键词热力图数量
N_KEYWORDS=6

# ==========================================

echo "Starting Attention Map Visualization..."
echo "TSV:        ${BATCH_TSV}"
echo "Output:     ${OUTPUT_ROOT}"
echo "Caption:    ${CAPTION_FIELD}"
echo "Max:        ${MAX_SAMPLES}"
echo "Checkpoint: ${CKPT_PATH}"
echo "DPI:        ${DPI}"
echo "Keywords:   ${N_KEYWORDS}"

python3 scripts/visualize_attention_map.py \
    --ckpt_path "${CKPT_PATH}" \
    --tsv_path "${BATCH_TSV}" \
    --caption_field "${CAPTION_FIELD}" \
    --max_samples "${MAX_SAMPLES}" \
    --output_dir "${OUTPUT_ROOT}" \
    --pointcloud_dir "${POINTCLOUD_DIR}" \
    --dpi "${DPI}" \
    --n_keywords "${N_KEYWORDS}"
