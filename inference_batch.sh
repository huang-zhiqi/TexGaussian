# 1. 初始化 Conda 环境
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活安装了 bpy 的环境
conda activate texgaussian

#（可选）只编译 3090/4090 的算力，缩短时间 #但其实没用
export TORCH_CUDA_ARCH_LIST="8.6;8.9"

# 让 nvcc / cicc / ptxas 都在 PATH 里
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$CONDA_PREFIX/bin:$PATH"

# 头/库路径（确保 crypt.h 等能被找到）
export CPATH="$CONDA_PREFIX/include:${CPATH}"
export C_INCLUDE_PATH="$CONDA_PREFIX/include:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

# ================= 配置区 =================

# 实验名称 (将作为文件夹名创建在 experiments 下)
EXP_NAME="texgaussian_baseline_mini"

# TSV 路径 (建议绝对路径，或相对于 texGaussian 的路径)
BATCH_TSV="../experiments/common_splits/test.tsv"

# 输出根目录 (指向 project_root/experiments/EXP_NAME)
# 假设脚本在 project_root/texGaussian 下运行
OUTPUT_ROOT="../experiments/${EXP_NAME}"

# 文本字段（caption_short 或 caption_long）
CAPTION_FIELD="caption_short"

# 是否使用 LongCLIP（True/False）
# LongCLIP 支持更长的文本描述（最长248 tokens），适合 caption_long
# 标准 CLIP 上下文长度为 77 tokens，适合 caption_short
USE_LONGCLIP="False"

# 最大处理样本数 (-1 表示处理所有样本)
# 用于快速测试或部分推理
MAX_SAMPLES=20

# 多GPU配置
# GPU_IDS: 使用的GPU编号，逗号分隔 (例如: "0,1,2,3")
# NUM_GPUS: 实际使用的GPU数量 (会自动取 GPU_IDS 和 NUM_GPUS 的较小值)
# WORKERS_PER_GPU: 每张GPU上并行运行的进程数
#   - "auto": 根据GPU显存自动计算最优值 (推荐)
#   - 数字 (如 "2"): 手动指定固定数量
GPU_IDS="0,1"
NUM_GPUS=2
WORKERS_PER_GPU="auto"

# ==========================================

echo "Starting Batch Inference..."
echo "Config: ${BATCH_TSV}"
echo "Output: ${OUTPUT_ROOT}"
echo "Caption: ${CAPTION_FIELD}"
echo "LongCLIP: ${USE_LONGCLIP}"
echo "Max Samples: ${MAX_SAMPLES}"
echo "GPU IDs: ${GPU_IDS}, Num GPUs: ${NUM_GPUS}, Workers/GPU: ${WORKERS_PER_GPU}"
echo "Total parallel workers: $((NUM_GPUS * WORKERS_PER_GPU))"
echo "Textures will be stored under: ${OUTPUT_ROOT}/textures"

python3 texture.py objaverse \
--tsv-path "${BATCH_TSV}" \
--caption-field "${CAPTION_FIELD}" \
--ckpt_path ./assets/ckpts/PBR_model.safetensors \
--output_dir "${OUTPUT_ROOT}" \
--save_image False \
--use_longclip "${USE_LONGCLIP}" \
--max-samples "${MAX_SAMPLES}" \
--gpu-ids "${GPU_IDS}" \
--num-gpus "${NUM_GPUS}" \
--workers-per-gpu "${WORKERS_PER_GPU}"
