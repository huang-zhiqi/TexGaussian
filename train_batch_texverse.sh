# 1. 初始化 Conda 环境
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活安装了 texgaussian 的环境
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

# =========================
# Runtime
# =========================
GPU_IDS="0"
NUM_GPUS=1
ACC_CONFIG="acc_configs/gpu${NUM_GPUS}.yaml"
MAIN_PORT=8878

# =========================
# Experiment
# =========================
EXP_NAME="texverse_longclip_heads"
WORKSPACE="../experiments/${EXP_NAME}"
RESUME_CKPT="./assets/ckpts/PBR_model.safetensors"  # leave empty to train from scratch

# =========================
# Data
# =========================
SPLIT_DIR="../experiments/common_splits"
TRAIN_LIST="${SPLIT_DIR}/train.tsv"
TEST_LIST="${SPLIT_DIR}/test.tsv"
CAPTION_FIELD="caption_long"  # caption_long | caption_short
TRAIN_IMAGE_DIR="../datasets/texverse_rendered_train"
TEST_IMAGE_DIR="../datasets/texverse_rendered_test"
POINTCLOUD_DIR="../datasets/texverse_pointcloud_npz"

# =========================
# Feature switches
# =========================
USE_TEXT="True"
USE_MATERIAL="True"
USE_LONGCLIP="True"
LONGCLIP_MODEL="third_party/Long-CLIP/checkpoints/longclip-L.pt"
LONGCLIP_CONTEXT_LENGTH=248
USE_NORMAL_HEAD="True"
USE_ROTATION_HEAD="True"

# =========================
# Optimization
# =========================
BATCH_SIZE=1
GRAD_ACC=8
NUM_EPOCHS=1
LR=4e-4
MIXED_PRECISION="bf16"
LAMBDA_GEO_NORMAL=1.0
LAMBDA_TEX_NORMAL=1.0

echo "[INFO] Training config"
echo "  GPUs: ${GPU_IDS} (num_processes=${NUM_GPUS})"
echo "  Accelerate config: ${ACC_CONFIG}"
echo "  Workspace: ${WORKSPACE}"
echo "  Resume: ${RESUME_CKPT}"
echo "  LongCLIP: ${USE_LONGCLIP} (${LONGCLIP_MODEL})"
echo "  Heads: normal=${USE_NORMAL_HEAD}, rotation=${USE_ROTATION_HEAD}"
echo "  Train list: ${TRAIN_LIST}"
echo "  Test list: ${TEST_LIST}"
echo "  Caption field: ${CAPTION_FIELD}"

if [[ ! -f "${TRAIN_LIST}" ]]; then
  echo "[ERROR] train split not found: ${TRAIN_LIST}"
  exit 1
fi
if [[ ! -f "${TEST_LIST}" ]]; then
  echo "[ERROR] test split not found: ${TEST_LIST}"
  exit 1
fi
if [[ ! -f "${ACC_CONFIG}" ]]; then
  echo "[ERROR] accelerate config not found: ${ACC_CONFIG}"
  exit 1
fi

if [[ "${TRAIN_IMAGE_DIR}" == "path_to_texverse_image_dir" ]]; then
  echo "[ERROR] Please set TRAIN_IMAGE_DIR to your rendered train multi-view data root."
  exit 1
fi
if [[ ! -d "${TRAIN_IMAGE_DIR}" ]]; then
  echo "[ERROR] TRAIN_IMAGE_DIR does not exist: ${TRAIN_IMAGE_DIR}"
  exit 1
fi
if [[ ! -d "${TEST_IMAGE_DIR}" ]]; then
  echo "[ERROR] TEST_IMAGE_DIR does not exist: ${TEST_IMAGE_DIR}"
  exit 1
fi
if [[ -z "${POINTCLOUD_DIR}" ]]; then
  echo "[ERROR] POINTCLOUD_DIR must be set to your precomputed pointcloud root."
  exit 1
fi
if [[ ! -d "${POINTCLOUD_DIR}" ]]; then
  echo "[ERROR] POINTCLOUD_DIR does not exist: ${POINTCLOUD_DIR}"
  exit 1
fi
if [[ -n "${RESUME_CKPT}" && ! -f "${RESUME_CKPT}" ]]; then
  echo "[ERROR] RESUME_CKPT not found: ${RESUME_CKPT}"
  exit 1
fi
if [[ "${USE_LONGCLIP}" == "True" && ! -f "${LONGCLIP_MODEL}" ]]; then
  echo "[ERROR] LongCLIP checkpoint not found: ${LONGCLIP_MODEL}"
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT=0
for g in "${GPU_ARRAY[@]}"; do
  g_trim="$(echo "${g}" | xargs)"
  if [[ -n "${g_trim}" ]]; then
    GPU_COUNT=$((GPU_COUNT + 1))
  fi
done
if [[ "${GPU_COUNT}" -ne "${NUM_GPUS}" ]]; then
  echo "[ERROR] NUM_GPUS=${NUM_GPUS} but GPU_IDS='${GPU_IDS}' has ${GPU_COUNT} entries."
  exit 1
fi

echo "[INFO] Running dataset preflight checks..."
python - "${TRAIN_LIST}" "${TRAIN_IMAGE_DIR}" "${TEST_LIST}" "${TEST_IMAGE_DIR}" "${POINTCLOUD_DIR}" "${CAPTION_FIELD}" "${USE_MATERIAL}" "${USE_NORMAL_HEAD}" <<'PY'
import csv
import glob
import os
import sys
import numpy as np

train_tsv, train_image_dir, test_tsv, test_image_dir, pointcloud_dir, caption_field, use_material, use_normal_head = sys.argv[1:]
use_material = use_material.lower() == "true"
use_normal_head = use_normal_head.lower() == "true"

with open(train_tsv, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    cols = set(reader.fieldnames or [])
    missing = sorted({"obj_id"} - cols)
    if missing:
        raise SystemExit(f"[ERROR] train TSV missing required columns: {missing}")
    if caption_field not in cols:
        raise SystemExit(f"[ERROR] caption field '{caption_field}' not in TSV columns")

    try:
        row = next(reader)
    except StopIteration:
        raise SystemExit("[ERROR] train TSV is empty")

def has_multiview_channel_from_image_dir(image_obj_dir, channel):
    aliases = [channel]
    if channel == "rough":
        aliases = ["rough", "roughness"]
    elif channel == "metal":
        aliases = ["metal", "metallic", "matellic"]
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
    if not uid:
        raise SystemExit(f"[ERROR] first row has empty obj_id: {tsv_path}")
    return uid

def check_split(split_name, tsv_path, image_dir):
    uid = first_uid(tsv_path)

    pc_path = os.path.join(pointcloud_dir, uid + ".npz")
    if not os.path.isfile(pc_path):
        raise SystemExit(f"[ERROR] missing pointcloud npz for {split_name} first sample uid={uid}: {pc_path}")
    try:
        with np.load(pc_path) as npz:
            if "points" not in npz or "normals" not in npz:
                raise SystemExit(f"[ERROR] invalid pointcloud npz (need keys points/normals): {pc_path}")
    except Exception as e:
        raise SystemExit(f"[ERROR] failed to read pointcloud npz: {pc_path}, err={e}")

    cam_dir = os.path.join(image_dir, uid)
    cam_ok = os.path.isfile(os.path.join(cam_dir, "cameras.npz")) or os.path.isfile(os.path.join(cam_dir, "transforms.json"))
    if not cam_ok:
        raise SystemExit(
            f"[ERROR] camera metadata missing for {split_name} first sample: {cam_dir} "
            "(need cameras.npz or transforms.json)"
        )

    bad = []
    if not has_multiview_channel_from_image_dir(cam_dir, "albedo"):
        bad.append("albedo")
    if use_material:
        if not has_multiview_channel_from_image_dir(cam_dir, "rough"):
            bad.append("rough")
        if not has_multiview_channel_from_image_dir(cam_dir, "metal"):
            bad.append("metal")
    if use_normal_head:
        if not has_multiview_channel_from_image_dir(cam_dir, "normal"):
            bad.append("normal")
    if bad:
        raise SystemExit(
            f"[ERROR] {split_name} first sample does not expose channels: "
            + ", ".join(bad)
            + f". Expected files like {image_dir}/{{uid}}/unlit/000_<channel>.png."
        )

    print(f"[INFO] Preflight passed for {split_name} first sample uid={uid}")

check_split("train", train_tsv, train_image_dir)
check_split("test", test_tsv, test_image_dir)
PY

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
  --use_normal_head "${USE_NORMAL_HEAD}"
  --use_rotation_head "${USE_ROTATION_HEAD}"
  --lambda_geo_normal "${LAMBDA_GEO_NORMAL}"
  --lambda_tex_normal "${LAMBDA_TEX_NORMAL}"
)

if [[ -n "${RESUME_CKPT}" ]]; then
  ARGS+=(--resume "${RESUME_CKPT}")
fi

# Keep stderr readable by dropping a known non-fatal libpng metadata warning.
CUDA_VISIBLE_DEVICES="${GPU_IDS}" accelerate launch "${ARGS[@]}" \
  2> >(grep -v -E '^libpng warning: eXIf: duplicate$' >&2)
