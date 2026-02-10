# =========================
# Inputs
# =========================
SPLIT_DIR="../experiments/common_splits"
TRAIN_TSV="${SPLIT_DIR}/train.tsv"
TEST_TSV="${SPLIT_DIR}/test.tsv"

# =========================
# Output
# =========================
OUTPUT_DIR="../datasets/texverse_pointcloud_npz"

# =========================
# Processing
# =========================
MAX_POINTS=200000
MAX_SAMPLES=10     # -1 means all, 10 means only convert first 10 unique samples
OVERWRITE="False"  # True | False

echo "[INFO] Pointcloud precompute config"
echo "  Train TSV: ${TRAIN_TSV}"
echo "  Test TSV:  ${TEST_TSV}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  MaxPoints: ${MAX_POINTS}"
echo "  MaxSamples:${MAX_SAMPLES}"
echo "  Overwrite: ${OVERWRITE}"

if [[ ! -f "${TRAIN_TSV}" ]]; then
  echo "[ERROR] train TSV not found: ${TRAIN_TSV}"
  exit 1
fi
if [[ ! -f "${TEST_TSV}" ]]; then
  echo "[ERROR] test TSV not found: ${TEST_TSV}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  scripts/precompute_pointclouds.py
  --tsv "${TRAIN_TSV}" "${TEST_TSV}"
  --output_dir "${OUTPUT_DIR}"
  --max_points "${MAX_POINTS}"
  --max_samples "${MAX_SAMPLES}"
)

if [[ "${OVERWRITE}" == "True" ]]; then
  ARGS+=(--overwrite)
fi

python "${ARGS[@]}"
