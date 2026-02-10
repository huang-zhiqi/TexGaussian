# =========================
# Inputs
# =========================
SPLIT_DIR="../experiments/common_splits"
TRAIN_TSV="${SPLIT_DIR}/train.tsv"
TEST_TSV="${SPLIT_DIR}/test.tsv"

# =========================
# Pointcloud dir
# =========================
POINTCLOUD_DIR="../datasets/texverse_pointcloud_npz"

# =========================
# Validation config
# =========================
MAX_SAMPLES=10        # -1 means all
MAX_POINTS=200000     # must match precompute setting
CHECK_MESH_REBUILD="True"
STRICT_FLOAT32="True"
CENTROID_TOL=-1       # <=0 disables centroid check
RADIUS_MIN=0.05
RADIUS_MAX=1.10
EXTENT_MAX_MIN=0.95
EXTENT_MAX_MAX=1.05
CENTER_TOL=0.10
NORMAL_LEN_MIN=0.50
NORMAL_LEN_MAX=1.50
COMPARE_TOL=1e-5
FAIL_FAST="False"
EXPORT_VIZ="True"
VIZ_MODE="all"   # failures | all
VIZ_LIMIT=50
VIZ_MAX_POINTS=20000
VIZ_DIR="${POINTCLOUD_DIR}/validate_viz"

echo "[INFO] Pointcloud validate config"
echo "  Train TSV: ${TRAIN_TSV}"
echo "  Test TSV:  ${TEST_TSV}"
echo "  Pointcloud:${POINTCLOUD_DIR}"
echo "  MaxSamples:${MAX_SAMPLES}"
echo "  MaxPoints: ${MAX_POINTS}"
echo "  Rebuild:   ${CHECK_MESH_REBUILD}"
echo "  StrictFP32:${STRICT_FLOAT32}"
echo "  ExtentMax: [${EXTENT_MAX_MIN}, ${EXTENT_MAX_MAX}]"
echo "  ExportViz: ${EXPORT_VIZ} (${VIZ_MODE}, limit=${VIZ_LIMIT})"

if [[ ! -d "${POINTCLOUD_DIR}" ]]; then
  echo "[ERROR] POINTCLOUD_DIR not found: ${POINTCLOUD_DIR}"
  exit 1
fi
if [[ ! -f "${TRAIN_TSV}" ]]; then
  echo "[ERROR] train TSV not found: ${TRAIN_TSV}"
  exit 1
fi
if [[ ! -f "${TEST_TSV}" ]]; then
  echo "[ERROR] test TSV not found: ${TEST_TSV}"
  exit 1
fi

ARGS=(
  scripts/validate_pointclouds.py
  --pointcloud_dir "${POINTCLOUD_DIR}"
  --tsv "${TRAIN_TSV}" "${TEST_TSV}"
  --max_samples "${MAX_SAMPLES}"
  --max_points "${MAX_POINTS}"
  --radius_min "${RADIUS_MIN}"
  --radius_max "${RADIUS_MAX}"
  --extent_max_min "${EXTENT_MAX_MIN}"
  --extent_max_max "${EXTENT_MAX_MAX}"
  --center_tol "${CENTER_TOL}"
  --centroid_tol "${CENTROID_TOL}"
  --normal_len_min "${NORMAL_LEN_MIN}"
  --normal_len_max "${NORMAL_LEN_MAX}"
  --compare_tol "${COMPARE_TOL}"
  --viz_mode "${VIZ_MODE}"
  --viz_limit "${VIZ_LIMIT}"
  --viz_max_points "${VIZ_MAX_POINTS}"
  --viz_dir "${VIZ_DIR}"
)

if [[ "${CHECK_MESH_REBUILD}" == "True" ]]; then
  ARGS+=(--check_mesh_rebuild)
fi
if [[ "${STRICT_FLOAT32}" == "True" ]]; then
  ARGS+=(--strict_float32)
fi
if [[ "${FAIL_FAST}" == "True" ]]; then
  ARGS+=(--fail_fast)
fi
if [[ "${EXPORT_VIZ}" == "True" ]]; then
  ARGS+=(--export_viz)
fi

python "${ARGS[@]}"
