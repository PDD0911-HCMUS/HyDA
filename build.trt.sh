#!/usr/bin/env bash
set -euo pipefail

#################### CONFIG ####################
ONNX_PATH="/home/map4/ThisPC/PhD_Journey/Controller/HyDAController/checkpoint/hyda_r50_e93.onnx"
ENGINE_PATH="/home/map4/ThisPC/PhD_Journey/Controller/HyDAController/checkpoint/hyda_r50_e93_fp32.trt"
WORKSPACE_MB=4096           # 4GB
LOG_DIR="./checkpoint/logs"
#################### END CONFIG ################

mkdir -p "${LOG_DIR}"

# Timestamp để log mỗi lần một file riêng
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/build_trt_${TIMESTAMP}.log"

echo "======================================="
echo "  TensorRT Engine Builder (TRT 8.6.1)  "
echo "======================================="
echo "ONNX:   ${ONNX_PATH}"
echo "ENGINE: ${ENGINE_PATH}"
echo "LOG:    ${LOG_FILE}"
echo "WS:     ${WORKSPACE_MB} MiB"
echo

# TẮT MYELIN để tránh crash munmap_chunk()
export TRT_MYELIN_DISABLE=1

# (Bạn có thể bật thêm mấy cái debug khác nếu muốn)
# export CUDA_LAUNCH_BLOCKING=1

# In version để chắc chắn đang dùng TRT 8
echo "[INFO] trtexec version:"
trtexec --version || true
echo

echo "[INFO] Starting build..."
echo

# Build engine
trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --workspace=${WORKSPACE_MB} \
  --buildOnly \
  --verbose \
  
  2>&1 | tee "${LOG_FILE}"

BUILD_RET=${PIPESTATUS[0]}

echo
if [[ ${BUILD_RET} -ne 0 ]]; then
  echo "[ERROR] trtexec failed with code ${BUILD_RET}."
  echo "        Check log file: ${LOG_FILE}"
  exit ${BUILD_RET}
else
  echo "[OK] Engine built successfully."
  echo "     Engine: ${ENGINE_PATH}"
  echo "     Log:    ${LOG_FILE}"
fi
