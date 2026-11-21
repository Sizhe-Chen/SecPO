#!/usr/bin/env bash
# Serve a target or a trained attacker with the pinned vLLM. Extra arguments go to vLLM unchanged.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 MODEL_DIR SERVED_NAME HOST PORT [VLLM_ARGS...]" >&2
  exit 2
fi

: "${PISMITH_HOME:?Set PISMITH_HOME to the directory created by scripts/setup_pismith.sh}"
readonly PYTHON="$PISMITH_HOME/envs/eval/bin/python"
readonly MODEL_DIR="$1"
readonly SERVED_NAME="$2"
readonly HOST="$3"
readonly PORT="$4"

[[ -x "$PYTHON" ]] || { echo "missing evaluation environment: $PYTHON" >&2; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "missing model directory: $MODEL_DIR" >&2; exit 1; }
[[ -x "${CUDA_HOME:-}/bin/nvcc" ]] || { echo "CUDA_HOME must name a CUDA 12.8 toolkit that contains bin/nvcc" >&2; exit 1; }

# vLLM compiles Qwen attention kernels at startup with this environment's ninja and CUDA_HOME's nvcc.
export PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --trust-remote-code \
  "${@:5}"
