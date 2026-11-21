#!/usr/bin/env bash
# Serve a target or a trained attacker with a pinned vLLM environment. Extra arguments go to vLLM unchanged.
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 ENVIRONMENT MODEL_DIR SERVED_NAME HOST PORT [VLLM_ARGS...]" >&2
  exit 2
fi

: "${PISMITH_HOME:?Set PISMITH_HOME to the directory created by scripts/setup_pismith.sh}"
readonly ENVIRONMENT="$1"
readonly PYTHON="$PISMITH_HOME/envs/$ENVIRONMENT/bin/python"
readonly MODEL_DIR="$2"
readonly SERVED_NAME="$3"
readonly HOST="$4"
readonly PORT="$5"

[[ "$ENVIRONMENT" = eval || "$ENVIRONMENT" = target ]] || { echo "ENVIRONMENT must be eval (vLLM 0.19, SEP) or target (vLLM 0.29, AgentDojo)" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "missing environment: $PYTHON" >&2; exit 1; }
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
  "${@:6}"
