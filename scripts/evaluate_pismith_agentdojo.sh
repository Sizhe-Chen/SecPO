#!/usr/bin/env bash
# Pass@NUM_SAMPLES AgentDojo evaluation of a trained PI-Smith attacker on all four suites (949 pairs).
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 CHECKPOINT ATTACKER_URL TARGET_URL TARGET_MODEL_ID TOOL_DELIMITER BENCHMARK_VERSION ATTACKER_TEMPERATURE NUM_SAMPLES OUTPUT_DIR" >&2
  exit 2
fi

: "${PISMITH_HOME:?Set PISMITH_HOME to the directory created by scripts/setup_pismith.sh}"
readonly PYTHON="$PISMITH_HOME/envs/agentdojo/bin/python"
[[ -x "$PYTHON" ]] || { echo "missing AgentDojo environment: $PYTHON" >&2; exit 1; }
[[ -d "$1" ]] || { echo "missing attacker checkpoint: $1" >&2; exit 1; }
[[ "$8" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_SAMPLES must be positive" >&2; exit 1; }
[[ ! -e "$9" ]] || { echo "output directory already exists: $9" >&2; exit 1; }

export PYTHONPATH="$PISMITH_HOME:$PISMITH_HOME/agentdojo/src"
export AGENTDOJO_QWEN_STRIP_THINKING=1
cd "$PISMITH_HOME/rl_attack"
exec env CUDA_VISIBLE_DEVICES='' "$PYTHON" -m eval.eval_agentdojo \
  --attacker_model "$1" \
  --attacker_server_url "$2" \
  --target_model local \
  --target_model_url "$3" \
  --target_model_id "$4" \
  --target_tool_delimiter "$5" \
  --eval_suites workspace,banking,travel,slack \
  --benchmark_version "$6" \
  --max_tokens 1024 \
  --temperature "$7" \
  --num_samples "$8" \
  --max_workers 16 \
  --output_dir "$9" \
  --logdir "$9/traces" \
  --seed 42
