#!/usr/bin/env bash
# Train a PI-Smith attacker against a served target. BENCHMARK is piarena (also used for SEP data) or agentdojo.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 BENCHMARK CONFIG TARGET_URL NUM_PROCESSES" >&2
  exit 2
fi

: "${PISMITH_HOME:?Set PISMITH_HOME to the directory created by scripts/setup_pismith.sh}"
readonly REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly BENCHMARK="$1"
readonly CONFIG="$2"
readonly TARGET_URL="$3"
readonly NUM_PROCESSES="$4"
readonly RENDERED="$PISMITH_HOME/configs/$(basename "$CONFIG")"

[[ -f "$CONFIG" ]] || { echo "missing training config: $CONFIG" >&2; exit 1; }
[[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_PROCESSES must be positive" >&2; exit 1; }

cp "$CONFIG" "$RENDERED"
printf 'target_model_url: %s\n' "$TARGET_URL" >> "$RENDERED"
case "$BENCHMARK" in
  piarena)
    : "${GEMINI_API_KEY:?Set GEMINI_API_KEY for the training judge}"
    readonly PYTHON="$PISMITH_HOME/envs/train/bin/python"
    export PYTHONPATH="$PISMITH_HOME"
    printf 'judge_model_config: %s\n' "$REPO_ROOT/configs/pismith/judge_gemini.yaml" >> "$RENDERED"
    if grep -q '^target_force_close: true' "$CONFIG"; then
      printf 'target_chat_template_path: %s\n' "$REPO_ROOT/templates/qwen38_input_role.jinja" >> "$RENDERED"
    fi
    ;;
  agentdojo)
    : "${OPENROUTER_BASE_URL:?Set OPENROUTER_BASE_URL for the reward judge}"
    : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for the reward judge}"
    readonly PYTHON="$PISMITH_HOME/envs/agentdojo/bin/python"
    export PYTHONPATH="$PISMITH_HOME:$PISMITH_HOME/agentdojo/src"
    export AGENTDOJO_QWEN_ENABLE_THINKING=0
    export AGENTDOJO_QWEN_STRIP_THINKING=1
    ;;
  *) echo "BENCHMARK must be piarena or agentdojo" >&2; exit 2 ;;
esac
[[ -x "$PYTHON" ]] || { echo "missing training environment: $PYTHON" >&2; exit 1; }

cd "$PISMITH_HOME/rl_attack"
exec "$PYTHON" -m accelerate.commands.launch \
  --config_file configs/accelerate.yaml \
  --num_processes "$NUM_PROCESSES" \
  -m train \
  --benchmark "$BENCHMARK" \
  --config_file "$RENDERED" \
  ${RESUME_FROM:+--resume_from_checkpoint "$RESUME_FROM"}
