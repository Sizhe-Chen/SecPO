#!/usr/bin/env bash
# Build the PI-Smith working directory: patched PI-Smith and AgentDojo sources and three pinned environments.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PISMITH_HOME" >&2
  exit 2
fi

readonly REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
readonly PISMITH_HOME="$1"
readonly PISMITH_URL="https://github.com/albert-y1n/PISmith.git"
readonly PISMITH_COMMIT="16865a73b1930ac98fc90d517f1ab20d532f912b"
readonly AGENTDOJO_URL="https://github.com/ethz-spylab/agentdojo.git"
readonly AGENTDOJO_COMMIT="d3640b5b03a88eb44dad96852e1de1ef437d836e"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
[[ ! -e "$PISMITH_HOME" ]] || { echo "PISMITH_HOME already exists: $PISMITH_HOME" >&2; exit 1; }

mkdir -p "$PISMITH_HOME"/{envs,models,data,runs,configs}

checkout() {  # URL COMMIT PATCH DESTINATION
  git clone "$1" "$4"
  git -C "$4" checkout --detach "$2"
  git -C "$4" apply "$3"
}
checkout "$PISMITH_URL" "$PISMITH_COMMIT" "$REPO_ROOT/third_party/patches/pismith.patch" "$PISMITH_HOME/rl_attack"
checkout "$AGENTDOJO_URL" "$AGENTDOJO_COMMIT" "$REPO_ROOT/third_party/patches/pismith_agentdojo.patch" "$PISMITH_HOME/agentdojo"

uv python install 3.10.20 3.12.13
install() {  # NAME PYTHON_VERSION
  uv venv --python "$2" "$PISMITH_HOME/envs/$1"
  # Every package is pinned, so each version may come from whichever index has it.
  uv pip install \
    --python "$PISMITH_HOME/envs/$1/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match \
    -r "$REPO_ROOT/requirements/pismith_$1.lock.txt"
}
install train 3.10.20
install agentdojo 3.10.20
install eval 3.12.13

echo "PI-Smith sources and environments are ready in $PISMITH_HOME"
