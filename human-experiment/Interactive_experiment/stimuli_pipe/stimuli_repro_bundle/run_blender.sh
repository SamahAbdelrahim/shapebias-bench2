#!/usr/bin/env bash
# Wrapper for repo Blender: prepends LD_LIBRARY_PATH for user-local libs (no sudo).
# Usage: ./run_blender.sh [same args as blender, e.g. -b -P animate_alice_stl.py]

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"
BLENDER=""
PATH_BLENDER_CANDIDATE=""

pick_blender() {
  local candidate="$1"
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    BLENDER="$candidate"
    return 0
  fi
  return 1
}

# 1) Prefer repo-local Blender if executable.
pick_blender "${REPO}/blender-4.5.0-linux-x64/blender" || true

# 2) macOS app bundle path.
if [[ -z "$BLENDER" && "$OS" == "Darwin" ]]; then
  pick_blender "/Applications/Blender.app/Contents/MacOS/Blender" || true
fi

# 3) Fallback to blender on PATH.
if [[ -z "$BLENDER" ]] && command -v blender >/dev/null 2>&1; then
  PATH_BLENDER_CANDIDATE="$(command -v blender)"
  # Accept PATH blender only if it can actually start.
  if "${PATH_BLENDER_CANDIDATE}" --version >/dev/null 2>&1; then
    BLENDER="${PATH_BLENDER_CANDIDATE}"
  fi
fi

USER_LIB_ROOT="${HOME}/.local/share/BlenderObjects-bundled-libs/usr"
case "$(uname -m)" in
  x86_64)  SUB="lib/x86_64-linux-gnu" ;;
  aarch64|arm64) SUB="lib/aarch64-linux-gnu" ;;
  *)       SUB="" ;;
esac

if [[ "$OS" == "Linux" ]]; then
  if [[ -n "$SUB" && -d "${USER_LIB_ROOT}/${SUB}" ]]; then
    export LD_LIBRARY_PATH="${USER_LIB_ROOT}/${SUB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi

  # Optional: conda/mamba env that provides libxkbcommon (conda-forge: libxkbcommon)
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

if [[ -z "$BLENDER" ]]; then
  echo "ERROR: Blender executable not found." >&2
  echo "Checked:" >&2
  echo "  - ${REPO}/blender-4.5.0-linux-x64/blender" >&2
  if [[ "$OS" == "Darwin" ]]; then
    echo "  - /Applications/Blender.app/Contents/MacOS/Blender" >&2
  fi
  echo "  - blender on PATH${PATH_BLENDER_CANDIDATE:+ (${PATH_BLENDER_CANDIDATE})}" >&2
  exit 1
fi

exec "$BLENDER" "$@"
