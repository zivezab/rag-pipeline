#!/usr/bin/env bash
# One-off setup: creates .venv in this folder and installs requirements.txt.
set -euo pipefail
cd "$(dirname "$0")"

# Pick a Python 3.10+ interpreter (haystack-ai needs it; the macOS system
# python3 is often older). Override by exporting PYTHON_BIN before running.
PYTHON_BIN="${PYTHON_BIN:-python3}"
version_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 || ! version_ok "$PYTHON_BIN"; then
  for candidate in python3.13 python3.12 python3.11 python3.10 \
                   /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
                   /usr/local/bin/python3.13 /usr/local/bin/python3.12; do
    if command -v "$candidate" >/dev/null 2>&1 && version_ok "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if ! version_ok "$PYTHON_BIN"; then
  echo "error: could not find a Python 3.10+ interpreter. Install one and re-run with PYTHON_BIN=/path/to/python3.x ./setup.sh" >&2
  exit 1
fi

echo "Using $("$PYTHON_BIN" --version) ($PYTHON_BIN)"

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
