#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

echo "=== Creating Python virtual environment ==="
python3 -m venv "$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/.venv/bin/activate"

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing requirements ==="
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=== Environment ready at $SCRIPT_DIR/.venv ==="
echo "To activate manually: source $SCRIPT_DIR/.venv/bin/activate"
