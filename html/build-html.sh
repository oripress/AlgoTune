#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOGS_DIR="${LOGS_DIR:-$ROOT_DIR/logs}"
HTML_GEN_PROCESSES="${HTML_GEN_PROCESSES:-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "❌ ERROR: Python executable not found: $PYTHON_BIN"
  exit 1
fi

export LOGS_DIR
export HTML_GEN_PROCESSES
exec "$PYTHON_BIN" "$SCRIPT_DIR/build_html_local.py"
