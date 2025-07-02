#!/usr/bin/env bash

# Simple launcher that delegates to scripts/algotune.sh
# This allows users to run ./algotune.sh from project root

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/scripts/algotune.sh" "$@"