#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-algotune:latest}"

require_dockerignore_entry() {
  local dockerignore_path="$1"
  local entry="$2"

  if [[ ! -f "$dockerignore_path" ]]; then
    echo "❌ ERROR: Missing $dockerignore_path"
    exit 1
  fi

  if ! grep -Fxq "$entry" "$dockerignore_path"; then
    echo "❌ ERROR: $dockerignore_path must include '$entry' to avoid leaking secrets"
    exit 1
  fi
}

require_dockerignore_entry "$ROOT_DIR/.dockerignore" ".env"
require_dockerignore_entry "$ROOT_DIR/.dockerignore" "aws/.env"
require_dockerignore_entry "$SCRIPT_DIR/.dockerignore" ".env"

echo "→ Building Docker image: $IMAGE_TAG"
docker build -t "$IMAGE_TAG" -f "$SCRIPT_DIR/Dockerfile" "$ROOT_DIR"
