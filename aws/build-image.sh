#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-algotune:latest}"

workspace_manifest_file="$(mktemp)"
image_manifest_file="$(mktemp)"

cleanup() {
  rm -f "$workspace_manifest_file" "$image_manifest_file"
}
trap cleanup EXIT

generate_workspace_manifest() {
  (
    cd "$ROOT_DIR"
    {
      find AlgoTuner AlgoTuneTasks scripts -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
      sha256sum pyproject.toml algotune.py algotune.sh
    }
  ) > "$workspace_manifest_file"
}

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

git_sha="$(git -C "$ROOT_DIR" rev-parse HEAD)"
if [[ -n "$(git -C "$ROOT_DIR" status --short --untracked-files=no)" ]]; then
  git_status="dirty"
else
  git_status="clean"
fi
build_time="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

generate_workspace_manifest

echo "→ Building Docker image: $IMAGE_TAG"
docker build \
  --build-arg "ALGOTUNE_GIT_SHA=$git_sha" \
  --build-arg "ALGOTUNE_GIT_STATUS=$git_status" \
  --build-arg "ALGOTUNE_BUILD_TIME=$build_time" \
  -t "$IMAGE_TAG" \
  -f "$SCRIPT_DIR/Dockerfile" \
  "$ROOT_DIR"

echo "→ Verifying built image contents against workspace"
docker run --rm --entrypoint bash "$IMAGE_TAG" -lc 'cat /app/.image-manifest.sha256' > "$image_manifest_file"

if ! cmp -s "$workspace_manifest_file" "$image_manifest_file"; then
  echo "❌ ERROR: built image manifest does not match workspace contents"
  diff -u "$workspace_manifest_file" "$image_manifest_file" || true
  exit 1
fi

echo "✓ Image manifest matches workspace"
echo "✓ Build metadata: git_sha=$git_sha git_status=$git_status build_time=$build_time"
