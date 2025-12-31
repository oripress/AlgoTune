#!/usr/bin/env bash

resolve_singularity_image() {
    if [ -z "${SINGULARITY_IMAGE:-}" ]; then
        echo "Error: SINGULARITY_IMAGE is not set."
        exit 1
    fi

    if [[ "$SINGULARITY_IMAGE" =~ ^(docker|oras|library):// ]]; then
        local cache_base="${TEMP_DIR_STORAGE:-/tmp}"
        local cache_dir="${cache_base%/}/singularity_images"
        local safe_name
        safe_name=$(echo "$SINGULARITY_IMAGE" | sed -e 's#^[a-zA-Z]\+://##' -e 's#[^A-Za-z0-9._-]#_#g')
        local image_path="${cache_dir}/${safe_name}.sif"

        mkdir -p "$cache_dir"
        if [ ! -f "$image_path" ]; then
            echo "Pulling Singularity image to $image_path"
            singularity pull "$image_path" "$SINGULARITY_IMAGE"
        fi

        SINGULARITY_IMAGE="$image_path"
    fi

    if [ ! -f "$SINGULARITY_IMAGE" ]; then
        echo "Error: Singularity image not found: $SINGULARITY_IMAGE"
        exit 1
    fi
}
