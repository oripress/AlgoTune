"""Helpers for pulling AlgoTune datasets from Hugging Face."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv_token() -> str | None:
    # Try HF_TOKEN first (official HuggingFace env var), then fall back to old names
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Backward compatibility
    token = os.environ.get("HUGGING_FACE_TOKEN")
    if token:
        return token

    dotenv_path = _project_root() / ".env"
    if not dotenv_path.exists():
        return None

    try:
        for raw in dotenv_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            # Check for HF_TOKEN first, then HUGGING_FACE_TOKEN for backward compat
            if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_TOKEN="):
                _, value = line.split("=", 1)
                return value.strip().strip('"').strip("'")
    except Exception:
        return None

    return None


def _get_repo_id() -> str:
    return os.environ.get("ALGOTUNE_HF_DATASET", "oripress/AlgoTune")


def _get_revision() -> str:
    return os.environ.get("ALGOTUNE_HF_REVISION", "main")


def _get_cache_dir() -> Path:
    cache_dir = os.environ.get("ALGOTUNE_HF_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return _project_root() / ".hf_datasets"


def _is_lazy_mode() -> bool:
    """Check if lazy download mode is enabled (only download .jsonl initially)."""
    return os.environ.get("ALGOTUNE_HF_LAZY") == "1"


def _data_dir_has_task_dataset(data_dir: Path, task_name: str) -> bool:
    """Return True if data_dir appears to contain a dataset for task_name."""
    import glob

    pattern = f"{task_name}_T*ms_n*_size*_train.jsonl"
    for base in (data_dir / task_name, data_dir):
        if base.is_dir() and glob.glob(str(base / pattern)):
            return True
    return False


def _maybe_set_data_dir_for_hf(data_dir: Path, task_name: str | None) -> None:
    """Point DATA_DIR at HF cache when it is the only place with the dataset."""
    current = os.environ.get("DATA_DIR")
    if not current:
        os.environ["DATA_DIR"] = str(data_dir)
        logging.info("DATA_DIR not set; using HF dataset dir %s", data_dir)
        return

    if task_name:
        try:
            current_path = Path(current)
        except Exception:
            current_path = None
        if current_path is None or not _data_dir_has_task_dataset(current_path, task_name):
            os.environ["DATA_DIR"] = str(data_dir)
            logging.info(
                "DATA_DIR has no dataset for %s; switching to HF dataset dir %s",
                task_name,
                data_dir,
            )


def ensure_hf_dataset(task_name: str | None = None) -> Path | None:
    """
    Ensure HF datasets are available locally and return the data directory to use.

    Returns:
        Path to the local "data" directory (or the task subdirectory), or None.
    """
    if os.environ.get("ALGOTUNE_HF_DISABLE") == "1":
        return None

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        logging.warning("HF download unavailable (huggingface_hub import failed): %s", exc)
        return None

    repo_id = _get_repo_id()
    revision = _get_revision()
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_dir = cache_dir / repo_id.replace("/", "__")
    allow_patterns = None
    lazy_mode = _is_lazy_mode()

    # Build allow_patterns based on lazy mode and task selection
    if lazy_mode:
        # In lazy mode, only download .jsonl files initially (metadata only)
        if task_name:
            allow_patterns = [f"data/{task_name}/**/*.jsonl"]
        else:
            allow_patterns = ["data/**/*.jsonl"]
    elif task_name:
        # Non-lazy mode: download entire task directory
        allow_patterns = [f"data/{task_name}/**"]

    token = _load_dotenv_token()
    force = os.environ.get("ALGOTUNE_HF_FORCE_DOWNLOAD") == "1"

    mode_str = "metadata-only (lazy)" if lazy_mode else "full"
    logging.info(
        "HF dataset download requested (repo=%s, revision=%s, task=%s, force=%s, mode=%s)",
        repo_id,
        revision,
        task_name or "all",
        force,
        mode_str,
    )

    if lazy_mode:
        print(f"📥 Downloading {task_name or 'all tasks'} metadata from HuggingFace ({repo_id})...")
        print("   (Lazy mode: .npy files will be downloaded on-demand)")
    else:
        print(f"📥 Downloading {task_name or 'all tasks'} dataset from HuggingFace ({repo_id})...")
        print("   This may take several minutes for large datasets...")

    try:
        # Import tqdm for progress display if available
        try:
            from tqdm import tqdm

            tqdm_class = tqdm
        except ImportError:
            tqdm_class = None
            print("   (Install tqdm for progress bars: pip install tqdm)")

        # Set cache_dir to be within local_dir to avoid duplicating space
        # Use symlinks to prevent storing files twice (cache + local_dir)
        cache_dir = local_dir / ".hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            cache_dir=str(cache_dir),
            local_dir_use_symlinks=True,  # Use symlinks to save 113GB disk space
            allow_patterns=allow_patterns,
            token=token,
            force_download=force,
            tqdm_class=tqdm_class,
        )
        print(f"✓ Download complete! Dataset cached at {local_dir}")
    except Exception as exc:
        logging.warning("HF snapshot download failed for %s: %s", repo_id, exc)
        print(f"❌ HF download failed: {exc}")
        return None

    data_dir = Path(snapshot_path) / "data"
    if data_dir.is_dir():
        _maybe_set_data_dir_for_hf(data_dir, task_name)
        if task_name:
            task_dir = data_dir / task_name
            if task_dir.is_dir():
                logging.info("HF dataset available at %s", task_dir)
                return task_dir
        logging.info("HF dataset available at %s", data_dir)
        return data_dir

    logging.warning("HF snapshot missing expected data directory: %s", data_dir)
    return None


def download_npy_files(task_name: str) -> bool:
    """
    Download .npy files for a specific task (for lazy loading mode).
    Call this after ensure_hf_dataset() in lazy mode to get the actual data files.

    Args:
        task_name: The task name to download .npy files for

    Returns:
        True if successful, False otherwise
    """
    if not _is_lazy_mode():
        logging.info("Not in lazy mode, .npy files should already be downloaded")
        return True

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        logging.warning("HF download unavailable (huggingface_hub import failed): %s", exc)
        return False

    repo_id = _get_repo_id()
    revision = _get_revision()
    cache_dir = _get_cache_dir()
    local_dir = cache_dir / repo_id.replace("/", "__")

    allow_patterns = [f"data/{task_name}/**/*.npy"]
    token = _load_dotenv_token()

    print(f"📥 Downloading {task_name} .npy files from HuggingFace...")
    print("   This may take several minutes for large datasets...")

    try:
        # Import tqdm for progress display if available
        try:
            from tqdm import tqdm

            tqdm_class = tqdm
        except ImportError:
            tqdm_class = None

        # Set cache_dir to be within local_dir to avoid duplicating space
        # Use symlinks to prevent storing files twice (cache + local_dir)
        hf_cache_dir = local_dir / ".hf_cache"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            cache_dir=str(hf_cache_dir),
            local_dir_use_symlinks=True,  # Use symlinks to save 113GB disk space
            allow_patterns=allow_patterns,
            token=token,
            force_download=False,  # Use cached files if available
            tqdm_class=tqdm_class,
        )
        print(f"✓ Download complete for {task_name} .npy files!")
        return True
    except Exception as exc:
        logging.warning("HF .npy download failed for %s: %s", task_name, exc)
        print(f"❌ .npy download failed: {exc}")
        return False


def cleanup_npy_files(task_name: str | None = None) -> None:
    """
    Clean up .npy files to free disk space (manual cleanup utility).

    This keeps .jsonl files but removes large .npy files to free disk space.
    Use this when you need to free up disk space manually.

    Note: The HF cache is normally kept and reused across jobs. Only clean up
    when you need to free disk space and won't be running the same task again soon.

    Args:
        task_name: If provided, only clean up files for this task.
                   If None, clean up all .npy files.
    """
    cache_dir = _get_cache_dir()
    repo_id = _get_repo_id()
    local_dir = cache_dir / repo_id.replace("/", "__") / "data"

    if not local_dir.exists():
        logging.info("No HF cache directory found, nothing to clean up")
        return

    cleaned_count = 0
    freed_bytes = 0

    if task_name:
        # Clean up specific task
        task_dir = local_dir / task_name
        if task_dir.exists():
            for npy_file in task_dir.glob("**/*.npy"):
                size = npy_file.stat().st_size
                npy_file.unlink()
                cleaned_count += 1
                freed_bytes += size
            logging.info(
                "Cleaned up %d .npy files for task %s (freed %.2f GB)",
                cleaned_count,
                task_name,
                freed_bytes / 1e9,
            )
    else:
        # Clean up all tasks
        for npy_file in local_dir.glob("**/*.npy"):
            size = npy_file.stat().st_size
            npy_file.unlink()
            cleaned_count += 1
            freed_bytes += size
        logging.info("Cleaned up %d .npy files (freed %.2f GB)", cleaned_count, freed_bytes / 1e9)

    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} .npy files (freed {freed_bytes / 1e9:.2f} GB)")
    else:
        print("🧹 No .npy files found to clean up")
