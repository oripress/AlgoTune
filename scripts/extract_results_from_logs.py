#!/usr/bin/env python3
"""
Extract best-performing code from local logs and write to results/.
Uses the same code-extraction logic as the HTML generator.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "html_logs" / ".mplconfig"))


def parse_log_name(filename: str) -> tuple[str | None, str | None]:
    match = re.match(r"^(?P<task>.+)_(?P<model>.+)_\d{8}_\d{6}\.log$", filename)
    if not match:
        return None, None
    return match.group("task"), match.group("model")


def display_model_name(raw_model: str) -> str:
    script_dir = REPO_ROOT / "html"
    sys.path.insert(0, str(script_dir))
    import batch_html_generator  # local import to avoid global heavy deps

    cleaned = batch_html_generator.clean_model_name(raw_model)
    if raw_model.strip().lower() == "gpt-5.2":
        return "GPT-5.2"
    return cleaned


def write_results(task: str, model: str, files: dict[str, str], results_dir: Path) -> None:
    output_dir = results_dir / display_model_name(model) / task
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in files.items():
        dest = output_dir / filename
        dest.write_text(content, encoding="utf-8")


def main() -> int:
    log_dir = Path("logs")
    results_dir = Path("results")

    if not log_dir.exists():
        print(f"ERROR: Log directory not found: {log_dir}")
        return 1

    log_paths = sorted(log_dir.glob("*.log"))
    if not log_paths:
        print("No logs found.")
        return 0

    processed = 0
    skipped = 0

    for log_path in log_paths:
        task, model = parse_log_name(log_path.name)
        if task is None or model is None:
            skipped += 1
            continue

        log_content = log_path.read_text(encoding="utf-8", errors="ignore")
        script_dir = Path(__file__).resolve().parent.parent / "html"
        sys.path.insert(0, str(script_dir))
        import batch_html_generator

        best_code = batch_html_generator.extract_best_code(log_content)
        filtered = batch_html_generator._filter_best_code(best_code)
        if not filtered:
            skipped += 1
            continue

        write_results(task, model, filtered, results_dir)
        processed += 1

    print(f"Processed logs: {processed}")
    print(f"Skipped logs:   {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
