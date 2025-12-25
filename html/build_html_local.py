#!/usr/bin/env python3
"""
Build the HTML report from local logs without extra inputs.
Defaults to repo/logs and a timestamped results directory.
"""
from __future__ import annotations

import os
import sys
import shutil
from datetime import datetime
from pathlib import Path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    os.environ.setdefault("MPLCONFIGDIR", str(repo_root / "html_logs" / ".mplconfig"))

    logs_dir = Path(os.environ.get("LOGS_DIR", repo_root / "logs"))
    output_dir_env = os.environ.get("OUTPUT_DIR")
    if output_dir_env:
        output_dir = Path(output_dir_env)
    else:
        output_dir = repo_root / "html_logs"

    if not logs_dir.exists():
        print(f"ERROR: Logs directory not found: {logs_dir}")
        return 1

    log_files = list(logs_dir.glob("*.log"))
    if not log_files:
        print(f"ERROR: No .log files found in: {logs_dir}")
        return 1

    # Clean up old HTML files before generating new ones
    if output_dir.exists():
        print(f"🧹 Cleaning up old HTML files in: {output_dir}")
        for item in output_dir.iterdir():
            if item.is_file() and item.suffix in {'.html', '.css'}:
                item.unlink()
                print(f"   Deleted: {item.name}")
            elif item.is_dir() and item.name in {'assets', '.mplconfig'}:
                shutil.rmtree(item)
                print(f"   Deleted directory: {item.name}")
        print(f"✓ Cleanup complete\n")

    sys.path.insert(0, str(script_dir))
    import batch_html_generator

    batch_html_generator.LOGS_DIR = str(logs_dir) + os.sep
    batch_html_generator.OUTPUT_DIR = str(output_dir)
    batch_html_generator.PLOTS_DIR = os.path.join(str(output_dir), "assets", "plots")
    os.makedirs(batch_html_generator.PLOTS_DIR, exist_ok=True)

    styles_source = script_dir / "styles.css"
    styles_dest = output_dir / "styles.css"
    if styles_source.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(styles_source, styles_dest)

    assets_source = script_dir / "assets"
    assets_dest = output_dir / "assets"
    if assets_source.exists():
        shutil.copytree(assets_source, assets_dest, dirs_exist_ok=True)

    batch_html_generator.main()

    print("")
    print("=" * 50)
    print("✅ HTML visualization generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Open in browser: {output_dir}/index.html")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
