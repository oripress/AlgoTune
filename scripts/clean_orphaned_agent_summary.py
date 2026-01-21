#!/usr/bin/env python3
"""
Clean orphaned entries from agent_summary.json

Removes entries for specific models (gemini-3-pro-preview, gpt-5.2) where:
1. No log file exists for that task+model combination
2. If log exists, validates the speedup value matches the log

Usage:
    python scripts/clean_orphaned_agent_summary.py [--dry-run] [--models MODEL1,MODEL2]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path


def find_log_file(task_name: str, model_key: str, logs_dir: Path) -> Path | None:
    """
    Find the log file for a given task and model.

    Args:
        task_name: Task name (e.g., 'convex_hull')
        model_key: Model key from agent_summary (e.g., 'gpt-5.2', 'gemini-3-pro-preview')
        logs_dir: Directory containing log files

    Returns:
        Path to log file if found, None otherwise
    """
    # Try exact match first
    pattern = f"{task_name}_{model_key}_*.log"
    matches = list(logs_dir.glob(pattern))

    if matches:
        # Return most recent if multiple matches
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return None


def extract_speedup_from_log(log_path: Path) -> str | None:
    """
    Extract final speedup value from log file.

    Args:
        log_path: Path to log file

    Returns:
        Speedup value as string, or None if not found
    """
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read last 1000 lines for efficiency
            lines = f.readlines()
            last_lines = lines[-1000:] if len(lines) > 1000 else lines

            # Look for patterns like:
            # - "Final Test Performance" or "test speedup"
            # - "Recording test speedup (X.XXXX)"
            # - "best_speedup": X.XXXX

            for line in reversed(last_lines):
                # Pattern: "Recording test speedup (1.4426)"
                match = re.search(r'Recording test speedup \(([0-9.]+)\)', line)
                if match:
                    return match.group(1)

                # Pattern: "best_speedup": 1.4426
                match = re.search(r'"best_speedup":\s*([0-9.]+)', line)
                if match:
                    return match.group(1)

                # Pattern: Using test dataset speedup for summary: 1.4426
                match = re.search(r'Using test dataset speedup for summary:\s*([0-9.]+)', line)
                if match:
                    return match.group(1)

    except Exception as e:
        print(f"  Warning: Error reading log {log_path}: {e}", file=sys.stderr)
        return None

    return None


def clean_agent_summary(
    agent_summary_path: Path,
    logs_dir: Path,
    models_to_check: list[str],
    dry_run: bool = False,
) -> tuple[dict, int, int, int]:
    """
    Clean orphaned entries from agent_summary.json

    Args:
        agent_summary_path: Path to agent_summary.json
        logs_dir: Directory containing log files
        models_to_check: List of model keys to check (e.g., ['gpt-5.2', 'gemini-3-pro-preview'])
        dry_run: If True, don't modify files

    Returns:
        Tuple of (updated_data, removed_count, validated_count, mismatched_count)
    """
    with open(agent_summary_path, 'r') as f:
        data = json.load(f)

    removed_count = 0
    validated_count = 0
    mismatched_count = 0

    for task_name in list(data.keys()):
        for model_key in list(data[task_name].keys()):
            # Only check specified models
            if not any(model in model_key for model in models_to_check):
                continue

            # Check if log file exists
            log_file = find_log_file(task_name, model_key, logs_dir)

            if log_file is None:
                # No log file - remove entry
                print(f"❌ {task_name} / {model_key}: No log file found, removing entry")
                if not dry_run:
                    del data[task_name][model_key]
                    # Remove task entirely if no models left
                    if not data[task_name]:
                        del data[task_name]
                removed_count += 1
            else:
                # Log exists - validate speedup
                current_speedup = data[task_name][model_key].get('final_speedup', 'N/A')
                log_speedup = extract_speedup_from_log(log_file)

                if log_speedup is None:
                    print(f"⚠️  {task_name} / {model_key}: Log exists but couldn't extract speedup (current: {current_speedup})")
                    validated_count += 1
                elif current_speedup == 'N/A':
                    print(f"🔄 {task_name} / {model_key}: Log has {log_speedup}, but agent_summary has N/A")
                    if not dry_run:
                        data[task_name][model_key]['final_speedup'] = log_speedup
                    mismatched_count += 1
                elif abs(float(current_speedup) - float(log_speedup)) > 0.01:
                    print(f"🔄 {task_name} / {model_key}: Mismatch! agent_summary={current_speedup}, log={log_speedup}, updating")
                    if not dry_run:
                        data[task_name][model_key]['final_speedup'] = log_speedup
                    mismatched_count += 1
                else:
                    print(f"✅ {task_name} / {model_key}: Validated (speedup={current_speedup})")
                    validated_count += 1

    return data, removed_count, validated_count, mismatched_count


def main():
    parser = argparse.ArgumentParser(description='Clean orphaned entries from agent_summary.json')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    parser.add_argument(
        '--models',
        default='gemini-3-pro-preview,gpt-5.2',
        help='Comma-separated list of model keys to check (default: gemini-3-pro-preview,gpt-5.2)'
    )
    parser.add_argument(
        '--sync-s3',
        action='store_true',
        help='Sync updated agent_summary.json to S3 after cleaning'
    )
    args = parser.parse_args()

    # Paths
    repo_root = Path(__file__).parent.parent
    agent_summary_path = repo_root / 'reports' / 'agent_summary.json'
    logs_dir = repo_root / 'logs'

    if not agent_summary_path.exists():
        print(f"Error: {agent_summary_path} not found", file=sys.stderr)
        sys.exit(1)

    if not logs_dir.exists():
        print(f"Error: {logs_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Parse models list
    models_to_check = [m.strip() for m in args.models.split(',')]

    print(f"Checking models: {', '.join(models_to_check)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Clean agent summary
    updated_data, removed_count, validated_count, mismatched_count = clean_agent_summary(
        agent_summary_path,
        logs_dir,
        models_to_check,
        dry_run=args.dry_run,
    )

    # Summary
    print()
    print("=" * 60)
    print(f"Removed entries (no log): {removed_count}")
    print(f"Updated entries (mismatched speedup): {mismatched_count}")
    print(f"Validated entries (matched): {validated_count}")
    print(f"Total processed: {removed_count + validated_count + mismatched_count}")
    print("=" * 60)

    # Write updated file
    if not args.dry_run and (removed_count > 0 or mismatched_count > 0):
        print(f"\nWriting updated agent_summary.json...")
        with open(agent_summary_path, 'w') as f:
            json.dump(updated_data, f, indent=2, sort_keys=True)
        print("✅ agent_summary.json updated")

        # Sync to S3 if requested
        if args.sync_s3:
            print("\nSyncing to S3...")
            import subprocess
            result = subprocess.run(
                ['aws', 's3', 'cp', str(agent_summary_path), 's3://algotune-results-926341122891/agent_summary.json'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ Synced to S3")
            else:
                print(f"❌ S3 sync failed: {result.stderr}", file=sys.stderr)
                sys.exit(1)
    elif args.dry_run:
        print("\n[DRY RUN] No changes made. Run without --dry-run to apply changes.")
    else:
        print("\nNo changes needed.")


if __name__ == '__main__':
    main()
