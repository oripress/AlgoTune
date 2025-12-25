#!/usr/bin/env python3
"""
Analyze local AlgoTune log files to determine success/failure.
Heuristic: if a log does not consume its full budget, mark as failed.
"""
import argparse
import re
from pathlib import Path


BUDGET_TOTAL_RE = re.compile(r"Budget:\s*\$([0-9]+(?:\.[0-9]+)?)")
USED_UP_RE = re.compile(
    r"used up\s*\$([0-9]+(?:\.[0-9]+)?)[^$]*You have\s*\$([0-9]+(?:\.[0-9]+)?)\s*remaining",
    re.IGNORECASE,
)
CURRENT_SPEND_RE = re.compile(r"Current spend:\s*\$([0-9]+(?:\.[0-9]+)?)")
SPEND_LIMIT_RE = re.compile(r"Spend limit of\s*\$([0-9]+(?:\.[0-9]+)?)\s*reached", re.IGNORECASE)
ERROR_MARKERS = [
    "BadRequestError",
    "Exception in model.query",
    "Solver validation failed",
    "No response received",
    "Traceback (most recent call last):",
]


def parse_log_file(log_path: Path):
    budget_total = None
    budget_used = None
    used_remaining = None
    spend_limit_reached = False
    errors = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if budget_total is None:
                match = BUDGET_TOTAL_RE.search(line)
                if match:
                    budget_total = float(match.group(1))

            if budget_used is None:
                match = CURRENT_SPEND_RE.search(line)
                if match:
                    budget_used = float(match.group(1))

            if not spend_limit_reached and SPEND_LIMIT_RE.search(line):
                spend_limit_reached = True

            if used_remaining is None:
                match = USED_UP_RE.search(line)
                if match:
                    used = float(match.group(1))
                    remaining = float(match.group(2))
                    used_remaining = (used, remaining)

            for marker in ERROR_MARKERS:
                if marker in line:
                    errors.append(marker)
                    break

    if used_remaining and budget_used is None:
        budget_used = used_remaining[0]
    if used_remaining and budget_total is None:
        budget_total = sum(used_remaining)

    return budget_total, budget_used, spend_limit_reached, errors


def parse_log_name(filename: str):
    match = re.match(r"^(?P<task>.+)_(?P<model>.+)_\d{8}_\d{6}\.log$", filename)
    if not match:
        return None, None
    return match.group("task"), match.group("model")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze local logs and classify success/failure based on budget usage."
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory containing .log files (default: logs/)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter by model display name (e.g., gemini-3-flash)"
    )
    parser.add_argument(
        "--tasks-file",
        default=None,
        help="Optional file listing tasks to include (one per line)"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    tasks_filter = None
    if args.tasks_file:
        tasks_path = Path(args.tasks_file)
        tasks_filter = {line.strip() for line in tasks_path.read_text().splitlines() if line.strip()}

    results = []
    for log_path in sorted(log_dir.glob("*.log")):
        task, model = parse_log_name(log_path.name)
        if task is None:
            continue
        if args.model and model != args.model:
            continue
        if tasks_filter is not None and task not in tasks_filter:
            continue

        budget_total, budget_used, spend_limit_reached, errors = parse_log_file(log_path)
        used_full_budget = False
        if spend_limit_reached:
            used_full_budget = True
        elif budget_total is not None and budget_used is not None:
            used_full_budget = budget_used >= (budget_total - 1e-6)

        status = "SUCCESS" if used_full_budget else "FAILED"
        reason_parts = []
        if not used_full_budget:
            if budget_total is None or budget_used is None:
                reason_parts.append("budget not fully consumed (missing usage info)")
            else:
                reason_parts.append(f"budget used ${budget_used:.4f} / ${budget_total:.4f}")
        if errors:
            reason_parts.append("errors: " + ", ".join(sorted(set(errors))))

        results.append({
            "task": task,
            "model": model,
            "status": status,
            "reason": "; ".join(reason_parts) if reason_parts else "",
            "log_path": str(log_path),
            "budget_total": budget_total,
            "budget_used": budget_used,
        })

    if not results:
        print("No matching logs found.")
        return

    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_count = len(results) - success_count
    finished_count = len(results)

    usage_ratios = []
    for result in results:
        budget_total = result.get("budget_total")
        budget_used = result.get("budget_used")
        if budget_total and budget_used is not None:
            usage_ratios.append(budget_used / budget_total)

    print("LOG ANALYSIS")
    print("────────────")
    print(f"Finished:   {finished_count}")
    print(f"Success:    {success_count}")
    print(f"Failed:     {failed_count}")
    if usage_ratios:
        avg_usage = sum(usage_ratios) / len(usage_ratios)
        print(f"Avg budget: {avg_usage:.1%}")
    print()

    for result in results:
        line = f"{result['status']}: {result['task']} ({result['model']})"
        if result["reason"]:
            line += f" - {result['reason']}"
        print(line)


if __name__ == "__main__":
    main()
