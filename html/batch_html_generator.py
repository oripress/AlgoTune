import base64
import html
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from multiprocessing import cpu_count, Pool

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# --- Log Current Working Directory ---
CWD = os.getcwd()
print(f"Script starting. Current Working Directory: {CWD}")
# -----------------------------------

# Configuration
LOGS_DIR = "logs/"
OUTPUT_DIR = "static_site/site/"  # Changed to generate static site structure
PLOTS_DIR = os.path.join(OUTPUT_DIR, "assets", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
TASKS_DIR = "AlgoTuneTasks"
SYSTEM_PROMPT_FILE = "messages/initial_system_message.txt"
GENERATION_FILE = os.path.join("reports", "generation.json")  # Primary task source
WHITELIST_FILE = os.path.join(os.path.dirname(__file__), "whitelist.txt")  # Fallback task list

# --- Regex Constants ---
LOG_LINE_REGEX = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+(?P<level>\w+)\s+-\s+(?P<rest>.*)",
    re.DOTALL,
)
CONFIG_REGEX = re.compile(r"Config loaded:\s*([^\n]+)", re.IGNORECASE)
PROMPT_REGEX = re.compile(r"--BEGIN DEMONSTRATION--.*?--END OF DEMONSTRATION--", re.DOTALL)
FAILED_EDIT_REGEX = re.compile(r"(Edit failed|Cannot apply edit)", re.IGNORECASE)

# --- Mappings for Template Rendering ---
ICON_MAPPING = {
    "ls": "📄",
    "view_file": "👁️",
    "edit": "✏️",
    "oracle": "🔮",
    "eval_input": "🧪",
    "revert": "↩️",
    "other": "",
}
CMD_DISPLAY = {
    "ls": "List Files",
    "view_file": "View File",
    "edit": "Code Edit",
    "oracle": "Oracle Query",
    "profile": "Profile",
    "profile_lines": "Profile Lines",
    "eval_input": "Input Eval",
    "eval": "Training Set Eval",
    "revert": "Revert Changes",
    "other": "Invalid Command",
}


def get_speedup_color(speedup_value):
    """Return an RGB color string for a given speed-up.

    Mapping:
    • Fail / None / invalid  → deep dark red (#B22222)
    • <1.005×               → orange (#D2691E)
    • ≥1.005×               → green (#006400)
    """

    # Fail or missing score
    try:
        if speedup_value is None:
            raise ValueError
        val = float(speedup_value)
        if math.isnan(val):
            raise ValueError
    except (ValueError, TypeError):
        return "#B22222"  # Darker red for fail - better contrast

    if val < 1.005:
        return "#D2691E"  # Dark orange for <1.005x
    else:
        return "#006400"  # Dark green for speedups >= 1.005x


def strip_thinking_blocks(text):
    """
    Remove <think>...</think> blocks from the text.
    """
    import re

    # Remove <think>...</think> blocks (case-insensitive, multiline)
    pattern = r"<think\b[^>]*>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()


# --- Model Name Mapping ---
def clean_model_name(raw_model_name):
    """Convert internal model identifiers to readable display names."""
    if not raw_model_name:
        return "Unknown Model"

    # Mapping dictionary for model name cleanup
    model_mappings = {
        # Gemini models
        "gemini-2.5-pro-preview-06-05": "Gemini 2.5 Pro 0605",
        "gemini/gemini-2.5-pro": "Gemini 2.5 Pro",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "gemini-3-flash-preview": "Gemini 3 Flash Preview",
        "gemini-3-pro-preview": "Gemini 3 Pro Preview",
        # DeepSeek models
        "DeepSeek-R1": "DeepSeek R1",
        "deepseek-ai/DeepSeek-R1": "DeepSeek R1",
        "deepseek-r1": "DeepSeek R1",
        "deepseek-reasoner": "DeepSeek R1",
        "deepseek/deepseek-reasoner": "DeepSeek R1",
        # Claude models
        "claude-opus-4-20250514": "Claude Opus 4",
        "claude-opus-4": "Claude Opus 4",
        "claude-opus-4.1": "Claude Opus 4.1",
        "claude-opus-4-1-20250805": "Claude Opus 4.1",
        "anthropic/claude-opus-4-1-20250805": "Claude Opus 4.1",
        "claude-opus-4.5": "Claude Opus 4.5 (medium)",
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
        "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
        "anthropic/claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
        "claude-sonnet-4-5": "Claude Sonnet 4.5",
        "claude-sonet-4-5": "Claude Sonnet 4.5",
        "claude-sonet-4.5": "Claude Sonnet 4.5",
        "sonet-4-5": "Claude Sonnet 4.5",
        "sonet-4.5": "Claude Sonnet 4.5",
        # GPT models
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4o": "GPT-4o",
        "gpt-5-mini": "GPT-5 Mini",  # Check gpt-5-mini BEFORE gpt-5
        "gpt-5": "GPT-5",
        "gpt-5-pro": "GPT-5 Pro (medium)",
        "gpt-5.2": "GPT-5.2 (medium)",
        "gpt-oss-120b": "GPT-OSS-120B",
        # OpenAI o-series
        "o3": "o3",
        "o4-mini": "o4-mini",
        # Chinese models
        "qwen3-coder": "Qwen3 Coder",
        "glm-4.5": "GLM-4.5",
        "glm-4.7": "GLM-4.7",
        "kimi-k2": "Kimi K2",
        # Minimax
        "minimax-m2.1": "Minimax M2.1",
    }

    # Check for exact matches first
    if raw_model_name in model_mappings:
        return model_mappings[raw_model_name]

    # Extract model identifier from log filename if it's a full filename
    # Pattern: taskname_modelname_timestamp.log
    import re

    filename_pattern = re.compile(r"^[^_]+_(.+?)_\d{8}_\d{6}(?:\.log)?$")
    filename_match = filename_pattern.match(raw_model_name)
    if filename_match:
        model_part = filename_match.group(1)
        # Check if this extracted model part matches any mapping
        if model_part in model_mappings:
            return model_mappings[model_part]

    # Smart partial matching - check longer patterns first to avoid substring confusion
    # Sort by length descending to match longer patterns first (gpt-5-mini before gpt-5)
    sorted_mappings = sorted(model_mappings.items(), key=lambda x: len(x[0]), reverse=True)
    for pattern, display_name in sorted_mappings:
        # Use underscore or space as delimiters for matching
        # This ensures "gpt-5" doesn't match in "task_gpt-5-mini_timestamp"
        if pattern.lower() in raw_model_name.lower():
            # Check if it's a proper match (not part of a longer model name)
            idx = raw_model_name.lower().find(pattern.lower())
            if idx != -1:
                # Check what comes after the pattern
                after_idx = idx + len(pattern)
                if after_idx >= len(raw_model_name) or raw_model_name[after_idx] in (
                    "_",
                    " ",
                    "-",
                    ".",
                ):
                    # Also check what comes before for completeness
                    if idx == 0 or raw_model_name[idx - 1] in ("_", " ", "/", "-"):
                        return display_name

    # Fallback: clean up the raw name
    # Remove common prefixes and make more readable
    cleaned = raw_model_name.replace("-", " ").replace("_", " ")
    # Capitalize words
    cleaned = " ".join(word.capitalize() for word in cleaned.split())

    return cleaned


def get_model_logo(clean_model_name):
    """Get the logo filename for a given clean model name."""
    logo_mapping = {
        "Claude Opus 4": "claude_logo.png",
        "Claude Opus 4.1": "claude_logo.png",
        "Claude Opus 4.5": "claude_logo.png",
        "Claude 3.5 Sonnet": "claude_logo.png",
        "Claude 3.7 Sonnet": "claude_logo.png",
        "Claude Sonnet 4.5": "claude_logo.png",
        "DeepSeek R1": "deepseek_logo.png",
        "DeepSeek Reasoner": "deepseek_logo.png",
        "Gemini 2.5 Pro": "gemini_logo.png",
        "Gemini 2.5 Pro 0605": "gemini_logo.png",
        "Gemini 3 Flash Preview": "gemini_logo.png",
        "Gemini 3 Pro Preview": "gemini_logo.png",
        "o3": "openai_logo.png",
        "o4-mini": "openai_logo.png",
        "GPT-4o Mini": "openai_logo.png",
        "GPT-4o": "openai_logo.png",
        "GPT-5": "openai_logo.png",
        "GPT-5 Mini": "openai_logo.png",
        "GPT-5 Pro": "openai_logo.png",
        "GPT-5.2": "openai_logo.png",
        "GPT-OSS-120B": "openai_logo.png",
        "Qwen3 Coder": "qwen_logo.png",
        "GLM-4.5": "z_logo.png",
        "GLM-4.7": "z_logo.png",
        "Kimi K2": "moonshot_logo.png",
        "Minimax M2.1": "minimax_logo.png",
    }

    return logo_mapping.get(clean_model_name, None)


# --- Data Structures ---
@dataclass
class TrainingEvaluationResult:
    performance_score: float | None = None
    used_budget: float | None = None
    your_time: float | None = None
    oracle_time: float | None = None
    is_new_best: bool = False


@dataclass
class TestEvaluationResult:
    performance_score: float | None = None
    your_time: float | None = None
    oracle_time: float | None = None


@dataclass
class LogExtractionResult:
    conversation: list = field(default_factory=list)
    training_results: list[TrainingEvaluationResult] = field(default_factory=list)
    command_sequence: list = field(default_factory=list)
    initial_prompts: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    dates: list = field(default_factory=list)
    test_score_result: TestEvaluationResult | None = None
    # Internal state tracked during processing, might not be needed in final result
    best_so_far_score: float | None = None
    pending_edit_index: int | None = None
    last_used_budget: float | None = None
    # New fields for collapsible sections
    initial_system_prompt: str = ""
    task_description: str = ""
    reference_implementation: str = ""
    # Map of filename -> code for the best snapshot(s)
    best_code: dict[str, str] = field(default_factory=dict)


def extract_total_budget_from_log(content):
    """
    Extract the total budget (used + remaining) from log content.
    """
    budget_pattern = re.compile(
        r"used up\s*\$([0-9]+(?:\.[0-9]+)?)[^$]*You have\s*\$([0-9]+(?:\.[0-9]+)?)\s*remaining[\.\,]?",
        re.DOTALL | re.IGNORECASE,
    )
    match = budget_pattern.search(content)
    if match:
        try:
            spent = float(match.group(1))
            remaining = float(match.group(2))
            total = spent + remaining
            # Debug info available if needed
            pass
            return total
        except (ValueError, TypeError):
            pass
    pass  # Total budget not found
    return None


def extract_used_budget_from_log(content):
    """
    Extract only the used (spent) amount from log content.
    It looks for the first occurrence of "used up $<number>".
    """
    used_pattern = re.compile(r"used up\s*\$([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    match = used_pattern.search(content)
    if match:
        try:
            used = float(match.group(1))
            pass  # Used budget extracted
            return used
        except (ValueError, TypeError):
            pass
    pass  # Used budget not found
    return None


def get_baseline_implementation(task_name):
    """
    Searches the tasks directory for a folder corresponding to the task.
    In that folder, finds a .py file (excluding __init__.py) that has a
    @register_task('<task_name>') marker, and extracts the "solve" function.
    The regex used allows leading whitespace (to capture indented definitions)
    and captures until the next top-level (or same-level) definition.
    Returns the extracted code as a string.
    """
    tasks_dir = "tasks"
    for entry in os.scandir(tasks_dir):
        if entry.is_dir() and entry.name.startswith("task_"):
            for file in os.scandir(entry.path):
                if file.name.endswith(".py") and not file.name.startswith("__"):
                    try:
                        with open(file.path, encoding="utf-8") as f:
                            content = f.read()
                            if (
                                f"@register_task('{task_name}')" in content
                                or f'@register_task("{task_name}")' in content
                            ):
                                # Find the class that contains the solve method
                                class_pattern = re.compile(
                                    r"@register_task\(['\"]"
                                    + re.escape(task_name)
                                    + r"['\"]\)\s*\n\s*class\s+(\w+)\(.*?\):",
                                    re.DOTALL,
                                )
                                class_match = class_pattern.search(content)
                                if class_match:
                                    class_name = class_match.group(1)
                                    # Match "def solve(" with possible indentation within the class
                                    solve_pattern = re.compile(
                                        r"(^\s+def\s+solve\(self,.*?\):\s*\n(?:\s+.*\n)+?)(?=\s+def\s|\Z|^\s*$)",
                                        re.MULTILINE | re.DOTALL,
                                    )
                                    solve_match = solve_pattern.search(content)
                                    if solve_match:
                                        baseline = solve_match.group(1).rstrip()
                                        # Remove indentation from the extracted code
                                        baseline_lines = baseline.splitlines()
                                        # Find the minimum indentation level (excluding empty lines)
                                        min_indent = min(
                                            (len(line) - len(line.lstrip()))
                                            for line in baseline_lines
                                            if line.strip()
                                        )
                                        # Remove that amount of indentation from each line
                                        baseline = "\n".join(
                                            line[min_indent:] if line.strip() else line
                                            for line in baseline_lines
                                        )
                                        # Baseline implementation extracted successfully
                                        pass
                                        return baseline
                    except Exception as e:
                        if logging.getLogger().isEnabledFor(logging.WARNING):
                            logging.warning(f"Error reading {file.path}: {e}")
    # Baseline implementation not found
    return "Baseline implementation not found for task " + task_name


def parse_log_file(filepath):
    """
    Reads the log file, splits it into blocks, extracts task and model names,
    and computes the total budget for display.
    Returns (list_of_blocks, task_name, best_score_test, model_name, total_budget).
    """
    # Parsing log file
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    total_budget = extract_total_budget_from_log(content)

    # Extract model name - comprehensive regex patterns for all log formats
    model_name = None

    # List of regex patterns to try in order
    model_patterns = [
        re.compile(
            r"Using max_completion_tokens from config for model '([^']+)'"
        ),  # GPT-5-mini pattern
        re.compile(r"Using default max_output_tokens for model '([^']+)'"),  # GPT-5 pattern
        re.compile(r"Using max_tokens from config for model '([^']+)'"),
        re.compile(r"Max tokens for model\s+'([^']+)'"),
        re.compile(r"Initialized LiteLLMModel with model_name:\s+(\S+)"),
        re.compile(
            r"BaseLLMInterface __init__ started for model:\s+([\S]+)"
        ),  # Another common pattern
        re.compile(
            r"Initializing LiteLLMModel with params:.*'model_name':\s*'([^']+)'"
        ),  # From dict params
    ]

    # Try each pattern until we find a match
    for pattern in model_patterns:
        match = pattern.search(content)
        if match:
            model_name = match.group(1)
            break

    if not model_name:
        fallback_name = os.path.basename(filepath).replace(".log", "")
        model_name = fallback_name

    # Keep raw model name for accurate identification
    # Don't clean it here - clean it only when displaying

    # Extract task name with multiple fallback methods
    task_name = None

    # Method 1: Running LLM interface
    run_regex = re.compile(r"Running LLM interface for task ([^\.]+)\.{3}")
    run_match = run_regex.search(content)
    if run_match:
        task_name = run_match.group(1)

    # Method 2: Creating task instance
    if not task_name:
        create_regex = re.compile(r"Creating task instance for ([^\.]+)\.{3}")
        task_match = create_regex.search(content)
        if task_match:
            task_name = task_match.group(1)

    # Method 2b: TaskFactory pattern (new models use this)
    if not task_name:
        factory_regex = re.compile(r"TaskFactory: Set task_instance\.task_name to raw '([^']+)'")
        factory_match = factory_regex.search(content)
        if factory_match:
            task_name = factory_match.group(1)

    # Method 3: Extract from filename pattern
    if not task_name:
        filename = os.path.basename(filepath)
        # Pattern: task_name_model_timestamp.log
        # Examples: count_connected_components_o4-mini_20250610_030911.log
        #           count_riemann_zeta_zeros_DeepSeek-R1_20250611_225011.log

        # Remove .log extension
        name_part = filename.replace(".log", "")

        # Known model prefixes to help identify where task name ends
        # Order by length (longest first) to match most specific patterns first
        known_models = [
            "claude-opus-4-1-20250805",
            "claude-opus-4-20250514",
            "gemini-2.5-pro-preview",
            "deepseek-reasoner",
            "gemini-2.5-pro",
            "qwen3-coder",
            "gpt-oss-120b",
            "claude-opus-4",
            "DeepSeek-R1",
            "glm-4.5",
            "o4-mini",
        ]

        # Try to find where model name starts
        for model in known_models:
            if model in name_part:
                task_part = name_part.split(model)[0].rstrip("_")
                if task_part and len(task_part) > 2:
                    task_name = task_part
                    break

        # Fallback: assume task name is everything before the first date pattern
        if not task_name:
            # Look for date pattern like _20250610_
            date_match = re.search(r"_\d{8}_", name_part)
            if date_match:
                # Everything before the date pattern
                before_date = name_part[: date_match.start()]
                # Remove the last part (likely model name)
                parts = before_date.split("_")
                if len(parts) >= 2:
                    # Task name is likely first few parts, excluding the last (model)
                    task_name = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
            else:
                # No date pattern, try first part
                parts = name_part.split("_")
                if len(parts) > 1:
                    # Treat the final segment as the model name; everything before that is the task
                    task_name = "_".join(parts[:-1])
                elif parts:
                    candidate = parts[0]
                    if len(candidate) > 2:
                        task_name = candidate

    # Method 4: Look for task patterns in content
    if not task_name:
        task_patterns = [
            r"task[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"Problem[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"Solving[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)",
        ]
        for pattern in task_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                task_name = match.group(1)
                break

    if not task_name:
        if logging.getLogger().isEnabledFor(logging.WARNING):
            logging.warning(f"No task name found in {filepath}; defaulting to 'unknown'")
        task_name = "unknown"

    # --- Extract Final Test Performance Score ---
    final_test_score_result = None
    final_results_marker = "Final Test Results:"
    if final_results_marker in content:
        final_results_part = content.split(final_results_marker, 1)[1]

        # Use the existing regexes, but search only within this part
        perf_match = re.compile(
            r"Performance score(?:\s*\(.*\))?:\s*([\d\.]+)(?:x)?", re.IGNORECASE
        ).search(final_results_part)
        time_match = re.search(
            r"Average solver time:\s*([\d\.]+)\s*ms", final_results_part, re.IGNORECASE
        )
        # Look for Baseline time instead
        baseline_match = re.search(
            r"Average baseline time:\s*([\d\.]+)\s*ms", final_results_part, re.IGNORECASE
        )

        if perf_match:
            try:
                perf_score = float(perf_match.group(1))
                your_time = float(time_match.group(1)) if time_match else None
                baseline_time = float(baseline_match.group(1)) if baseline_match else None
                final_test_score_result = TestEvaluationResult(
                    performance_score=perf_score, your_time=your_time, oracle_time=baseline_time
                )
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(
                        f"Extracted Final Test Score from dedicated block: {final_test_score_result}"
                    )
            except (ValueError, AttributeError):
                if logging.getLogger().isEnabledFor(logging.WARNING):
                    logging.warning("Found 'Final Test Results:' but failed to parse score/times.")
                pass  # Fall through to other methods if parsing fails
        else:
            pass  # Final Test Results block found but no score
    else:
        pass  # Final Test Results marker not found

    # --- Legacy/Fallback Score Extraction (kept for now, might be removable) ---
    # Extract best test performance score from "After Run Results:" if final score wasn't found
    initial_best_test_score_val = None  # Used if final_test_score_result isn't populated
    if final_test_score_result is None:
        pass  # Falling back to After Run Results
        AFTER_RUN_MARKER = "After Run Results:"
        DATA_SUBSET_TEST = "Data Subset: test"
        # PERF_SCORE_REGEX defined earlier
        after_run_parts = content.split(AFTER_RUN_MARKER)
        if len(after_run_parts) > 1:
            after_run_content = after_run_parts[1]
            if DATA_SUBSET_TEST in after_run_content:
                test_part = after_run_content.split(DATA_SUBSET_TEST, 1)[1]
                perf_match_fallback = re.compile(
                    r"Performance score(?:\s*\(.*\))?:\s*([\d\.]+)(?:x)?", re.IGNORECASE
                ).search(test_part)
                if perf_match_fallback:
                    try:
                        initial_best_test_score_val = float(perf_match_fallback.group(1))
                    except ValueError:
                        pass
            if (
                initial_best_test_score_val is None
            ):  # If not found in test subset or no subset exists
                perf_match_fallback = re.compile(
                    r"Performance score(?:\s*\(.*\))?:\s*([\d\.]+)(?:x)?", re.IGNORECASE
                ).search(after_run_content)
                if perf_match_fallback:
                    try:
                        initial_best_test_score_val = float(perf_match_fallback.group(1))
                    except ValueError:
                        pass
        # Fallback to searching the entire content if still no match
        if initial_best_test_score_val is None:
            BEST_SCORE_REGEX_FALLBACK = re.compile(
                r"Performance Score \((?:ratio|speedup), (?:lower|higher) is better!\):\s*([\d\.]+)x",
                re.IGNORECASE,
            )
            fallback_match = BEST_SCORE_REGEX_FALLBACK.search(content)
            if fallback_match:
                try:
                    initial_best_test_score_val = float(fallback_match.group(1))
                except ValueError:
                    pass
        if initial_best_test_score_val is not None:
            pass  # Found fallback test score
        else:
            pass  # No test score found via fallback

    # Define PERF_SCORE_REGEX here if not already global, needed by score extraction logic below
    PERF_SCORE_REGEX = re.compile(
        r"Performance score(?:\s*\(.*\))?:\s*([\d\.]+)(?:x)?", re.IGNORECASE
    )

    # Split content into log blocks
    LOG_MESSAGE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+-\s+\w+\s+-\s+")
    messages = []
    current_block = []
    lines = content.split("\n")
    for line in lines:
        if LOG_MESSAGE_REGEX.match(line):
            if current_block:
                messages.append("\n".join(current_block))
            current_block = [line]
        else:
            if current_block is not None:
                current_block.append(line)
    if current_block:
        messages.append("\n".join(current_block))

    # Log file parsing completed
    # Return the TestEvaluationResult object if found, otherwise the fallback float value
    # The caller (process_log_file_data) will prioritize the object if present
    return (
        messages,
        task_name,
        final_test_score_result if final_test_score_result else initial_best_test_score_val,
        model_name,
        total_budget,
    )


def extract_budget_from_filename(filename):
    parts = filename.replace(".log", "").split("_")
    if len(parts) >= 6 and parts[-2] == "budget":
        return parts[-1]
    return "N/A"


def get_error_context(lines, error_line_index, context_before=3, context_after=2):
    start_idx = max(0, error_line_index - context_before)
    end_idx = min(len(lines), error_line_index + context_after + 1)
    context_lines = []
    for i in range(start_idx, end_idx):
        is_error = i == error_line_index
        context_lines.append((lines[i], is_error))
    return context_lines


def format_and_escape(text):
    # Early return for empty text
    if not text or not text.strip():
        return ""

    text = text.replace("\\n", "\n")

    # Lazy evaluation: only compile regex if text contains code blocks
    if "```" not in text:
        # Simple case: just escape and convert newlines
        escaped = html.escape(text)
        # Replace literal <br> tags with newlines before escaping
        escaped = escaped.replace("&lt;br&gt;", "\n")
        return escaped.replace("\n", "<br>")

    code_pattern = re.compile(r"```(\w*)\s*\n?(.*?)\n?```", re.DOTALL | re.MULTILINE)
    segments = []
    last_end = 0
    for match in code_pattern.finditer(text):
        if match.start() > last_end:
            segment = text[last_end : match.start()]
            segment = html.escape(segment)
            # Replace literal <br> tags with newlines before escaping
            segment = segment.replace("&lt;br&gt;", "\n")
            segment = segment.replace("\n", "<br>")
            segments.append(segment)
        language = match.group(1).strip()
        code = match.group(2).strip("\n")
        code = html.escape(code)
        lang_display = language.upper() if language else "PYTHON"
        lang_class = f"language-{language.lower()}" if language else "language-python"

        # Generate unique ID for each code block
        import hashlib

        code_id = hashlib.md5(code.encode()).hexdigest()[:8]

        # Determine if code needs expand/collapse (more than 15 lines)
        code_lines = code.count("\n") + 1
        needs_expand = code_lines > 15
        expand_class = " code-expandable" if needs_expand else ""

        expand_button = (
            f'<button class="code-expand-btn" onclick="toggleExpand(\'{code_id}\')"></button>'
            if needs_expand
            else ""
        )
        content_class = " expanded" if not needs_expand else ""

        # Use a much simpler markup for code blocks (no header or copy button)
        segments.append(f'<pre><code class="{lang_class}">{code}</code></pre>')
        last_end = match.end()
    if last_end < len(text):
        segment = text[last_end:]
        segment = html.escape(segment)
        # Replace literal <br> tags with newlines before escaping
        segment = segment.replace("&lt;br&gt;", "\n")
        segment = segment.replace("\n", "<br>")
        segments.append(segment)
    result = "".join(segments)
    lines = result.split("<br>")
    formatted_lines = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        is_error = "ERROR" in line or "Exception" in line
        if is_error and not line.startswith("<pre"):
            context = get_error_context(lines, idx)
            for ctx_line, is_err in context:
                if is_err:
                    formatted_lines.append(f'<span class="error-line">{ctx_line}</span>')
                else:
                    formatted_lines.append(f'<span class="context-line">{ctx_line}</span>')
            idx += len(context)
        else:
            formatted_lines.append(line)
            idx += 1
    return "<br>".join(formatted_lines)


def remove_code_blocks(text):
    """
    Remove all code blocks (```...```) from text, leaving only the regular text.
    """
    import re

    # Pattern to match code blocks with optional language specifier
    code_pattern = re.compile(r"```(\w*)\s*\n?(.*?)\n?```", re.DOTALL | re.MULTILINE)

    # Replace all code blocks with empty string
    result = code_pattern.sub("", text)

    # Clean up any extra whitespace left behind
    lines = result.split("\n")
    cleaned_lines = []

    for line in lines:
        # Keep the line but strip trailing whitespace
        cleaned_lines.append(line.rstrip())

    # Join back and remove excessive blank lines
    result = "\n".join(cleaned_lines)

    # Replace multiple consecutive newlines with at most 2
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def format_system_text(text):
    """
    Format system messages so that any content after the first two lines is
    rendered inside a <pre><code> block. This is mainly to improve the
    readability of "Contents of …" and "Proposed changes …" messages where the
    bulk of the message is source code or a diff. On both desktop and mobile we
    want the following behaviour:

    1. Preserve the first two lines as plain text (they provide context such as
       the filename and the line-range information).
    2. Wrap all subsequent lines in a syntax-highlighted code block.  We default
       to the *python* language as these snippets are almost always Python /
       unified diff.  Prism will gracefully fall back if the language token is
       not perfect.

    If the system message is shorter than three lines we fall back to the old
    behaviour (plain escaped text).
    """

    # Early return for empty or whitespace-only messages
    if not text or not text.strip():
        return ""

    # Normalise escaped newlines that sometimes appear in the raw logs
    text = text.replace("\\n", "\n")

    # Split into individual lines **before** any HTML escaping so indices are
    # accurate.
    lines = text.splitlines()

    # Determine how many lines constitute the non-code header.  For most system
    # messages we keep the first **two** lines (e.g. "Contents of …" and the
    # range line).  For "Code Context:" messages we keep only the very first
    # line as header so that the numbered lines appear inside the code block.

    import re

    first_line = lines[0].strip()
    header_line_count = 2  # default
    if re.match(r"^(Code Context:|Invalid Example\b)", first_line):
        header_line_count = 1

    # If we have enough lines, move the remainder into a code block.
    if len(lines) > header_line_count:
        header_raw = "\n".join(lines[:header_line_count])
        code_raw = "\n".join(lines[header_line_count:])
    else:
        header_raw = "\n".join(lines)
        code_raw = ""

    # Escape the header (but keep newlines as <br>) and color speedup numbers
    def _color_speedup_in_text(text: str) -> str:
        """Color speedup numbers in regular text (not code blocks)"""

        # Regex to capture "Speedup: <value>" (case-insensitive, optional whitespace, optional trailing 'x')
        def replace_speedup(match):
            prefix = match.group(1)
            value_str = match.group(2)
            suffix = match.group(3)

            # Determine colour
            value_lower = value_str.lower()
            if value_lower in {"n/a", "na"}:
                colour = get_speedup_color(None)
            else:
                try:
                    val = float(value_str)
                except ValueError:
                    colour = get_speedup_color(None)
                else:
                    colour = get_speedup_color(val)

            return (
                f'{prefix}<span style="color:{colour}; font-weight:600;">{value_str}</span>{suffix}'
            )

        return re.sub(
            r"(Speedup:\s*)([-+]?[0-9]*\.?[0-9]+|N/?A)(x?)",
            replace_speedup,
            text,
            flags=re.IGNORECASE,
        )

    header_html = _color_speedup_in_text(html.escape(header_raw)).replace("\n", "<br>")

    # --- Identify and wrap only the "true" code lines ---------------------
    def _is_code_like(l: str) -> bool:
        l_strip = l.lstrip()
        # Treat lines that begin with an ellipsis ("...", optionally followed by other chars) as code so that
        # they stay within the surrounding <pre><code> block.  This prevents stray ellipsis lines from being
        # pushed outside when the code context is truncated with "..." placeholders.
        if l_strip.startswith("..."):
            return True

        # Matches code context lines that typically look like "92: code", "| 07: code", "> 107: code", "6 : code" etc.
        # Allow an optional space between the digits and the colon to handle variants such as "6 :".
        return bool(re.match(r"^[|>!?]?\s*\d+\s*:", l_strip))

    formatted_parts: list[str] = [header_html]

    if code_raw:
        current_buf: list[str] = []
        current_is_code = None  # unknown

        for ln in code_raw.split("\n"):
            line_is_code = _is_code_like(ln)

            if current_is_code is None:
                current_is_code = line_is_code
            elif line_is_code != current_is_code:
                # Flush buffer
                segment = "\n".join(current_buf)
                if current_is_code:
                    formatted_parts.append(_build_code_block(segment))
                else:
                    formatted_parts.append(
                        _color_speedup_in_text(html.escape(segment)).replace("\n", "<br>")
                    )
                current_buf = []
                current_is_code = line_is_code

            current_buf.append(ln)

        # Flush last segment
        if current_buf:
            segment = "\n".join(current_buf)
            if current_is_code:
                formatted_parts.append(_build_code_block(segment))
            else:
                formatted_parts.append(
                    _color_speedup_in_text(html.escape(segment)).replace("\n", "<br>")
                )

    return "<br>".join(part for part in formatted_parts if part)


def _build_code_block(code_text: str) -> str:
    """Return a <pre><code> block with Speedup numbers colour-coded."""

    def _colour_speedup_line(line: str) -> str:
        # Regex to capture "Speedup: <value>" (case-insensitive, optional whitespace, optional trailing 'x')
        m = re.search(r"(Speedup:\s*)([-+]?[0-9]*\.?[0-9]+|N/?A)(x?)", line, re.IGNORECASE)
        if not m:
            return html.escape(line)

        prefix, value_str, suffix = m.groups()

        # Determine colour
        value_lower = value_str.lower()
        if value_lower in {"n/a", "na"}:
            colour = get_speedup_color(None)
        else:
            try:
                val = float(value_str)
            except ValueError:
                colour = get_speedup_color(None)
            else:
                colour = get_speedup_color(val)

        # Assemble with HTML escaping for non-value parts
        pre_part = html.escape(line[: m.start(1)])
        post_part = html.escape(line[m.end(3) :])

        return (
            pre_part
            + html.escape(prefix)
            + f'<span style="color:{colour}; font-weight:600;">{html.escape(value_str)}</span>'
            + html.escape(suffix)
            + post_part
        )

    coloured_lines = [_colour_speedup_line(ln) for ln in code_text.split("\n")]
    code_html = "\n".join(coloured_lines)
    return f'<pre><code class="language-python">{code_html}</code></pre>'


def _filter_best_code(best_code: dict[str, str]) -> dict[str, str]:
    """Return a subset of *best_code* containing only solver.py and any
    other files that are (directly or transitively) imported by it.

    The heuristic is simple: we scan for top-level ``import foo`` or ``from foo``
    statements, map *foo* → ``foo.py`` (first dotted component), and keep the
    file if it exists in *best_code*.  We then repeat the process for each new
    file discovered until no new files are added.
    """

    # Early exit if solver.py not present or best_code is not a mapping
    if not isinstance(best_code, dict) or "solver.py" not in best_code:
        return best_code

    import_pattern = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][\w\.]*)", re.MULTILINE)

    # Extensions we consider as source files for a Python import – this covers
    # regular Python, Cython and generated C/C++ as well as headers.
    _SOURCE_EXTS: tuple[str, ...] = (
        ".py",
        ".pyx",
        ".pxd",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
    )

    used_files: set[str] = {"solver.py"}
    queue: list[str] = ["solver.py"]

    while queue:
        fname = queue.pop()
        code_text = best_code.get(fname, "")
        if not isinstance(code_text, str):
            continue

        # Look for top-level import statements and map them to candidate files
        for match in import_pattern.finditer(code_text):
            module_path = match.group(1).lstrip(".")  # strip any relative dots
            base_mod = module_path.split(".")[0]
            for ext in _SOURCE_EXTS:
                candidate = f"{base_mod}{ext}"
                if candidate in best_code and candidate not in used_files:
                    used_files.add(candidate)
                    queue.append(candidate)

    # Include setup.py only if we detected any non-Python source files (Cython/C/C++)
    compilation_exts = (".pyx", ".pxd", ".c", ".cc", ".cpp", ".h", ".hpp")
    needs_setup = any(fname.endswith(compilation_exts) for fname in used_files)
    if needs_setup and "setup.py" in best_code:
        used_files.add("setup.py")

    # Build filtered dict preserving original order where possible
    return {fname: best_code[fname] for fname in best_code if fname in used_files}


def extract_command(raw_message: str) -> str:
    # Early return if no code blocks
    if "```" not in raw_message:
        return "other"

    # Quick check for common commands in the message
    message_lower = raw_message.lower()
    for cmd in ["ls", "view_file", "edit", "oracle", "eval_input", "revert"]:
        if cmd in message_lower:
            break
    else:
        return "other"  # No command keywords found

    # Updated regex pattern to ensure ``` are on their own line
    # This pattern requires:
    # 1. ``` to be at the start of a line (after optional whitespace)
    # 2. Optional language identifier (python, etc)
    # 3. Content
    # 4. ``` to be on its own line at the end
    backtick_blocks = re.finditer(
        r"^[ \t]*```(?:python)?\s*\n(.*?)\n[ \t]*```[ \t]*$", raw_message, re.DOTALL | re.MULTILINE
    )
    for block in backtick_blocks:
        content = block.group(1).strip()
        first_line = content.split("\n")[0].strip()
        first_word = first_line.split()[0] if first_line else ""
        if first_word in ["ls", "view_file", "edit", "oracle", "eval_input", "revert"]:
            return first_word
    return "other"


def parse_training_evaluation(block: str) -> TrainingEvaluationResult:
    """
    Parse a single training evaluation block to extract:
      - performance_score (float)
      - used_budget (float)
      - your_average_time (float)
      - optimal_average_time (float)
      - is_new_best (bool)

    Returns a TrainingEvaluationResult instance.
    If the block does not match the expected patterns, returns an instance
    with default (None) values.
    """
    result = TrainingEvaluationResult()  # Initialize with defaults
    if not block:
        return result

    # Block parsing (debug info removed for performance)

    # Look for something like "Performance Score (...): 0.1x" or "Performance score: 6.67" or "Speedup: 2.86x"
    # Make the parenthesized part and the trailing 'x' optional
    perf_match = re.search(
        r"(?:Performance score(?:\s*\(.*\))?|Speedup):\s*([\d\.]+)(?:x)?", block, re.IGNORECASE
    )
    # Look for "Your average time: 152.8 ms" or "Average solver time: ..."
    your_time_match = re.search(
        r"(?:Your average time|Average solver time):\s*([\d\.]+)\s*ms", block, re.IGNORECASE
    )
    # Look for "Average baseline time: ..." instead of Oracle/Optimal time
    baseline_time_match = re.search(
        r"Average baseline time:\s*([\d\.]+)\s*ms", block, re.IGNORECASE
    )

    # Extract used budget from log text (you had this function in your code).
    used_budget = extract_used_budget_from_log(block)

    # Check for "New best performance achieved"
    is_new_best = "New best performance achieved" in block

    # Regex matches processed (debug removed for performance)

    # If there's no performance score, skip (this block doesn't represent training eval)
    if not perf_match:
        return result  # Return default instance

    # Convert the captures to floats (or None if invalid)
    try:
        result.performance_score = float(perf_match.group(1))
    except (ValueError, AttributeError):
        pass  # Keep default None

    try:
        result.your_time = float(your_time_match.group(1)) if your_time_match else None
    except ValueError:
        pass  # Keep default None

    try:
        result.oracle_time = float(baseline_time_match.group(1)) if baseline_time_match else None
    except ValueError:
        pass  # Keep default None

    result.used_budget = used_budget  # Already extracted
    result.is_new_best = is_new_best  # Already extracted

    # Training evaluation parsed

    return result


def parse_test_score_from_block(
    block: str, current_test_result: TestEvaluationResult | None
) -> TestEvaluationResult | None:
    """Parse test evaluation block to extract performance score, your average time, and optimal average time.

    This function treats a block as a test evaluation block if it:
      - Contains "Final Test Speedup (mean):" (highest priority)
      - OR does NOT contain "Sent to LLM:" AND contains both "Performance Score" and "Your average time:" markers.

    Returns a TestEvaluationResult instance if found, otherwise returns the current_test_result.
    """
    if not block:
        return current_test_result

    # First check for "Final Test Speedup (mean)" - this takes priority
    final_test_match = re.search(r"Final Test Speedup \(mean\):\s*([\d\.]+)", block, re.IGNORECASE)
    if final_test_match:
        try:
            perf_score = float(final_test_match.group(1))
            test_result = TestEvaluationResult(
                performance_score=perf_score,
                your_time=None,  # Final test speedup doesn't include timing details
                oracle_time=None,
            )
            return test_result
        except (ValueError, AttributeError):
            pass

    # Skip blocks that belong to training evaluations
    if "Sent to LLM:" in block:
        return current_test_result

    # Ensure the block likely contains test evaluation details.
    if (
        "Performance Score" not in block and "Speedup:" not in block
    ) or "Your average time:" not in block:
        return current_test_result

    performance_score_regex = re.compile(
        r"(?:Performance Score \((?:ratio|speedup), (?:lower|higher) is better!\)|Speedup):\s*([\d\.]+)x",
        re.IGNORECASE,
    )
    time_regex = re.compile(r"Your average time:\s*([\d\.]+)\s*ms", re.IGNORECASE)
    # optimal_time_regex = re.compile(r"(?:Optimal average|Average oracle) time:\s*([\d\.]+)\s*ms", re.IGNORECASE) # OLD REGEX
    baseline_time_regex = re.compile(
        r"Average baseline time:\s*([\d\.]+)\s*ms", re.IGNORECASE
    )  # NEW REGEX

    perf_match = performance_score_regex.search(block)
    time_match = time_regex.search(block)
    # optimal_match = optimal_time_regex.search(block) # OLD
    baseline_match = baseline_time_regex.search(block)  # NEW

    if perf_match:
        try:
            perf_score = float(perf_match.group(1))
            your_time = float(time_match.group(1)) if time_match else None
            # optimal_time = float(optimal_match.group(1)) if optimal_match else None # OLD
            baseline_time = float(baseline_match.group(1)) if baseline_match else None  # NEW
            test_result = TestEvaluationResult(
                performance_score=perf_score,
                your_time=your_time,
                # oracle_time=optimal_time # OLD
                oracle_time=baseline_time,  # NEW - Store Baseline time in oracle_time field
            )
            # Test evaluation parsed
            return test_result
        except (ValueError, AttributeError):
            pass

    return current_test_result


# --- Log Block Processing Helpers ---


def _extract_config_from_block(block: str, current_config: dict):
    """Extracts config key-value pairs from a log block if present."""
    if "Config loaded:" in block:  # Ensure this line has exactly 4 leading spaces
        match = CONFIG_REGEX.search(block)  # Ensure this line has exactly 8 leading spaces
        if match:
            config_str = match.group(1).strip()
            pairs = config_str.split()
            for p in pairs:
                if "=" in p:
                    parts = p.split("=", 1)
                    key = parts[0]
                    value = parts[1]
                    try:
                        current_config[key] = float(value)
                    except ValueError:
                        current_config[key] = value


def _extract_prompt_from_block(block: str, current_prompts: list):
    """Extracts formatted initial prompt from a log block if present."""
    if "--BEGIN DEMONSTRATION--" in block:
        match = PROMPT_REGEX.search(block)
        if match:
            d_text = match.group(0)
            d_text = d_text.replace("\\n", "\n")
            d_text = format_and_escape(d_text)  # Assuming format_and_escape exists
            current_prompts.append(d_text)


def _process_log_block_for_conversation(block: str, result: LogExtractionResult):
    """Processes a single log block to extract conversation messages, dates, and commands."""
    lines = block.split("\n", 1)
    top_line = lines[0]
    match = LOG_LINE_REGEX.match(top_line)
    if not match:
        return  # Not a standard log message line

    date_str = match.group("date")
    result.dates.append(date_str.split()[0])
    rest = match.group("rest")

    role = None
    text_html = None
    command_type = None
    is_edit_command = False

    if "Sent to LLM:" in rest:
        splitted = rest.split("Sent to LLM:", 1)
        full_text = splitted[-1].strip()
        if len(lines) > 1:
            full_text += "\n" + lines[1]
        # Strip thinking blocks before formatting
        full_text = strip_thinking_blocks(full_text)
        text_html = format_system_text(full_text)  # Assuming format_system_text exists
        role = "system"

    elif "Received from LLM:" in rest:
        splitted = rest.split("Received from LLM:", 1)
        full_text = splitted[-1].strip()
        if len(lines) > 1:
            full_text += "\n" + lines[1]
        # Strip thinking blocks before formatting
        full_text = strip_thinking_blocks(full_text)
        text_html = format_and_escape(full_text)  # Assuming format_and_escape exists
        command_type = extract_command(full_text)  # Assuming extract_command exists
        role = "assistant"
        if command_type == "edit":
            is_edit_command = True
        elif command_type != "other":
            # Record non-edit, non-other commands immediately
            result.command_sequence.append((len(result.conversation), command_type, None))

    if role and text_html:
        message_data = {"role": role, "text": text_html}
        if role == "assistant":
            message_data["command"] = command_type
        result.conversation.append(message_data)

        # If this was an edit command, mark it for later status update
        if is_edit_command:
            result.command_sequence.append(
                (len(result.conversation) - 1, command_type, None)
            )  # Mark with None status initially
            result.pending_edit_index = len(result.command_sequence) - 1


def extract_system_message_sections(log_content: str) -> tuple[str, str, str]:
    """Extract initial system prompt, task description, and reference implementation from log content.

    Returns:
        tuple: (initial_prompt, task_description, reference_implementation)
    """
    initial_prompt = ""
    task_description = ""
    reference_impl = ""

    # Extract initial system message
    system_start = log_content.find("--- Full Initial System Message Content ---")
    if system_start != -1:
        # Move to the next line after the marker
        system_start = log_content.find("\n", system_start) + 1
        # Find the end marker - it's a line of dashes after the content
        system_end = log_content.find("\n---", system_start)
        if system_end != -1:
            initial_prompt = log_content[system_start:system_end].strip()

    # Extract task description
    task_desc_start = log_content.find("**TASK DESCRIPTION:**")
    if task_desc_start != -1:
        # Find the end of task description (usually before reference implementation)
        task_desc_end = log_content.find("Below is the reference implementation", task_desc_start)
        if task_desc_end == -1:
            # Try alternative end marker
            task_desc_end = log_content.find("This function will be used to check", task_desc_start)
        if task_desc_end != -1:
            task_description = log_content[task_desc_start:task_desc_end].strip()
            # Strip leading '**TASK DESCRIPTION:**' if present
            import re

            task_description = re.sub(
                r"^\*\*?TASK DESCRIPTION:?\*\*?\s*", "", task_description, flags=re.IGNORECASE
            )

    # Extract reference implementation
    ref_impl_start = log_content.find("Below is the reference implementation")
    if ref_impl_start != -1:
        # Find the validation function section as the end
        ref_impl_end = log_content.find(
            "This function will be used to check if your solution is valid", ref_impl_start
        )
        if ref_impl_end != -1:
            reference_impl = log_content[ref_impl_start:ref_impl_end].strip()

    return initial_prompt, task_description, reference_impl


def extract_best_code(log_content: str) -> dict[str, str]:
    """Extract the best (final) code snapshot from the log.

    The log produced by AlgoTuner can expose code in several different ways:
    1.  After a run completes it prints blocks like::

            FILE IN CODE DIR solver.py:
            <code lines>

        There can be multiple such blocks (one per file).  These blocks are
        considered the *authoritative* final snapshot because they are emitted
        **after** the test evaluation has finished.

    2.  Earlier in the trajectory the assistant may show code via an *edit*
        command or by *Viewing file solver.py*.  Those snapshots are used as a
        fallback when no final snapshot is found.

    The function returns a mapping ``{filename: code}`` where::
        • Only plausible source files (e.g. ``*.py`` / ``*.c`` / ``*.cpp`` …)
          are included to avoid bloating the HTML with binary / metadata files.
        • ``solver.py`` is guaranteed to be the first key if it exists, because
          it is the most important file when inspecting a solution.
    """

    import re

    best_code: dict[str, str] = {}
    log_line_re = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+-\s+\w+\s+-\s+")

    def _sanitize_code_block(block: str) -> str:
        if not block:
            return ""
        cleaned_lines = []
        for line in block.splitlines():
            if log_line_re.match(line):
                break
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines).rstrip()
        if "extract_code_blocks" in cleaned or "PARSER_ENTRY" in cleaned:
            return ""
        return cleaned

    # --- 1. Parse the final snapshot blocks ---------------------------------
    # Each block starts with "FILE IN CODE DIR <filename>:" and runs until the
    # next timestamped log line ("YYYY-MM-DD HH:MM:SS,") *or* the next "FILE
    # IN CODE DIR" header or end-of-file.  We use a look-ahead to stop the
    # non-greedy capture at those boundaries.
    snapshot_regex = re.compile(
        r"(?:FILE IN CODE DIR\s+|CODE_DIR_FILE:?_?)(?P<filename>[^\n:]+):?\s*\n"
        r"(?P<code>.*?)(?=^\d{4}-\d{2}-\d{2}\s|^FILE IN CODE DIR|^CODE_DIR_FILE|\Z)",
        re.DOTALL | re.MULTILINE,
    )

    for m in snapshot_regex.finditer(log_content):
        fname = m.group("filename").strip()
        # Some logs emit an underscore prefix (e.g. "_solver.py").  Remove it
        # so the file is displayed with its real name.
        fname = fname.lstrip("_")
        code_block = _sanitize_code_block(m.group("code").rstrip())
        if not fname or not code_block:
            continue
        if (
            fname == "solver.py"
            and "class Solver" not in code_block
            and "def solve" not in code_block
        ):
            continue

        # Heuristic: keep only typical source files to avoid huge HTML blobs.
        allowed_exts = (".py", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".h", ".hpp")
        if fname == "solver.py" or fname.endswith(allowed_exts):
            # The log is chronological, so later occurrences should override
            # earlier ones – simply assign each time we encounter the file.
            best_code[fname] = code_block

    # --- 2. Fallback – harvest solver.py from edit / view commands -----------
    if "solver.py" not in best_code:
        # a) edit command blocks
        edit_regex = re.compile(
            r"edit\s*\nfile:\s*solver\.py\s*\nlines:.*?\n---\n(.*?)\n---",
            re.DOTALL,
        )
        edit_matches = list(edit_regex.finditer(log_content))
        if edit_matches:
            candidates = []
            for match in edit_matches:
                candidate = _sanitize_code_block(match.group(1))
                if not candidate:
                    continue
                if "class Solver" in candidate or "def solve" in candidate:
                    candidates.append(candidate)
            if candidates:
                best_code["solver.py"] = max(candidates, key=len)

        # b) view file blocks
        view_regex = re.compile(
            r"Viewing file solver\.py.*?\n(.*?)(?=\n\d{4}-\d{2}-\d{2}|\Z)",
            re.DOTALL,
        )
        for m in view_regex.finditer(log_content):
            view_content = _sanitize_code_block(m.group(1).strip())
            if view_content and ("class Solver" in view_content or "def solve" in view_content):
                if len(view_content) > len(best_code.get("solver.py", "")):
                    best_code["solver.py"] = view_content

    # --- 3. Re-order dict so that solver.py is first -------------------------
    if "solver.py" in best_code:
        ordered: dict[str, str] = {"solver.py": best_code["solver.py"]}
        for k, v in best_code.items():
            if k != "solver.py":
                ordered[k] = v
        return ordered

    return best_code


def extract_conversation_from_messages(messages) -> LogExtractionResult:
    """Extracts various data points from a list of log message blocks."""
    result = LogExtractionResult()
    # Removed temporary storage - add directly to result.training_results
    # potential_training_results = []
    # pending_edit_eval_map = {}

    for block_idx, block in enumerate(messages):
        # 1. Track last seen budget
        used_budget_in_block = extract_used_budget_from_log(block)
        if used_budget_in_block is not None:
            result.last_used_budget = used_budget_in_block

        # 2. Parse Training Evaluation - Add directly if valid
        tr_eval_result = parse_training_evaluation(block)

        # --- Condition Update: Check for score AND "Sent to LLM:" marker ---
        if tr_eval_result.performance_score is not None and "Sent to LLM:" in block:
            # Performance data point added
            # Apply budget fallback if necessary
            if (
                tr_eval_result.used_budget is None or tr_eval_result.used_budget == 0
            ) and result.last_used_budget is not None:
                # Applying fallback budget
                tr_eval_result.used_budget = result.last_used_budget

            # Add if score is valid AND block contained "Sent to LLM:".
            # Budget check moved implicitly to _prepare_performance_data
            result.training_results.append(tr_eval_result)
        elif tr_eval_result.performance_score is not None:
            # Log why a score is being ignored if it doesn't meet the new criteria
            # Performance score found but no 'Sent to LLM:' marker
            pass

        # --- Separate logic for updating edit command status ---
        # This affects the Action Sequence plot legend, not the perf plot points
        # Only update edit status if a valid score was found (regardless of 'Sent to LLM:')
        if tr_eval_result.performance_score is not None:
            current_perf = tr_eval_result.performance_score  # Score is not None here
            if result.pending_edit_index is not None:  # Check if an edit is pending status update
                # Decide status based on best_so_far_score
                new_status = None
                if result.best_so_far_score is None or current_perf > result.best_so_far_score:
                    result.best_so_far_score = current_perf  # Update best score seen
                    new_status = "edit(best)"
                else:
                    # This score did not improve over the best so far - Edit still succeeded (non-failure) but didn't improve the best
                    new_status = "edit"

                # Find and update the pending edit in command_sequence
                if result.pending_edit_index < len(result.command_sequence):
                    old_entry = result.command_sequence[result.pending_edit_index]
                    result.command_sequence[result.pending_edit_index] = (
                        old_entry[0],
                        old_entry[1],
                        new_status,
                    )

                result.pending_edit_index = None  # Clear pending edit
                # Updated pending edit status

        # 3. Check for edit failures separately (for updating command status)
        if FAILED_EDIT_REGEX.search(block) and result.pending_edit_index is not None:
            # Mark the pending edit as failed
            if result.pending_edit_index < len(result.command_sequence):
                old_entry = result.command_sequence[result.pending_edit_index]
                result.command_sequence[result.pending_edit_index] = (
                    old_entry[0],
                    old_entry[1],
                    "edit(failed)",
                )
            result.pending_edit_index = None
            # Marked pending edit as failed

        # 4. Parse test score (kept separate, doesn't affect training results or command status)
        result.test_score_result = parse_test_score_from_block(block, result.test_score_result)

        # 5. Extract config and prompts
        _extract_config_from_block(block, result.config)
        _extract_prompt_from_block(block, result.initial_prompts)

        # 6. Process conversation and update command_sequence
        _process_log_block_for_conversation(block, result)

    # Extraction completed
    return result


def _prepare_performance_data(training_results: list[TrainingEvaluationResult]):
    """
    Processes a list of TrainingEvaluationResult objects to prepare data for plotting.
    Only includes results with both performance_score and used_budget.

    Returns (budgets_list, performances_list, times_list, oracle_times_list).
    """
    budgets, performances, times, oracle_times = [], [], [], []

    for result in training_results:
        # Only include results that have both performance score and budget
        if result.performance_score is not None and result.used_budget is not None:
            budgets.append(result.used_budget)
            performances.append(result.performance_score)
            times.append(result.your_time if result.your_time is not None else 0.0)
            oracle_times.append(result.oracle_time if result.oracle_time is not None else 0.0)
            # Performance data point included
        else:
            # Performance data point excluded
            pass

    # Performance data prepared for plotting
    return budgets, performances, times, oracle_times


def create_performance_plot(
    training_results: list[TrainingEvaluationResult],
    is_higher_better=True,
    mobile: bool = False,
    output_path: str | None = None,
):
    """Create a clean, minimal performance vs budget plot."""
    if not training_results:
        return ""

    budgets, performances, times, oracle_times = _prepare_performance_data(training_results)

    if not budgets:  # No valid data points
        return ""

    # Slightly larger canvas for mobile because tick labels/legends are scaled up
    if mobile:
        fig, ax = plt.subplots(figsize=(8, 5))  # keep same physical size, fonts will increase
    else:
        fig, ax = plt.subplots(figsize=(8, 5))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Use blue dots for all points
    blue_color = "#2196F3"

    marker_size = 180 if mobile else 120
    ax.scatter(
        budgets,
        performances,
        s=marker_size,
        c=blue_color,
        alpha=0.8,
        edgecolors="white",
        linewidth=1,
    )

    label_fs = 22 if mobile else 16
    ax.set_xlabel("Budget Used", fontsize=label_fs)
    ax.set_ylabel("Speedup", fontsize=label_fs)

    # Format y-axis ticks to nearest hundredth for speedup
    if is_higher_better:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.2f}x"))
        # Limit the number of y-axis ticks and round to reasonable values
        y_min, y_max = ax.get_ylim()
        tick_count = min(6, max(3, int((y_max - y_min) / 0.5) + 1))  # Reasonable number of ticks
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=tick_count, prune="lower"))
    else:
        # For "lower is better" metrics, we might want different formatting
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:.2f}"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="lower"))

    # Format x-axis ticks with dollar signs and proper padding
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.2f}"))
    # Limit x-axis ticks as well
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune="lower"))

    # Show only bottom and left spines (x and y axes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Style the visible spines
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)

    # Remove grid
    ax.grid(False)

    tick_fs = 20 if mobile else 16
    tick_len = 10 if mobile else 8
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fs,
        colors="black",
        length=tick_len,
        pad=10,
        width=2,
    )

    plt.tight_layout()

    # Convert to base64
    dpi_val = 140 if mobile else 120
    if output_path:
        # Ensure directory exists and save to disk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            format="png",
            dpi=dpi_val,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        # Return path relative to OUTPUT_DIR for HTML embedding
        return os.path.relpath(output_path, OUTPUT_DIR).replace(os.sep, "/")
    else:
        buf = BytesIO()
        plt.savefig(
            buf, format="png", dpi=dpi_val, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"


def _clean_reference_implementation(ref_impl_text):
    """Clean up reference implementation text for display.

    Removes:
    - The "Below is the reference implementation" header
    - Line numbers in format "| 01: " or "| 02: "
    - Extra whitespace
    """
    if not ref_impl_text:
        return ""

    # Remove the header line
    lines = ref_impl_text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Skip the header line
        if line.strip().startswith("Below is the reference implementation"):
            continue

        # Remove line numbers (format: "| 01: code" or "| 02:     code")
        # Check if line contains the pattern (may have leading spaces)
        if "|" in line and ":" in line:
            pipe_idx = line.find("|")
            colon_idx = line.find(":", pipe_idx)
            # Check if this looks like a line number pattern
            if colon_idx > pipe_idx and colon_idx - pipe_idx < 6:  # "| 01: " is 6 chars
                # Extract everything after the colon and space
                cleaned_line = line[colon_idx + 1 :]
                cleaned_lines.append(cleaned_line)
            else:
                # Not a line number pattern, keep as is
                cleaned_lines.append(line)
        else:
            # Keep lines that don't start with |
            cleaned_lines.append(line)

    # Join back and strip extra whitespace
    result = "\n".join(cleaned_lines)

    # Remove any leading/trailing whitespace
    result = result.strip()

    return result


def create_action_sequence_plot(
    command_sequence, mobile: bool = False, output_path: str | None = None
):
    """Create a minimal action sequence plot with 5 actions per row and consistent colors.
    If `mobile` is True, use larger marker sizes, tick labels and legend fonts for better readability on small screens."""
    if not command_sequence:
        return ""

    # Prepare data for plotting
    colors = []
    markers = []
    display_commands = []

    # Bold, contrasting color palette - consistent across all trajectories
    color_map = {
        "ls": "#FFC107",  # Bold Yellow
        "view_file": "#20B2AA",  # Turquoise
        "edit": "#0077B6",  # Bold Blue (all edit types use same color)
        "edit(best)": "#0077B6",  # Bold Blue (same as edit)
        "edit(failed)": "#0077B6",  # Bold Blue (same as edit)
        "eval_input": "#E91E63",  # Deep Pink
        "revert": "#6C757D",  # Medium Gray
        "reference": "#9C27B0",  # Purple
        "delete": "#F44336",  # Red
        "profile": "#4CAF50",  # Green (same color for both profile types)
        "profile_lines": "#4CAF50",  # Green (same as profile)
    }

    # Different markers for edit types and profile variants
    marker_map = {
        "ls": "s",  # Square
        "view_file": "s",  # Square
        "edit": "s",  # Square
        "edit(best)": "P",  # Plus (filled) - more visible than star
        "edit(failed)": "X",  # X (filled) - more visible than lowercase x
        "eval_input": "s",  # Square
        "revert": "s",  # Square
        "reference": "s",  # Square
        "delete": "s",  # Square
        "profile": "o",  # Circle
        "profile_lines": "D",  # Diamond
    }

    for i, (msg_idx, cmd_type, status) in enumerate(command_sequence):
        # Determine the display command and color
        if cmd_type == "edit" and status:
            display_cmd = status  # Use status like 'edit(best)' or 'edit(failed)'
        else:
            display_cmd = cmd_type

        # Use mapped color or default to gray for unmapped commands
        colors.append(color_map.get(display_cmd, "#6C757D"))  # Default gray for unmapped
        markers.append(marker_map.get(display_cmd, "s"))  # Default square for unmapped
        display_commands.append(display_cmd)

    if not colors:
        return ""

    # Calculate grid dimensions (5 actions per row)
    actions_per_row = 5
    total_actions = len(command_sequence)
    num_rows = (total_actions + actions_per_row - 1) // actions_per_row  # Ceiling division

    # Minimal figure sizing (same physical size; fonts/markers scale for mobile)
    fig_width = 6
    fig_height = max(3, num_rows * 0.6 + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Create grid positions
    x_positions = []
    y_positions = []

    for i in range(total_actions):
        row = i // actions_per_row
        col = i % actions_per_row
        x_positions.append(col)
        y_positions.append(num_rows - row - 1)  # Invert y to start from top

    # Plot each marker individually to use different shapes
    for i, (x, y, color, marker) in enumerate(zip(x_positions, y_positions, colors, markers)):
        base_size_special = 260 if mobile else 200
        base_size = 200 if mobile else 150
        marker_size = base_size_special if marker in ["P", "X"] else base_size
        ax.scatter(
            x, y, c=color, s=marker_size, alpha=0.9, edgecolors="white", linewidth=2, marker=marker
        )

    # Set up the plot - completely despined and minimal
    ax.set_xlim(-0.5, actions_per_row - 0.5)
    ax.set_ylim(-0.5, num_rows - 0.5)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove x-axis elements but keep y-axis labels
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Add y-axis labels for action ranges (1-10, 11-20, etc.)
    y_tick_positions = []
    y_tick_labels = []

    for row in range(num_rows):
        y_pos = num_rows - row - 1  # Same inversion as the plotting
        start_action = row * actions_per_row + 1
        end_action = min((row + 1) * actions_per_row, total_actions)

        y_tick_positions.append(y_pos)
        if start_action == end_action:
            y_tick_labels.append(f"{start_action}")
        else:
            y_tick_labels.append(f"{start_action}-{end_action}")

    tick_fs = 14 if mobile else 10
    # Explicitly set tick positions before assigning the labels to ensure they line up correctly.
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, fontsize=tick_fs, ha="right")
    ax.tick_params(axis="y", length=0, pad=8, labelsize=tick_fs)

    # Define consistent legend order - edit actions first, then view, ls, then others
    legend_order = [
        "edit",  # Regular edit
        "edit(best)",  # Best edit
        "edit(failed)",  # Failed edit
        "view_file",  # View file
        "ls",  # List files
        "eval_input",  # Input evaluation
        "profile",  # Profile
        "profile_lines",  # Profile lines
        "revert",  # Revert changes
        "reference",  # Reference
        "delete",  # Delete
        # Any other commands will be added at the end in alphabetical order
    ]

    # Find unique commands that actually appear in this plot
    unique_commands_set = set(display_commands)

    # Create ordered list of commands that appear, following the defined order
    ordered_commands = []
    for cmd in legend_order:
        if cmd in unique_commands_set:
            ordered_commands.append(cmd)

    # Add any remaining commands not in the predefined order (alphabetically)
    remaining_commands = sorted([cmd for cmd in unique_commands_set if cmd not in legend_order])
    ordered_commands.extend(remaining_commands)

    # Create legend handles with consistent ordering
    legend_handles = []
    for cmd in ordered_commands:
        color = color_map.get(cmd, "#6C757D")  # Default gray for unmapped
        marker = marker_map.get(cmd, "s")  # Default square for unmapped

        # Create clean labels for legend
        if cmd.startswith("edit("):
            if "best" in cmd:
                label = "Best Edit"
            elif "failed" in cmd:
                label = "Failed Edit"
            else:
                label = "Edit"
        else:
            label = CMD_DISPLAY.get(cmd, cmd.title())

        legend_ms = 18 if mobile else 14 if marker in ["P", "X"] else 14 if mobile else 10
        marker_size = legend_ms
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=color,
                markersize=marker_size,
                label=label,
                markeredgecolor="white",
                markeredgewidth=1.5,
                linestyle="None",
            )
        )

    # Place compact legend at bottom
    legend_fs = 14 if mobile else 10
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, len(legend_handles)),
        frameon=False,
        fontsize=legend_fs,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=1,
    )

    plt.tight_layout()

    # Convert to base64
    dpi_val = 140 if mobile else 120
    if output_path:
        # Ensure directory exists and save to disk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(
            output_path,
            format="png",
            dpi=dpi_val,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        # Return path relative to OUTPUT_DIR for HTML embedding
        return os.path.relpath(output_path, OUTPUT_DIR).replace(os.sep, "/")
    else:
        buf = BytesIO()
        plt.savefig(
            buf, format="png", dpi=dpi_val, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{img_base64}"


def process_log_file_data(filepath):
    """Process a single log file and return structured data for template rendering."""
    # Processing log file

    try:
        messages, task_name, test_score_result, model_name, total_budget = parse_log_file(filepath)
        extraction_result = extract_conversation_from_messages(messages)

        # Use test_score_result from parse_log_file if extraction didn't find one
        if extraction_result.test_score_result is None:
            extraction_result.test_score_result = test_score_result

        # Extract collapsible sections from raw log content
        with open(filepath, encoding="utf-8") as f:
            log_content = f.read()

        initial_prompt, task_description, reference_impl = extract_system_message_sections(
            log_content
        )
        best_code = extract_best_code(log_content)

        # Store in extraction result
        extraction_result.initial_system_prompt = initial_prompt
        extraction_result.task_description = task_description
        extraction_result.reference_implementation = reference_impl
        extraction_result.best_code = best_code

        # Create plots if we have data
        performance_plot = ""
        action_plot = ""
        slug = os.path.splitext(os.path.basename(filepath))[0]
        if extraction_result.training_results:
            perf_abs = os.path.join(PLOTS_DIR, f"{slug}_perf.png")
            performance_plot = create_performance_plot(
                extraction_result.training_results, mobile=True, output_path=perf_abs
            )
        if extraction_result.command_sequence:
            act_abs = os.path.join(PLOTS_DIR, f"{slug}_actions.png")
            action_plot = create_action_sequence_plot(
                extraction_result.command_sequence, mobile=True, output_path=act_abs
            )

        # Calculate final test score value for display
        final_test_score = None
        if isinstance(extraction_result.test_score_result, TestEvaluationResult):
            final_test_score = extraction_result.test_score_result.performance_score
        elif isinstance(extraction_result.test_score_result, (int, float)):
            final_test_score = extraction_result.test_score_result

        # Calculate invalid commands count
        # Count both "other" commands and system error responses indicating parsing failures
        invalid_commands_count = 0

        # Count commands with type "other"
        invalid_commands_count += sum(
            1 for _, cmd_type, _ in extraction_result.command_sequence if cmd_type == "other"
        )

        # Count system messages that indicate command parsing errors
        error_patterns = [
            "Command parsing failed",
            "Error: Command parsing failed",
            "parsing failed",
            "Invalid command",
            "Unrecognized command",
            "Command not recognized",
            "Unknown command",
            "Failed to parse command",
            "Could not parse command",
        ]

        for msg in extraction_result.conversation:
            if msg.get("role") == "system":
                text = msg.get("text", "").lower()
                for pattern in error_patterns:
                    if pattern.lower() in text:
                        invalid_commands_count += 1
                        break  # Only count once per message

        # Calculate average reference time from generation.json
        average_reference_time = "N/A"
        try:
            import json

            generation_file_path = os.path.join(
                os.path.dirname(os.path.dirname(filepath)), "reports", "generation.json"
            )
            if os.path.exists(generation_file_path):
                with open(generation_file_path) as f:
                    generation_data = json.load(f)

                task_data = generation_data.get(task_name)
                if task_data and "baseline_runs" in task_data:
                    # Get all avg_min_ms values from baseline runs
                    avg_times = []
                    for run_data in task_data["baseline_runs"].values():
                        if isinstance(run_data, dict) and "avg_min_ms" in run_data:
                            avg_times.append(run_data["avg_min_ms"])

                    if avg_times:
                        # Calculate mean of all baseline runs
                        mean_reference_time = sum(avg_times) / len(avg_times)
                        average_reference_time = f"{mean_reference_time:.1f} ms"
        except Exception as e:
            logging.debug(f"Could not load reference time for {task_name}: {e}")
            average_reference_time = "N/A"

        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "task_name": task_name,
            "model_name": model_name,
            "conversation": extraction_result.conversation,
            "training_results_count": len(extraction_result.training_results),
            "command_sequence_count": len(extraction_result.command_sequence),
            "invalid_commands_count": invalid_commands_count,
            "average_reference_time": average_reference_time,
            "config": extraction_result.config,
            "initial_prompts": extraction_result.initial_prompts,
            "dates": list(set(extraction_result.dates)),  # unique dates
            "performance_plot": performance_plot,
            "action_plot": action_plot,
            "final_test_score": final_test_score,
            "total_budget": total_budget,
            "budget_from_filename": extract_budget_from_filename(os.path.basename(filepath)),
            # New collapsible sections
            "initial_system_prompt": extraction_result.initial_system_prompt,
            "task_description": extraction_result.task_description,
            "reference_implementation": extraction_result.reference_implementation,
            "best_code": extraction_result.best_code,
        }

    except Exception as e:
        if logging.getLogger().isEnabledFor(logging.ERROR):
            logging.error(f"Error processing {filepath}: {e}")
        return None


def generate_single_log_html(data, all_data=None):
    """Generate HTML content for a single log file with sidebar navigation."""
    # Format final test score
    final_test_score_display = "Fail"
    if data["final_test_score"] is not None:
        final_test_score_display = f"{data['final_test_score']:.3f}x"

    # Format budget info
    budget_display = f"${data['total_budget']:.2f}" if data["total_budget"] else "N/A"

    # Generate sidebar HTML - filter to show only current task trajectories
    sidebar_html = ""
    if all_data:
        # Filter data to only include trajectories for the current task
        current_task_logs = [
            log_data
            for log_data in all_data
            if log_data
            and log_data["task_name"] == data["task_name"]
            and "dummy" not in log_data["model_name"].lower()
        ]

        # NEW: Deduplicate by (task, model) keeping only the newest trajectory
        deduped_logs: dict[str, dict] = {}
        for log in current_task_logs:
            # Normalize model key to lower-case for grouping
            model_key = log.get("model_name", "").lower()
            try:
                mtime = os.path.getmtime(log.get("filepath", ""))
            except Exception:
                mtime = 0
            # Keep the log if we haven't seen this model yet, or if it is newer
            if model_key not in deduped_logs or mtime > deduped_logs[model_key]["_mtime"]:
                # Store a shallow copy with mtime for comparison
                temp = log.copy()
                temp["_mtime"] = mtime
                deduped_logs[model_key] = temp
        # Replace current_task_logs with the newest-only versions (drop helper key)
        current_task_logs = [
            {k: v for k, v in log.items() if k != "_mtime"} for log in deduped_logs.values()
        ]

        if current_task_logs:
            # Sort by score (highest first)
            valid_logs = [log for log in current_task_logs if log["final_test_score"] is not None]
            valid_logs.sort(key=lambda x: x["final_test_score"], reverse=True)

            # Add remaining logs without scores
            invalid_logs = [log for log in current_task_logs if log["final_test_score"] is None]
            all_task_logs = valid_logs + invalid_logs

            runs_html = ""
            for log in all_task_logs:
                clean_filename = f"{log['task_name']}_{log['model_name'].replace('/', '_').replace(' ', '_')}.html"

                # Determine display score with lower bound 1x for color and text
                if log["final_test_score"] is not None:
                    score_color = get_speedup_color(log["final_test_score"])
                    clean_model = clean_model_name(log["model_name"])
                    display_text = f"{clean_model} ({log['final_test_score']:.2f}x)"
                else:
                    score_color = get_speedup_color(None)
                    clean_model = clean_model_name(log["model_name"])
                    display_text = f"{clean_model} (Fail)"

                # Check if this is the current trajectory being viewed
                current_class = " current" if log["filename"] == data["filename"] else ""

                runs_html += f"""
                <div class="sidebar-run{current_class}">
                    <a href="{clean_filename}">
                        <div class="run-score" style="background-color: {score_color}; color: #ffffff; padding: 12px 16px; font-size: 0.95rem; border-radius: 8px; font-weight: 600; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1); letter-spacing: 0.025em; min-height: 24px; width: 100%; box-sizing: border-box;">{display_text}</div>
                    </a>
                </div>"""

            sidebar_items = f"""
            <div class="sidebar-task">
                <div class="task-runs">
                    {runs_html}
                </div>
            </div>"""
        else:
            sidebar_items = """
            <div class="sidebar-task">
                <div class="task-runs">
                    <div class="no-runs">No other runs found for this task</div>
                </div>
            </div>"""

        sidebar_html = f"""
        <div class="sidebar">
            <div class="sidebar-header">
                <a href="index.html" class="back-link">← Back to Speedup Table</a>
            </div>
            <div class="sidebar-content">
                <h3 style="color: black;">{data["task_name"]}</h3>
                {sidebar_items}
            </div>
        </div>"""

    # Generate conversation HTML
    conversation_html = ""
    for msg in data["conversation"]:
        role_class = msg["role"]

        # Update role display names
        if msg["role"] == "system":
            role_display = "System"
        elif msg["role"] == "assistant":
            role_display = "Language Model"
        else:
            role_display = msg["role"].title()

        command_badge = ""
        if msg["role"] == "assistant" and "command" in msg:
            command = msg["command"]
            command_badge = f'<span class="command-badge {command}">{ICON_MAPPING.get(command, "")} {CMD_DISPLAY.get(command, command.title())}</span>'

        # msg['text'] is already formatted HTML (thinking blocks removed earlier)
        cleaned_text = msg["text"]

        conversation_html += f"""
        <div class="message {role_class}">
            <div class="message-header">
                {role_display} {command_badge}
            </div>
            <div class="message-content">
                {cleaned_text}
            </div>
        </div>"""

    # Generate plots HTML - side by side layout
    plots_html = f"""
    <div class="plots-container">
        <div class="plot-section plot-half">
            <h3>Speedup vs Budget</h3>
            <div class="plot-container">
                {f'<img src="{data["performance_plot"]}" alt="Speedup vs Budget Plot" />' if data["performance_plot"] else '<div class="no-plot">No performance data available for plotting</div>'}
            </div>
        </div>
        <div class="plot-section plot-half">
            <h3>Action Sequence</h3>
            <div class="plot-container">
                {f'<img src="{data["action_plot"]}" alt="Action Sequence Plot" />' if data["action_plot"] else '<div class="no-plot">No action sequence data available</div>'}
            </div>
        </div>
    </div>"""

    # Prepare "Best Code" HTML – handle single or multiple files
    # Filter best_code to keep only solver.py and actual dependencies
    if isinstance(data.get("best_code"), dict):
        filtered_code_dict = _filter_best_code(data["best_code"])
    else:
        filtered_code_dict = data.get("best_code")

    best_code_html = ""
    if isinstance(filtered_code_dict, dict):
        if filtered_code_dict:
            blocks = []
            # Put solver.py first, then the rest alphabetically
            filenames = sorted(
                filtered_code_dict, key=lambda n: (0 if n == "solver.py" else 1, n.lower())
            )
            for fname in filenames:
                code = filtered_code_dict[fname]
                blocks.append(
                    f'<div class="best-file">'
                    f'<div class="file-name" style="font-weight:600; margin-bottom:0.25rem;">{html.escape(fname)}</div>'
                    f'<pre class="best-code"><code class="language-python">{html.escape(code)}</code></pre>'
                    f"</div>"
                )
            best_code_html = "\n".join(blocks)
        else:
            best_code_html = "N/A"
    else:
        best_code_str = filtered_code_dict if isinstance(filtered_code_dict, str) else ""
        best_code_html = f'<pre class="best-code"><code class="language-python">{html.escape(best_code_str) if best_code_str.strip() else "N/A"}</code></pre>'

    # Generate collapsible sections HTML
    collapsible_sections_html = f"""
    <div class="collapsible-sections">
        <details class="collapsible-section">
            <summary>Initial System Prompt</summary>
            <div class="section-content">
                <pre>{html.escape(data["initial_system_prompt"])}</pre>
            </div>
        </details>
        
        <details class="collapsible-section">
            <summary>AlgoTune Task Description</summary>
            <div class="section-content">
                <pre>{html.escape(data["task_description"])}</pre>
            </div>
        </details>
        
        <details class="collapsible-section">
            <summary>Reference Implementation</summary>
            <div class="section-content">
                <pre class="reference-code"><code class="language-python">{html.escape(_clean_reference_implementation(data["reference_implementation"]))}</code></pre>
            </div>
        </details>
        
        <details class="collapsible-section">
            <summary>Best AlgoTuner-Generated Code</summary>
            <div class="section-content">
                {best_code_html}
            </div>
        </details>
        
        <details class="collapsible-section">
            <summary>Speedup vs Budget Plot</summary>
            <div class="section-content plot-section-content">
                {plots_html}
            </div>
        </details>
    </div>"""

    # Generate the complete HTML for individual log file using clean layout
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no"/>
    <title>AlgoTuner Log – {data["task_name"]} – {data["model_name"]}</title>
    <link rel="icon" type="image/png" href="assets/AlgoTunerMascot.png">
    
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-7XSBWH5NQF"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());

      gtag('config', 'G-7XSBWH5NQF');
    </script>
    
    <!-- Prism.js for syntax highlighting - loaded after styles.css to ensure proper precedence -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-sql.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
    <link rel="stylesheet" href="styles.css">
    
    <!-- Basic Styling & Layout -->
    <style>
    /* Basic Styling & Layout */
    :root {{
        --primary-color: #2196F3;
        --primary-light: #E3F2FD;
        --text-color: #333;
        --border-color: #eaeaea;
        --content-bg: #ffffff;
        --error-border: #dc3545;
        --code-bg: #f6f8fa;
        --code-border: #d0d7de;
        --code-text: #24292e;
        
        /* Glass-morphism variables for light mode */
        --glass-bg: rgba(255, 255, 255, 0.12);
        --glass-border: rgba(255, 255, 255, 0.05);
        --glass-header-bg: rgba(0, 0, 0, 0.03);
        --glass-header-border: rgba(255, 255, 255, 0.08);
        --glass-btn-bg: rgba(255, 255, 255, 0.1);
        --glass-btn-border: rgba(255, 255, 255, 0.2);
        --glass-btn-hover: rgba(255, 255, 255, 0.2);
        --glass-expand-bg: linear-gradient(to top, rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.8));
        --glass-text: rgba(0, 0, 0, 0.8);
        --glass-text-secondary: rgba(0, 0, 0, 0.6);
    }}
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {{
        :root {{
            --glass-bg: rgba(0, 0, 0, 0.15);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-header-bg: rgba(255, 255, 255, 0.05);
            --glass-header-border: rgba(255, 255, 255, 0.12);
            --glass-btn-bg: rgba(255, 255, 255, 0.08);
            --glass-btn-border: rgba(255, 255, 255, 0.15);
            --glass-btn-hover: rgba(255, 255, 255, 0.15);
            --glass-expand-bg: linear-gradient(to top, rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.7));
            --glass-text: rgba(255, 255, 255, 0.9);
            --glass-text-secondary: rgba(255, 255, 255, 0.7);
        }}
    }}
    
    body {{
        margin: 0;
        padding: 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: var(--text-color);
        line-height: 1.5;
        background: var(--content-bg);
        display: flex;
        min-height: 100vh;
    }}
    
    /* Sidebar - Desktop only, hidden by default on mobile */
    .sidebar {{
        /* Slightly narrower sidebar to give more room to main content */
        width: 180px;
        background: #f8f9fa;
        border-right: 1px solid var(--border-color);
        position: fixed;
        left: 0;
        top: 0;
        height: 100vh;
        overflow-y: auto;
        z-index: 1000;
        display: none; /* Hidden by default */
    }}
    
    /* Show sidebar only on large screens (1025px and up) */
    @media (min-width: 1025px) {{
        .sidebar {{
            display: block;
        }}
    }}
    
    /* --------------------------- */
    /* Sidebar header & back link  */
    /* --------------------------- */
    .sidebar-header {{
        padding: 1.25rem;
        display: flex;
        justify-content: center;  /* Centre the back link horizontally */
    }}

    .back-link {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;               /* Space between arrow and label */
        padding: 0.45rem 0.9rem;
        background: #2196F3;       /* Primary blue */
        border-radius: 8px;
        color: #ffffff;            /* White text */
        font-weight: 600;
        font-size: 0.9rem;
        text-decoration: none;
        transition: background 0.2s ease, box-shadow 0.2s ease;
    }}

    .back-link:hover {{
        background: #1976D2;       /* Darker blue on hover */
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
        text-decoration: none; 
        color: #ffffff;
    }}
    
    .sidebar-content {{
        padding: 1rem;
    }}
    
    .sidebar-content h3 {{
        margin: 0 0 1rem 0;
        font-size: 1rem;
        color: var(--text-color);
        text-align: left;
        /* Allow long task names with underscores to wrap onto multiple lines */
        white-space: normal;
        word-wrap: break-word;
        overflow-wrap: anywhere;
        line-height: 1.3;
    }}
    
    .sidebar-task {{
        margin-bottom: 1.5rem;
    }}
    
    .task-name {{
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        padding: 0.25rem 0;
        border-bottom: 1px solid #e0e0e0;
        /* Allow very long task names to wrap instead of overflowing */
        white-space: normal;
        word-wrap: break-word;
        overflow-wrap: anywhere;
        line-height: 1.3;
    }}
    
    .sidebar-run {{
        margin-bottom: 8px;
    }}
    
    /* Make sidebar run links occupy full width */
    .sidebar-run a {{
        display: block;
        width: 100%;
        text-decoration: none;
    }}
    
    .sidebar-run a:hover, .sidebar-run a:focus, .sidebar-run a:visited {{
        text-decoration: none;
    }}
    
    /* Ensure the coloured badge stretches the whole column */
    .run-score {{
        width: 100%;
    }}
    
    /* Thicker highlight for the currently selected run on desktop */
    @media (min-width: 769px) {{
        .sidebar-run.current a {{
            border-left: 5px solid #2196F3 !important;
        }}
    }}
    
    .main-content {{
        flex: 1;
        margin-left: 180px;
        padding: 0;
        max-width: calc(100vw - 180px);
    }}
    
    .container {{
        /* Allow the main conversation area to take up the full width that is
           available once the fixed sidebar is accounted for. */
        max-width: 100%;
        margin: 0 auto;
        padding: 0 15px;
    }}
    
    h1 {{
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 0.5rem;
    }}
    
    .info-section {{
        background: var(--primary-light);
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        display: flex;
        flex-wrap: nowrap;
        gap: 1rem;
        overflow-x: auto;
        white-space: nowrap;
    }}
    
    .info-item {{
        display: flex;
        flex-direction: column;
        flex-shrink: 0;
        min-width: 140px;
    }}
    
    .info-label {{
        font-weight: 600;
        color: var(--primary-color);
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
        white-space: nowrap;
    }}
    
    .info-value {{
        font-size: 0.9rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    
    .task-info-line, .model-info-line {{
        font-size: 1.3rem;
        word-wrap: break-word;
        overflow-wrap: break-word;
        text-align: left !important;
        display: block;
    }}
    
    .task-name-display {{
        font-weight: 600;
        font-size: clamp(0.9rem, 4vw, 1.3rem);
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.2;
    }}
    
    .plots-container {{
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }}
    
    .plot-section {{
        margin: 0;
    }}
    
    .plot-half {{
        flex: 1;
        width: 50%;
    }}
    
    .plot-section h3 {{
        margin-bottom: 0.8rem;
        color: var(--text-color);
    }}
    
    .plot-container {{
        text-align: center;
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    
    .plot-container img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
    }}
    
    .no-plot {{
        color: #666;
        font-style: italic;
        padding: 2rem;
        text-align: center;
    }}
    
    .conversation-section {{
        margin: 1.5rem 0;
    }}
    
    .message {{
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 18px;
        /* Give the chat bubbles more breathing room. 90 % looks good on both
           desktop and tablet while still leaving a small margin on the side. */
        max-width: 90%;
        position: relative;
    }}
    
    .message.system {{
        background: #e5e5ea;
        color: #000;
        margin-left: auto;
        margin-right: 0;
        border-radius: 18px 18px 4px 18px;
    }}
    
    .message.assistant {{
        background: #007aff;
        color: white;
        margin-left: 0;
        margin-right: auto;
        border-radius: 18px 18px 18px 4px;
    }}
    
    .message-header {{
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .command-badge {{
        background: rgba(255, 255, 255, 0.15);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        opacity: 0.9;
    }}
    
    .message.system .command-badge {{
        background: rgba(0, 0, 0, 0.1);
        color: white;
    }}
    
    .message.assistant .command-badge {{
        background: rgba(255, 255, 255, 0.15);
        color: white;
    }}
    
    /* Premium Glass-Morphism Code Block Container */
    .code-block {{
        position: relative;
        margin: clamp(1.5rem, 2vw, 2rem) 0;
        border-radius: clamp(12px, 3vw, 20px);
        padding: 0;
        overflow: hidden;
        max-width: 100%;
        box-sizing: border-box;
        
        /* Glass-morphism backdrop effects */
        background: var(--code-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        
        /* Multi-layer shadows for depth */
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.12),
            0 2px 8px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.15),
            0 0 0 1px var(--code-border);
        
        /* Smooth animations */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .code-block:hover {{
        transform: translateY(-2px);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.16),
            0 4px 12px rgba(0, 0, 0, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.08);
    }}
    
    /* Code Block Header with Copy Button */
    .code-block-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: clamp(0.75rem, 2vw, 1rem) clamp(1rem, 3vw, 1.5rem);
        background: var(--glass-header-bg);
        border-bottom: 1px solid var(--glass-header-border);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }}
    
    .code-language-tag {{
        font-size: clamp(0.7rem, 1.5vw, 0.75rem);
        font-weight: 600;
        color: var(--glass-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif;
    }}
    
    .code-copy-btn {{
        padding: clamp(0.4rem, 1vw, 0.5rem) clamp(0.6rem, 1.5vw, 0.8rem);
        background: var(--glass-btn-bg);
        border: 1px solid var(--glass-btn-border);
        border-radius: clamp(6px, 1.5vw, 8px);
        color: var(--glass-text-secondary);
        font-size: clamp(0.7rem, 1.5vw, 0.75rem);
        font-weight: 500;
        cursor: pointer;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        touch-action: manipulation;
        user-select: none;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif;
    }}
    
    .code-copy-btn:hover {{
        background: var(--glass-btn-hover);
        border-color: var(--glass-btn-border);
        transform: scale(1.02);
    }}
    
    .code-copy-btn:active {{
        transform: scale(0.98);
    }}
    
    .code-copy-btn.copied {{
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.3);
        color: #059669;
    }}
    
    /* Code Content Container */
    .code-content {{
        position: relative;
        overflow: hidden;
    }}
    
    /* Code Block Content (pre/code tags) */
    .code-block pre, .code-block code {{
        margin: 0;
        padding: 0;
        background: none !important;
        font-family: 'SF Mono', 'Fira Code', 'Menlo', 'Consolas', monospace;
        font-size: clamp(0.8rem, 2vw, 0.85rem);
        line-height: 1.6;
        color: var(--code-text);
        text-shadow: none;
    }}
    
    .code-block pre {{
        padding: clamp(1rem, 3vw, 1.5rem);
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }}
    
    /* Expand/Collapse functionality for long code blocks */
    .code-expandable .code-content {{
        max-height: 400px; /* Default collapsed height */
        transition: max-height 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .code-expandable .code-content.expanded {{
        max-height: 2000px; /* Expanded height */
    }}
    
    .code-expand-overlay {{
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 100px;
        background: var(--glass-expand-bg);
        display: flex;
        justify-content: center;
        align-items: flex-end;
        padding-bottom: 1rem;
        pointer-events: none;
        opacity: 1;
        transition: opacity 0.3s;
    }}
    
    .code-expandable .code-content.expanded + .code-expand-overlay {{
        opacity: 0;
    }}
    
    .code-expand-btn {{
        padding: 0.5rem 1rem;
        background: var(--glass-btn-bg);
        border: 1px solid var(--glass-btn-border);
        border-radius: 8px;
        color: var(--glass-text-secondary);
        font-size: 0.8rem;
        font-weight: 500;
        cursor: pointer;
        pointer-events: all;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.2s;
    }}
    
    .code-expand-btn:hover {{
        background: var(--glass-btn-hover);
    }}
    
    .code-expand-btn::after {{
        content: 'Show More';
    }}
    
    .code-expandable .code-content.expanded + .code-expand-overlay .code-expand-btn::after {{
        content: 'Show Less';
    }}
    
    /* Collapsible Sections */
    .collapsible-sections {{
        margin: 2rem 0;
    }}
    
    .collapsible-section {{
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin-bottom: 1rem;
        overflow: hidden;
    }}
    
    .collapsible-section summary {{
        padding: 1rem;
        font-weight: 600;
        cursor: pointer;
        background: #f8f9fa;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    
    .collapsible-section summary::after {{
        content: '▼';
        font-size: 0.8rem;
        transition: transform 0.2s;
    }}
    
    .collapsible-section[open] summary::after {{
        transform: rotate(180deg);
    }}
    
    .collapsible-section .section-content {{
        padding: 1rem;
        background: white;
    }}
    
    /* Special styling for plot sections to avoid double frames */
    .plot-section-content {{
        background: transparent !important;
        padding: 0 !important;
    }}
    
    .collapsible-section pre {{
        background: var(--code-bg);
        padding: 1rem;
        border-radius: 6px;
        overflow-x: auto;
    }}
    
    .best-code, .reference-code {{
        max-height: 500px;
        overflow-y: auto;
    }}
    
    /* Desktop-specific adjustments for collapsible sections */
    @media (min-width: 769px) {{
        .collapsible-section {{
            margin-bottom: 0.5rem;  /* Reduced from 1rem */
        }}
        
        .collapsible-section summary {{
            padding: 0.75rem 1rem;  /* Reduced vertical padding */
            font-size: 0.95rem;  /* Slightly smaller font */
        }}
        
        .collapsible-section .section-content {{
            padding: 0.75rem 1rem;  /* Reduced padding */
        }}
        
        .collapsible-section pre {{
            font-size: 0.85rem;  /* Smaller font for code blocks */
            line-height: 1.4;
            padding: 0.75rem;
        }}
        
        /* Larger font size for reference and best code on desktop */
        .best-code {{
            font-size: 1rem !important;  /* Increase from default */
            line-height: 1.5;
        }}
        
        .reference-code {{
            font-size: 1rem !important;  /* Increase from default */
            line-height: 1.5;
        }}
        
        .collapsible-sections {{
            margin: 1.5rem 0;  /* Reduced from 2rem */
        }}
    }}
    
    /* Floating back button - hidden by default */
    .mobile-back-button {{
        display: none;
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 9999;  /* ensure it stays above all content */
        background: #2196F3;
        color: white;
        text-decoration: none;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
    }}
    
    .mobile-back-button:hover {{
        background: #1976D2;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }}
    
    .mobile-back-button:active {{
        transform: scale(0.95);
    }}
    
    .mobile-back-button svg {{
        width: 24px;
        height: 24px;
    }}
    
    /* Responsive adjustments for mobile */
    @media (max-width: 768px) {{
        /* Show floating back button on mobile */
        .mobile-back-button {{
            display: flex;
        }}
        
        .main-content {{
            margin-left: 0;
            padding: 60px 10px 0 10px;  /* Added top padding to account for floating button */
            max-width: 100vw;
        }}
        
        .container {{
            padding: 0 5px;
        }}
        
        .plots-container {{
            flex-direction: column;
        }}
        
        .plot-half {{
            width: 100%;
        }}
        
        /* Keep plots within container on mobile */
        .plot-container {{
            overflow: hidden;
            padding: 0.5rem;
        }}
        
        .plot-container img {{
            max-width: 100%;
            height: auto;
        }}
        
        .message {{
            max-width: 100%;
        }}
        
        .header-section {{
            margin-bottom: 0.5rem;
            text-align: left !important;
        }}
        
        /* Mobile trajectory page adjustments */
        .task-info-line, .model-info-line {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-bottom: 6px;
            text-align: left !important;
            align-items: flex-start !important;
        }}
        
        .task-info-line span:first-child,
        .model-info-line span:first-child {{
            font-size: 0.9rem;
            font-weight: 500;
        }}
        
        .task-name-display,
        .model-name-display {{
            font-size: 1.1rem !important;
            font-weight: 600;
        }}
        
        .info-value {{
            font-size: 1.1rem !important;
            font-weight: 600;
        }}
        
        .header-section {{
            margin-bottom: 0.5rem !important;
        }}
        
        .header-section > div:first-child {{
            margin-bottom: 8px !important;
        }}
    }}
    
    /* Ensure container doesn't center content on desktop */
    @media (min-width: 769px) {{
        .container {{
            margin: 0 !important;
            text-align: left !important;
        }}
        .header-section {{
            text-align: left !important;
        }}
        .task-info-line, .model-info-line {{
            text-align: left !important;
        }}
    }}

    /* Additional mobile adjustments for very small screens */
    @media (max-width: 480px) {{
        .header-section {{
            margin-bottom: 0.25rem !important;
        }}
        
        .header-section > div:first-child {{
            margin-bottom: 6px !important;
        }}
        
        .task-info-line, .model-info-line {{
            margin-bottom: 4px !important;
        }}
        
        .info-section {{
            margin-bottom: 0.25rem !important;
        }}
    }}

    .info-section {{
        flex-wrap: wrap;
        justify-content: flex-start;
        margin-bottom: 0.5rem;
    }}
    
    .info-item {{
        min-width: 120px;
        flex-grow: 1;
    }}

    .hide-on-mobile {{
        display: flex;
    }}

    /* Mobile adjustments */
    @media (max-width: 768px) {{
        .hide-on-mobile {{
            display: none !important;
        }}
        
        /* Reduce gap between collapsible sections on mobile */
        .collapsible-sections {{
            margin: 1rem 0;
        }}
        
        .collapsible-section {{
            margin-bottom: 0.5rem;
        }}
        
        .collapsible-section summary {{
            padding: 0.75rem;
            font-size: 0.9rem;
        }}
        
        .collapsible-section .section-content {{
            padding: 0.75rem;
        }}
    }}
    </style>
    <script>
        function copyCode(button, codeId) {{
            const code = document.getElementById(codeId).textContent;
            navigator.clipboard.writeText(code).then(() => {{
                button.textContent = 'Copied!';
                button.classList.add('copied');
                setTimeout(() => {{
                    button.textContent = 'Copy';
                    button.classList.remove('copied');
                }}, 2000);
            }});
        }}

        function toggleExpand(codeBlockId) {{
            const content = document.getElementById('content-' + codeBlockId);
            const overlay = document.getElementById('overlay-' + codeBlockId);
            content.classList.toggle('expanded');
            if (overlay) {{
                overlay.style.display = content.classList.contains('expanded') ? 'none' : 'flex';
            }}
        }}
        
        document.addEventListener('DOMContentLoaded', () => {{
            Prism.highlightAll();
        }});
    </script>
</head>
<body>
    {sidebar_html}
    
    <!-- Floating back button for mobile -->
    <a href="index.html" class="mobile-back-button" aria-label="Back to Speedup Table">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </a>
    
    <div class="main-content">
        <div class="container">
            <div class="header-section" style="text-align: left !important; align-items: flex-start !important; justify-content: flex-start !important;">
                <div style="margin-bottom: 20px; display: flex; align-items: center; gap: 10px; justify-content: flex-start; text-align: left;">
                    <img src="assets/AlgoTunerMascot.png" alt="AlgoTune Mascot" style="height: 32px; width: auto;">
                    <span style="font-weight: 700; font-size: 1.5rem;">AlgoTuner Trajectory</span>
                </div>
                <div class="task-info-line" style="text-align: left !important; margin-bottom: 8px; display: block;">
                    <span style="color: #6c757d; font-weight: 400;">AlgoTune Task:</span>
                    <span class="task-name-display">{data["task_name"]}</span>
                </div>
                <div class="model-info-line" style="text-align: left !important; display: block;">
                    <span style="color: #6c757d; font-weight: 400;">Model:</span>
                    <span class="model-name-display" style="font-weight: 500;">{clean_model_name(data["model_name"])}</span>
                </div>
            </div>
            
            <div class="info-section">
                <div class="info-item">
                    <div class="info-label">Speedup</div>
                    <div class="info-value" style="color: {get_speedup_color(data["final_test_score"])}; font-weight: 600;">{final_test_score_display}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Total Budget</div>
                    <div class="info-value">{budget_display}</div>
                </div>
                <div class="info-item hide-on-mobile">
                    <div class="info-label">Commands Executed</div>
                    <div class="info-value">{data["command_sequence_count"]}</div>
                </div>
                <div class="info-item hide-on-mobile">
                    <div class="info-label">Invalid Commands</div>
                    <div class="info-value">{data["invalid_commands_count"]}</div>
                </div>
                <div class="info-item hide-on-mobile">
                    <div class="info-label">Average Reference Time (ms)</div>
                    <div class="info-value">{data["average_reference_time"]}</div>
                </div>
            </div>
            
            {collapsible_sections_html}
            
            <div class="conversation-section">
                <h2>Conversation Log</h2>
                {conversation_html}
            </div>
        </div>
    </div>
</body>
</html>"""
    return html_content


def generate_individual_log_file(data, output_dir, all_data=None):
    """Generate HTML file for a single log entry."""
    if not data:
        return None

    # Create filename based on task, model, and original filename
    task_name = data["task_name"]
    model_name = data["model_name"]
    original_filename = data["filename"]

    # Extract timestamp from original filename
    # Format: {task}_{model}.html
    clean_filename = f"{task_name}_{model_name.replace('/', '_').replace(' ', '_')}.html"

    filepath = os.path.join(output_dir, clean_filename)

    # Generate individual log HTML content
    html_content = generate_single_log_html(data, all_data)

    # --- Post-processing fixes: remove duplicate university logos ---
    # a) Remove the mobile-only logo strip entirely
    html_content = re.sub(
        r'\s*<div class="mobile-logos mobile-only">[\s\S]*?</div>',
        "",
        html_content,
        flags=re.DOTALL,
    )

    # b) Strip the now-unneeded "desktop-only" class from logo <img> tags
    html_content = html_content.replace(" desktop-only", "")

    # Write the file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return clean_filename


def _load_task_whitelist():
    """Load task names from generation.json, falling back to whitelist.txt if not found."""
    tasks = set()

    # Try to load from generation.json first
    if os.path.exists(GENERATION_FILE):
        try:
            with open(GENERATION_FILE, encoding="utf-8") as f:
                generation_data = json.load(f)
                tasks = set(generation_data.keys())
                logging.info(f"Loaded {len(tasks)} tasks from generation.json")
                return tasks
        except (OSError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load {GENERATION_FILE}: {e}, falling back to whitelist")

    # Fallback to whitelist.txt
    if not os.path.exists(WHITELIST_FILE):
        logging.warning("Neither generation.json nor whitelist file found")
        return tasks

    try:
        with open(WHITELIST_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # Skip empty lines and comments
                    tasks.add(line)
        logging.info(f"Loaded {len(tasks)} tasks from whitelist (fallback)")
    except Exception as e:
        logging.error(f"Error reading whitelist file: {e}")

    return tasks


def _load_task_description_from_file(task_name: str) -> str:
    """Attempt to read tasks/<task_dir>/description.txt for the given task."""
    tasks_root = "tasks"
    pattern = re.compile(rf"task_\d+_{re.escape(task_name)}$")
    for entry in os.scandir(tasks_root):
        if entry.is_dir() and pattern.match(entry.name):
            desc_path = os.path.join(entry.path, "description.txt")
            if os.path.exists(desc_path):
                try:
                    with open(desc_path, encoding="utf-8") as f:
                        return f.read().strip()
                except Exception:
                    pass
    return ""


def _clean_task_description(desc: str) -> str:
    """Clean task description for use in HTML tooltips."""
    import re

    # Strip leading '**TASK DESCRIPTION:**' if present
    desc = re.sub(r"^\*\*?TASK DESCRIPTION:?\*\*?\s*", "", desc.strip(), flags=re.IGNORECASE)
    # Remove duplicate newlines/whitespace and code markers
    lines = desc.split("\n")
    cleaned_lines = []
    for line in lines:
        # Remove code block markers
        if line.strip() in ["```python", "```", "```plaintext"]:
            continue
        # Keep non-empty lines
        if line.strip():
            cleaned_lines.append(line.strip())

    # Join with space and truncate if too long
    result = " ".join(cleaned_lines)
    if len(result) > 300:
        result = result[:297] + "..."
    return result


def generate_index_html(log_data_list, output_dir):
    """Generate index.html with overview tables and navigation."""
    # Load whitelist
    whitelist = _load_task_whitelist()

    if not whitelist:
        logging.warning("Whitelist is empty - no tasks will be shown!")

    # Group data by task for the index, only including whitelisted tasks
    tasks_data = {}
    for data in log_data_list:
        if data:
            task_name = data["task_name"]
            # Skip non-whitelisted tasks
            if task_name not in whitelist:
                continue
            if task_name not in tasks_data:
                tasks_data[task_name] = []
            tasks_data[task_name].append(data)

    # Generate index HTML content
    html_content = generate_index_html_content(tasks_data, log_data_list)

    # Write to file
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_content


def generate_index_html_content(tasks_data, all_data):
    """Generate the main index HTML content."""
    # Generate summary tables
    model_summary_html = generate_model_summary_table(all_data)
    task_summary_html = generate_task_summary_table(tasks_data)

    # Use a plain triple-quoted template so we don't have to escape every {{ }} in code snippets.
    # We will insert the dynamic tables via simple string replacement after constructing the template.
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>AlgoTune</title>
    <link rel="icon" type="image/png" href="assets/AlgoTunerMascot.png">
    
    <!-- Social Preview Tags -->
    <meta property="og:title" content="AlgoTune">
    <meta property="og:description" content="Can Language Models Speed Up General-Purpose Numerical Programs?">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://algotune.io">
    <meta property="og:image" content="https://algotune.io/assets/algotune_banner.png">
    <meta property="og:image:alt" content="AlgoTune - Can Language Models Speed Up General-Purpose Numerical Programs?">
    <meta property="og:site_name" content="AlgoTune">
    
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="AlgoTune">
    <meta name="twitter:description" content="Can Language Models Speed Up General-Purpose Numerical Programs?">
    <meta name="twitter:image" content="https://algotune.io/assets/algotune_banner.png">
    <meta name="twitter:image:alt" content="AlgoTune - Can Language Models Speed Up General-Purpose Numerical Programs?">
    
    <meta name="description" content="AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs?">
    
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <link rel="stylesheet" href="styles.css">
    
    <!-- Prism.js for syntax highlighting - loaded after styles.css to ensure proper precedence -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-sql.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
    
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-7XSBWH5NQF"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());

      gtag('config', 'G-7XSBWH5NQF');
    </script>

    <style>
      /* Responsive styles for leaderboard table */
      @media (max-width: 768px) {
        .leaderboard-table-container {
          max-width: 400px !important;
        }
      }
      
      @media (max-width: 480px) {
        .leaderboard-table-container {
          max-width: 280px !important;
        }
        
        .leaderboard-table-container table {
          font-size: 0.85rem;
        }
        
        .leaderboard-table-container th,
        .leaderboard-table-container td {
          padding: 8px 6px;
        }
      }
      
      /* Make code block larger on desktop */
      .algotune-code-example {
        font-size: 1rem !important;
        line-height: 1.4 !important;
      }
      
      @media (max-width: 768px) {
        .algotune-code-example {
          font-size: 0.8rem !important;
          line-height: 1.25 !important;
        }
      }
      
    </style>

    <style>
      /* ---------------------------------------------------- */
      /* Additional mobile tweaks (logos, buttons, table)    */
      /* ---------------------------------------------------- */

      /* Narrower leaderboard table widths */
      @media (max-width: 768px) {
        /* Keep leaderboard adjustments */
        .leaderboard-table-container {
          max-width: 320px !important;
        }

        /* Stack header elements vertically on tablets & mobile */
        .authors-section {
          flex-direction: column !important;
          gap: 10px;
          align-items: center;
          justify-content: center;
        }

        /* Ensure authors list comes first */
        .authors-container {
          order: 0;
          width: 100%;
          margin-bottom: 10px;
        }

        /* Container holding both logos side-by-side */
        .university-logos {
          order: 1;
          display: flex !important;
          flex-direction: row !important;
          gap: 12px;
          justify-content: center;
          align-items: center;
          width: 100%;
        }

        .university-logo {
          height: 30px;
          width: 30px;
          object-fit: contain;
        }

        /* Slightly enlarge Princeton logo for visual balance - override external CSS */
        .university-logos .university-logo.right-logo {
          height: 40px !important;
          width: auto !important;
          max-height: 40px !important;
          min-height: 40px !important;
        }
        
        /* Ensure both logos have consistent sizing on mobile */
        .university-logos .university-logo {
          height: 40px !important;
          width: auto !important;
          max-height: 40px !important;
          min-height: 40px !important;
        }
        

        /* Buttons row */
        .buttons-container {
          order: 2;
          width: 100%;
          display: flex;
          justify-content: center;
          gap: 16px;
          margin-top: 16px;
          flex-wrap: nowrap;
        }

        .buttons-container .button {
          flex: 1;
          min-width: 120px;
          max-width: 150px;
          text-align: center;
          padding: 10px 16px;
        }
      }

      @media (max-width: 480px) {
        .leaderboard-table-container {
          width: 100% !important;          /* fill available space */
          max-width: 100% !important;     /* override inline 600px limit */
          overflow-x: hidden;             /* no horizontal scroll */
          padding: 0 6px;                 /* keep some breathing room */
          box-sizing: border-box;
        }
        .leaderboard-table-container table {
          font-size: 0.9rem;
          table-layout: fixed;
          width: 100%;
          min-width: 280px;               /* prevent excessive squeezing */
        }
        .leaderboard-table-container th,
        .leaderboard-table-container td {
          padding: 2px 1px;
        }
        
        /* Allow first column (model names) to wrap on mobile */
        .leaderboard-table-container td:first-child {
          white-space: normal;
          word-break: break-word;
          line-height: 1.2;
        }
        
        /* Keep second column (scores) single line and centered */
        .leaderboard-table-container td:nth-child(2),
        .leaderboard-table-container th:nth-child(2) {
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        /* Buttons side by side on small mobile */
        .buttons-container {
          flex-direction: row !important;
          gap: 10px;
          justify-content: center;
        }
      }

      /* Desktop: place Tübingen logo to the left of the authors and
         Princeton logo to the right by adjusting flex order */
      .university-logo.left-logo {
        order: -1;   /* precedes the author list */
      }

      .university-logo.right-logo {
        order: 1;    /* follows the author list */
      }
    </style>

    <!-- Mobile clean-up: remove emojis / markers in speedup timeline -->
    <style>
      @media (max-width: 768px) {
        .timeline-marker {
          display: none !important;
        }
        .performance-rank {
          display: none !important;  /* hide numeric order on mobile */
        }
        .mobile-hint::before {
          content: "" !important;
        }
      }
    </style>

    <style>
      /* Custom mobile overrides for leaderboard table */
      @media (max-width: 480px) {
        .leaderboard-table-container {
          overflow-y: hidden !important; /* eliminate vertical scrollbar */
        }
        .leaderboard-table-container table {
          font-size: 1rem !important; /* slightly larger font */
        }
        /* Center align the second column (AlgoTune Score) on mobile */
        .leaderboard-table-container td:nth-child(2),
        .leaderboard-table-container th:nth-child(2) {
          text-align: center !important;
        }
        
        /* Adjust logo sizes for mobile */
        .model-logo {
          height: 16px !important;
        }
        
        .model-logo-small {
          height: 12px !important;
        }
      }
      
      /* Center align the second column (AlgoTune Score) on desktop too */
      .leaderboard-table-container td:nth-child(2),
      .leaderboard-table-container th:nth-child(2) {
        text-align: center;
      }
      
      /* Model logo styles */
      .model-logo {
        height: 18px;
        width: auto;
        vertical-align: middle;
        margin-right: 6px;
        border-radius: 2px;
      }
      
      .model-logo-small {
        height: 14px;
        width: auto;
        vertical-align: middle;
        margin-right: 4px;
        border-radius: 2px;
      }

      /* Job Market Banner Styles */
      .job-market-banner {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        text-align: center;
        padding: 8px 16px;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
        z-index: 1000;
      }

      /* NeurIPS label under title */
      .algotune-conference {
        display: inline-block;
        align-self: center;
        text-align: center;
        font-weight: 600; /* a bit quieter than 700 */
        font-size: 1.05rem; /* slightly smaller */
        letter-spacing: 0.06em;
        text-transform: none; /* keep mixed-case 'NeurIPS' */
        padding: 6px 12px;
        /* tighten spacing below the NeurIPS label */
        margin: 12px 0 0 0; /* no extra space underneath */
        color: #000000; /* black font for better contrast */
        /* softer, neutral background instead of vivid gradient */
        background: linear-gradient(135deg, #f4f6f8 0%, #eef2f6 100%);
        border-radius: 999px; /* pill */
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border: 2px solid #001f3f; /* navy blue border for NeurIPS 2025 */
      }

      /* Remove extra gap between NeurIPS label and authors row */
      .authors-section {
        padding-top: 0 !important;
        margin-top: 0 !important;
      }
      
      .banner-content {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
      }
      
      .job-market-banner a {
        color: #ffeb3b;
        text-decoration: underline;
        font-weight: 500;
      }
      
      .job-market-banner a:hover {
        color: #fff;
        text-decoration: none;
      }
      
      /* Mobile responsive adjustments */
      @media (max-width: 768px) {
        .job-market-banner {
          padding: 6px 12px;
          font-size: 0.85rem;
        }
        
        .banner-content {
          flex-direction: column;
          gap: 4px;
          text-align: center;
        }
      }
    </style>
</head>
<body>

    <header class="header-section">
        <div class="header-content">
            <div class="algotune-banner">
                <div class="algotune-text">
                    <h1 class="algotune-title">
                        <img src="assets/AlgoTunerMascot.png" alt="AlgoTune Mascot" class="algotune-mascot-inline">
                        AlgoTune
                    </h1>
                    <p class="algotune-subtitle">Can Language Models Speed Up General-Purpose Numerical Programs?</p>
                    <div class="algotune-conference">NeurIPS 2025</div>
                </div>
            </div>
            
            <div class="authors-section">
                <div class="authors-container">
                    <a class="author-name" href="https://oripress.com" target="_blank" rel="noopener noreferrer">Ori Press</a>
                    <a class="author-name" href="https://bamos.github.io/" target="_blank" rel="noopener noreferrer">Brandon Amos</a>
                    <a class="author-name" href="https://hyzhao.me/" target="_blank" rel="noopener noreferrer">Haoyu Zhao</a>
                    <a class="author-name" href="https://yikai-wu.github.io/" target="_blank" rel="noopener noreferrer">Yikai Wu</a>
                    <a class="author-name" href="https://samlikes.pizza/" target="_blank" rel="noopener noreferrer">Samuel K. Ainsworth</a>
                    <a class="author-name" href="https://krupke.cc/" target="_blank" rel="noopener noreferrer">Dominik Krupke</a>
                    <a class="author-name" href="https://kidger.site/" target="_blank" rel="noopener noreferrer">Patrick Kidger</a>
                    <span class="author-name">Touqir Sajed</span>
                    <a class="author-name" href="https://stellato.io/" target="_blank" rel="noopener noreferrer">Bartolomeo Stellato</a>
                    <a class="author-name" href="https://jisunp515.github.io/" target="_blank" rel="noopener noreferrer">Jisun Park</a>
                    <a class="author-name" href="https://nathanaelbosch.github.io/" target="_blank" rel="noopener noreferrer">Nathanael Bosch</a>
                    <span class="author-name">Eli Meril</span>
                    <span class="author-name">Albert Steppi</span>
                    <a class="author-name" href="https://arman-z.github.io/" target="_blank" rel="noopener noreferrer">Arman Zharmagambetov</a>
                    <a class="author-name" href="https://fangzhaoz.github.io/" target="_blank" rel="noopener noreferrer">Fangzhao Zhang</a>
                    <a class="author-name" href="https://davidppineiro.com/" target="_blank" rel="noopener noreferrer">David Pérez-Piñeiro</a>
                    <span class="author-name">Alberto Mercurio</span>
                    <a class="author-name" href="https://jennyzhanni.com/" target="_blank" rel="noopener noreferrer">Ni Zhan</a>
                    <a class="author-name" href="https://talorabr.github.io/" target="_blank" rel="noopener noreferrer">Talor Abramovich</a>
                    <a class="author-name" href="https://www.lieret.net/" target="_blank" rel="noopener noreferrer">Kilian Lieret</a>
                    <a class="author-name" href="https://hanlin-zhang.com/" target="_blank" rel="noopener noreferrer">Hanlin Zhang</a>
                    <span class="author-name">Shirley Huang</span>
                    <a class="author-name" href="https://bethgelab.org/" target="_blank" rel="noopener noreferrer">Matthias Bethge</a>
                    <a class="author-name" href="https://ofir.io/about/" target="_blank" rel="noopener noreferrer">Ofir Press</a>
                </div>
                <div class="university-logos">
                    <img src="assets/uni_tuebingen_logo.png" alt="University of Tübingen Logo" class="university-logo left-logo">
                    <img src="assets/uni_princeton_logo.png" alt="Princeton University Logo" class="university-logo right-logo">
                </div>
            </div>
            
            <div class="buttons-container"> 
                <a href="https://arxiv.org/abs/2507.15887" target="_blank" class="button">
                    <i class="fas fa-file-pdf"></i> Paper
                </a>
                <a href="https://github.com/oripress/AlgoTune" target="_blank" class="button">
                    <i class="fab fa-github"></i> Code
                </a>
            </div>
        </div>
    </header>

    <main class="overview-section" id="overview">
        <div class="overview-content">
            <p class="overview-text">
                Can language models optimize the runtime of popular algorithms like gzip compression, AES encryption or SVD?
                To answer this, we built AlgoTune, a benchmark consisting of more than one hundred widely used math, physics, and computer science functions. 
                For each function, the goal is to write code that is faster than the reference implementation while producing the same outputs as the reference, on a held-out test set of inputs.
                In addition to the benchmark, we also developed AlgoTuner, an agent which enables language models to iteratively optimize code.
            </p>
            
            <img src="assets/algotune_banner.png" alt="AlgoTune Banner" class="algotune-banner" style="width: 78%; max-width: none;">
            
            <p class="overview-text">
                This site contains AlgoTuner trajectories for all AlgoTune tasks. Each entry shows the complete conversation between the model and the AlgoTune environment, including code edits, timing evaluations, and the iterative optimization process.
            </p>
        </div>
    </main>

    <section class="tables-section">
        <div class="tables-content">
            <h2 class="tables-heading" id="leaderboard"><a href="#leaderboard">Leaderboard</a></h2>
            
            <p class="overview-text">
                We use our agent, called AlgoTuner, to optimize functions in AlgoTune, using ten state-of-the-art models. 
                AlgoTuner, using these models, is able to achieve impressive surface-level speedups on many tasks, but is unable to come up with novel algorithms.
            </p>
            
            {model_summary_html}
            
            <p class="overview-text" style="margin-top: 20px; text-align: left;">
                The AlgoTune score for each model is the harmonic mean of its speedups across all AlgoTune tasks. In the table at the bottom of this page, you can find the speedups achieved by each model on each AlgoTune task.
            </p>
        </div>
    </section>

    <section class="overview-section" id="what-is-task">
        <div class="overview-content">
            <div style="text-align: center; width: 100%;">
                <h3 class="overview-heading" style="font-size: 2em; margin-bottom: 20px; text-align: center !important; text-decoration: none; border: none; border-bottom: none; display: inline-block;">AlgoTune Task Implementation</h3>
            </div>
            
            <p class="overview-text" style="text-align: left; margin-bottom: 6px;">
                To measure speedups for the algorithms in AlgoTune, we implement a class containing three functions for each algorithm.
                One generates problem instances (i.e. in the case of PCA this a matrix and number of components), one method checks that the problem has been solved (i.e. for PCA, we check that the matrix is orthonormal), and the last function is a reference solver (for the PCA task, we just use a PCA solver from scikit-learn).
            </p>
            
            
            <!-- Minimal example collapsible code block -->
            <details class="task-impl-details" style="max-width: 95%; margin: 0;">
              <summary style="cursor: pointer; font-weight: 600; margin: 6px 0 12px 0;">Show code example</summary>
              <pre class="algotune-code-example" style="overflow-x: auto; white-space: pre-wrap; word-break: break-word; margin: 0;"><code class="language-python">    def generate_problem(self, n: int, random_seed: int = 1) -> dict[str, Any]:
        """
        Generate random data matrix using n to control the hardness
        """
        np.random.seed(random_seed)
        # 50 * n samples
        m = 50 * n

        r = max(2, n * 5)  # factorization rank
        # Step 1: Generate non-negative W and H
        W = np.random.rand(m, r)  # m x r
        H = np.random.rand(r, 10 * n)  # r x (10 n)

        # Step 2: Generate Y = W H + small noise
        Y = W @ H
        noise_level = 0.01

        Y += noise_level * np.random.rand(
            m, 10 * n
        )  # additive small noise to simulate imperfection

        return dict(X=Y.tolist(), n_components=r)

    def solve(self, problem: dict[str, Any]) -> list[list[float]]:
        try:
            # use sklearn.decomposition.PCA to solve the task
            model = sklearn.decomposition.PCA(n_components=problem["n_components"])
            X = np.array(problem["X"])
            X = X - np.mean(X, axis=0)
            model.fit(X)
            V = model.components_
            return V.tolist()
        except Exception as e:
            logging.error(f"Error: {e}")
            n_components = problem["n_components"]
            n, d = np.array(problem["X"]).shape
            V = np.zeros((n_components, n))
            id = np.eye(n_components)
            V[:, :n_components] = id
            return V.tolist()  # return trivial answer

    def is_solution(self, problem: dict[str, Any], solution: list[list[float]]) -> bool:
        try:
            n_components = problem["n_components"]
            V = np.array(solution)
            X = np.array(problem["X"])
            X = X - np.mean(X, axis=0)

            r, n = V.shape
            # make sure that the number of components is satisfied
            if n_components != r:
                return False
            # check shape
            if n != X.shape[1]:
                return False

            tol = 1e-4
            # check if the matrix V is orthonormal
            VVT = V @ V.T
            if not np.allclose(VVT, np.eye(n_components), rtol=tol, atol=tol / 10):
                return False

            # check objective
            res = self.solve(problem)
            V_solver = np.array(res)

            obj_solver = np.linalg.norm(X @ V_solver.T) ** 2
            obj_sol = np.linalg.norm(X @ V.T) ** 2
            if np.allclose(obj_sol, obj_solver, rtol=tol, atol=tol / 10):
                return True
            return False

        except Exception as e:
            logging.error(f"Error when verifying solution: {e}")
            return False</code></pre>
            </details>
        </div>
    </section>

    <section class="tables-section">
        <div class="tables-content">
            <h2 class="tables-heading" id="results-logs"><a href="#results-logs">Results & Logs</a></h2>
    
            {task_summary_html}
        </div>
    </section>

</body>
</html>'''

    # Inject the dynamically generated tables into the template.
    html_content = html_template.replace("{model_summary_html}", model_summary_html).replace(
        "{task_summary_html}", task_summary_html
    )

    # The following reordering previously caused duplicate table sections in some builds.
    # Keeping tables in their original position ensures a single, consistent rendering.

    return html_content


def get_model_earliest_dates():
    """Extract the earliest log date for each model from log files."""
    model_dates = {}

    # Process all log files
    if os.path.exists(LOGS_DIR):
        for file in os.listdir(LOGS_DIR):
            if file.endswith(".log"):
                # Extract model and date from filename
                # Pattern: task_model_YYYYMMDD_HHMMSS.log
                basename = file[:-4]  # Remove .log extension

                # Find the date part (8 digits followed by 6 digits at the end)
                match = re.search(r"^(.+?)_(\d{8})_(\d{6})$", basename)
                if match:
                    prefix = match.group(1)  # Everything before the date
                    date_str = match.group(2)

                    # Now we need to extract the model name from the prefix
                    # The prefix is in format: task_name_model_name
                    # We need to figure out where the task name ends and model name begins

                    # Load task names to help identify them
                    task_name = None
                    if os.path.exists(TASKS_DIR):
                        for task_dir in os.listdir(TASKS_DIR):
                            if prefix.startswith(task_dir + "_"):
                                task_name = task_dir
                                break

                    if task_name:
                        # Extract model name by removing task prefix
                        model_name = prefix[len(task_name) + 1 :]  # +1 for underscore
                    else:
                        # Fallback: assume everything after first underscore-delimited segment is model
                        parts = prefix.split("_", 1)
                        if len(parts) > 1:
                            model_name = parts[1]
                        else:
                            continue

                    if model_name:
                        # Convert to datetime for comparison
                        try:
                            log_date = datetime.strptime(date_str, "%Y%m%d")

                            # Update earliest date for this model
                            if model_name not in model_dates or log_date < model_dates[model_name]:
                                model_dates[model_name] = log_date
                        except ValueError:
                            continue

    # Convert dates to string format
    return {model: date.strftime("%Y-%m-%d") for model, date in model_dates.items()}


def generate_model_summary_table(_all_data_unused=None):
    """Generate the leaderboard table from reports/agent_summary.json.

    Harmonic-mean per-model is computed over (whitelisted) tasks.  Each task
    contributes *1×* if the speed-up is "N/A" or below 1.
    """

    summary_path = os.path.join("reports", "agent_summary.json")
    if not os.path.exists(summary_path):
        logging.error("agent_summary.json not found – unable to build leaderboard")
        return "<p style='color:red;'>agent_summary.json missing</p>"

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    # Get earliest dates for each model
    model_dates = get_model_earliest_dates()

    whitelist = _load_task_whitelist()

    # --------------------------------------------------------------
    # Build per-model lists that include *all* whitelist tasks.
    # Any task missing for a model contributes 1×.
    # --------------------------------------------------------------

    # First pass: collect any speedups that are actually present
    # task_map[task][model] = speed
    task_map: dict[str, dict[str, float]] = {}

    for task_name, model_dict in summary.items():
        if task_name not in whitelist:
            continue  # non-whitelisted task

        for raw_model_name, metrics in model_dict.items():
            if "dummy" in raw_model_name.lower():
                continue

            speedup_str = metrics.get("final_speedup", "N/A")
            try:
                speed_val = float(speedup_str)
            except ValueError:
                speed_val = 1.0  # treat N/A as 1

            if speed_val < 1.0:
                speed_val = 1.0  # clamp

            task_map.setdefault(task_name, {})[raw_model_name] = speed_val

    # Derive full set of models that appeared at least once
    all_models: set[str] = set()
    for m_dict in task_map.values():
        all_models.update(m_dict.keys())

    total_tasks = len(whitelist)

    model_harmonics: dict[str, float] = {}
    for model in sorted(all_models):
        vals: list[float] = []
        for task in whitelist:
            vals.append(task_map.get(task, {}).get(model, 1.0))

        # harmonic mean (vals guaranteed non-empty)
        denom = sum(1.0 / v for v in vals)
        model_harmonics[model] = total_tasks / denom

    # Sort descending
    sorted_models = sorted(model_harmonics.items(), key=lambda x: x[1], reverse=True)

    rows_html = ""
    for raw_name, harmonic in sorted_models:
        display_name = clean_model_name(raw_name)
        logo_filename = get_model_logo(display_name)

        if logo_filename:
            logo_html = (
                f'<img src="assets/{logo_filename}" alt="{display_name} logo" class="model-logo"> '
            )
        else:
            logo_html = ""

        # Get the earliest date for this model (if available)
        earliest_date = model_dates.get(raw_name, "")
        date_attr = f'data-earliest-date="{earliest_date}"' if earliest_date else ""

        rows_html += f"""
        <tr {date_attr}>
          <td>{logo_html}{display_name}</td>
          <td>{harmonic:.2f}x</td>
        </tr>"""

    return f"""
    <div class='table-container leaderboard-table-container' style='max-width: 600px; margin: 0 auto; overflow-y: visible; max-height: none;'>
      <table style='cursor: default;' id='model-summary-table'>
        <thead>
          <tr><th>Model Name</th><th>AlgoTune Score</th></tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>"""


def generate_task_summary_table(tasks_data):
    """Generate task vs model performance table with sorting options."""

    # Prepare task data with top speedup for sorting
    task_data_with_speedup = []
    for task_name, task_logs in tasks_data.items():
        # Remove any runs produced by a dummy model
        task_logs = [log for log in task_logs if "dummy" not in log["model_name"].lower()]

        # Deduplicate by (task, model): keep only the newest trajectory per model
        deduped_logs: dict[str, dict] = {}
        for log in task_logs:
            model_key = log.get("model_name", "").lower()
            try:
                mtime = os.path.getmtime(log.get("filepath", ""))
            except Exception:
                mtime = 0
            if model_key not in deduped_logs or mtime > deduped_logs[model_key]["_mtime"]:
                tmp = log.copy()
                tmp["_mtime"] = mtime
                deduped_logs[model_key] = tmp
        task_logs = [
            {k: v for k, v in log.items() if k != "_mtime"} for log in deduped_logs.values()
        ]

        # Successful runs only
        valid_logs = [log for log in task_logs if log["final_test_score"] is not None]

        # Sort by actual speedup descending
        valid_logs.sort(key=lambda x: x["final_test_score"], reverse=True)

        top_speedup = valid_logs[0]["final_test_score"] if valid_logs else 0

        task_data_with_speedup.append((task_name, task_logs, valid_logs, top_speedup))

    # Sort alphabetically by task name by default (case-insensitive)
    task_data_with_speedup.sort(key=lambda x: x[0].lower())

    # Generate sorting controls
    sorting_controls = """
    <div class="sorting-controls">
        <label style="font-weight: 600; margin-right: 1rem;">Sort by:</label>
        <div class="sort-buttons-container">
            <button onclick="sortTable('alphabetical')" id="sort-alphabetical" class="sort-btn active">Alphabetical</button>
            <button onclick="sortTable('speedup-desc')" id="sort-speedup-desc" class="sort-btn">Top Speedup ↓</button>
            <button onclick="sortTable('speedup-asc')" id="sort-speedup-asc" class="sort-btn">Top Speedup ↑</button>
        </div>
    </div>"""

    rows_html = ""

    for task_name, task_logs, valid_logs, top_speedup in task_data_with_speedup:
        # Get task description from the first available log
        task_description = ""
        if task_logs:
            # Try to get task description from any log that has it
            for log in task_logs:
                if log.get("task_description"):
                    task_description = log["task_description"]
                    break

        # If not found in logs, fallback to tasks/ description file
        if not task_description:
            task_description = _load_task_description_from_file(task_name)

        # Clean the task description for tooltip
        cleaned_description = _clean_task_description(task_description)

        rows_html += f"""
        <tr data-task-name="{task_name}" data-top-speedup="{top_speedup}">
          <td data-tooltip="{html.escape(cleaned_description)}" class="task-name-cell">{task_name}</td>"""

        # Show top 4 performers (successful runs first, then failed runs)
        all_logs_sorted = valid_logs.copy()  # Start with successful runs
        failed_logs = [log for log in task_logs if log["final_test_score"] is None]
        all_logs_sorted.extend(failed_logs)  # Add failed runs after successful ones

        for i in range(4):
            if i < len(all_logs_sorted):
                log = all_logs_sorted[i]
                clean_filename = f"{log['task_name']}_{log['model_name'].replace('/', '_').replace(' ', '_')}.html"

                # Determine display text and color
                if log["final_test_score"] is not None:
                    score_color = get_speedup_color(log["final_test_score"])
                    # Use clean model name for display
                    clean_model = clean_model_name(log["model_name"])
                    display_text = f"{clean_model} ({log['final_test_score']:.2f}x)"
                else:
                    score_color = get_speedup_color(None)
                    clean_model = clean_model_name(log["model_name"])
                    display_text = f"{clean_model} (Fail)"

                rows_html += f"""
          <td>
            <a href="{clean_filename}" style="color: #ffffff; text-decoration: none; font-weight: 500; display: block;">
              <span class="score-badge" style="background-color: {score_color}; color: #ffffff;">
                {display_text}
              </span>
            </a>
          </td>"""
            else:
                na_color = get_speedup_color(None)
                rows_html += f"""
          <td>
            <span class="score-badge" style="background-color: {na_color}; color: #ffffff;">-</span>
          </td>"""

        rows_html += """
        </tr>"""

    # Generate Performance Timeline HTML for mobile (Approach #2)
    mobile_timeline_html = ""
    for task_name, task_logs, valid_logs, top_speedup in task_data_with_speedup:
        # Get task description
        task_description = ""
        if task_logs:
            for log in task_logs:
                if log.get("task_description"):
                    task_description = log["task_description"]
                    break
        if not task_description:
            task_description = _load_task_description_from_file(task_name)
        cleaned_description = _clean_task_description(task_description)

        # Prepare speedup items with emoji rankings
        all_logs_sorted = valid_logs.copy()
        failed_logs = [log for log in task_logs if log["final_test_score"] is None]
        all_logs_sorted.extend(failed_logs)

        # Get best speedup for header summary
        best_speedup_text = f"{top_speedup:.2f}x" if top_speedup > 0 else "No Success"

        # Create horizontal performance strips
        performance_strips = []
        # Generate numeric ranks for all entries (not just top 4)
        numeric_ranks = [str(i + 1) for i in range(len(all_logs_sorted))]

        for i in range(len(all_logs_sorted)):
            log = all_logs_sorted[i]
            clean_filename = (
                f"{log['task_name']}_{log['model_name'].replace('/', '_').replace(' ', '_')}.html"
            )
            score_color = get_speedup_color(log["final_test_score"])

            clean_model = clean_model_name(log["model_name"])

            if log["final_test_score"] is not None:
                display_text = f"{clean_model} - {log['final_test_score']:.2f}x"
            else:
                display_text = f"{clean_model} - Failed"

            performance_strips.append(f"""
            <div class="performance-strip">
              <div class="performance-rank">{numeric_ranks[i]}</div>
              <div class="performance-info">
                <a href="{clean_filename}" class="performance-link">
                  <div class="performance-badge" style="background-color: {score_color};">
                    {display_text}
                  </div>
                </a>
              </div>
            </div>""")

        # On mobile, show all available trajectories (no need to fill empty slots)

        mobile_timeline_html += f"""
        <div class="timeline-item" data-task-name="{task_name}" data-top-speedup="{top_speedup}">
          <div class="timeline-header" onclick="toggleTimelineItem(this)">
            <div class="timeline-marker"></div>
            <div class="timeline-title-section">
              <div class="timeline-task-name" data-tooltip="{html.escape(cleaned_description)}">
                {task_name}
              </div>
              <div class="timeline-summary">
                Best: {best_speedup_text}
              </div>
            </div>
            <div class="timeline-toggle">
              <span class="timeline-arrow">▼</span>
            </div>
          </div>
          <div class="timeline-content">
            <div class="performance-strips-container">
              {"".join(performance_strips)}
            </div>
          </div>
        </div>"""

    # Split into parts to avoid f-string parsing issues with complex JavaScript
    table_html = (
        sorting_controls
        + """
    <div id="speedup-section" class="table-container task-table-container" style="position: relative;">
      <!-- Mobile hint -->
      <div class="mobile-hint">
        Click speedups to see full trajectories
      </div>
      
      <!-- Mobile Performance Timeline layout (hidden on desktop) -->
      <div class="mobile-performance-timeline">
        """
        + mobile_timeline_html
        + """
      </div>
      
      <!-- Desktop table layout (hidden on mobile) -->
      <div class="table-wrapper">
        <table id="task-summary-table">
          <thead>
            <tr>
              <th>Task Name</th>
              <th>Best Speedup</th>
              <th>2nd Best Speedup</th>
              <th>3rd Best Speedup</th>
              <th>4th Best Speedup</th>
            </tr>
          </thead>
          <tbody id="task-table-body">
            """
        + rows_html
        + """
          </tbody>
        </table>
      </div>
      <div class="click-hint-container" style="position: absolute; left: calc(100% + 12px); top: 165px;">
        <span class="click-hint" style="font-size: 0.9rem; color: #666; font-style: italic; display: flex; align-items: center; gap: 8px;">
          <span style="font-size: 1.2rem;">👈</span>
          Click to see full trajectories
        </span>
      </div>
    </div>
    
    """
    )

    # Add CSS and JavaScript as separate strings to avoid f-string parsing issues
    table_html += """
    <style>
    /* Wider container specifically for the big task table */
    .table-container { overflow-x: auto; }
    .task-table-container { width: 90%; margin: 0 auto; position: relative; overflow: visible !important; }
    .task-table-container .table-wrapper { overflow-x: auto; }
    .task-name-cell:hover { background-color: #f8f9fa; }

    /* Make first column wider and keep full table on narrow screens via horizontal scroll */
    #task-summary-table { table-layout: fixed; width: 100%; min-width: 700px; }
    #task-summary-table th:first-child,
    #task-summary-table td:first-child { width: 30%; }

    /* White text inside speedup badges */
    .score-badge, .score-badge *, a .score-badge, a .score-badge * { 
        color: #ffffff !important; 
        cursor: pointer; 
        white-space: nowrap; /* prevent text wrapping so badges don't compress */
    }
    /* Ensure numeric columns have enough width and allow horizontal scroll instead of squeezing */
    #task-summary-table td:nth-child(n+2),
    #task-summary-table th:nth-child(n+2) {
        min-width: 135px;
    }
    /* Make leaderboard table non-clickable */
    #model-summary-table, #model-summary-table *, #model-summary-table td, #model-summary-table th {
        cursor: default !important;
    }
    /* Click hint container and styling - positioned to the right of table */
    .click-hint-container {
        position: absolute !important;
        left: calc(100% + 12px) !important;
        top: 105px !important;
        z-index: 1000 !important;
        white-space: nowrap !important;
        pointer-events: none !important;
        min-width: 200px !important;
    }
    
    .click-hint { 
        font-size: 0.9rem; 
        color: #666; 
        font-style: italic;
        animation: bounce 1.5s ease-in-out infinite; 
        display: inline-block;
    }
    
    @keyframes bounce { 
         0%, 100% { transform: translateY(0px); } 
         50% { transform: translateY(-4px); } 
     }
    
    /* Hide mobile timeline on desktop by default */
    .mobile-performance-timeline {
        display: none;
    }
    
    .mobile-hint {
        display: none;
        text-align: center;
        color: #666;
        font-style: italic;
        margin-bottom: 15px;
        font-size: 0.9rem;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Base styles for sorting controls */
    .sorting-controls {
        margin-bottom: 20px;
        padding: 15px;
        background: transparent;
        border: none;
    }
    
    .sorting-controls label {
        font-weight: 600;
        margin-right: 1rem;
        color: #333;
        font-size: 1rem;
    }
    
    .sort-buttons-container {
        display: inline-flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .sort-btn {
        padding: 8px 16px;
        font-size: 0.9rem;
        border: 1px solid #ddd;
        background: white;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-family: inherit;
        font-weight: 500;
        min-height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    
    .sort-btn.active {
        background: #2196F3;
        color: white;
        border-color: #2196F3;
        box-shadow: 0 2px 4px rgba(33, 150, 243, 0.3);
    }
    
    .sort-btn:hover:not(.active) {
        background: #f0f0f0;
        border-color: #ccc;
    }
    
    .sort-btn:focus {
        outline: 2px solid #2196F3;
        outline-offset: 2px;
    }
    
    /* Tooltip styles for task descriptions */
    [data-tooltip] {
        position: relative;
        cursor: help;
    }
    
    [data-tooltip]:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        white-space: normal;
        max-width: 300px;
        width: max-content;
        z-index: 1000;
        line-height: 1.4;
        word-wrap: break-word;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    [data-tooltip]:hover::before {
        content: '';
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%) translateY(100%);
        border: 5px solid transparent;
        border-top-color: rgba(0, 0, 0, 0.9);
        z-index: 1001;
    }
    
    /* Disable default CSS tooltip – JS-powered tooltip handles display */
    [data-tooltip]:hover::after,
    [data-tooltip]:hover::before {
        display: none !important;
        content: none !important;
    }
    
    /* JavaScript tooltip styles */
    .tooltip {
        position: fixed;
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        max-width: 300px;
        width: max-content;
        z-index: 10000;
        line-height: 1.4;
        word-wrap: break-word;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        pointer-events: none;
        opacity: 0;
        transform: translateY(5px);
        transition: opacity 0.2s ease, transform 0.2s ease;
    }
    
    .tooltip.show {
        opacity: 1;
        transform: translateY(0);
    }
    
    .tooltip.bottom {
        /* Additional styles for bottom-positioned tooltips if needed */
    }
    
    /* Mobile Performance Timeline Layout - Professional Accordion Design */
    @media (max-width: 992px) {{
        /* Hide the traditional table and its wrapper on mobile */
        .table-wrapper {{
            display: none !important;
        }}
        
        /* Show timeline layout instead */
        .mobile-performance-timeline {{
            display: block !important;
            margin: 0;
            padding: 0;
        }}
        
        /* Show mobile hint */
        .mobile-hint {{
            display: block !important;
        }}
        
        .task-table-container {{
            width: 100% !important;
            margin: 0 !important;
            padding: 0 10px;
        }}
        
        /* Hide click hint on mobile */
        .click-hint-container {{
            display: none;
        }}
        
        /* Mobile sorting controls improvements */
        .sorting-controls {{
            margin-bottom: 20px !important;
            padding: 15px !important;
            background: #f8f9fa !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        }}
        
        .sorting-controls label {{
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #333 !important;
            display: block !important;
            text-align: center !important;
            margin-bottom: 12px !important;
        }}
        
        .sort-buttons-container {{
            display: flex !important;
            gap: 10px !important;
            flex-wrap: wrap !important;
            justify-content: center !important;
        }}
        
        .sort-btn {{
            padding: 12px 18px !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            border: 2px solid #e0e0e0 !important;
            background: white !important;
            border-radius: 25px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            min-height: 44px !important;
            flex: 1 !important;
            min-width: 120px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }}
        
        .sort-btn.active {{
            background: #2196F3 !important;
            color: white !important;
            border-color: #2196F3 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(33,150,243,0.3) !important;
        }}
        
        .sort-btn:hover:not(.active) {{
            background: #f5f5f5 !important;
            border-color: #ccc !important;
            transform: translateY(-1px) !important;
        }}
        /* NEW: First sort button full width on mobile */
        .sort-buttons-container .sort-btn:first-child {{
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }}
        
        /* Performance Timeline Styles */
        .timeline-item {{
            background: white;
            border-radius: 12px;
            margin-bottom: 12px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.08);
            border: 1px solid #e8e8e8;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .timeline-item:hover {{
            box-shadow: 0 5px 15px rgba(0,0,0,0.12);
            border-color: #2196F3;
        }}
        
        .timeline-header {{
            display: flex;
            align-items: center;
            padding: 16px;
            cursor: pointer;
            background: #fafafa;
            border-bottom: 1px solid #f0f0f0;
            min-height: 64px;
            user-select: none;
            transition: background-color 0.2s ease;
        }}
        
        .timeline-header:hover {{
            background: #f5f5f5;
        }}
        
        .timeline-header:active {{
            background: #f0f0f0;
        }}
        
        .timeline-marker {{
            width: 12px;
            height: 12px;
            background: linear-gradient(135deg, #2196F3, #1976D2);
            border-radius: 50%;
            margin-right: 16px;
            flex-shrink: 0;
            box-shadow: 0 2px 4px rgba(33,150,243,0.3);
        }}
        
        .timeline-title-section {{
            flex: 1;
            min-width: 0;
        }}
        
        .timeline-task-name {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 4px;
            line-height: 1.3;
        }}
        
        .timeline-summary {{
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
        }}
        
        .timeline-toggle {{
            margin-left: 12px;
            flex-shrink: 0;
        }}
        
        .timeline-arrow {{
            font-size: 1.2rem;
            color: #666;
            transition: transform 0.3s ease;
            display: inline-block;
        }}
        
        .timeline-item.expanded .timeline-arrow {{
            transform: rotate(180deg);
        }}
        
        .timeline-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease-out, padding 0.3s ease;
            background: white;
        }}
        
        .timeline-item.expanded .timeline-content {{
            max-height: none;
            padding: 16px;
        }}
        
        /* Performance Strips */
        .performance-strips-container {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .performance-strip {{
            display: flex;
            align-items: center;
            padding: 12px 16px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            transition: all 0.2s ease;
        }}
        
        .performance-strip:hover {{
            background: #e3f2fd;
            border-color: #2196F3;
            transform: translateX(4px);
        }}
        
        .performance-rank {{
            font-size: 1.5rem;
            margin-right: 16px;
            flex-shrink: 0;
        }}
        
        .performance-info {{
            flex: 1;
            min-width: 0;
        }}
        
        .performance-link {{
            text-decoration: none;
            display: block;
        }}
        
        .performance-badge {{
            padding: 10px 16px;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: transform 0.2s ease;
        }}
        
        .performance-link:hover .performance-badge {{
            transform: scale(1.02);
        }}
        
        /* Mobile interaction hint */
        .mobile-hint {{
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            color: #1976d2;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 0.85rem;
            text-align: center;
            margin-bottom: 16px;
            border: 1px solid #bbdefb;
            font-weight: 500;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }}
        
        .mobile-hint::before {{
            content: "📱 ";
            font-size: 1rem;
        }}
    }}
    
    /* Extra mobile optimizations for smaller screens */
    @media (max-width: 480px) {{
        .task-table-container {{
            padding: 0 5px !important;
        }}
        
        .timeline-item {{
            margin-bottom: 10px !important;
            border-radius: 10px !important;
        }}
        
        .timeline-header {{
            padding: 12px !important;
            min-height: 56px !important;
        }}
        
        .timeline-marker {{
            width: 10px !important;
            height: 10px !important;
            margin-right: 12px !important;
        }}
        
        .timeline-task-name {{
            font-size: 1rem !important;
            line-height: 1.2 !important;
        }}
        
        .timeline-summary {{
            font-size: 0.8rem !important;
        }}
        
        .timeline-item.expanded .timeline-content {{
            padding: 12px !important;
        }}
        
        .performance-strips-container {{
            gap: 8px !important;
        }}
        
        .performance-strip {{
            padding: 10px 12px !important;
            border-radius: 8px !important;
        }}
        
        .performance-rank {{
            font-size: 1.3rem !important;
            margin-right: 12px !important;
        }}
        
        .performance-badge {{
            padding: 8px 12px !important;
            font-size: 0.8rem !important;
        }}
        
        /* Compact sorting controls for very small screens */
        .sorting-controls {{
            padding: 12px !important;
            margin-bottom: 16px !important;
        }}
        
        .sorting-controls label {{
            font-size: 1rem !important;
            margin-bottom: 10px !important;
        }}
        
        .sort-btn {{
            font-size: 0.8rem !important;
            padding: 10px 14px !important;
            min-height: 40px !important;
            min-width: 100px !important;
        }}
        
        .mobile-hint {{
            font-size: 0.8rem !important;
            padding: 10px 12px !important;
            margin-bottom: 12px !important;
        }}
    }}
    
    /* Tablet-specific optimizations */
    @media (min-width: 769px) and (max-width: 1024px) {{
        .task-table-container {{
            width: 95% !important;
        }}
        
        #task-summary-table {{
            font-size: 0.95rem;
        }}
        
        .score-badge {{
            font-size: 0.85rem !important;
            padding: 5px 8px !important;
        }}
    }}
    </style>
    
    <script>
    function sortTable(sortType) {
        // Save sort state to localStorage
        localStorage.setItem('taskTableSortType', sortType);
        
        // Remove active class from all buttons
        document.querySelectorAll('.sort-btn').forEach(btn => btn.classList.remove('active'));
        
        // Add active class to clicked button
        document.getElementById('sort-' + sortType).classList.add('active');
        
        // Sort desktop table
        const tableBody = document.getElementById('task-table-body');
        if (tableBody) {
            const rows = Array.from(tableBody.getElementsByTagName('tr'));
            
            rows.sort((a, b) => {
                if (sortType === 'alphabetical') {
                    const nameA = a.dataset.taskName.toLowerCase();
                    const nameB = b.dataset.taskName.toLowerCase();
                    return nameA.localeCompare(nameB);
                } else if (sortType === 'speedup-desc') {
                    const speedupA = parseFloat(a.dataset.topSpeedup) || 0;
                    const speedupB = parseFloat(b.dataset.topSpeedup) || 0;
                    return speedupB - speedupA;
                } else if (sortType === 'speedup-asc') {
                    const speedupA = parseFloat(a.dataset.topSpeedup) || 0;
                    const speedupB = parseFloat(b.dataset.topSpeedup) || 0;
                    return speedupA - speedupB;
                }
            });
            
            rows.forEach(row => tableBody.appendChild(row));
        }
        
        // Sort mobile timeline items
        const mobileTimeline = document.querySelector('.mobile-performance-timeline');
        if (mobileTimeline) {
            const timelineItems = Array.from(mobileTimeline.getElementsByClassName('timeline-item'));
            
            timelineItems.sort((a, b) => {
                if (sortType === 'alphabetical') {
                    const nameA = a.dataset.taskName.toLowerCase();
                    const nameB = b.dataset.taskName.toLowerCase();
                    return nameA.localeCompare(nameB);
                } else if (sortType === 'speedup-desc') {
                    const speedupA = parseFloat(a.dataset.topSpeedup) || 0;
                    const speedupB = parseFloat(b.dataset.topSpeedup) || 0;
                    return speedupB - speedupA;
                } else if (sortType === 'speedup-asc') {
                    const speedupA = parseFloat(a.dataset.topSpeedup) || 0;
                    const speedupB = parseFloat(b.dataset.topSpeedup) || 0;
                    return speedupA - speedupB;
                }
            });
            
            timelineItems.forEach(item => mobileTimeline.appendChild(item));
        }
    }
    
    // Restore sort state on page load
    function restoreSortState() {
        const savedSortType = localStorage.getItem('taskTableSortType');
        if (savedSortType && ['alphabetical', 'speedup-desc', 'speedup-asc'].includes(savedSortType)) {
            sortTable(savedSortType);
        } else {
            // Default to alphabetical if no saved state
            sortTable('alphabetical');
        }
        
        // Restore scroll position
        const savedScrollPosition = localStorage.getItem('taskTableScrollPosition');
        if (savedScrollPosition) {
            // Use setTimeout to ensure DOM is ready and sorting is complete
            setTimeout(() => {
                window.scrollTo(0, parseInt(savedScrollPosition));
                // Clear the saved position after restoring
                localStorage.removeItem('taskTableScrollPosition');
            }, 100);
        }
    }
    
    // Save current sort state (called before navigation)
    function saveCurrentSortState() {
        const activeSortBtn = document.querySelector('.sort-btn.active');
        if (activeSortBtn) {
            const sortType = activeSortBtn.id.replace('sort-', '');
            localStorage.setItem('taskTableSortType', sortType);
        }
        // Save scroll position
        const speedupSection = document.getElementById('speedup-section');
        if (speedupSection) {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            localStorage.setItem('taskTableScrollPosition', scrollTop);
        }
    }
    
    // Performance Timeline Accordion functionality
    function toggleTimelineItem(headerElement) {
        const timelineItem = headerElement.closest('.timeline-item');
        const isExpanded = timelineItem.classList.contains('expanded');
        
        if (isExpanded) {
            timelineItem.classList.remove('expanded');
        } else {
            timelineItem.classList.add('expanded');
        }
    }
    
    // Smart tooltip functionality
    let currentTooltip = null;
    let tooltipTimeout = null;
    let currentCell = null;
    
    function createTooltip(element, text) {
        // Clear any pending removal
        if (tooltipTimeout) {
            clearTimeout(tooltipTimeout);
            tooltipTimeout = null;
        }
        
        // If tooltip already exists for the same element, don't recreate
        if (currentTooltip && currentCell === element) {
            return;
        }
        
        // Remove any existing tooltip
        if (currentTooltip) {
            currentTooltip.remove();
            currentTooltip = null;
        }
        
        currentCell = element;
        
        // Truncate text to 500 characters and add ellipsis if needed
        let displayText = text;
        if (text.length > 500) {
            displayText = text.substring(0, 500) + '...';
        }
        
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = displayText;
        document.body.appendChild(tooltip);
        
        // Position tooltip
        const rect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        
        // Calculate preferred position (above the element)
        let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
        let top = rect.top - tooltipRect.height - 10;
        
        // Check if tooltip goes outside viewport and adjust
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        // Adjust horizontal position if needed
        if (left < 10) {
            left = 10;
        } else if (left + tooltipRect.width > viewportWidth - 10) {
            left = viewportWidth - tooltipRect.width - 10;
        }
        
        // If tooltip would go above viewport, show it below instead
        if (top < 10) {
            top = rect.bottom + 10;
            tooltip.classList.add('bottom');
        }
        
        // Ensure tooltip doesn't go below viewport
        if (top + tooltipRect.height > viewportHeight - 10) {
            top = viewportHeight - tooltipRect.height - 10;
        }
        
        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
        
        // Show tooltip with animation
        setTimeout(() => tooltip.classList.add('show'), 10);
        
        currentTooltip = tooltip;
    }
    
    function removeTooltip(immediate = false) {
        if (tooltipTimeout) {
            clearTimeout(tooltipTimeout);
            tooltipTimeout = null;
        }
        
        if (immediate) {
            if (currentTooltip) {
                currentTooltip.classList.remove('show');
                setTimeout(() => {
                    if (currentTooltip) {
                        currentTooltip.remove();
                        currentTooltip = null;
                        currentCell = null;
                    }
                }, 200);
            }
        } else {
            // Add a small delay to prevent flickering when moving within the cell
            tooltipTimeout = setTimeout(() => {
                if (currentTooltip) {
                    currentTooltip.classList.remove('show');
                    setTimeout(() => {
                        if (currentTooltip) {
                            currentTooltip.remove();
                            currentTooltip = null;
                            currentCell = null;
                        }
                    }, 200);
                }
                tooltipTimeout = null;
            }, 100);
        }
    }
    
    function isMouseOverCell(cell, event) {
        const rect = cell.getBoundingClientRect();
        return event.clientX >= rect.left && 
               event.clientX <= rect.right && 
               event.clientY >= rect.top && 
               event.clientY <= rect.bottom;
    }
    
    // Detect touch device
    function isTouchDevice() {
        return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    }
    
    // Add event listeners to task name cells  
    document.addEventListener('DOMContentLoaded', function() {
        // Restore table sort state
        restoreSortState();
        
        // Save sort state before navigating to any trajectory page
        document.addEventListener('click', function(e) {
            const link = e.target.closest('a[href$=".html"]');
            if (link && link.href.includes('.html') && !link.href.includes('index.html')) {
                saveCurrentSortState();
            }
        });
        const taskCells = document.querySelectorAll('.task-name-cell[data-tooltip]');
        const isTouch = isTouchDevice();
        
        taskCells.forEach(cell => {
            if (isTouch) {
                // Touch device handling
                let touchStartTime = 0;
                let isLongPress = false;
                let touchTimer = null;
                
                cell.addEventListener('touchstart', function(e) {
                    touchStartTime = Date.now();
                    isLongPress = false;
                    
                    // Set up long press detection
                    touchTimer = setTimeout(() => {
                        isLongPress = true;
                        const tooltipText = this.getAttribute('data-tooltip');
                        if (tooltipText) {
                            // Prevent default to avoid triggering click
                            e.preventDefault();
                            createTooltip(this, tooltipText);
                            
                            // Provide haptic feedback if available
                            if (navigator.vibrate) {
                                navigator.vibrate(50);
                            }
                        }
                    }, 600); // 600ms long press
                });
                
                cell.addEventListener('touchend', function(e) {
                    if (touchTimer) {
                        clearTimeout(touchTimer);
                        touchTimer = null;
                    }
                    
                    const touchDuration = Date.now() - touchStartTime;
                    
                    if (isLongPress) {
                        // If it was a long press, don't trigger click
                        e.preventDefault();
                        e.stopPropagation();
                        
                        // Auto-hide tooltip after 5 seconds on touch devices
                        setTimeout(() => {
                            removeTooltip(true);
                        }, 5000);
                    }
                });
                
                cell.addEventListener('touchcancel', function(e) {
                    if (touchTimer) {
                        clearTimeout(touchTimer);
                        touchTimer = null;
                    }
                    isLongPress = false;
                });
                
                // Double tap to navigate, single tap to show tooltip
                let lastTap = 0;
                cell.addEventListener('click', function(e) {
                    if (!isLongPress) {
                        const currentTime = new Date().getTime();
                        const tapLength = currentTime - lastTap;
                        
                        if (tapLength < 500 && tapLength > 0) {
                            // Double tap - allow navigation
                            lastTap = 0;
                            removeTooltip(true);
                            return true; // Allow default click behavior
                        } else {
                            // Single tap - show tooltip
                            lastTap = currentTime;
                            
                            const tooltipText = this.getAttribute('data-tooltip');
                            if (tooltipText) {
                                if (currentTooltip && currentCell === this) {
                                    // Hide existing tooltip
                                    removeTooltip(true);
                                } else {
                                    // Show new tooltip
                                    createTooltip(this, tooltipText);
                                    
                                    // Auto-hide after 4 seconds on mobile
                                    setTimeout(() => {
                                        removeTooltip(true);
                                    }, 4000);
                                }
                            }
                            
                            // Prevent navigation on single tap
                            e.preventDefault();
                            e.stopPropagation();
                            return false;
                        }
                    }
                });
                
            } else {
                // Desktop mouse handling
                cell.addEventListener('mouseenter', function(e) {
                    const tooltipText = this.getAttribute('data-tooltip');
                    if (tooltipText) {
                        createTooltip(this, tooltipText);
                    }
                });
                
                cell.addEventListener('mouseleave', function(e) {
                    // Check if mouse is actually leaving the cell area
                    if (!isMouseOverCell(this, e)) {
                        removeTooltip();
                    }
                });
                
                // Handle mouse movement within the cell
                cell.addEventListener('mousemove', function(e) {
                    // If we have a pending removal and mouse is still in cell, cancel it
                    if (tooltipTimeout && isMouseOverCell(this, e)) {
                        clearTimeout(tooltipTimeout);
                        tooltipTimeout = null;
                    }
                    
                    // Ensure tooltip is shown if we're in the cell
                    const tooltipText = this.getAttribute('data-tooltip');
                    if (tooltipText && (!currentTooltip || currentCell !== this)) {
                        createTooltip(this, tooltipText);
                    }
                });
            }
        });
        
        if (!isTouch) {
            // Global mouse move handler to clean up tooltips when mouse leaves the table area (desktop only)
            document.addEventListener('mousemove', function(e) {
                if (currentTooltip && currentCell) {
                    if (!isMouseOverCell(currentCell, e)) {
                        // Check if mouse is over the tooltip itself
                        const tooltipRect = currentTooltip.getBoundingClientRect();
                        const overTooltip = e.clientX >= tooltipRect.left && 
                                          e.clientX <= tooltipRect.right && 
                                          e.clientY >= tooltipRect.top && 
                                          e.clientY <= tooltipRect.bottom;
                        
                        if (!overTooltip) {
                            removeTooltip(true);
                        }
                    }
                }
            });
        } else {
            // Touch-specific: tap outside to hide tooltip
            document.addEventListener('touchstart', function(e) {
                if (currentTooltip && currentCell) {
                    const rect = currentCell.getBoundingClientRect();
                    const touch = e.touches[0];
                    const isOutside = touch.clientX < rect.left || 
                                    touch.clientX > rect.right || 
                                    touch.clientY < rect.top || 
                                    touch.clientY > rect.bottom;
                    
                    if (isOutside) {
                        removeTooltip(true);
                    }
                }
            });
        }
    });
    
    // Timeline task name tooltip functionality for mobile
    document.addEventListener('DOMContentLoaded', function() {
        const timelineTaskNames = document.querySelectorAll('.timeline-task-name[data-tooltip]');
        const isTouch = isTouchDevice();
        
        timelineTaskNames.forEach(taskName => {
            if (isTouch) {
                // Touch device handling for timeline task names
                let touchStartTime = 0;
                let isLongPress = false;
                let touchTimer = null;
                
                taskName.addEventListener('touchstart', function(e) {
                    touchStartTime = Date.now();
                    isLongPress = false;
                    
                    touchTimer = setTimeout(() => {
                        isLongPress = true;
                        const tooltipText = this.getAttribute('data-tooltip');
                        if (tooltipText) {
                            e.preventDefault();
                            createTooltip(this, tooltipText);
                            
                            if (navigator.vibrate) {
                                navigator.vibrate(50);
                            }
                        }
                    }, 600);
                });
                
                taskName.addEventListener('touchend', function(e) {
                    if (touchTimer) {
                        clearTimeout(touchTimer);
                        touchTimer = null;
                    }
                    
                    if (isLongPress) {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        setTimeout(() => {
                            removeTooltip(true);
                        }, 5000);
                    }
                });
                
                taskName.addEventListener('touchcancel', function(e) {
                    if (touchTimer) {
                        clearTimeout(touchTimer);
                        touchTimer = null;
                    }
                    isLongPress = false;
                });
            } else {
                // Desktop mouse handling for timeline task names
                taskName.addEventListener('mouseenter', function(e) {
                    const tooltipText = this.getAttribute('data-tooltip');
                    if (tooltipText) {
                        createTooltip(this, tooltipText);
                    }
                });
                
                taskName.addEventListener('mouseleave', function(e) {
                    if (!isMouseOverCell(this, e)) {
                        removeTooltip();
                    }
                });
                
                taskName.addEventListener('mousemove', function(e) {
                    if (tooltipTimeout && isMouseOverCell(this, e)) {
                        clearTimeout(tooltipTimeout);
                        tooltipTimeout = null;
                    }
                    
                    const tooltipText = this.getAttribute('data-tooltip');
                    if (tooltipText && (!currentTooltip || currentCell !== this)) {
                        createTooltip(this, tooltipText);
                    }
                });
            }
        });
    });
    </script>"""

    # Fix doubled curly braces that were added to avoid interfering with earlier f-string formatting. Since
    # the final HTML string `table_html` is returned without additional string interpolation, we can safely
    # convert all instances of `{{` / `}}` back to single curly braces so that the CSS and JavaScript blocks
    # are syntactically correct.
    table_html = table_html.replace("{{", "{").replace("}}", "}")

    return table_html


def process_single_log_wrapper(log_file):
    """Wrapper function for parallel log processing."""
    try:
        logging.info(f"Processing: {log_file}")
        # Exclude known bad/obsolete runs explicitly marked in filename
        base = os.path.basename(log_file)
        if "BAD" in base:
            logging.info(f"Skipping BAD-marked log: {base}")
            return None
        data = process_log_file_data(log_file)

        # Skip if we got no data
        if not data:
            return None

        # Exclude logs generated by dummy models
        if "dummy" in data.get("model_name", "").lower():
            return None

        return data
    except Exception as e:
        logging.error(f"Error processing {log_file}: {e}")
        return None


def generate_single_html_wrapper(args):
    """Wrapper function for parallel HTML generation."""
    data, output_dir, all_data = args
    try:
        filename = generate_individual_log_file(data, output_dir, all_data)
        return filename
    except Exception as e:
        logging.error(f"Error generating HTML for {data.get('task_name', 'unknown')}: {e}")
        return None


def main():
    """Main function to process logs and generate HTML visualization."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting trajectory visualization generation...")

    # Check if logs directory exists
    if not os.path.exists(LOGS_DIR):
        logging.error(f"Logs directory '{LOGS_DIR}' not found!")
        return

    # Find all log files
    log_files = []
    for file in os.listdir(LOGS_DIR):
        if file.endswith(".log"):
            log_files.append(os.path.join(LOGS_DIR, file))

    if not log_files:
        logging.error(f"No .log files found in '{LOGS_DIR}'!")
        return

    logging.info(f"Found {len(log_files)} log files to process")

    # Determine number of processes to use (use available CPUs but cap at 16)
    override = os.getenv("HTML_GEN_PROCESSES")
    if override:
        try:
            n_processes = max(1, int(override))
        except ValueError:
            n_processes = 1
    else:
        n_processes = min(cpu_count(), 16, len(log_files))
    logging.info(f"Using {n_processes} processes for parallel log processing")

    # Process log files (optionally in parallel)
    if n_processes == 1:
        log_data_list = [process_single_log_wrapper(p) for p in sorted(log_files)]
    else:
        with Pool(processes=n_processes) as pool:
            log_data_list = pool.map(process_single_log_wrapper, sorted(log_files))

    # Filter out None entries (failed processing)
    valid_data = [data for data in log_data_list if data is not None]
    logging.info(f"Successfully processed {len(valid_data)} out of {len(log_files)} log files")

    if not valid_data:
        logging.error("No log files were successfully processed!")
        return

    # Generate multi-file HTML structure
    print("\nGenerating multi-file HTML structure...")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ensure assets directory exists and copy mascot
    assets_dir = os.path.join(OUTPUT_DIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Copy all assets from assets directory
    import shutil

    # Copy from main assets directory
    if os.path.exists("./assets"):
        for asset_file in os.listdir("./assets"):
            source_path = os.path.join("./assets", asset_file)
            dest_path = os.path.join(assets_dir, asset_file)

            if os.path.isfile(source_path):
                try:
                    shutil.copy2(source_path, dest_path)
                    logging.info(f"Copied asset: {asset_file}")
                except Exception as e:
                    logging.warning(f"Failed to copy {asset_file}: {e}")

    # Also copy company logos from static_site/site/assets directory
    static_assets_dir = "./static_site/site/assets"
    if os.path.exists(static_assets_dir):
        logo_files = [
            "claude_logo.png",
            "deepseek_logo.png",
            "gemini_logo.png",
            "openai_logo.png",
            "qwen_logo.png",
            "z_logo.png",
        ]
        for logo_file in logo_files:
            source_path = os.path.join(static_assets_dir, logo_file)
            dest_path = os.path.join(assets_dir, logo_file)

            # Skip if source and destination are the same file
            if os.path.abspath(source_path) == os.path.abspath(dest_path):
                logging.info(f"Skipping {logo_file} - already in destination")
                continue

            if os.path.isfile(source_path):
                try:
                    shutil.copy2(source_path, dest_path)
                    logging.info(f"Copied company logo: {logo_file}")
                except Exception as e:
                    logging.warning(f"Failed to copy {logo_file}: {e}")

    # Generate individual log files (optionally in parallel)
    logging.info(
        f"Generating {len(valid_data)} individual HTML files using {n_processes} processes"
    )
    html_gen_args = [(data, OUTPUT_DIR, valid_data) for data in valid_data]

    if n_processes == 1:
        individual_files = [generate_single_html_wrapper(args) for args in html_gen_args]
    else:
        with Pool(processes=n_processes) as pool:
            individual_files = pool.map(generate_single_html_wrapper, html_gen_args)

    # Filter out None entries
    individual_files = [filename for filename in individual_files if filename is not None]
    logging.info(f"Generated {len(individual_files)} individual HTML files")

    # Generate index page (if you have this function)
    try:
        generate_index_html(valid_data, OUTPUT_DIR)
        logging.info("Generated index.html")
    except NameError:
        logging.warning("generate_index_html function not found - skipping index generation")

    print("\nHTML visualization generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Individual files: {len(individual_files)}")


if __name__ == "__main__":
    main()
