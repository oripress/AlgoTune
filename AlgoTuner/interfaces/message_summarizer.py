import re
from AlgoTuner.interfaces.commands.types import EditStatus, EvalStatus
from typing import Optional
import logging
from AlgoTuner.utils.message_writer import MessageWriter
from AlgoTuner.utils.snippet_utils import compute_centered_snippet_bounds, compute_snippet_bounds
from AlgoTuner.utils.profiler import TaskProfiler


def extract_eval_info(content, eval_status: Optional[str] = None):
    """Extract evaluation information from a message."""
    avg_diff_match = re.search(r"Average Difference: ([\d.]+)", content)
    best_speedup_match = re.search(r"Snapshot .* saved as new best", content)

    if avg_diff_match:
        avg_diff = float(avg_diff_match.group(1))
        if best_speedup_match:
            return f"Best speedup: Eval score: {avg_diff:.2f}"
        return f"Eval score: {avg_diff:.2f}"
    elif eval_status == EvalStatus.FAILURE.value:
        return "Eval Failed"
    elif eval_status == EvalStatus.TIMEOUT.value:
        return "Eval Timeout"
    return None


def extract_edit_info(content, edit_status: Optional[str] = None):
    """Extract information about code edits."""
    lines = [
        line.strip()
        for line in content.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    edited_lines = len(lines)
    total_lines = len(content.splitlines())

    if edit_status == EditStatus.FAILURE.value:
        # Extract error message if present
        error_start = content.find("Your changes were not applied.") + len(
            "Your changes were not applied."
        )
        error_end = content.find("\n", error_start)
        if error_end != -1:
            error_msg = content[error_start:error_end].strip()
            return f"Edit Failed: {error_msg}"
        return "Edit Failed"
    elif edit_status == EditStatus.SUCCESS.value:
        return f"{edited_lines} lines changed" if edited_lines > 0 else None

    return None


def extract_command_info(content):
    """Extract information about commands used."""
    if content.startswith("edit "):
        return "Edit command"
    elif content.startswith("oracle "):
        return "Oracle command"
    elif content.startswith("eval_input "):
        return "Eval command"
    elif content.startswith("revert "):
        return "Revert command"
    elif content.startswith("ls "):
        return "List command"
    elif content.startswith("view_file "):
        return "View command"
    return None


def generate_message_summary(role, content):
    """Generate a meaningful summary for a message based on its content.
    Handles duplicated content by identifying and removing repetitions.
    For messages sent to LLM (user/system roles), skips budget information.
    For messages from LLM (assistant role), summarizes key actions.
    """
    # Split content into lines for analysis
    lines = content.splitlines()

    # Skip empty messages
    if not lines:
        return f"Empty {role} message"

    # For messages sent to LLM (user/system roles), skip budget information
    if role in ["user", "system"] and len(lines) > 1:
        # Skip budget status line if present
        if "You have so far sent" in lines[0] and "remaining" in lines[0]:
            lines = lines[1:]

    # Remove duplicate blocks of content
    unique_blocks = []
    current_block = []

    for line in lines:
        # Skip empty lines between duplicated blocks
        if not line.strip():
            if current_block:
                block_content = "\n".join(current_block)
                if block_content not in unique_blocks:
                    unique_blocks.append(block_content)
                current_block = []
            continue

        current_block.append(line)

    # Add the last block if it exists
    if current_block:
        block_content = "\n".join(current_block)
        if block_content not in unique_blocks:
            unique_blocks.append(block_content)

    # Join unique blocks with newlines
    unique_content = "\n\n".join(unique_blocks)

    # For assistant messages, try to extract key information
    if role == "assistant":
        # Look for command patterns
        if "edit" in unique_content.lower():
            return "Assistant: Edit command"
        elif "eval" in unique_content.lower():
            return "Assistant: Evaluation command"
        elif any(cmd in unique_content.lower() for cmd in ["run", "execute", "test"]):
            return "Assistant: Run/Execute command"

    # For user messages with file content, summarize appropriately
    if role == "user" and "Current file content:" in unique_content:
        return "User: File content update"

    # Return truncated unique content with role
    max_summary_length = 100
    if len(unique_content) > max_summary_length:
        return f"{role.capitalize()}: {unique_content[:max_summary_length]}..."

    return f"{role.capitalize()}: {unique_content}"

    if role == "system":
        if "eval" in content.lower():
            eval_info = extract_eval_info(content)
            if eval_info:
                return eval_info
        return "System message"
    elif role == "assistant":
        # Try to extract command info first
        command_info = extract_command_info(content)
        if command_info:
            return command_info

        # Then try edit info
        edit_info = extract_edit_info(content)
        if edit_info:
            return edit_info
        return "Assistant message"
    elif role == "user":
        # Keep user messages simple
        return "User message"

    # Default fallback
    return f"{role.capitalize()} message"
