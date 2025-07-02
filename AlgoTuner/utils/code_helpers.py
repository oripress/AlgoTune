import re
import logging
# Import the function from its NEW location
from AlgoTuner.utils.error_helpers import get_bad_response_error_message

def extract_code_blocks(content: str) -> list[tuple[str, str]]:
    """
    Extract code blocks from a string, leniently ignoring extra text on backtick lines.
    Now supports any fence length ≥ 3 and treats blank interior lines as inline commands.

    Args:
        content: The string to extract code blocks from.

    Returns:
        A list of tuples (language, content). Extracts content between matching backtick fence pairs.
        Returns empty list if fewer than two fence lines are found.
        Returns [("MULTIPLE_COMMANDS", error_msg)] if more than two fence lines found.
    """
    if not content:
        logging.info("extract_code_blocks: Empty content")
        return []

    # Early exit for formatted diff content - these should not be parsed as code blocks
    # This prevents the parser from getting confused by system-generated diffs
    lines = content.splitlines()
    if len(lines) > 10:  # Only check if we have enough lines to be a diff
        diff_line_count = sum(1 for line in lines[:20] if re.match(r'^\s*[|>]\s*\d+:', line))
        if diff_line_count >= 5:  # If we see 5+ diff-style lines in the first 20 lines
            logging.info("extract_code_blocks: Detected formatted diff content, skipping code block extraction")
            return []
    
    # Early exit for error messages that contain command parsing failure text
    if "Command parsing failed" in content and "triple backticks" in content:
        logging.info("extract_code_blocks: Detected error message content, skipping code block extraction")
        return []

    # Single-line fence support: recognize ```cmd args``` on a single line
    stripped = content.strip()
    single_m = re.match(r'^\s*(`{3,})([^\s`]+)(?:\s+(.+?))?\1\s*$', stripped)
    if single_m:
        lang = single_m.group(2)
        body = single_m.group(3) or ""
        return [(lang, body)]

    fence_lines: list[tuple[int, str, str]] = []
    # Find lines with a backtick fence of length ≥ 3
    for idx, line in enumerate(lines):
        # Only log if we're in debug mode or if we find a fence
        # Allow optional leading whitespace before the backtick fence
        m = re.match(r'^\s*(`{3,})(.*)$', line)
        if m:
            logging.info(f"  Found fence at line {idx}: groups={m.groups()}")
            fence_lines.append((idx, m.group(1), m.group(2).strip()))

    # Need at least one opener and one closer
    if len(fence_lines) < 2:
        logging.info("extract_code_blocks: Fewer than two fence lines found.")
        return []
    # If odd number of fence lines, ignore the last unmatched fence
    total_fences = len(fence_lines)
    if total_fences % 2 != 0:
        logging.warning(f"extract_code_blocks: odd number of fence lines ({total_fences}) found; ignoring last fence")
    blocks: list[tuple[str, str]] = []
    # Iterate over each pair of opener/closer fences
    pairs = total_fences // 2
    for i in range(pairs):
        start_idx, fence_str, opener_rest = fence_lines[2*i]
        end_idx, _, _ = fence_lines[2*i + 1]
        opener_line = lines[start_idx].strip()
        block_lines = lines[start_idx + 1 : end_idx]

        # Unified opener_rest + block_lines logic
        lang_match = re.match(r'^`{3,}\s*(\w+)', opener_line)
        if lang_match:
            lang = lang_match.group(1)
        else:
            parts_lang = opener_rest.split(maxsplit=1)
            lang = parts_lang[0] if parts_lang and parts_lang[0] else ""
        # Extract opener_rest text beyond language token
        extra = ""
        if opener_rest:
            parts_extra = opener_rest.split(maxsplit=1)
            if parts_extra[0] == lang:
                extra = parts_extra[1] if len(parts_extra) > 1 else ""
            else:
                extra = opener_rest
        # Build block content: opener_rest extra first, then interior lines
        block_content_lines = []
        if extra:
            block_content_lines.append(extra)
        block_content_lines.extend(block_lines)
        clean_block = "\n".join(block_content_lines).strip()
        logging.debug(f"extract_code_blocks: Extracted block {i+1}/{pairs} with language '{lang}' and content: {clean_block[:50]}...")

        # Logging summary for this block
        log_msg = f"extract_code_blocks: Extracted block {i+1}/{pairs}"
        if lang:
            log_msg += f" with language '{lang}'"
        if not clean_block:
            log_msg += ": [empty block]"
            logging.warning("extract_code_blocks: Extracted empty block")
        elif len(clean_block) > 50:
            log_msg += f": {clean_block[:50]}..."
        else:
            log_msg += f": {clean_block}"
        logging.info(log_msg)
        blocks.append((lang, clean_block))
    return blocks


def check_for_text_after_command(command: str) -> tuple[bool, str]:
    """
    Check if there is text after a command in a code block.
    
    This is a wrapper for backward compatibility that delegates to 
    the implementation in command_helpers.py to avoid code duplication.
    
    Args:
        command: The command string to check
        
    Returns:
        A tuple of (has_text_after_command, warning_message)
        If has_text_after_command is True, warning_message contains the warning
        If has_text_after_command is False, warning_message is empty
    """
    from AlgoTuner.utils.command_helpers import check_for_text_after_command as cmd_check
    result = cmd_check(command)
    # Convert the return type to match the expected tuple[bool, str] return type
    # instead of Tuple[bool, Optional[str]] from command_helpers
    return (result[0], result[1] or "") 