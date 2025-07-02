import re
import logging
from typing import List, Optional, Tuple

# Import from utils.py for common utilities
from AlgoTuner.utils.utils import extract_line_numbers
from AlgoTuner.interfaces.commands.types import COMMAND_FORMATS, COMMAND_PATTERNS
from AlgoTuner.utils.error_helpers import get_error_messages_cached

def get_default_error_message():
    try:
        with open("messages/error_message.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "An unexpected error occurred. Please contact support."

# Define error message here to avoid circular imports (using default error message from file)
ERROR_MESSAGES_EMPTY_COMMAND = get_default_error_message()

def extract_all_line_numbers_from_string(error_msg: str) -> List[int]:
    """
    Extract all line numbers from the error message string.
    Returns a sorted list of unique line numbers as integers.
    """
    return extract_line_numbers(error_msg)


def check_for_text_after_command(command_str: str) -> Tuple[bool, Optional[str]]:
    """
    Check if there is any text after a command block in the input.
    
    Args:
        command_str: The full string containing the command
        
    Returns:
        Tuple of (has_extra_text, error_message)
    """
    lines = command_str.strip().split('\n')
    in_command_block = False
    command_end_line = None
    
    # Find the end of the command block
    for i, line in enumerate(lines):
        if line.strip() == "---":
            # If we're in a command block and find ---, this is the end
            if in_command_block:
                command_end_line = i
                break
            # Otherwise, we're entering a command block
            in_command_block = True
    
    # If no command block or no end found, check for single-line commands
    if command_end_line is None:
        # Check for single line commands (ls, revert, etc.)
        single_line_commands = ["ls", "revert", "eval"]
        for cmd in single_line_commands:
            if lines and lines[0].strip() == cmd:
                command_end_line = 0
                break
                
        # Check for commands with arguments on a single line
        if command_end_line is None and lines:
            first_word = lines[0].split()[0] if lines[0].split() else ""
            
            # Special handling for commands with multiline code blocks or inputs
            # These commands allow multiline inputs without treating them as separate commands
            multiline_input_commands = ["eval_input", "profile", "profile_lines", "oracle"]
            if first_word in multiline_input_commands:
                return False, None
                
            # Other commands with arguments
            other_commands = ["view_file"]
            if first_word in other_commands:
                command_end_line = 0
                
    # If we found the end of a command, check if there's text after it
    if command_end_line is not None and command_end_line < len(lines) - 1:
        # Check if there's any non-whitespace content after the command
        remaining_text = '\n'.join(lines[command_end_line + 1:]).strip()
        if remaining_text:
            # Check if there's another command in the remaining text
            # by looking for common command patterns
            command_patterns = list(COMMAND_PATTERNS.keys())
            has_another_command = False
            
            # Look for code blocks that might contain commands
            code_block_markers = ["```", "'''"]
            for marker in code_block_markers:
                if marker in remaining_text:
                    has_another_command = True
                    break
            
            # Look for specific command keywords
            for cmd in command_patterns:
                if cmd + " " in remaining_text or cmd + "\n" in remaining_text or remaining_text.strip() == cmd:
                    has_another_command = True
                    break
            
            error_msg = get_default_error_message()
            
            return True, error_msg
    
    return False, None


def find_command(
    input_str: str, valid_commands: Optional[List[str]] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Find a command in the input string.
    Returns a tuple of (command, args, error_message).
    If no command is found, returns (None, None, error_message).
    If a command is found but has invalid format, returns (command, None, error_message).
    If a command is found and valid, returns (command, args, None).

    Args:
        input_str: The input string to search for commands
        valid_commands: Optional list of valid command names. If not provided, uses all commands from COMMAND_PATTERNS
    """
    command_str = input_str.strip()
    if not command_str:
        from AlgoTuner.interfaces.commands.types import ERROR_MESSAGES
        return None, None, ERROR_MESSAGES["empty_command"]

    # Use provided valid_commands or all commands from COMMAND_PATTERNS
    if valid_commands is None:
        valid_commands = list(COMMAND_PATTERNS.keys())

    # Special handling for commands that don't take arguments
    no_arg_commands = {"ls", "revert", "eval"}
    first_word = command_str.split()[0] if command_str.split() else ""

    if first_word in no_arg_commands:
        if first_word not in valid_commands:
            # Move import and call inside function to avoid circular dependency
            error_msgs = get_error_messages_cached()
            return None, None, f"Unknown command: {first_word}\n\n{error_msgs}"
        # For these commands, the entire string must match the pattern exactly
        pattern = COMMAND_PATTERNS[first_word]
        if not re.match(pattern, command_str):
            return (
                first_word,
                None,
                f"Invalid {first_word} format. Expected: {first_word} (no arguments)",
            )
        return first_word, "", None

    # For other commands, proceed with normal parsing
    parts = command_str.split(maxsplit=1)
    if not parts:
        from AlgoTuner.interfaces.commands.types import ERROR_MESSAGES
        return None, None, ERROR_MESSAGES["empty_command"]

    command = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    if command not in valid_commands:
        # Move import and call inside function to avoid circular dependency
        error_msgs = get_error_messages_cached()
        return None, None, f"Unknown command: {command}\n\n{error_msgs}"

    if command not in COMMAND_PATTERNS:
        # Move import and call inside function to avoid circular dependency
        error_msgs = get_error_messages_cached()
        return None, None, f"Unknown command: {command}\n\n{error_msgs}"

    # For commands that take arguments, validate the full pattern
    pattern = COMMAND_PATTERNS[command]
    if not re.match(pattern, command_str):
        expected_format = COMMAND_FORMATS[command].example
        return command, None, f"Bad command format for '{command}'. Expected format:\n{expected_format}"

    return command, args, None
