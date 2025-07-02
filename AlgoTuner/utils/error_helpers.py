import traceback
import logging
import os
import re

_error_messages_cache = {}
_error_message_path = os.path.join("AlgoTuner", "messages", "error_message.txt")

def get_error_messages_cached():
    """Load error messages from file, caching the result."""
    global _error_messages_cache
    if _error_message_path in _error_messages_cache:
        return _error_messages_cache[_error_message_path]

    try:
        # Find the project root (directory containing AlgoTuner/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = script_dir
        while not os.path.isdir(os.path.join(project_root, "AlgoTuner")) and os.path.dirname(project_root) != project_root:
            project_root = os.path.dirname(project_root)
        full_path = os.path.join(project_root, _error_message_path)

        if not os.path.exists(full_path):
            logging.error(f"Error message file not found at: {full_path}")
            default_msg = "Error: Invalid command format. Could not load detailed instructions."
            _error_messages_cache[_error_message_path] = default_msg
            return default_msg
        else:
            with open(full_path, 'r') as f:
                content = f.read()
                _error_messages_cache[_error_message_path] = content
                return content
    except Exception as e:
        logging.error(f"Failed to read error message file {_error_message_path}: {e}")
        default_msg = "Error: Invalid command format. Failed to load detailed instructions."
        _error_messages_cache[_error_message_path] = default_msg
        return default_msg

def get_bad_response_error_message():
    """Alias for get_error_messages_cached for consistent bad response errors."""
    # This function essentially just calls the cached loader now.
    # The logic is kept separate in get_error_messages_cached.
    return get_error_messages_cached()


def extract_error_context(tb_str, error_msg):
    """Extracts context like filename and line number from traceback or error message."""
    context = {}
    # Try parsing traceback first (more reliable)
    try:
        lines = tb_str.strip().split('\n')
        # Look for the last "File ... line ..." entry
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('File'):
                match = re.search(r'File "(.*?)", line (\d+)', line)
                if match:
                    context['file_path'] = match.group(1)
                    context['line_number'] = int(match.group(2))
                    # Maybe extract the line content below it?
                    if i + 1 < len(lines):
                        context['error_line_content'] = lines[i+1].strip()
                    break # Found the most recent file context
        
        # Refine error message if possible (e.g., remove traceback prefix if present)
        if error_msg and tb_str in error_msg:
             context['error'] = error_msg.split(tb_str)[-1].strip()
        elif error_msg:
             context['error'] = error_msg # Keep original if no traceback found within
             
    except Exception as e:
        logging.warning(f"Failed to parse traceback for context: {e}")
        context['error'] = error_msg # Fallback to original error message

    # If traceback parsing failed, try simple regex on error message itself
    if 'file_path' not in context:
        # Example: Look for patterns like "file.py:10: error: ..."
        match = re.search(r'^([\./\w-]+):(\d+):\s*(.*)', error_msg)
        if match:
            context['file_path'] = match.group(1)
            context['line_number'] = int(match.group(2))
            context['error'] = match.group(3).strip()
            
    # --- Ensure 'error' key exists --- 
    if 'error' not in context:
        context['error'] = error_msg # Ensure the original message is always there

    return context 