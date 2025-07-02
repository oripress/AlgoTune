import os
from typing import Optional, List, Union
import re

def clean_traceback(
    traceback: Union[str, List[str]], 
    code_dir: Optional[str] = None,
    include_full_paths: bool = False
) -> str:
    """Clean and format a traceback to show relevant information.
    
    Args:
        traceback: The traceback string or list of lines to clean
        code_dir: Optional directory to filter traceback lines to only those containing this path.
                 If None, all lines are included.
        include_full_paths: Whether to keep full paths or strip to relative paths.
                          Only applies if code_dir is provided.
    
    Returns:
        A cleaned traceback string with only relevant information.
    """
    if not traceback:
        return "No traceback available"
    
    # Convert string to lines if needed
    if isinstance(traceback, str):
        lines = traceback.split('\n')
    else:
        lines = traceback
    
    # Filter and clean lines
    cleaned_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
            
        # If it's a file line
        if line.lstrip().startswith('File "'):
            # Only include if it's from our code directory
            if code_dir and code_dir not in line:
                i += 1
                continue
                
            # Clean up the path but preserve the structure
            if code_dir and not include_full_paths:
                parts = line.split(code_dir + '/')
                if len(parts) > 1:
                    indent = len(line) - len(line.lstrip())
                    line = ' ' * indent + 'File "' + parts[1]
            
            cleaned_lines.append(line)
            
            # Add the "in function" line if it exists
            if i + 1 < len(lines) and lines[i + 1].strip():
                cleaned_lines.append(lines[i + 1].rstrip())
                i += 1
                
            # Add the code line if it exists
            if i + 2 < len(lines) and lines[i + 2].strip():
                cleaned_lines.append(lines[i + 2].rstrip())
                i += 2
            
        # Keep error message lines
        elif not line.startswith(' '):
            cleaned_lines.append(line)
            
        i += 1
    
    # If no relevant lines found
    if not cleaned_lines:
        return "No relevant traceback lines found"
    
    return '\n'.join(cleaned_lines)

def format_error(
    error_msg: str,
    traceback: Optional[str] = None,
    code_dir: Optional[str] = None,
    include_full_paths: bool = False,
    prefix: str = "Error"
) -> str:
    """Format an error message with its traceback.
    
    Args:
        error_msg: The main error message
        traceback: Optional traceback string
        code_dir: Optional directory to filter traceback lines
        include_full_paths: Whether to keep full paths in traceback
        prefix: Prefix to use for the error message (e.g., "Error", "Solver error", etc.)
    
    Returns:
        A formatted error string with message and relevant traceback
    """
    # If error_msg already contains the prefix, don't add it again
    if not error_msg.startswith(prefix):
        error_msg = f"{prefix}: {error_msg}"
    
    parts = [error_msg]
    
    if traceback:
        cleaned_tb = clean_traceback(traceback, code_dir, include_full_paths)
        if cleaned_tb != "No relevant traceback lines found":
            parts.append("Traceback:")
            parts.append(cleaned_tb)
    
    return '\n'.join(parts)

def clean_build_output(output: Optional[str]) -> str:
    """Clean build output by stripping full file paths, leaving only file names."""
    if not output:
        return ""
    # Pattern to match absolute file paths
    pattern = re.compile(r'/[^\s:,]+')
    cleaned_lines = []
    for line in output.splitlines():
        # Replace each absolute path with its basename
        cleaned_line = pattern.sub(lambda m: os.path.basename(m.group(0)), line)
        cleaned_lines.append(cleaned_line)
    return '\n'.join(cleaned_lines) 