import os
import logging
import traceback
from typing import Optional, Dict, Any
from concurrent.futures import TimeoutError as FuturesTimeoutError # Need this for type check
import errno as _errno

from AlgoTuner.utils.utils import clean_traceback

CODE_DIR = os.environ.get("CODE_DIR", "llm_src")

# --- New Custom Exception and Message for Solver Not Found ---
class SolverFileNotFoundError(FileNotFoundError):
    """Custom exception for when solver.py is not found."""
    pass

SOLVER_NOT_FOUND_GENERIC_MSG = "Error: solver.py not found."
# --- End New Custom Exception and Message ---

def extract_error_context(tb_str, error_msg):
    """Extracts error context (line number, file, code snippet) from a traceback string.
    
    Args:
        tb_str: The traceback string
        error_msg: The error message
        
    Returns:
        A dictionary containing:
        - enhanced_error_message: Error message potentially enriched with file/line info.
        - code_context_snippet: String containing surrounding code lines, or None.
    """
    logging.info(f"DIAG: extract_error_context called with error_msg: {error_msg[:100]}...")
    try:
        # Clean the traceback first
        cleaned_tb_str = clean_traceback(tb_str)
        logging.info(f"DIAG: Cleaned traceback (first 100 chars): {cleaned_tb_str[:100]}...")
        
        # Extract traceback to find the actual line causing the issue
        tb_lines = cleaned_tb_str.strip().split('\n')
        logging.info(f"DIAG: Found {len(tb_lines)} lines in traceback")
        
        # Initialize error_type
        error_type = ""
        
        # Attempt to extract error type ONLY if we have a non-empty traceback
        # and the last line looks like a potential exception message
        if tb_lines and tb_lines[-1].strip():
            actual_error_line = tb_lines[-1]
            logging.info(f"DIAG: Last line of traceback: {actual_error_line}")

            # Check if the last line looks like an Exception: message format
            # Also ensure it's not just the start of the traceback itself
            if ":" in actual_error_line and not actual_error_line.startswith(" ") and not actual_error_line.startswith("Traceback"):
                error_parts = actual_error_line.split(":", 1)
                if len(error_parts) == 2:
                    extracted_error_type = error_parts[0].strip()
                    extracted_actual_error = error_parts[1].strip()
                    # Only overwrite the original error_msg if we extracted a non-empty error description
                    # AND the extracted type isn't something generic like "NoneType" which can happen with empty tracebacks
                    if extracted_actual_error and extracted_error_type != "NoneType":
                        error_type = extracted_error_type # Store the extracted type
                        error_msg = f"{error_type}: {extracted_actual_error}" # Overwrite original error_msg
                        logging.info(f"DIAG: Refined error message based on traceback: {error_msg}")
                    else:
                         logging.info(f"DIAG: Not refining error message; extracted error type '{extracted_error_type}' or actual error '{extracted_actual_error}' was empty or invalid.")
            else:
                logging.info(f"DIAG: Last line of traceback ('{actual_error_line}') does not appear to be a standard exception message. Not attempting to refine error message.")
        else:
             logging.info(f"DIAG: Traceback appears empty or invalid. Using original error message: {error_msg}")
        
        # Look for line number and file that's relevant to the user's code
        error_line_info = ""
        code_snippet_line = ""
        user_file_path: Optional[str] = None
        user_line_no: int = 0
        func_name = "<unknown>"
        
        # First pass: Look for user code frames
        for i in range(len(tb_lines) - 1, -1, -1):
            line = tb_lines[i]
            if "File " in line and ".py" in line:
                try:
                    current_file_path = line.split('"')[1] if '"' in line else line.split("'")[1] if "'" in line else ""
                    current_line_no = int(line.split(", line ")[1].split(",")[0])
                    
                    logging.info(f"DIAG: Found error in file: {current_file_path}, line: {current_line_no}")
                    
                    # Check if this file is likely user code
                    if "solver.py" in current_file_path or current_file_path.startswith(CODE_DIR) or "AlgoTune/tasks/" in current_file_path or "AlgoTuneTasks/" in current_file_path:
                        # Try to get function name from the same line
                        if ", in " in line:
                            func_name = line.split(", in ")[1].strip()
                        # Try to get the code snippet line following the File line
                        code_snippet_line = ""
                        if i + 1 < len(tb_lines) and tb_lines[i+1].strip():
                            code_snippet_line = tb_lines[i+1].strip()
                            
                        # Construct error line info including function name
                        file_name_for_msg = os.path.basename(current_file_path)
                        error_line_info = f" in function '{func_name}' at line {current_line_no} in {file_name_for_msg}"
                        
                        # MODIFIED: Store identified user code location
                        user_file_path = current_file_path
                        user_line_no = current_line_no
                            
                        logging.info(f"DIAG: Identified as user code, error_line_info: {error_line_info}")
                        break # Found the most relevant frame
                except (IndexError, ValueError) as e:
                    logging.info(f"DIAG: Error parsing file/line info: {e}")
                    continue # Ignore lines that don't parse correctly
        
        # Second pass: If no user code found, show system code where error occurred (for system-level errors like MemoryError)
        if not user_file_path:
            logging.info(f"DIAG: No user code found in traceback, looking for system code where error occurred")
            for i in range(len(tb_lines) - 1, -1, -1):
                line = tb_lines[i]
                if "File " in line and ".py" in line:
                    try:
                        current_file_path = line.split('"')[1] if '"' in line else line.split("'")[1] if "'" in line else ""
                        current_line_no = int(line.split(", line ")[1].split(",")[0])
                        
                        # Accept any Python file as fallback for system errors
                        if current_file_path.endswith('.py'):
                            if ", in " in line:
                                func_name = line.split(", in ")[1].strip()
                            if i + 1 < len(tb_lines) and tb_lines[i+1].strip():
                                code_snippet_line = tb_lines[i+1].strip()
                                
                            file_name_for_msg = os.path.basename(current_file_path)
                            error_line_info = f" in function '{func_name}' at line {current_line_no} in {file_name_for_msg} (system code)"
                            
                            user_file_path = current_file_path
                            user_line_no = current_line_no
                                
                            logging.info(f"DIAG: Using system code for context, error_line_info: {error_line_info}")
                            break
                    except (IndexError, ValueError) as e:
                        logging.info(f"DIAG: Error parsing system file/line info: {e}")
                        continue
        
        # Get surrounding code if we found a valid file path and line number in user code
        surrounding_code = None
        if user_file_path and user_line_no > 0:
            # --- Make Path Absolute (More Robust Method) ---
            # Use os.path.abspath to handle both relative (to CWD) and absolute paths
            abs_file_path = os.path.abspath(user_file_path)
            logging.info(f"DIAG: Resolved path from traceback ('{user_file_path}') to absolute path '{abs_file_path}'")
            # --- End Make Path Absolute ---
            
            # Check existence of the resolved absolute path
            if os.path.exists(abs_file_path):
                logging.info(f"DIAG: File {abs_file_path} exists, attempting to read surrounding code")
                try:
                    with open(abs_file_path, 'r') as f:
                        logging.info("DIAG: Successfully opened file for reading")
                        lines = f.readlines()
                        logging.info(f"DIAG: Read {len(lines)} lines from file")
                        
                        start_line_idx = max(0, user_line_no - 11)
                        end_line_idx = min(len(lines), user_line_no + 10)
                        logging.info(f"DIAG: Will extract lines {start_line_idx+1} to {end_line_idx}")
                        
                        # Determine the maximum line number width for padding
                        max_line_num_width = len(str(end_line_idx))
                        
                        # Format the code snippet
                        context_list = []
                        for idx in range(start_line_idx, end_line_idx):
                            current_line_num = idx + 1
                            line_content = lines[idx].rstrip() # Keep leading whitespace, remove trailing
                            
                            # Format line number with padding
                            padded_line_num = f"{current_line_num:<{max_line_num_width}}" # Pad on right now

                            # Determine prefix and always include number and colon
                            if current_line_num == user_line_no:
                                prefix = " ! "  # Error line prefix (space-exclam-space)
                            else:
                                prefix = "   "  # Normal line prefix (3 spaces)
                            # Include prefix, padded number, colon, space, and content
                            context_list.append(f"{prefix}{padded_line_num}: {line_content}")
                                
                        # Join with '\n' for a proper newline character
                        surrounding_code = "\n".join(context_list)
                        logging.info(f"DIAG: Generated {len(context_list)} lines of context")
                except Exception as e:
                    logging.warning(f"Failed to extract surrounding code context: {e}")
                    logging.info(f"DIAG: Exception while reading file: {str(e)}")
            elif not os.path.exists(abs_file_path):
                logging.info(f"DIAG: Absolute file '{abs_file_path}' (derived from '{user_file_path}') does not exist.")
        
        # Return the enhanced error message and the surrounding code context separately
        enhanced_error = f"{error_msg}{error_line_info}"
        
        # Return as dict to easily separate context
        result = {
            "enhanced_error_message": enhanced_error,
            "code_context_snippet": surrounding_code
        }
        
        logging.info(f"DIAG: Returning result with context snippet: {'present' if surrounding_code else 'None'}")
        if surrounding_code:
            logging.info(f"DIAG: Context snippet length: {len(surrounding_code)}")
            
        return result
        
    except Exception as e:
        logging.error(f"Unexpected error during extract_error_context: {e}")
        logging.info(f"DIAG: Caught exception in extract_error_context: {str(e)}")
        import traceback
        logging.info(f"DIAG: Exception traceback: {traceback.format_exc()}")
        # Fallback to original error message if anything goes wrong
        return {
            "enhanced_error_message": error_msg,
            "code_context_snippet": None
        } 

# --- New Function ---
def create_standard_error_result(
    exception: Exception,
    traceback_str: str,
    error_type_override: Optional[str] = None,
    elapsed_ms: Optional[float] = 0,
    stdout: Optional[str] = "",
    stderr: Optional[str] = "",
    default_error_msg: str = "An unexpected error occurred",
) -> Dict[str, Any]:
    """
    Creates a standardized dictionary representing a failed operation.

    Args:
        exception: The exception object caught.
        traceback_str: The raw traceback string (from traceback.format_exc()).
        error_type_override: Explicitly set the error type (e.g., 'timeout').
        elapsed_ms: Elapsed time before failure (optional).
        stdout: Captured stdout (optional).
        stderr: Captured stderr (optional).
        default_error_msg: Message to use if str(exception) is empty.

    Returns:
        A dictionary with standard failure keys.
    """
    logging.debug(f"create_standard_error_result called for exception type: {type(exception).__name__}")
    
    error_msg_from_exception = str(exception) if str(exception) else default_error_msg
    error_type = error_type_override # Use override if provided

    if not error_type:
        # Specific checks
        if isinstance(exception, OSError) and getattr(exception, 'errno', None) in {_errno.EBADF, _errno.EFBIG, _errno.EPERM, _errno.EACCES, _errno.EROFS}:
            error_type = "filesystem_access_error"
            # Standardized evaluator message for any read/write attempt
            error_msg_from_exception = "Error: Your code cannot read/write files."
        elif isinstance(exception, SolverFileNotFoundError): # Check for our new custom error first
            error_type = "solver_not_found_error"
            # Use only the clean generic message, ignore the verbose exception details
            error_msg_from_exception = SOLVER_NOT_FOUND_GENERIC_MSG 
        elif isinstance(exception, FuturesTimeoutError): # Check for concurrent.futures timeout
            error_type = "timeout"
            error_msg_from_exception = "Execution timed out" # Standardize timeout message
        elif type(exception).__name__ == "TimeoutError": # Check for custom TimeoutError (if any)
            error_type = "timeout"
            error_msg_from_exception = "Execution timed out"
        elif type(exception).__name__ == "ValidationException": # Check for custom ValidationException
            error_type = "validation_error"
        elif isinstance(exception, MemoryError): # Check for memory errors
            error_type = "memory_error"
            error_msg_from_exception = "Execution failed due to insufficient memory"
        # Add more specific checks here if needed (e.g., ImportError, FileNotFoundError)
        else:
            error_type = type(exception).__name__ # Default to exception class name

    logging.debug(f"Determined error_type: {error_type}")

    # Get enhanced message and code context
    enhanced_error_msg = error_msg_from_exception
    code_context = None
    traceback_for_dict = traceback_str # Default to raw traceback

    # For MemoryError, check if we have a custom user traceback
    if error_type == "memory_error" and hasattr(exception, 'user_traceback'):
        # Use the captured user traceback instead of the system traceback
        user_traceback_str = ''.join(exception.user_traceback)
        traceback_for_dict = user_traceback_str
        logging.debug(f"Using custom user traceback for MemoryError: {len(user_traceback_str)} chars")

    # Don't extract context for timeouts, OOM kills, or solver_not_found_error, as traceback/context is often irrelevant or misleading
    # However, do extract context for memory_error to show user where the memory limit was exceeded
    if error_type not in ["timeout", "oom_kill", "solver_not_found_error"]:
        try:
            # Use the appropriate traceback for context extraction
            context_traceback = traceback_for_dict if error_type == "memory_error" and hasattr(exception, 'user_traceback') else traceback_str
            context_info = extract_error_context(context_traceback, error_msg_from_exception)
            enhanced_error_msg = context_info.get("enhanced_error_message", error_msg_from_exception)
            code_context = context_info.get("code_context_snippet")
            # For filesystem access errors, preserve the standardized message exactly
            if error_type == "filesystem_access_error":
                enhanced_error_msg = error_msg_from_exception
            # Optionally use cleaned traceback if desired:
            # traceback_for_dict = clean_traceback(traceback_str)
            logging.debug(f"Context extracted. Enhanced msg: '{enhanced_error_msg[:100]}...', Context present: {bool(code_context)}")
        except Exception as context_exc:
            logging.error(f"Failed during context extraction within create_standard_error_result: {context_exc}", exc_info=True)
            # Fallback to original message and raw traceback if context extraction fails

    # Construct the standard dictionary
    result_dict = {
        "success": False,
        "error": enhanced_error_msg,
        "error_type": error_type,
        "traceback": traceback_for_dict,
        "code_context": code_context,
        "elapsed_ms": elapsed_ms if elapsed_ms is not None else 0,
        "stdout": stdout if stdout is not None else "",
        "stderr": stderr if stderr is not None else "",
    }
    logging.debug(f"Returning standard error result: { {k: f'{type(v).__name__} ({len(v)} chars)' if isinstance(v, str) else type(v).__name__ for k, v in result_dict.items()} }")
    return result_dict 