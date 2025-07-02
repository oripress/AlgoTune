from typing import List, Optional, Tuple, Any, Dict
from AlgoTuner.utils.snippet_utils import compute_centered_snippet_bounds, compute_snippet_bounds
from threading import Lock
import logging
import os
import numpy as np
import traceback
import re
from AlgoTuner.interfaces.commands.types import COMMAND_FORMATS


class MessageWriter:
    """
    Formats editor commands, outputs, code snippets, etc.
    Thread-safe singleton implementation for consistent message formatting.
    """

    _instance = None
    _lock = Lock()

    ERROR_CATEGORIES = {
        "file": "File Operation",
        "command": "Command Execution",
        "parsing": "Command Parsing",
        "validation": "Input Validation",
        "system": "System Operation",
        "api": "API Communication",
        "budget": "Budget Management",
        "solver": "Solver Operation",
    }

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MessageWriter, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        with self._lock:
            if not self._initialized:
                self._initialized = True

    @staticmethod
    def _format_snippet(
        file_path: str,
        lines: List[str],
        snippet_start: int,
        snippet_end: int,
        highlight_range: Optional[Tuple[int, int]],
        show_header: bool = False,
        header_text: Optional[str] = None,
        is_new_file: bool = False,
    ) -> str:
        """
        Given a precomputed snippet range [snippet_start..snippet_end],
        format lines with markers. highlight_range is a (start,end) for lines
        that should be marked '>'.
        If is_new_file is True, all lines will be marked with '>' since they're all new.
        """
        # Debug logging
        logging.info(f"_format_snippet called with:")
        logging.info(f"  file_path: {file_path}")
        logging.info(f"  num_lines: {len(lines)}")
        logging.info(f"  snippet_range: {snippet_start}-{snippet_end}")
        logging.info(f"  highlight_range: {highlight_range}")
        logging.info(f"  show_header: {show_header}")
        logging.info(f"  header_text: {header_text}")
        logging.info(f"  is_new_file: {is_new_file}")

        total_lines = len(lines)
        width = len(str(total_lines))

        out = []
        if show_header:
            if header_text:
                out.append(header_text)
            else:
                filename = MessageWriter._get_filename(file_path)
                out.append(
                    f"Contents of {filename} (lines {snippet_start}-{snippet_end} out of {total_lines})"
                )
            out.append("(| = existing code, > = modified code)\n")

        if snippet_start > 1:
            out.append("...")

        hr_start = None
        hr_end = None
        if highlight_range:
            hr_start, hr_end = highlight_range

        for i in range(snippet_start, snippet_end + 1):
            marker = ">"
            if not is_new_file:
                marker = "|"
                if hr_start and hr_end and (hr_start <= i <= hr_end):
                    marker = ">"
            line_text = lines[i - 1].rstrip("\r\n")
            out.append(f"{marker} {str(i).zfill(width)}: {line_text}")

        if snippet_end < total_lines:
            out.append("...")

        result = "\n".join(out)
        logging.info(f"_format_snippet output:\n{result}")
        return result

    @staticmethod
    def _compute_center_for_proposed(
        error_line: Optional[int],
        changed_range: Optional[Tuple[int, int]],
        total_lines: int,
    ) -> int:
        """
        Decide which line to center around for the proposed code snippet:
          - If error_line is present, we center around that.
          - Else if changed_range is present, center around the midpoint of that range.
          - Else fallback to line=1
        """
        if error_line is not None and 1 <= error_line <= total_lines:
            return error_line
        if changed_range:
            cstart, cend = changed_range
            midpoint = (cstart + cend) // 2
            if midpoint < 1:
                return 1
            if midpoint > total_lines:
                return total_lines
            return midpoint
        return 1

    @staticmethod
    def _compute_center_for_current(
        changed_range: Optional[Tuple[int, int]], total_lines: int
    ) -> int:
        """
        Per your request: "the current code snippet should be centered around the start edit line."
        So if changed_range=(start,end), we center around start.
        If there's no changed_range, we center at line=1
        """
        if changed_range:
            start_line = changed_range[0]
            if start_line < 1:
                return 1
            if start_line > total_lines:
                return total_lines
            return start_line
        return 1

    ########################################################################
    # Public API
    ########################################################################

    @staticmethod
    def _get_filename(file_path: str) -> str:
        """
        Extract just the filename from a file path.
        """
        return file_path.split("/")[-1]

    @staticmethod
    def format_edit_result(raw_result: dict) -> str:
        """
        Format the result of an edit operation with sections in this order:
        1. Budget status (handled by caller)
        2. Error message (if failed)
        3. Proposed changes (if failed)
        4. Current code (always show)
        5. Performance score (if available)
        """
        # Deferred import to avoid circular dependencies
        from AlgoTuner.interfaces.commands.types import EditStatus, SnapshotStatus, EvalStatus
        lines_out = []
        file_path = raw_result.get("file_path", "unknown file")
        filename = MessageWriter._get_filename(file_path)
        success = raw_result.get("success", False)
        edit_status = raw_result.get(
            "edit_status",
            EditStatus.FAILURE.value if not success else EditStatus.SUCCESS.value,
        )

        # Debug logging
        logging.debug(f"format_edit_result called with raw_result: {raw_result}")

        if edit_status == EditStatus.SUCCESS.value:
            lines_out.append(f"Edit successful for {filename}.")
            
            compilation_status = raw_result.get("compilation_status")
            if compilation_status:
                if compilation_status.get("success"):
                    command = compilation_status.get("command", "")
                    if "pythran" in command:
                        lines_out.append("Pythran compilation successful.")
                    elif "pip install" in command:
                        lines_out.append("Cython compilation successful.")
                else:
                    # Compilation failed
                    command = compilation_status.get("command", "")
                    error = compilation_status.get("error", "Compilation failed")
                    if "pythran" in command:
                        lines_out.append(f"Pythran compilation failed: {str(error) if error is not None else 'Unknown error'}")
                    elif "pip install" in command:
                        lines_out.append(f"Cython compilation failed: {str(error) if error is not None else 'Unknown error'}")
                    
                    # If file was reverted due to compilation failure
                    if raw_result.get("reverted_due_to_compilation"):
                        lines_out.append("File reverted to previous state due to compilation failure.")
            # Insert code to include cleaned stderr for Pythran compilation failures
            compilation_status = raw_result.get("compilation_status", {})
            command = compilation_status.get("command", "")
            if "pythran" in command:
                stderr = compilation_status.get("stderr", "")
                if stderr:
                    lines_out.append("Compilation stderr:")
                    # Use the existing clean_build_output function to clean file paths
                    from AlgoTuner.utils.trace_cleaner import clean_build_output
                    cleaned_stderr = clean_build_output(stderr)
                    for line in cleaned_stderr.strip().splitlines():
                        lines_out.append(line)
        else:
            # 2. Error message
            err_msg = raw_result.get("error", "Unknown error")
            # Replace the exact string '<unknown>' with an empty string
            err_msg = err_msg.replace("<unknown>", "")
            lines_out.append(
                f"Edit failed (and thus not applied) for {filename}: {err_msg}"
            )

            # Debug logging for proposed changes section
            logging.info(
                f"Processing proposed changes. temp_file_content: {raw_result.get('temp_file_content')}"
            )
            logging.info(f"error_line: {raw_result.get('temp_file_error_line')}")
            logging.info(f"changed_range: {raw_result.get('changed_range')}")

            # 3. Proposed changes
            temp_content = raw_result.get("temp_file_content") # Get the value, could be None
            proposed = (temp_content if temp_content is not None else "").strip() # Ensure it's a string before stripping
            if proposed:
                logging.info(f"Found proposed content of length {len(proposed)}")
                # Center around error_line if present, otherwise around the middle of changed_range
                error_line = raw_result.get("temp_file_error_line")
                changed_range = raw_result.get("changed_range")
                proposed_lines = proposed.splitlines()
                total = len(proposed_lines)

                # Compute center line and snippet bounds
                center_line = MessageWriter._compute_center_for_proposed(
                    error_line, changed_range, total
                )
                snippet_start, snippet_end = compute_centered_snippet_bounds(
                    center_line, total, 50
                )

                # Compute highlight range for the proposed changes based ONLY on changed_range
                highlight_range = None
                if changed_range:
                    cstart, cend = changed_range
                    # Handle None values safely (though editor should provide valid ints)
                    if cstart is None:
                        cstart = 1 
                    if cend is None:
                        cend = cstart # Default end to start if None
                    # Ensure start <= end (should already be true from parser/editor)
                    if cend < cstart:
                         cend = cstart 
                    # Clamp to valid line numbers for the proposed content
                    if cstart < 1:
                        cstart = 1
                    if cend > total:
                        cend = total
                    # Assign the corrected range based ONLY on changed_range
                    highlight_range = (cstart, cend)

                # Check if this would be a new file
                is_new_file = not raw_result.get("old_content", "").strip()

                # Format the snippet with proper bounds
                snippet_text = MessageWriter._format_snippet(
                    file_path=file_path,
                    lines=proposed_lines,
                    snippet_start=max(1, min(snippet_start, total)),
                    snippet_end=max(1, min(snippet_end, total)),
                    highlight_range=highlight_range,
                    show_header=True,
                    header_text=f"\nProposed changes - This is what you tried to apply (lines {max(1, min(snippet_start, total))}-{max(1, min(snippet_end, total))} out of {total}):",
                    is_new_file=is_new_file,
                )
                lines_out.append(snippet_text)
            else:
                pass
        logging.info("No proposed content found")

        lines_out.append("")

        # 4. Current code (show for both success and failure)
        current_content_raw = raw_result.get(
            "old_content" if not success else "formatted" # Get raw value, could be None
        )
        # Ensure it's a string before stripping
        current_content = (current_content_raw if current_content_raw is not None else "").strip()

        if current_content:
            lines_current = current_content.splitlines()
            total_current = len(lines_current)

            if total_current > 0:  # Only show snippet if we have content
                # Center around changed_range[0] if we have one
                c_range = raw_result.get("changed_range")
                center_line = MessageWriter._compute_center_for_current(
                    c_range, total_current
                )
                snippet_start, snippet_end = compute_centered_snippet_bounds(
                    center_line, total_current, 50
                )

                # We highlight the changed_range only if success
                highlight_range = None
                if success and c_range:
                    cs, ce = c_range
                    if cs and ce:  # Ensure both values are not None
                        # clamp if out of range
                        if cs < 1:
                            cs = 1
                        if ce > total_current:
                            ce = total_current
                        highlight_range = (cs, ce)

                # Check if this is a new file by looking at old_content
                is_new_file = success and not raw_result.get("old_content", "").strip()

                # Add a clear header for current content - only use the distinguishing header for failed edits
                header_text = None
                if not success:
                    header_text = f"CURRENT FILE - This is what's actually in the file (lines {snippet_start}-{snippet_end} out of {total_current}):"

                snippet_text = MessageWriter._format_snippet(
                    file_path=file_path,
                    lines=lines_current,
                    snippet_start=snippet_start,
                    snippet_end=snippet_end,
                    highlight_range=highlight_range,
                    show_header=True,
                    header_text=header_text,
                    is_new_file=is_new_file,
                )
                lines_out.append(snippet_text)
            else:
                lines_out.append("Contents of current file:")
                lines_out.append(f"File {filename} is empty.")
        else:
            lines_out.append("Contents of current file:")
            lines_out.append(f"File {filename} is empty.")
            
        # 5. Add performance statistics if available
        perf_score = raw_result.get("performance_score")
        if perf_score:
            lines_out.append(f"\nPerformance Score: {perf_score:.4f}")
            lines_out.append("(Lower score is better - ratio of your solution's runtime to oracle solution)")
            
            # Add timing information if available
            your_time = raw_result.get("your_average_ms")
            oracle_time = raw_result.get("oracle_average_ms")
            if your_time is not None and oracle_time is not None:
                lines_out.append(f"Your average time: {your_time:.4f}ms")
                lines_out.append(f"Oracle average time: {oracle_time:.4f}ms")

        return "\n".join(lines_out)

    ########################################################################
    # The rest of the existing methods remain the same
    ########################################################################

    @staticmethod
    def _format_error_message(
        error_msg: str, context: Optional[str] = None, include_traceback: bool = False, category: str = None
    ) -> str:
        """Format an error message with optional context and traceback."""
        parts = []

        # Clean file paths from error message
        from AlgoTuner.utils.trace_cleaner import clean_build_output
        cleaned_error_msg = clean_build_output(error_msg) if error_msg else error_msg

        # Check if the error message already starts with "Error:"
        if cleaned_error_msg.strip().startswith("Error:"):
            # If it does, don't add another error prefix
            parts.append(cleaned_error_msg)
        else:
            # Format the error header
            if context:
                parts.append(f"Error {context}:")
            else:
                parts.append("Error:")

            # Add the main error message
            parts.append(cleaned_error_msg)

        # If traceback is included in the error message and we want to show it
        if include_traceback and "Traceback" in error_msg:
            # The error message already contains the traceback, no need to modify
            pass
        elif include_traceback:
            # Try to extract traceback if it exists in a different format
            tb_start = error_msg.find("\n")
            if tb_start != -1:
                error_text = error_msg[:tb_start]
                traceback_text = error_msg[tb_start:].strip()
                if traceback_text:
                    parts = [parts[0], error_text, "\nTraceback:", traceback_text]

        return "\n".join(parts)

    @staticmethod
    def format_error(
        error_msg: str, context: Optional[str] = None, include_traceback: bool = False
    ) -> str:
        """Format a generic error message.

        Note: This method should not be called directly. Instead use format_message_with_budget
        to ensure budget info is included. This method is kept for backward compatibility
        and internal use.
        """
        return MessageWriter._format_error_message(
            error_msg, context=context, include_traceback=include_traceback
        )

    @staticmethod
    def format_file_error(
        error_msg: str, context: Optional[str] = None, include_traceback: bool = False
    ) -> str:
        """Format a file operation error message."""
        return MessageWriter._format_error_message(
            error_msg, context=context, include_traceback=include_traceback
        )

    @staticmethod
    def format_command_error(
        error_msg: str, context: str = None, include_traceback: bool = False
    ) -> str:
        """Format a command execution error message.

        Note: This method should not be called directly. Instead use format_message_with_budget
        to ensure budget info is included. This method is kept for backward compatibility
        and internal use.
        """
        # If the error message already starts with "Error:", don't add context
        if error_msg.strip().startswith("Error:"):
            return error_msg
        else:
            return MessageWriter._format_error_message(
                error_msg, context=context, include_traceback=include_traceback
            )

    @staticmethod
    def format_api_error(
        error_msg: str, context: Optional[str] = None, include_traceback: bool = False
    ) -> str:
        """Format an API communication error message."""
        return MessageWriter._format_error_message(
            error_msg, context=context, include_traceback=include_traceback, category="api"
        )

    @staticmethod
    def format_solver_error(error_dict: Dict[str, Any]) -> str:
        """
        Formats errors specifically originating from the solver execution or validation.
        Uses the internal _format_error_details helper.
        """
        # Use _format_error_details which now includes context and traceback
        error_lines = MessageWriter._format_error_details(error_dict)

        # Prepend header
        header = ""
        
        return header + "\n".join(error_lines)

    @staticmethod
    def format_validation_error(
        error_msg: str, context: Optional[str] = None, include_traceback: bool = False
    ) -> str:
        """Format a validation error message.

        Note: This method should not be called directly. Instead use format_message_with_budget
        to ensure budget info is included. This method is kept for backward compatibility
        and internal use.

        Args:
            error_msg: The error message to format
            context: Optional context for where the error occurred
            include_traceback: Whether to include traceback in output

        Returns:
            Formatted validation error message
        """
        return MessageWriter._format_error_message(
            error_msg, context=context, include_traceback=include_traceback
        )

    @staticmethod
    def format_budget_error(
        error_msg: str, context: Optional[str] = None, include_traceback: bool = False
    ) -> str:
        """Format a budget management error message."""
        return MessageWriter._format_error_message(
            error_msg, context=context, include_traceback=include_traceback
        )

    @staticmethod
    def format_multiple_code_blocks_warning(total_blocks: int) -> str:
        """Format a warning message about multiple code blocks."""
        if total_blocks <= 1:
            return ""
        return f"\nNote: Found {total_blocks} code blocks in the response. Only the first code block containing a valid command was processed. Please send only one command per message."

    @staticmethod
    def format_command_result(
        command: str,
        success: bool,
        result: Optional[str] = None,
        error: Optional[str] = None,
        total_code_blocks: Optional[int] = None,
        edit_status: Optional[str] = None,
        snapshot_status: Optional[str] = None,
        eval_status: Optional[str] = None,
    ) -> str:
        """Format a command result with consistent structure.

        Note: This method should not be called directly. Instead use format_message_with_budget
        to ensure budget info is included. This method is kept for backward compatibility
        and internal use.
        """
        output = []
        if success:
            if result:
                if command == "edit":
                    output.append(f"{command} completed successfully:\n{result}")
                else:
                    output.append(result)
            else:
                if command == "edit":
                    output.append(f"{command} completed successfully.")
                else:
                    output.append(result if result else "")
        else:
            if error:
                # Create a temporary instance to format the error
                writer = MessageWriter()
                output.append(writer.format_error(error, f"executing {command}"))
            else:
                writer = MessageWriter()
                output.append(writer.format_error(f"{command} failed", "execution"))

        # Add status information if present
        if edit_status:
            output.append(f"Edit status: {edit_status}")
        if snapshot_status:
            output.append(f"Snapshot status: {snapshot_status}")
        if eval_status:
            output.append(f"Evaluation status: {eval_status}")

        if total_code_blocks is not None:
            warning = MessageWriter.format_multiple_code_blocks_warning(
                total_code_blocks
            )
            if warning:
                output.append(warning)

        return "\n".join(output)

    @staticmethod
    def format_command_output(
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        command: Optional[str] = None,
        max_lines: int = 50,
    ) -> str:
        output = []
        if command:
            output.append(f"Output from {command}:")
            output.append("(| = stdout, ! = stderr)")

        def format_stream(content: str, marker: str, max_lines: int) -> List[str]:
            if not content:
                return []
            lines = content.splitlines()
            if len(lines) > max_lines:
                half = max_lines // 2
                return (
                    [f"{marker} {line}" for line in lines[:half]]
                    + ["..."]
                    + [f"{marker} {line}" for line in lines[-half:]]
                )
            return [f"{marker} {line}" for line in lines]

        if stdout:
            stdout_lines = format_stream(stdout, "|", max_lines)
            if stdout_lines:
                if not command:
                    output.append("Standard Output:")
                output.extend(stdout_lines)

        if stderr:
            stderr_lines = format_stream(stderr, "!", max_lines)
            if stderr_lines:
                output.append("Standard Error:")
                output.extend(stderr_lines)

        return "\n".join(output)

    @staticmethod
    def format_system_message(msg: str, msg_type: str = "info") -> str:
        if msg_type.lower() != "info":
            return f"[{msg_type.upper()}] {msg}"
        return msg

    @staticmethod
    def _clean_traceback(traceback_str: Optional[str]) -> Optional[str]:
        """Extracts the last frame from a traceback string and cleans the file path."""
        if not traceback_str or not traceback_str.strip():
            return None
        
        # Use the existing clean_build_output function to clean all file paths
        from AlgoTuner.utils.trace_cleaner import clean_build_output
        cleaned_tb = clean_build_output(traceback_str)
            
        lines = cleaned_tb.strip().split('\n')
        # Find the last line starting with "  File "
        last_frame_line = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("File "):
                last_frame_line = lines[i].strip()
                break
                
        if not last_frame_line:
            # If no "File" line found, return the last non-empty line as fallback
            return lines[-1] if lines else None

        # The line should already be cleaned by clean_build_output, so return as-is
        return last_frame_line 

    @staticmethod
    def format_evaluation_result_from_raw(evaluation_output: dict) -> str:
        """
        Format the result of an evaluation operation.
        Handles results from both single input and full dataset evaluations.
        For full dataset evaluations, always shows summary stats if possible,
        and includes details of the first actual error if one occurred.
        """
        lines = []
        success = evaluation_output.get("success", False)
        error_type = evaluation_output.get("error_type")
        traceback_str = evaluation_output.get("traceback")
        code_context = evaluation_output.get("code_context")
        cleaned_traceback = MessageWriter._clean_traceback(traceback_str)

        logging.info(f"format_evaluation_result_from_raw: START. Input keys: {list(evaluation_output.keys())}")
        is_full_dataset_eval = evaluation_output.get("evaluation_type") == "dataset"

        if is_full_dataset_eval:
            aggregate_metrics = evaluation_output.get("aggregate_metrics", {})
            num_evaluated = aggregate_metrics.get("num_evaluated")
            if num_evaluated is None:
                num_evaluated = evaluation_output.get("num_evaluated", 0)
                
            logging.info(f"format_evaluation_result_from_raw: is_full_dataset_eval=True. num_evaluated={num_evaluated}. aggregate_metrics keys: {list(aggregate_metrics.keys())}")
            logging.info(f"format_evaluation_result_from_raw: aggregate_metrics truthiness: {bool(aggregate_metrics)}")
                
            if num_evaluated > 0:
                logging.info("format_evaluation_result_from_raw: num_evaluated > 0 branch.")
                if aggregate_metrics:
                    logging.info("format_evaluation_result_from_raw: aggregate_metrics is non-empty branch.")
                    lines.extend(MessageWriter.format_evaluation_summary(aggregate_metrics))
                    
                    # Add invalid solution analysis if present
                    invalid_solution_analysis = evaluation_output.get("invalid_solution_analysis", [])
                    if invalid_solution_analysis:
                        logging.info(f"Adding {len(invalid_solution_analysis)} invalid solution examples")
                        lines.append("")
                        lines.append("")
                        lines.append("Snapshot not saved - invalid solutions present")
                        lines.append("")
                        # Show up to 3 random examples
                        import random
                        examples_to_show = random.sample(invalid_solution_analysis, min(3, len(invalid_solution_analysis)))
                        for i, context in enumerate(examples_to_show, 1):
                            lines.append(f"Invalid Example #{i}:")
                            lines.append("Error in 'is_solution':")
                            # Add the context on the next line
                            lines.append(context)
                            lines.append("")
                else:
                    logging.info("format_evaluation_result_from_raw: aggregate_metrics IS EMPTY branch (fallback).")
                    lines.append("Evaluation Summary:")
                    lines.append(f"  Problems Evaluated: {num_evaluated}")
                    lines.append("  (Detailed metrics unavailable, possibly due to early exit)")
                    lines.append("")
            else:  # num_evaluated is 0
                logging.info("format_evaluation_result_from_raw: num_evaluated == 0 branch.")
                lines.append("Evaluation Summary:")
                if evaluation_output.get("success", False):  # Should not happen if num_eval is 0?
                    lines.append("  No problems evaluated.")
                else:
                    # Special formatting for invalid_solution so the user still sees
                    # the output and runtime just like a successful run.
                    if error_type == "invalid_solution":
                        # Show the solver's raw output and timing first
                        result_value = evaluation_output.get('result')
                        if result_value is None:
                            result_value = 'N/A'
                        lines.append(f"Output: {result_value}")
                        runtime_ms = evaluation_output.get('elapsed_ms', 'N/A')
                        lines.append(f"Runtime: {runtime_ms} ms" if runtime_ms != 'N/A' else "Runtime: N/A")
                        # Then the standard invalid-solution explanation
                        err_msg = evaluation_output.get('error') or 'Solution is invalid.'
                        lines.append(err_msg)
                        # Include code context if available to help user understand what went wrong
                        MessageWriter._append_code_context_to_lines(lines, evaluation_output.get('code_context'))
                    else:
                        # Generic failure formatting (unchanged)
                        err_msg = evaluation_output.get('error') or (
                            "Solver class not found in solver.py" if error_type == 'missing_solver_class' else error_type or 'Unknown Error'
                        )
                        lines.append(f"Evaluation Failed: {err_msg}")
                        if cleaned_traceback:  # Use cleaned traceback
                            lines.append("\nTraceback:")
                            lines.append(f"  {cleaned_traceback}")
                        # Append code context if available
                        MessageWriter._append_code_context_to_lines(lines, evaluation_output.get('code_context'))
        else:  # Single input evaluation
            # --- ADDED DIAGNOSTIC LOG ---
            logging.info("format_evaluation_result_from_raw: is_single_input_eval branch.")
            stdout = evaluation_output.get("stdout")
            logging.info(f"STDOUT DEBUG: stdout length={len(stdout) if stdout else 0}, content='{stdout[:100] if stdout else None}'")
            
            if success:
                result_value = evaluation_output.get('result')
                if result_value is None:
                    result_value = 'N/A'
                lines.append(f"Output: {result_value}")
                
                # Add Stdout right after Output if present
                if stdout and stdout.strip():
                    lines.append(f"Stdout: {stdout.strip()}")
                
                runtime_ms = evaluation_output.get('elapsed_ms', 'N/A')
                lines.append(f"Runtime: {runtime_ms} ms" if runtime_ms != 'N/A' else "Runtime: N/A")
                is_valid = evaluation_output.get('is_valid', None)
                lines.append(f"Output is valid: {'Yes' if is_valid else 'No' if is_valid is not None else 'N/A'}")
                speedup = evaluation_output.get('speedup')
                if speedup is not None:
                    lines.append(f"Speedup: {MessageWriter._format_speedup(speedup)}")
            else:
                # Special formatting for invalid_solution so the user still sees
                # the output and runtime just like a successful run.
                if error_type == "invalid_solution":
                    # Show the solver's raw output and timing first
                    result_value = evaluation_output.get('result')
                    if result_value is None:
                        result_value = 'N/A'
                    lines.append(f"Output: {result_value}")
                    
                    # Add Stdout right after Output if present
                    if stdout and stdout.strip():
                        lines.append(f"Stdout: {stdout.strip()}")
                    
                    runtime_ms = evaluation_output.get('elapsed_ms', 'N/A')
                    lines.append(f"Runtime: {runtime_ms} ms" if runtime_ms != 'N/A' else "Runtime: N/A")
                    # Then the standard invalid-solution explanation
                    err_msg = evaluation_output.get('error') or 'Solution is invalid.'
                    # Removed "Output is not valid" line as requested
                    lines.append(err_msg)
                    # Include code context if available to help user understand what went wrong
                    MessageWriter._append_code_context_to_lines(lines, evaluation_output.get('code_context'))
                else:
                    # Generic failure formatting (unchanged)
                    err_msg = evaluation_output.get('error') or (
                        "Solver class not found in solver.py" if error_type == 'missing_solver_class' else error_type or 'Unknown Error'
                    )
                    lines.append(f"Evaluation Failed: {err_msg}")
                    if cleaned_traceback:  # Use cleaned traceback
                        lines.append("\nTraceback:")
                        lines.append(f"  {cleaned_traceback}")
                    # Append code context if available
                    MessageWriter._append_code_context_to_lines(lines, evaluation_output.get('code_context'))

            # Append Stdout if present and not already included in the message
            if stdout and stdout.strip():
                # Check if stdout is already included in the existing message
                current_message = "\n".join(lines)
                if "Stdout:" not in current_message:
                    lines.append(f"Stdout: {stdout.strip()}")

        # --- ADDED DIAGNOSTIC LOG ---
        logging.info(f"format_evaluation_result_from_raw: END. Returning {len(lines)} lines.")
        return "\n".join(lines)

    @staticmethod
    def format_evaluation_summary(aggregate_metrics: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        
        num_evaluated = aggregate_metrics.get("num_evaluated", 0)
        num_valid = aggregate_metrics.get("num_valid", 0)
        num_invalid = aggregate_metrics.get("num_invalid", 0)
        num_timeouts = aggregate_metrics.get("num_timeouts", 0)
        num_errors = aggregate_metrics.get("num_errors", 0)
        
        mean_speedup = aggregate_metrics.get("mean_speedup")
        
        # COMPREHENSIVE MESSAGE WRITER TIMING DEBUG: Log what timing values we're receiving
        timing_debug_fields = ["avg_solver_time_ms", "avg_oracle_time_ms", "avg_solver_time_on_mutual_valid", "avg_oracle_time_on_mutual_valid", "mean_speedup"]
        timing_debug = {field: aggregate_metrics.get(field) for field in timing_debug_fields}
        logging.info(f"MESSAGE_WRITER_TIMING_DEBUG: Aggregate metrics timing fields: {timing_debug}")
        
        # Average timing values
        avg_solver_time_valid = aggregate_metrics.get("avg_solver_time_on_mutual_valid", aggregate_metrics.get("avg_solver_time_ms"))
        avg_oracle_time_valid = aggregate_metrics.get("avg_oracle_time_on_mutual_valid", aggregate_metrics.get("avg_oracle_time_ms"))
        
        logging.info(f"MESSAGE_WRITER_TIMING_DEBUG: Final avg_solver_time_valid={avg_solver_time_valid}, avg_oracle_time_valid={avg_oracle_time_valid}")

        if num_evaluated > 0:
            # Count all errors as invalid solutions, but keep timeouts separate
            total_invalid = num_invalid + num_errors
            
            # Calculate percentages
            correct_pct = round(num_valid / num_evaluated * 100)
            invalid_pct = round(total_invalid / num_evaluated * 100)
            timeout_pct = round(num_timeouts / num_evaluated * 100)
             
            # Ensure percentages add up to 100%
            total_pct = correct_pct + invalid_pct + timeout_pct
            if total_pct != 100:
                # Adjust invalid percentage to make total 100%
                invalid_pct += (100 - total_pct)
        else:
            correct_pct = invalid_pct = timeout_pct = 0
        
        def format_speedup_value(val):
            if val is None: return "N/A"
            if val == float('inf'): return "Infinite"
            return f"{val:.2f}x"

        def format_time_value(val_ms):
            if val_ms is None: return "N/A"
            try:
                return f"{float(val_ms):.2f} ms"
            except (ValueError, TypeError):
                logging.warning(f"Could not format time value '{val_ms}' as float, returning as string.")
                return f"{val_ms} ms"

        # --- Summary header: mean speedup ---
        lines.append(f"Speedup: {format_speedup_value(mean_speedup)}")
        lines.append("  (Speedup = Baseline Time / Your Time; Higher is better)")
        # Stats block: only percentages
        lines.append("")
        if num_evaluated > 0:
            lines.append(f"  Valid Solutions: {correct_pct}%")
            lines.append(f"  Invalid Solutions: {invalid_pct}%")
            lines.append(f"  Timeouts: {timeout_pct}%")
        else:
            lines.append("  No problems were evaluated.")

        lines.append("") # Ensure blank line at the end
        return lines

    @staticmethod
    def format_single_error_summary(error_details: Dict[str, Any]) -> List[str]:
        """
        Format a single problem's error details into a list of formatted strings.
        Used to provide context on the first error in a dataset evaluation.

        Args:
            error_details: Dictionary containing the error information for a single problem.
                           Expected keys: problem_id, error_type, error, traceback, code_context.

        Returns:
            List of formatted strings representing the error details.
        """
        logging.info(f"format_single_error_summary called with error_type={error_details.get('error_type', 'None')}")
        lines = []
        
        # Get error type (but don't display it) - used for conditional behavior
        error_type = error_details.get("error_type", "unknown_error")
        
        if error_type == "invalid_solution":
            # For invalid solutions, focus on showing the code context
            
            # Check for is_solution context in different places
            # First check if we have the dedicated is_solution context field
            if "is_solution_context" in error_details:
                is_solution_context = error_details.get("is_solution_context")
                logging.info(f"Found is_solution_context in error_details (length: {len(is_solution_context)})")
                # Just show the context directly
                lines.append(is_solution_context)
                return lines
                
            # If we have code_context that contains is_solution, use that
            elif error_details.get("code_context") and "is_solution" in error_details.get("code_context", ""):
                is_solution_context = error_details.get("code_context")
                lines.append(is_solution_context)
                return lines
                
            # No specific context is available
            else:
                lines.append("# No detailed context available from task.is_solution")
                lines.append("# To see why your solution was rejected, examine the is_solution method")
                
                # Try to extract context from traceback if it mentions is_solution
                traceback_str = error_details.get("traceback", "")
                if traceback_str and "is_solution" in traceback_str:
                    # Extract a simplified version of the traceback with just is_solution lines
                    is_solution_tb_lines = [line for line in traceback_str.split('\n') if "is_solution" in line]
                    if is_solution_tb_lines:
                        for tb_line in is_solution_tb_lines[:3]:  # Show max 3 lines
                            lines.append(f"# {tb_line.strip()}")
                return lines
                
        elif error_type == "timeout":
            # Just show that execution timed out
            lines.append("Execution timed out.")
            
        else:
            # Generic handling for other error types
            lines.append(f"Error: {error_details.get('error', 'Unknown error')}")
            
            # Add cleaned traceback if available
            traceback_str = error_details.get("traceback", "")
            if traceback_str:
                cleaned_tb = MessageWriter._clean_traceback(traceback_str)
                if cleaned_tb:
                    lines.append(f"\n{cleaned_tb}")
        
        return lines

    @staticmethod
    def format_message_to_llm(message: str) -> str:
        return message

    @staticmethod
    def format_message_from_llm(message: str) -> str:
        return message

    @staticmethod
    def format_file_view(file_name: str, content: str, show_header: bool = True) -> str:
        """Format a file view with optional header."""
        filename = MessageWriter._get_filename(file_name)
        if show_header:
            lines = content.splitlines()
            total_lines = len(lines)
            return f"\nContents of {filename} (lines 1-{total_lines} out of {total_lines})\n\n{content}"
        return content

    @staticmethod
    def format_file_view_from_raw(raw: dict) -> str:
        """Format a raw file view dictionary into a readable string."""
        # Get filename from file_path if available, otherwise use filename field
        file_path = raw.get("file_path", raw.get("filename", "unknown"))
        filename = MessageWriter._get_filename(file_path)

        lines = raw.get("lines", [])
        total = len(lines)  # Get total number of lines
        changed_range = raw.get("changed_range")
        snippet_start = raw.get("snippet_start", 1)
        snippet_end = raw.get("snippet_end", total)
        cstart = raw.get("center_start")
        cend = raw.get("center_end")

        if cstart is not None and cend is not None:
            snippet_start, snippet_end = compute_snippet_bounds(cstart, cend, total, 50)
        else:
            # clamp if snippet_end - snippet_start + 1 is bigger than 50
            length = snippet_end - snippet_start + 1
            if length > 50:
                snippet_end = snippet_start + 50 - 1

        out = []
        out.append(
            f"Contents of {filename} (lines {snippet_start}-{snippet_end} out of {total})"
        )
        out.append("(| = existing code, > = modified code)\n")

        if snippet_start > 1:
            out.append("...")

        width = len(str(total))
        for i in range(snippet_start, snippet_end + 1):
            marker = "|"
            if changed_range and (changed_range[0] <= i <= changed_range[1]):
                marker = ">"
            # Ensure we preserve the actual line content without mangling newlines
            line_text = lines[i - 1]
            if isinstance(line_text, str):
                # Convert literal \n into actual newlines, then strip trailing newlines
                line_text = (
                    line_text.encode("utf-8").decode("unicode_escape").rstrip("\n\r")
                )
            out.append(f"{marker} {str(i).zfill(width)}: {line_text}")

        if snippet_end < total:
            out.append("...")

        return "\n".join(out)

    @staticmethod
    def format_task_status(status: str, details: Optional[str] = None) -> str:
        if details:
            return f"Task {status}: {details}"
        return f"Task {status}"

    @staticmethod
    def format_model_response(
        message: str,
        model_name: Optional[str] = None,
        tokens: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> str:
        lines = []
        if model_name:
            lines.append(f"Model: {model_name}")
        if tokens is not None:
            lines.append(f"Tokens: {tokens}")
        if cost is not None:
            lines.append(f"Cost: ${cost:.4f}")
        if lines:  # Only add a newline before message if we have metadata
            lines.append("")
        lines.append(message)
        return "\n".join(lines)

    @staticmethod
    def format_profile_result(
        profile_result_dict: Dict[str, Any]
    ) -> str:
        """Format profiling results, handling both success and failure cases.

        Args:
            profile_result_dict: Dictionary containing profiling results or error details.
                                   Expected keys on success: 'profile_output', 'focus_lines' (optional).
                                   Expected keys on failure: 'success' (False), 'error', 'traceback', 'code_context'.

        Returns:
            Formatted string with profiling results or error details.
        """
        
        # --- FIX: Check for success/failure --- 
        success = profile_result_dict.get("success", False)
        
        if not success:
            # Handle failure case: Format error, context, traceback
            error_lines = MessageWriter._format_error_details(profile_result_dict)
            # Prepend a header indicating profiling failure
            return "Profiling failed:\n" + "\n".join(error_lines)
        
        # --- Handle success case (original logic adapted) --- 
        profile_output = profile_result_dict.get("profile_output", "")
        focus_lines = profile_result_dict.get("focus_lines")
        
        # Format the output without showing the solution
        result_str = ""

        # Add profiling header
        if focus_lines:
            result_str += f"Profiling results (focusing on lines {', '.join(map(str, focus_lines))}):\n"
        else:
            result_str += "Profiling results:\n"

        # Add the profile output
        result_str += profile_output

        return result_str
        # --- END FIX --- 

    @staticmethod
    def format_command_parse_error(
        template: Optional[str] = None,
        command: Optional[str] = None,
        valid_commands: Optional[List[str]] = None,
    ) -> str:
        """Format command parsing errors with detailed explanations."""
        error_lines = ["Error: Command parsing failed"]

        if template:
            # Special handling for specific error types
            if "Warning:" in template:
                # Handle our new warning about text after commands
                error_lines = ["Error: Invalid command format"]
                clean_message = template.replace("Warning: ", "")
                # Check if this is about multiple commands or text after command
                if "Multiple commands" in clean_message:
                    error_lines.append(clean_message)
                    error_lines.append("\nPlease follow this structure:")
                    error_lines.append("1. You can have explanatory text before the command")
                    error_lines.append("2. Include only ONE command in a code block (```)")
                    error_lines.append("3. Do not put any text or additional commands after the command")
                    
                    # Add examples for the correct way to use commands
                    error_lines.append("\nCorrect examples:")
                    
                    error_lines.append("\nExample 1 - Single command only:")
                    error_lines.append("```")
                    error_lines.append("view_file solver.py")
                    error_lines.append("```")
                    
                    error_lines.append("\nExample 2 - Thoughts followed by a command:")
                    error_lines.append("I want to modify the solver to print the eigenvalues.")
                    error_lines.append("```")
                    error_lines.append("edit")
                    error_lines.append("file: solver.py")
                    error_lines.append("lines: 5-10")
                    error_lines.append("---")
                    error_lines.append("def solve(matrix):")
                    error_lines.append("    eigenvalues, eigenvectors = np.linalg.eig(matrix)")
                    error_lines.append("    print('Eigenvalues:', eigenvalues)")
                    error_lines.append("    return eigenvalues, eigenvectors")
                    error_lines.append("---")
                    error_lines.append("```")
                else:
                    error_lines.append(clean_message)
                    error_lines.append("\nPlease ensure your command follows the correct format:")
                    error_lines.append("1. You can have explanatory text before the command")
                    error_lines.append("2. The command should be in a code block (```)")
                    error_lines.append("3. There should be no text after the command")
            elif "triple backticks" in template:
                # Handle the specific error for improperly formatted code blocks
                error_lines.append(template)
            elif "unknown command" in template.lower():
                # Handle unknown command errors
                error_lines.append(template)
            elif "edit" in template.lower() and "format" in template.lower():
                # Handle edit format errors
                error_lines.append("Invalid edit command format:")
                error_lines.append(template)
            elif "end line" in template.lower() and "start line" in template.lower():
                # Extract the actual line numbers if present
                import re

                start_match = re.search(r"start line \((\d+)\)", template.lower())
                end_match = re.search(r"end line \((\d+)\)", template.lower())
                start_line = start_match.group(1) if start_match else None
                end_line = end_match.group(1) if end_match else None

                error_lines.append("Invalid line range in edit command:")
                if start_line and end_line:
                    error_lines.append(
                        f"- You specified end line ({end_line}) which is less than start line ({start_line})"
                    )
                    error_lines.append(
                        "- End line must be greater than or equal to start line"
                    )
                elif "prepend" in template.lower():
                    error_lines.append(
                        "- For prepending content (adding at the start of file), both start and end lines must be 0"
                    )
                    error_lines.append(
                        "- You specified a non-zero line number for prepend operation"
                    )
                else:
                    error_lines.append(
                        "- End line must be greater than or equal to start line"
                    )
                    error_lines.append(
                        "- For prepend operations, both start_line and end_line must be 0"
                    )

                error_lines.append("\nCorrect formats:")
                error_lines.append("1. To insert/replace content:")
                error_lines.append("edit: file.py")
                error_lines.append("lines: 1-5")
                error_lines.append("---")
                error_lines.append("new content")
                error_lines.append("---")
                error_lines.append("\n2. To prepend content:")
                error_lines.append("edit: file.py")
                error_lines.append("lines: 0-0")
                error_lines.append("---")
                error_lines.append("new content")
                error_lines.append("---")
            elif "no such file" in template.lower():
                error_lines.append("Note: A new file will be created")
                error_lines.append("- Use lines: 0-0 to start adding content")
                error_lines.append(
                    "- The file will be created when you make your first edit"
                )
                error_lines.append("\nExample for creating a new file:")
                error_lines.append("edit: new_file.py")
                error_lines.append("lines: 0-0")
                error_lines.append("---")
                error_lines.append("def example():")
                error_lines.append("    pass")
                error_lines.append("---")
            elif "line number" in template.lower():
                # Extract the actual line number if present
                import re

                line_match = re.search(r"got (\-?\d+)", template.lower())
                line_num = line_match.group(1) if line_match else None

                error_lines.append("Invalid line number in edit command:")
                if line_num:
                    error_lines.append(f"- You specified line number: {line_num}")
                    if int(line_num) < 0:
                        error_lines.append("- Line numbers cannot be negative")
                    error_lines.append("- Line numbers must be non-negative integers")
                else:
                    error_lines.append("- Line numbers must be non-negative integers")
                error_lines.append("- For new files, use lines: 0-0")
                error_lines.append(
                    "- For existing files, start line must be within file bounds"
                )
            else:
                error_lines.append(template)
        else:
            error_lines.append("Invalid command format")

        # Append example usage for the command, if available
        cmd_key = None
        if command:
            cmd_key = command.strip().split()[0]
        if cmd_key and cmd_key in COMMAND_FORMATS:
            example = COMMAND_FORMATS[cmd_key].example
            if example:
                error_lines.append("")
                error_lines.append("Example usage:")
                error_lines.append(example)

        if valid_commands:
            error_lines.append("\nValid commands and their formats:")
            for cmd in valid_commands:
                if cmd == "edit":
                    error_lines.append("1. edit - Modify file content:")
                    error_lines.append("   a) To insert/replace content:")
                    error_lines.append("      edit: file.py")
                    error_lines.append("      lines: start-end")
                    error_lines.append("      ---")
                    error_lines.append("      new content")
                    error_lines.append("      ---")
                elif cmd == "view_file":
                    error_lines.append("2. view_file - View file content:")
                    error_lines.append("   view_file filename [start_line]")
                    error_lines.append("   Example: view_file solver.py")
                    error_lines.append(
                        "   Example with start line: view_file solver.py 10"
                    )
                else:
                    error_lines.append(f"- {cmd}")

        # No longer echo back the command to avoid leaking sensitive information

        return "\n".join(error_lines)

    @staticmethod
    def format_command_validation_error(error_msg: str, command: str) -> str:
        """Format a command validation error with detailed explanation."""
        error_lines = ["Command validation failed:"]

        # Handle specific validation errors
        if "line" in error_msg.lower():
            if "range" in error_msg.lower():
                # Extract line numbers if present
                import re

                start_match = re.search(r"start line \((\d+)\)", error_msg.lower())
                end_match = re.search(r"end line \((\d+)\)", error_msg.lower())
                start_line = start_match.group(1) if start_match else None
                end_line = end_match.group(1) if end_match else None

                error_lines.append("Invalid line range:")
                if start_line and end_line:
                    error_lines.append(
                        f"- You specified end line {end_line} which is less than start line {start_line}"
                    )
                    error_lines.append(
                        "- End line must be greater than or equal to start line"
                    )
                    error_lines.append(
                        f"- To edit lines {start_line} to {end_line}, use: lines: {start_line}-{end_line}"
                    )
                else:
                    error_lines.append(
                        "- End line must be greater than or equal to start line"
                    )
                    error_lines.append(
                        "- For prepend operations (inserting at start), use lines: 0-0"
                    )
            elif "number" in error_msg.lower():
                # Extract the problematic line number if present
                import re

                line_match = re.search(r"got (\-?\d+)", error_msg.lower())
                line_num = line_match.group(1) if line_match else None

                error_lines.append("Invalid line number:")
                if line_num:
                    error_lines.append(f"- You specified: {line_num}")
                    if int(line_num) < 0:
                        error_lines.append("- Line numbers cannot be negative")
                    error_lines.append("- Line numbers must be non-negative integers")
                else:
                    error_lines.append("- Line numbers must be non-negative integers")
                error_lines.append("- For new files, use lines: 0-0")
                error_lines.append(
                    "- For existing files, line numbers must be within file bounds"
                )
            else:
                error_lines.append(error_msg)
        elif "file" in error_msg.lower():
            if "permission" in error_msg.lower():
                error_lines.append("File permission error:")
                error_lines.append("- Cannot access the specified file")
                error_lines.append("- Check file permissions and try again")
                error_lines.append("- Make sure you have write access to the directory")
            else:
                error_lines.append(error_msg)
        else:
            error_lines.append(error_msg)

        # Append example usage for the command, if available
        cmd_key = None
        if command:
            cmd_key = command.strip().split()[0]
        if cmd_key and cmd_key in COMMAND_FORMATS:
            example = COMMAND_FORMATS[cmd_key].example
            if example:
                error_lines.append("")
                error_lines.append("Example usage:")
                error_lines.append(example)

        # No longer echo back the command to avoid leaking sensitive information
        return "\n".join(error_lines)

    @staticmethod
    def format_budget_status(
        spend: float,
        remaining: float,
        messages_sent: Optional[int] = None,
        messages_remaining: Optional[int] = None,
    ) -> str:
        """
        Format the current budget status information.

        Args:
            spend: Current spend amount
            remaining: Remaining budget
            messages_sent: Number of messages sent (optional)
            messages_remaining: Number of messages remaining (optional)

        Returns:
            Formatted budget status string
        """
        try:
            # Format spend and remaining with exactly 4 decimal places
            spend_str = f"{spend:.4f}"
            remaining_str = f"{remaining:.4f}"

            # Default to 0 if messages_sent is None
            msg_count = messages_sent if messages_sent is not None else 0

            return f"You have sent {msg_count} messages and have used up ${spend_str}. You have ${remaining_str} remaining."

        except Exception as e:
            logging.error(f"Error formatting budget status: {str(e)}")
            return "[Budget status unavailable]"

    @staticmethod
    def format_warning(msg: str, context: Optional[str] = None) -> str:
        if context:
            msg = msg.replace("Warning:", "").strip()
            return f"Warning ({context}): {msg}"
        return f"Warning: {msg}"

    @staticmethod
    def format_conversation_state(
        total_messages: int, truncated_count: int, kept_messages: List[str]
    ) -> str:
        """Format a concise summary of conversation state."""
        if truncated_count == 0:
            return f"[{total_messages} msgs]"

        # Find indices of truncated messages by comparing with kept messages
        kept_indices = set()
        for msg in kept_messages:
            try:
                # Extract index from message if it exists
                if "[" in msg and "]" in msg:
                    idx = int(msg.split("[")[1].split("]")[0])
                    kept_indices.add(idx)
            except:
                continue

        # All indices not in kept_indices are truncated
        truncated_indices = sorted(
            [i for i in range(total_messages) if i not in kept_indices]
        )

        # Get the last truncated message if available
        last_truncated = None
        if truncated_indices:
            last_idx = truncated_indices[-1]
            # Try to find the message with this index
            for msg in kept_messages:
                if f"[{last_idx}]" in msg:
                    # Get the first 50 chars of the message content
                    msg_content = msg.split("]", 1)[
                        1
                    ].strip()  # Remove the index prefix
                    if len(msg_content) > 50:
                        last_truncated = f"msg[{last_idx}]: {msg_content[:50]}..."
                    else:
                        last_truncated = f"msg[{last_idx}]: {msg_content}"
                    break

        # Format the output
        base = f"[{total_messages} msgs, truncated: {truncated_indices}]"
        if last_truncated:
            return f"{base}\nLast truncated: {last_truncated}"
        return base

    @staticmethod
    def format_message_with_budget(budget_status: str, message: str) -> str:
        """
        Format a message with budget status, ensuring budget info is always first.
        
        Args:
            budget_status: Current budget status string
            message: The main message content
            
        Returns:
            Formatted message with budget status as the first line
        """
        if not budget_status:
            logging.warning("Budget status is empty when formatting message")
            budget_status = "[Budget status unavailable]"

        # Ensure message is not None
        if message is None:
            message = ""
            
        # Check if the message contains code context that should be preserved
        has_code_context = "Error context:" in message
        code_context_section = None
        
        if has_code_context:
            # Extract the code context section to preserve it
            lines = message.split("\n")
            context_start = None
            for i, line in enumerate(lines):
                if line.strip() == "Error context:":
                    context_start = i
                    break
            
            if context_start is not None:
                # Get the code context section (from "Error context:" to the end)
                code_context_section = "\n".join(lines[context_start:])
                # Remove it from the original message
                message = "\n".join(lines[:context_start]).rstrip()
        
        # Add the budget status to the message
        formatted = f"{budget_status}\n\n{message}"
        
        # Re-add the code context section if it was extracted
        if code_context_section:
            formatted += f"\n\n{code_context_section}"
            
        return formatted

    @staticmethod
    def format_command_message_with_budget(
        budget_status: str, result: "CommandResult"
    ) -> str:
        """
        Format a command result with budget status.

        Args:
            budget_status: Current budget status string
            result: CommandResult object or Dict containing command execution details

        Returns:
            Formatted command result with budget status
        """
        if not budget_status:
            logging.warning("Budget status is empty when formatting command result")
            budget_status = "[Budget status unavailable]"

        # Format the command result
        message_parts = []
        base_message = ""

        if isinstance(result, dict):
            # Always append the main message
            base_message = result.get("message", str(result))
            message_parts.append(base_message)
            # If the edit failed, append proposed and current code snippets
            if not result.get("success", True):
                proposed_code = result.get("proposed_code") or ""
                current_code = result.get("current_code") or ""
                if proposed_code:
                    message_parts.append(f"\n\nProposed Code:\n```\n{proposed_code}\n```")
                if current_code:
                    message_parts.append(f"\n\nCurrent Code:\n```\n{current_code}\n```")
        elif hasattr(result, 'message'): # Assuming CommandResult object
            base_message = str(result.message)
            message_parts.append(base_message)
            # Potentially add similar logic for CommandResult objects if they can carry edit failure details
            # For now, this focuses on the dict case as per the logs
        else:
            base_message = str(result)
            message_parts.append(base_message)
        
        final_message = "".join(message_parts)

        # Always include budget status first
        return f"{budget_status}\n\n{final_message}"

    @staticmethod
    def _format_speedup(speedup: float) -> str:
        """Helper to format speedup values with appropriate precision."""
        if not isinstance(speedup, (int, float)):
            return str(speedup)
        
        speedup = float(speedup)
        if speedup >= 1000:
            return f"{speedup:.0f}x"
        elif speedup >= 100:
            return f"{speedup:.1f}x"
        elif speedup >= 10:
            return f"{speedup:.2f}x"
        else:
            return f"{speedup:.3f}x"

    @staticmethod
    def _format_number(value: float) -> str:
        """Helper to format numbers with appropriate precision."""
        if not isinstance(value, (int, float)):
            return str(value)

        # Convert to float to handle both int and float inputs
        value = float(value)

        # Handle zero specially
        if value == 0:
            return "0"

        # Format with 4 significant figures
        # Use engineering notation for very large/small numbers
        formatted = f"{value:.4g}"

        # If it's in scientific notation, convert to decimal where reasonable
        if "e" in formatted.lower():
            exp = int(formatted.lower().split("e")[1])
            if -4 <= exp <= 4:  # Convert to decimal for reasonable exponents
                formatted = f"{value:f}"
                # Trim to 4 significant figures
                non_zero_idx = 0
                for i, c in enumerate(formatted.replace(".", "")):
                    if c != "0":
                        non_zero_idx = i
                        break
                sig_fig_end = non_zero_idx + 4
                if "." in formatted:
                    sig_fig_end += 1
                formatted = formatted[:sig_fig_end]
                # Remove trailing zeros after decimal
                if "." in formatted:
                    formatted = formatted.rstrip("0").rstrip(".")

        return formatted

    @staticmethod
    def format_performance_update(
        metric_name: str, current_value: float, best_value: float, status: str = None
    ) -> str:
        """
        Format a performance update message.

        Args:
            metric_name: Name of the metric being tracked
            current_value: Current value of the metric
            best_value: Best value achieved so far
            status: Optional status message to append
        """
        msg = f"Performance Update - {metric_name}: {current_value:.4f} (Best: {best_value:.4f})"
        if status:
            msg += f" - {status}"
        return msg

    @staticmethod
    def format_module_reload(module_name: str) -> str:
        """Format a message indicating a module has been reloaded."""
        return f"Reloaded module: {module_name}"

    @staticmethod
    def format_budget_limit_reached(budget_type: str, limit: float) -> str:
        """
        Format a message indicating that a budget limit has been reached.

        Args:
            budget_type: The type of budget (e.g., "cost", "tokens", "messages")
            limit: The limit that was reached
        """
        return f"Budget limit reached: {budget_type} limit of {MessageWriter._format_number(limit)} has been reached."

    @staticmethod
    def format_multi_part_message_with_budget(
        budget_status: str, parts: List[str]
    ) -> str:
        """Format a multi-part message with budget status at the start."""
        all_parts = [budget_status] + parts
        return "\n\n".join(all_parts)

    def get_formatted_budget_status(self, interface) -> str:
        """Get formatted budget status with error handling.

        Args:
            interface: The LLM interface instance

        Returns:
            Formatted budget status string, or error message if formatting fails
        """
        try:
            return self.format_budget_status(
                spend=interface.state.spend,
                remaining=interface.spend_limit - interface.state.spend,
                messages_sent=interface.state.messages_sent,
                messages_remaining=interface.total_messages
                - interface.state.messages_sent,
            )
        except Exception as e:
            logging.error(f"Failed to get budget status: {e}")
            return "[Budget status unavailable]"

    @staticmethod
    def format_command_response(interface, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a command response with budget status in a consistent way.
        Handles both success and error cases.

        Args:
            interface: The LLM interface instance (needed for budget status)
            result: Dict containing command result with at least 'success' and 'message' fields

        Returns:
            Dict containing formatted response with budget status
        """
        logging.info(f"DIAG: format_command_response START. Incoming result keys: {list(result.keys())}")
        
        # Check for code_context specifically
        if "code_context" in result:
            code_context = result["code_context"]
            logging.info(f"DIAG: code_context present in format_command_response result: {'Yes' if code_context else 'No'}")
            if code_context:
                if code_context is not None:
                    logging.info(f"DIAG: code_context length: {len(code_context)}")
                    logging.info(f"DIAG: First 100 chars of code_context: {code_context[:100]}")
                else:
                    logging.info("DIAG: code_context is None")
        else:
            logging.info("DIAG: No code_context key in format_command_response result")
            
        try:
            # Fix: Get instance and call instance method
            writer_instance = MessageWriter() 
            budget_status = writer_instance.get_formatted_budget_status(interface)
        except Exception as e:
            budget_status = None
            logging.warning(
                f"Could not get budget status when formatting command response: {str(e)}"
            )
            logging.debug(f"Command response being formatted: {result}")

        # Determine the core message: prefer pre-formatted evaluation message if available
        message = result.get("formatted_message")
        if not message:
            message = result.get("message", "")
            if not message and result.get("error"):
                # If no message but we have an error, use that
                message = result.get("error")
                logging.debug("Using error as message since no message field present and no formatted_message")

        # Check if message includes error context if we have code_context
        if "code_context" in result and result["code_context"]:
            context_snippet = result["code_context"]
            first_line = context_snippet.split("\n")[0] if "\n" in context_snippet else context_snippet
            logging.info(f"DIAG: Checking if message already contains code context '{first_line[:30]}...'")
            if first_line in message:
                logging.info("DIAG: Message already contains code context")
            else:
                logging.info("DIAG: Message does NOT contain code context")
                
                # If message doesn't include code context, should we add it?
                logging.info("DIAG: Checking if message includes string 'Error context:'")
                if "Error context:" not in message and result.get("code_context"):
                    logging.info("DIAG: Message doesn't include 'Error context:' - this might indicate code context was not included")

        formatted_message = (
            MessageWriter.format_message_with_budget(budget_status, message)
            if budget_status
            else message
        )
        
        # --- START: Append proposed and current code blocks on failure ---
        if not result.get("success", True):
            proposed = result.get("proposed_code")
            current = result.get("current_code")
            if proposed:
                formatted_message += f"\n\nProposed Code:\n```\n{proposed}\n```"
            if current:
                formatted_message += f"\n\nCurrent Code:\n```\n{current}\n```"
        # --- END ---

        # Base response always includes success and formatted message
        response = {
            "success": result.get("success", True),
            "message": formatted_message,
        }
        # Include error only if non-None
        if result.get("error") is not None:
            response["error"] = result.get("error")
        # Include data only if non-None
        if result.get("data") is not None:
            response["data"] = result.get("data")
        
        # Preserve code_context and traceback if available
        if "code_context" in result:
            response["code_context"] = result["code_context"]
            logging.info("DIAG: Preserved code_context in response")
        
        if "traceback" in result:
            response["traceback"] = result["traceback"]
            logging.info("DIAG: Preserved traceback in response")

        # Copy any additional status fields
        for field in [
            "edit_status",
            "snapshot_status",
            "eval_status",
            "file_status",
            "profile_status",
        ]:
            if field in result:
                response[field] = result[field]
        
        # Preserve proposed_code and current_code if present
        if "proposed_code" in result:
            response["proposed_code"] = result["proposed_code"]
        if "current_code" in result:
            response["current_code"] = result["current_code"]

        # --- FIX START ---
        # Also preserve top-level stdout and stderr if they exist in the input result
        # but only if they haven't been incorporated into the formatted message
        if "stdout" in result:
            stdout_content = result["stdout"]
            if stdout_content and stdout_content.strip():
                # Check if stdout is already included in the formatted message
                if "Stdout:" not in formatted_message:
                    response["stdout"] = stdout_content
                else:
                    # Stdout is already incorporated into the message, explicitly set empty to prevent duplication
                    response["stdout"] = ""
            else:
                # Empty stdout, preserve as-is for compatibility
                response["stdout"] = stdout_content
        if "stderr" in result:
            response["stderr"] = result["stderr"]
        # --- FIX END ---
        
        logging.info(f"DIAG: format_command_response END. Response keys: {list(response.keys())}")
        
        return response

    @staticmethod
    def format_oracle_result_from_raw(evaluation_output: dict) -> str:
        """Format oracle evaluation result from raw output.
        
        Args:
            evaluation_output: Raw oracle evaluation output
            
        Returns:
            Formatted oracle evaluation result
        """
        lines = []
        
        # Check for validation error information stored in builtins
        try:
            import builtins
            validation_error = getattr(builtins, 'last_validation_error', None)
            has_validation_error = validation_error is not None
        except Exception:
            validation_error = None
            has_validation_error = False
        
        # Catch empty input early
        if not evaluation_output:
            return "No oracle output provided."
        
        # Extract values from the raw output
        is_success = evaluation_output.get("success", False)
        elapsed_ms = evaluation_output.get("elapsed_ms")
        stdout = evaluation_output.get("stdout", "")
        result = evaluation_output.get("result")
        error = evaluation_output.get("error", "Unknown error")
        error_type = evaluation_output.get("error_type", "")
        # Ensure cleaned_traceback is defined to avoid NameError downstream
        raw_traceback = evaluation_output.get("traceback")
        cleaned_traceback = MessageWriter._clean_traceback(raw_traceback)
        
        # Get command source to customize formatting
        command_source = evaluation_output.get("command_source")
        
        # Get code context early - this is key for showing error context
        code_context = evaluation_output.get("code_context")
        
        # Log the timing information for debugging
        if elapsed_ms is not None:
            logging.info(f"DEBUG format_oracle_result_from_raw: found elapsed_ms={elapsed_ms}")
        else:
            logging.info("DEBUG format_oracle_result_from_raw: elapsed_ms is None")
            
            # Try to find elapsed_ms in other places
            if "output_logs" in evaluation_output and isinstance(evaluation_output["output_logs"], dict):
                output_logs_elapsed = evaluation_output["output_logs"].get("elapsed_ms")
                if output_logs_elapsed is not None:
                    logging.info(f"DEBUG format_oracle_result_from_raw: found elapsed_ms={output_logs_elapsed} in output_logs")
                    elapsed_ms = output_logs_elapsed
        
            if elapsed_ms is None and "first_run_result" in evaluation_output:
                first_run = evaluation_output["first_run_result"]
                if isinstance(first_run, dict):
                    first_run_elapsed = first_run.get("elapsed_ms")
                    if first_run_elapsed is not None:
                        logging.info(f"DEBUG format_oracle_result_from_raw: found elapsed_ms={first_run_elapsed} in first_run_result")
                        elapsed_ms = first_run_elapsed
        
        # Special handling for solver errors
        # FIX: Ensure 'error' is a string before using 'in' operator
        error_str = error if isinstance(error, str) else "" # Use empty string if error is None or not string
        is_solver_error = error_type == "solver_error" or (error_str and "solver.solve" in error_str)
        is_solution_error = error_type == "invalid_solution" or error_type == "validation_error" or (error_str and "is_solution" in error_str)
        
        # Show problem input if available - fixed to handle numpy arrays
        problem_input = evaluation_output.get("problem_input")
        if problem_input is not None:
            lines.append(f"Input: {problem_input}")
        
        # Process and clean up stdout if it exists
        clean_stdout = None
        if stdout and stdout.strip():
            # Clean up stdout to remove wrapper output
            if "=== SOLVER INPUT ===" in stdout:
                parts = stdout.split("=== USER OUTPUT BEGINS ===")
                if len(parts) > 1:
                    user_part = parts[1].split("=== USER OUTPUT ENDS ===")[0].strip()
                    if user_part:
                        clean_stdout = f"Stdout: {user_part}"
            elif "===" not in stdout:  # If no markers, use as is
                clean_stdout = f"Stdout: {stdout.strip()}"
        
        # For successful evaluations
        if is_success:
            if result is None:
                result = "N/A"
            lines.append(f"Reference Output: {result}")
            # Add clean stdout if available and not already included
            if clean_stdout:
                # Check if stdout is already included in the message
                current_message = "\n".join(lines)
                if "Stdout:" not in current_message:
                    lines.append(clean_stdout)

            # --- START: Add Runtime Here (Success Case) ---
            if elapsed_ms is not None:
                if isinstance(elapsed_ms, (int, float)):
                    lines.append(f"Runtime: {elapsed_ms:.5f} ms")
                else:
                    try: lines.append(f"Runtime: {float(elapsed_ms):.5f} ms")
                    except (ValueError, TypeError): lines.append(f"Runtime: {elapsed_ms} ms (could not format)")
            else:
                # Fallback check (simplified)
                timing_found = False
                for key in ["elapsed_ms", "direct_elapsed"]:
                    for source in [evaluation_output, evaluation_output.get("output_logs", {}), evaluation_output.get("first_run_result", {})]:
                        if isinstance(source, dict):
                            time_val = source.get(key)
                            if time_val is not None:
                                try:
                                    lines.append(f"Runtime: {float(time_val):.5f} ms")
                                    timing_found = True
                                    break # Found time in this source
                                except (ValueError, TypeError):
                                    pass
                        if timing_found: break
                    if timing_found: break
                if not timing_found:
                    lines.append("Runtime: N/A (timing information unavailable)")
            # --- END: Add Runtime Here (Success Case) ---
        else:
            # For failed evaluations
            error = evaluation_output.get("error", "Unknown error")
            error_type = evaluation_output.get("error_type", "")
            code_context = evaluation_output.get("code_context") # Get context early
            
            # Special handling for solver errors (might be redundant now but keep for clarity)
            is_solver_error = error_type == "solver_error" or "solver.solve" in error
            is_solution_error = error_type == "invalid_solution" or error_type == "validation_error" or "is_solution" in error
            
            # Check for validation error specifically from eval_input
            if error_type == "validation_error" and "validation_error" in evaluation_output:
                solution_type = evaluation_output.get("solution_type", "unknown")
                solution_shape = evaluation_output.get("solution_shape", "unknown")
                validation_traceback = evaluation_output.get("validation_traceback", "")
                validation_error_msg = evaluation_output.get("validation_error", "Unknown validation error")
                
                # If error is "Solution is invalid", don't print that line
                if error != "Solution is invalid":
                    lines.append(f"Solution validation error: {validation_error_msg}")
                
                lines.append(f"Solution type: {solution_type}")
                lines.append(f"Solution shape: {solution_shape}")
                
                # If we have validation error in builtins, it might have the extended context
                # which we should prefer over the basic traceback
                if has_validation_error and "is_solution" in validation_traceback:
                    # Let's rely on the general context/traceback handling below.
                    pass # Avoid adding potentially duplicate/less clean context here
                    
            # --- General Failure Case ---
            # Handle any other error type by displaying the error message
            else: 
                # Special formatting for invalid_solution so the user still sees
                # the output and runtime just like a successful run.
                if error_type == "invalid_solution":
                    # Show the solver's raw output and timing first
                    result_output = evaluation_output.get('result')
                    if result_output is None:
                        result_output = 'N/A'
                    lines.append(f"Output: {result_output}")
                    runtime_ms = evaluation_output.get('elapsed_ms', 'N/A')
                    lines.append(f"Runtime: {runtime_ms} ms" if runtime_ms != 'N/A' else "Runtime: N/A")
                    # Then the standard invalid-solution explanation
                    err_msg = evaluation_output.get('error') or 'Solution is invalid.'
                    lines.append(err_msg)
                    # Include code context if available to help user understand what went wrong
                    MessageWriter._append_code_context_to_lines(lines, evaluation_output.get('code_context'))
                else:
                    # Generic failure formatting (unchanged)
                    err_msg = evaluation_output.get('error') or (
                        "Solver class not found in solver.py" if error_type == 'missing_solver_class' else error_type or 'Unknown Error'
                    )
                    lines.append(f"Evaluation Failed: {err_msg}")
                    if cleaned_traceback:  # Use cleaned traceback
                        lines.append("\nTraceback:")
                        lines.append(f"  {cleaned_traceback}")
                    # Append code context if available
                    MessageWriter._append_code_context_to_lines(lines, evaluation_output.get('code_context'))

            # --- Add Context and Traceback for *all* failures in this block ---
            if code_context:
                # Check if context might already be embedded in a validation traceback from builtins
                already_has_context = False
                if error_type == "validation_error" and has_validation_error:
                    builtin_tb = validation_error.get('traceback', '')
                    if "--- Code Context ---" in builtin_tb:
                        already_has_context = True
                        
                if not already_has_context:
                    # FIX: Use helper to append context
                    MessageWriter._append_code_context_to_lines(lines, code_context)

            # Ensure this block is indented correctly, aligned with 'if code_context:' above
                exception_type = evaluation_output.get("exception_type", "")
                if exception_type and exception_type not in error:
                    lines.append(f"Exception type: {exception_type}")
                
                example_input = evaluation_output.get("example_input")
                if example_input is not None:
                    lines.append(f"Example valid input: {example_input}")
            
                if clean_stdout and "OUTPUT:" not in stdout:
                    # Check if stdout is already included in the message
                    current_message = "\n".join(lines)
                    if "Stdout:" not in current_message:
                        lines.append(clean_stdout)
        
        return "\n".join(lines)

    @staticmethod
    def _format_error_details(error_dict: Dict[str, Any]) -> List[str]:
        """
        Formats the common error details (message, code context, traceback)
        from an error result dictionary into a list of strings for consistent display.
        """
        lines = []
        # Use the basic error message
        error_msg = error_dict.get("error", "Unknown error detail.")
        code_context = error_dict.get("code_context")
        traceback_str = error_dict.get("traceback") # Get the raw traceback

        # Clean file paths from error message
        from AlgoTuner.utils.trace_cleaner import clean_build_output
        cleaned_error_msg = clean_build_output(error_msg) if error_msg else error_msg

        lines.append(f"Error: {cleaned_error_msg}") # Use the cleaned error message

        return lines

    # --- ADDED HELPER for Code Context Appending ---
    @staticmethod
    def _append_code_context_to_lines(lines: List[str], code_context: Optional[str]):
        """Appends the formatted code context block to a list of message lines."""
        if code_context:
            lines.append("\nCode Context:\n")
            lines.append(code_context)
    # --- END ADDED HELPER ---
