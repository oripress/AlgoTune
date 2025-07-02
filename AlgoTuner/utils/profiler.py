import os
import logging
from line_profiler import LineProfiler
from functools import wraps
import tempfile
import re
import io
import numpy as np
from AlgoTuner.utils.message_writer import MessageWriter
from AlgoTuner.utils.casting import cast_input
import traceback
from pathlib import Path
from AlgoTuner.utils.solver_loader import load_solver_module, get_solve_callable, with_working_dir
from AlgoTuner.utils.error_utils import extract_error_context, SolverFileNotFoundError, SOLVER_NOT_FOUND_GENERIC_MSG


class TaskProfiler:
    """Handles profiling of task solve methods using both line and memory profilers."""

    def __init__(self, task_instance):
        """Initialize profiler with a task instance."""
        self.task = task_instance
        self.line_profiler = LineProfiler()

    def profile_solve(self, problem, focus_lines=None, filename=None):
        """
        Profile the solve method of the task using both line and memory profilers.

        Args:
            problem: The problem instance to solve
            focus_lines: Optional list of line numbers to focus on
            filename: The Python file to profile (defaults to 'solver.py')

        Returns:
            Dict containing:
                - success: Whether profiling was successful
                - result: The solution returned by the solve function
                - profile_output: Formatted string containing the profiling information
                - error: Error message if profiling failed
        """
        try:
            logging.info(f"TaskProfiler.profile_solve: Problem type: {type(problem)}")
            if hasattr(problem, 'shape'):
                logging.info(f"TaskProfiler.profile_solve: Problem shape: {problem.shape}")
            if focus_lines:
                logging.info(f"TaskProfiler.profile_solve: Focus lines: {focus_lines}")
            
            # --- FILENAME HANDLING ---
            code_dir = Path(os.environ.get("CODE_DIR", "."))
            solver_filename = filename if filename else "solver.py"
            solver_path = code_dir / solver_filename
            if not solver_filename.endswith(".py"):
                return {
                    "success": False,
                    "error": f"Specified file '{solver_filename}' is not a Python (.py) file.",
                    "error_type": "invalid_file_type",
                    "file_path": str(solver_path)
                }
            if not solver_path.is_file():
                return {
                    "success": False,
                    "error": f"File '{solver_filename}' not found in code directory.",
                    "error_type": "file_not_found",
                    "file_path": solver_filename  # report only the filename
                }
            # --- END FILENAME HANDLING ---
            try:
                # Load solver module then get the Solver.solve callable
                with with_working_dir(code_dir):
                    solver_module = load_solver_module(code_dir, solver_filename=solver_filename)
                SolverClass = getattr(solver_module, "Solver", None)
                if SolverClass is None:
                    return {
                        "success": False,
                        "error": f"Class 'Solver' not found in {solver_filename}.",
                        "error_type": "solver_class_not_found",
                        "file_path": str(solver_path)
                    }
                solver_instance = SolverClass()
                solve_method = getattr(solver_instance, "solve", None)
                if not callable(solve_method):
                    return {
                        "success": False,
                        "error": f"Method 'solve' not found or not callable on Solver instance in {solver_filename}.",
                        "error_type": "solve_method_not_found",
                        "file_path": str(solver_path)
                    }
                # --- END NEW ---
                logging.info(f"TaskProfiler.profile_solve: Loaded Solver.solve from {solver_path}")
            except SolverFileNotFoundError as e:
                tb = traceback.format_exc()
                logging.error(f"SolverFileNotFoundError during profile_solve: {e} (Path: {solver_path})")
                return {
                    "success": False,
                    "error": SOLVER_NOT_FOUND_GENERIC_MSG,
                    "error_type": "solver_not_found_error",
                    "traceback": tb,
                    "code_context": None
                }
            except Exception as e:
                tb = traceback.format_exc()
                context_info = extract_error_context(tb, str(e))
                enhanced_message = context_info.get("enhanced_error_message", str(e))
                context_snippet = context_info.get("code_context_snippet")
                return {
                    "success": False,
                    "error": enhanced_message,
                    "error_type": "solver_load_error",
                    "traceback": tb,
                    "code_context": context_snippet
                }
            # Store original solve method
            original_solve = solve_method

            try:
                # First do line profiling
                logging.info(f"TaskProfiler.profile_solve: Applying line profiler to Solver.solve")
                profiled_solve = self.line_profiler(solve_method)
                # Call the profiled solve method
                logging.info(f"TaskProfiler.profile_solve: Calling profiled solve method with problem")
                solution = profiled_solve(problem)
                logging.info(f"TaskProfiler.profile_solve: Solve function returned result of type: {type(solution)}")

                # Get line profiling stats
                logging.info(f"TaskProfiler.profile_solve: Getting line profiler stats")
                with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
                    self.line_profiler.print_stats(tmp)
                    tmp.seek(0)
                    raw_output = tmp.read()
                os.unlink(tmp.name)
                
                # Log information about the line profiler output
                logging.info(f"TaskProfiler.profile_solve: Line profiler output length: {len(raw_output)} characters")
                logging.debug(f"TaskProfiler.profile_solve: First 200 chars of raw output: {raw_output[:200]}")
                if not raw_output:
                    logging.warning("TaskProfiler.profile_solve: Empty line profiler output received!")
                if "Total time:" not in raw_output:
                    logging.warning("TaskProfiler.profile_solve: 'Total time:' not found in line profiler output")

                # Parse and filter the line profiler output
                logging.info(f"TaskProfiler.profile_solve: Filtering line profiler output")
                line_output, missing_lines = self._filter_line_profiler_output(
                    raw_output,
                    focus_lines,
                    solver_file_path=str(solver_path),
                )

                # Build the profiling output
                combined_output = "=== Line-by-Line Timing ===\n\n"
                combined_output += line_output.replace(
                    "Timer unit: 1e-09 s", "Timer unit: 1e-06 ms"
                )

                # Correctly format the total time in milliseconds only
                total_time_match = re.search(r"Total time: (\d+\.\d+) s", raw_output)
                if total_time_match:
                    raw_total_time = float(total_time_match.group(1))
                    total_time_ms = raw_total_time * 1000
                    combined_output = re.sub(
                        r"Total time: .*",
                        f"Total time: {total_time_ms:.6f} ms",
                        combined_output,
                    )
                else:
                    # If total time not found in output, estimate it from the sum of line times
                    total_time_ms = 0.0
                    for line in line_output.split('\n'):
                        match = re.match(r"\s*\d+\s+\d+\s+(\d+\.\d+)", line)
                        if match:
                            total_time_ms += float(match.group(1))
                    
                    # Add total time line if it doesn't exist
                    combined_output += f"\nTotal time: {total_time_ms:.6f} ms"

                # Clean up file path to just show solver.py
                combined_output = re.sub(
                    r"File: .*?solver\.py", "File: solver.py", combined_output
                )

                # Remove 'at line X' from function descriptions
                combined_output = re.sub(
                    r"(Function:.*?)(?: at line \d+)", r"\1", combined_output
                )

                # Add warning about missing lines if any
                if missing_lines:
                    combined_output += f"\nNote: Lines {', '.join(map(str, missing_lines))} were not found in the code."

                # Format the result using MessageWriter
                logging.info(f"TaskProfiler.profile_solve: Formatting profile result")
                mw = MessageWriter()
                
                # Prepare dict for formatter
                profile_success_dict = {
                    "success": True,
                    "profile_output": combined_output, 
                    "focus_lines": focus_lines 
                }
                formatted_message = mw.format_profile_result(profile_success_dict)

                # Log the total time being returned
                logging.info(f"TaskProfiler.profile_solve: Total profiling time: {total_time_ms:.6f} ms")
                
                logging.info(f"TaskProfiler.profile_solve: Returning successful result")
                return {
                    "success": True,
                    "result": solution,
                    "profile_output": combined_output,
                    "formatted_message": formatted_message,
                    "elapsed_ms": total_time_ms,
                    "file_path": solver_path
                }

            except Exception as e:
                error_msg = f"Error during profiling: {str(e)}"
                tb = traceback.format_exc()
                logging.error(f"{error_msg}\n{tb}")
                
                # Call extractor
                context_info = extract_error_context(tb, str(e))
                enhanced_message = context_info.get("enhanced_error_message", error_msg)
                context_snippet = context_info.get("code_context_snippet")
                
                return {
                    "success": False,
                    "error": enhanced_message,
                    "error_type": "profiling_error",
                    "file_path": solver_path,  
                    "traceback": tb,
                    "code_context": context_snippet
                }
        except Exception as e:
            error_msg = f"Unexpected error in profile_solve: {str(e)}"
            tb = traceback.format_exc()
            logging.error(f"{error_msg}\n{tb}")
            
            # Call extractor
            context_info = extract_error_context(tb, str(e))
            enhanced_message = context_info.get("enhanced_error_message", error_msg)
            context_snippet = context_info.get("code_context_snippet")
            
            return {
                "success": False,
                "error": enhanced_message,
                "error_type": "unexpected_error",
                "traceback": tb,
                "code_context": context_snippet
            }
        finally:
            # Always restore original solve method
            try:
                solve_method = original_solve
            except:
                pass

    def _filter_line_profiler_output(self, raw_output, focus_lines=None, solver_file_path=None):
        """Filter line profiler output to show relevant lines."""
        # Split output into header and content
        header_match = re.search(
            r"(Timer.*?==============================================================\n)",
            raw_output,
            re.DOTALL,
        )
        if not header_match:
            return raw_output, []

        header = header_match.group(1)
        content = raw_output[header_match.end() :]

        # Parse lines into structured data
        lines = []
        for line in content.split("\n"):
            if not line.strip():
                continue
            # More flexible regex that handles lines with 0 hits or missing timing data
            match = re.match(
                r"\s*(\d+)\s+(\d+|\s*)\s*(\d+\.?\d*|\s*)\s*(\d+\.?\d*|\s*)\s*(\d+\.?\d*|\s*)\s*(.*)",
                line,
            )
            if match:
                line_no, hits, time, per_hit, percent, code = match.groups()
                # Handle cases where fields might be empty/whitespace
                try:
                    lines.append(
                        {
                            "line_no": int(line_no),
                            "hits": int(hits) if hits.strip() else 0,
                            "time": float(time) if time.strip() else 0.0,
                            "per_hit": float(per_hit) if per_hit.strip() else 0.0,
                            "percent": float(percent) if percent.strip() else 0.0,
                            "code": code,
                            "raw": line,
                        }
                    )
                except (ValueError, AttributeError):
                    # Skip lines that don't parse correctly
                    continue

        # Filter lines based on strategy
        if focus_lines:
            # Only show the exact lines requested
            focus_lines = [int(line) for line in focus_lines]
            filtered_lines = [line for line in lines if line["line_no"] in focus_lines]
            # Track which requested lines weren't found
            existing_lines = {line["line_no"] for line in filtered_lines}
            missing_lines = [line for line in focus_lines if line not in existing_lines]
            # Sort by line number
            filtered_lines.sort(key=lambda x: x["line_no"])
        else:
            # Show the 25 lines that take the most cumulative time
            sorted_lines = sorted(lines, key=lambda x: x["time"], reverse=True)
            line_numbers = sorted(list({line["line_no"] for line in sorted_lines[:25]}))
            filtered_lines = [line for line in lines if line["line_no"] in line_numbers]
            filtered_lines.sort(key=lambda x: x["line_no"])  # Resort by line number
            missing_lines = []

        # Rebuild output
        result = header + "\n".join(line["raw"] for line in filtered_lines)
        if len(filtered_lines) < len(lines):
            result += (
                "\n... (showing requested lines only)"
                if focus_lines
                else "\n... (showing most time-consuming lines)"
            )

        # --- NEW: Add placeholder rows for missing lines so they appear in output ---
        if missing_lines and solver_file_path and os.path.exists(solver_file_path):
            try:
                file_lines = Path(solver_file_path).read_text().splitlines()
                added_any = False
                still_missing: list[int] = []
                for ln in missing_lines:
                    if 1 <= ln <= len(file_lines):
                        code_txt = file_lines[ln - 1].rstrip()
                        placeholder_raw = (
                            f"{ln:>9} {0:>9} {0.0:>9.1f} {0.0:>9.1f} {0.0:>8.1f} {code_txt}"
                        )
                        lines.append({
                            "line_no": ln,
                            "hits": 0,
                            "time": 0.0,
                            "per_hit": 0.0,
                            "percent": 0.0,
                            "code": code_txt,
                            "raw": placeholder_raw,
                        })
                        added_any = True
                    else:
                        still_missing.append(ln)

                if added_any:
                    # Rebuild the filtered_lines and result including placeholders
                    filtered_lines = [l for l in lines if l["line_no"] in (focus_lines or [])]
                    filtered_lines.sort(key=lambda x: x["line_no"])
                    header_plus = header + "\n".join(l["raw"] for l in filtered_lines)
                    if len(filtered_lines) < len(lines):
                        header_plus += "\n... (showing requested lines only)"
                    result = header_plus
                    missing_lines = still_missing
            except Exception as _read_err:
                logging.debug(f"_filter_line_profiler_output: Could not add placeholders: {_read_err}")

        return result, missing_lines