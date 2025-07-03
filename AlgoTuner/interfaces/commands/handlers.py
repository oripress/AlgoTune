import logging
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from pathlib import Path
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
import os
import time
import numpy as np
import inspect
import sys
from enum import Enum
import importlib
import re
import builtins
import ast
import typing
import math
import subprocess
import shutil

from AlgoTuner.interfaces.commands.types import (
    ParsedCommand,
    CommandResult,
    SnapshotStatus,
    EditStatus,
    EvalStatus,
    ProfileStatus,
    FileStatus,
)
from AlgoTuner.interfaces.core.base_interface import BaseLLMInterface
from AlgoTuner.utils.evaluator import (
    evaluate_code_on_dataset,
    reload_all_llm_src,
    run_evaluation,
    run_oracle_evaluation,
    cast_input,
    warmup_evaluator,
    ProcessPoolManager
)
from AlgoTuner.utils.message_writer import MessageWriter
from AlgoTuner.utils.code_helpers import extract_code_blocks
from AlgoTuner.utils.error_helpers import get_error_messages_cached
from AlgoTuner.utils.profiler import TaskProfiler
from AlgoTuner.utils.trace_cleaner import clean_traceback, clean_build_output
from AlgoTuner.utils.type_inspection import describe_type, describe_annotation
from AlgoTuner.utils.evaluator.main import run_evaluation_on_input, _calculate_aggregate_metrics, evaluate_baseline_dataset
from AlgoTuner.utils.error_utils import extract_error_context
from AlgoTuner.utils.casting import parse_string
from AlgoTuner.utils.evaluator.runner import run_oracle_evaluation, _strip_bulky_fields
from AlgoTuner.utils.timing_config import DEV_RUNS, EVAL_RUNS
from AlgoTuner.utils.validation import validate_solver_setup
from AlgoTuner.utils.evaluator import load_task
from AlgoTuner.utils.profiler import TaskProfiler
from AlgoTuner.config.loader import load_config
import json, tempfile

class CommandHandlers:
    """Handlers for executing parsed commands.

    This class provides a unified interface for executing various commands
    in the system. It handles command routing, execution, error handling,
    and response formatting.

    Each command is handled by a dedicated execute method that:
    1. Validates and extracts command arguments
    2. Calls appropriate handler method
    3. Formats response with consistent structure
    4. Includes command-specific status fields
    5. Handles any errors that occur

    All responses include:
    - Success/failure status
    - Formatted message with budget information
    - Command-specific status (e.g. edit_status, eval_status)
    - Error details if applicable
    """

    def __init__(self, interface: BaseLLMInterface):
        """Initialize command handlers.

        Sets up the command handler with required dependencies for executing
        commands and formatting responses.

        Args:
            interface: Base interface providing core functionality
        """
        self.interface = interface
        self.message_writer = MessageWriter()

        self.baseline_times_train = None
        self.baseline_times_test = None

    def _run_with_cast_and_format(
        self,
        input_str: str,
        runner: Callable[..., CommandResult], # Expects a function that returns CommandResult
        status_field: str, # e.g., "eval_status", "profile_status"
        **runner_kwargs: Any # Additional keyword arguments for the runner function
    ) -> Dict[str, Any]:
        """Parses input, runs a given function, and formats the output.

        Args:
            input_str: The raw input string to parse.
            runner: The function to execute with the parsed input.
            status_field: The status field name for formatting.
            **runner_kwargs: Additional arguments to pass to the runner function.

        Returns:
            A formatted response dictionary.
        """
        try:
            # Attempt to parse and cast the input string using type hints
            try:
                # Use cast_input which handles parsing AND type hint casting
                casted_input = cast_input(input_str, self.interface.task_instance)
                logging.info(f"Successfully casted input string '{input_str[:50]}...' to type {type(casted_input)} using task hints.")
            except Exception as e:
                logging.error(f"Failed to parse/cast input string: '{input_str[:100]}...'. Error: {str(e) if e is not None else 'Unknown error'}")
                

                from AlgoTuner.utils.casting import parse_string # Helper to parse the input string
                parsed_for_error_check = None
                try:
                    parsed_for_error_check = parse_string(input_str)
                except Exception:
                    pass # Ignore if parsing for error check fails

                # Clean file paths from error message
                raw_error_msg = str(e)
                error_message_detail = clean_build_output(raw_error_msg) if raw_error_msg else raw_error_msg

                if isinstance(parsed_for_error_check, (list, tuple)) and len(parsed_for_error_check) > 20: # Threshold for "large"
                    type_name = type(parsed_for_error_check).__name__
                    length = len(parsed_for_error_check)
                    # Get the first line of the original error as a summary and clean it
                    original_error_summary = str(e).splitlines()[0] if str(e) else "Casting failed"
                    cleaned_summary = clean_build_output(original_error_summary) if original_error_summary else original_error_summary
                    error_message_detail = (
                        f"Failed to cast input (type: {type_name}, length: {length}) "
                        f"to the expected type. Detail: {cleaned_summary}"
                    )
                
                final_error_msg = f"Invalid input format: {error_message_detail}"


                # Add traceback to data for better debugging
                error_data = {"input_string": input_str, "traceback": traceback.format_exc()}
                return self._format_error_response(
                    error_msg=final_error_msg, # Use potentially shortened message
                    context=f"parsing input for {status_field.split('_')[0]} command",
                    status_value=EvalStatus.FAILURE.value if status_field == "eval_status" else ProfileStatus.FAILURE.value,
                    status_field=status_field,
                    data=error_data
                )
            
            # Run the provided runner function with casted input and other args
            logging.info(f"Running {runner.__name__} with casted input and kwargs: {runner_kwargs}")
            result: CommandResult = runner(casted_input, **runner_kwargs)
            logging.info(f"Runner {runner.__name__} completed. Success: {result.success}")

            # Format the response based on runner result
            if result.success:
                return self._format_success_response(result, status_field=status_field)
            else:
                # Runner failed, format error response using its details
                return self._format_error_response(
                    error_msg=result.error or "Runner failed without specific error message.",
                    context=f"running {runner.__name__}",
                    status_value=result.status if hasattr(result, 'status') else (EvalStatus.FAILURE.value if status_field == "eval_status" else ProfileStatus.FAILURE.value),
                    status_field=status_field,
                    data=result.data if hasattr(result, 'data') else {}
                )

        except Exception as e:
            # Catch unexpected errors during the process
            logging.error(f"Unexpected error during _run_with_cast_and_format for {runner.__name__}: {str(e) if e is not None else 'Unknown error'}")
            # Clean file paths from error message
            raw_error_msg = str(e) if e is not None else "Unknown error"
            cleaned_error_msg = clean_build_output(raw_error_msg)
            return self._format_error_response(
                error_msg=cleaned_error_msg,
                context=f"running {runner.__name__} pipeline",
                status_value=EvalStatus.FAILURE.value if status_field == "eval_status" else ProfileStatus.FAILURE.value,
                status_field=status_field,
                data={"input_string": input_str, "traceback": traceback.format_exc(), **runner_kwargs}
            )

    def handle_command(self, command_str: ParsedCommand) -> Dict[str, Any]:
        """Handle execution of a parsed command.

        Routes the command to the appropriate handler based on the command type.
        Handles any unexpected errors during command execution.

        Args:
            command_str: Parsed command containing command type and arguments

        Returns:
            Dict containing success status, formatted message, and command-specific status fields
        """
        try:
            # If command_str is a structured error response (parsing or validation error)
            if isinstance(command_str, dict) and not command_str.get("success", True):
                error_msg = command_str["error"]
                cmd = command_str.get("command", "command")

                # Format the error message based on error type
                if command_str.get("is_validation_error", True):
                    # For validation errors, format as validation error
                    # If error_msg is already a dict with error field, extract it
                    if isinstance(error_msg, dict) and "error" in error_msg:
                        error_msg = error_msg["error"]
                    error_msg = self.message_writer.format_validation_error(
                        error_msg, f"validating {cmd}"
                    )
                elif command_str.get("is_parsing_error", False):
                    # For parsing errors, show command help
                    error_msg = f"Command failed: {error_msg}"
                else:
                    # For other errors, use standard error formatting
                    error_msg = self.message_writer.format_command_error(
                        error_msg, f"executing {cmd}"
                    )

                # Format with budget info
                return self._format_error_response(error_msg, f"executing {cmd}")
                
            # Lazy import to avoid circular import
            def check_text_after_command(cmd_str):
                from AlgoTuner.utils.command_helpers import check_for_text_after_command
                return check_for_text_after_command(cmd_str)

            cmd = command_str.command
            result = None
            
            if cmd == "edit":
                result = self._execute_edit_command(command_str)
            elif cmd == "eval_input":
                result = self._execute_eval_input_command(command_str)
            elif cmd == "eval":
                # Delegate all eval formatting to helper for the train subset
                result = self._run_and_format_dataset_eval("train", "eval")
            elif cmd == "revert":
                result = self._execute_revert_command()
            elif cmd == "view_file":
                # Call the handler
                result = self._handle_view_file(command_str.args["file_name"], command_str.args.get("start_line"))
                # Already properly formatted, return directly
                return result
            elif cmd == "ls":
                # Call the handler
                handler_result = self._handle_ls(command_str.args.get("path"))
                # Format the result explicitly
                if handler_result.get('success'):
                    result = self._format_success_response(handler_result, status_field="file_status")
                else:
                    result = self._format_error_response(
                        error_msg=handler_result.get('error', 'List directory failed'),
                        context=f"listing directory {command_str.args.get('path') or '.'}",
                        status_value=FileStatus.FAILURE.value,
                        status_field="file_status",
                        data=handler_result # Pass full dict for potential context
                    )
            elif cmd == "delete":
                result = self._execute_delete_command(command_str)
            elif cmd == "profile":
                result = self._execute_profile_command(command_str)
            elif cmd == "profile_line":
                result = self._execute_profile_line_command(command_str)
            elif cmd == "profile_lines":
                result = self._execute_profile_lines_command(command_str)
            elif cmd == "evaluate":
                result = self._execute_eval_command(command_str)
            elif cmd == "reference":
                result = self._execute_baseline_command(command_str)
            elif cmd == "create_test_file":
                result = self._handle_create_test_file(command_str.args["file_name"], command_str.args.get("content"))
            else:
                result = self._unknown_command_error(cmd)
                
            # Return the result obtained from the specific handler
            return result

        except Exception as e:
            # Clean file paths from error message
            raw_error_msg = str(e) if e is not None else "Unknown error"
            cleaned_error_msg = clean_build_output(raw_error_msg)
            return self._format_error_response(
                error_msg=cleaned_error_msg, 
                context=f"handling command '{command_str.command if isinstance(command_str, ParsedCommand) else 'unknown'}'",
                data={"traceback": traceback.format_exc()} # Pass traceback explicitly
            )

    def _format_command_response(
        self,
        success: bool,
        message: str,
        error: Optional[str] = None,
        status_value: Optional[str] = None,
        status_field: Optional[str] = None,
        data: Optional[Any] = None,
        add_budget: bool = True, # Parameter kept for signature compatibility, but ignored
    ) -> Dict[str, Any]:
        """Format a command response for the LLM interface.

        Args:
            success: Whether the command succeeded
            message: The formatted message to display
            error: Optional error message
            status_value: Optional status value (e.g., EvalStatus.SUCCESS.value)
            status_field: Optional status field (e.g., "eval_status")
            data: Optional additional data from the command
            add_budget: Ignored - budget status is added in message_handler

        Returns:
            Formatted response dict
        """

        # Prepare arguments for send_message
        send_message_kwargs = {
            "content": message,
            "error_message": error,
        }

        # Add status field as a dynamic argument
        if status_field and status_value:
            send_message_kwargs[status_field] = status_value

        # Extract problem input from data if this is an evaluation-related response
        if (data and isinstance(data, dict) and
            status_field in ["eval_status", "profile_status"] and
            "problem_input" in data):
            send_message_kwargs["problem_input"] = data["problem_input"]

        # Return the message without budget formatting - message_handler will add it
        try:
            # Build the result dict with raw message - budget will be added by message_handler
            result = {
                "success": success,
                "message": message,  # Raw message without budget
                "spend": getattr(self.interface.state, 'spend', 0.0),
            }
            
            # Add error if provided
            if error:
                result["error"] = error
            
            # Add status field if provided
            if status_field and status_value:
                result[status_field] = status_value
            
            # Include the data in the response for upstream consumers
            if data is not None:
                result["data"] = data

            return result

        except Exception as e:
            logging.error(f"Error in _format_command_response: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to format response: {str(e)}",
                "spend": getattr(self.interface.state, 'spend', 0.0),
            }

    def _format_error_response(
        self,
        error_msg: str,
        context: str,
        status_value: Optional[str] = None,
        status_field: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format an error response with consistent structure."""
        # Handle None or empty error message
        if not error_msg:
            error_msg = f"Unknown error during {context}"
            
        # Determine which traceback to use: prefer caller-supplied, else capture current
        orig_tb = ""
        if data and isinstance(data, dict) and "traceback" in data:
            orig_tb = data.get("traceback") or ""
        elif not data or not isinstance(data, dict) or "traceback" not in data:
            # Only capture current traceback if not provided explicitly
            try:
                 current_tb = traceback.format_exc()
                 # Avoid adding default "NoneType: None\n" if no real exception occurred
                 if "NoneType: None" not in current_tb:
                      orig_tb = current_tb
            except Exception:
                 pass # Ignore errors during traceback capture

        from AlgoTuner.utils.error_utils import extract_error_context
        
        # Ensure error_msg is a string before potentially passing to extract_error_context
        error_msg_str = str(error_msg) if error_msg is not None else f"Unknown error during {context}"
        
        # Identify dataset evaluation by explicit data flag
        is_dataset_evaluation = (
            status_field == "eval_status"
            and isinstance(data, dict)
            and data.get("evaluation_type") == "dataset"
        )
        
        context_info = {}
        code_context = None
        enhanced_error = error_msg_str # Default to original message
        
        # Extract error context for all error types
        try:
            # Check if caller already provided code context (e.g., from evaluation results)
            if data and isinstance(data, dict) and data.get("code_context"):
                code_context = data.get("code_context")
                enhanced_error = error_msg_str  # Use original message when context is pre-provided
            else:
                # Only extract context if not already provided
                context_info = extract_error_context(orig_tb, error_msg_str)
                code_context = context_info.get("code_context_snippet")
                enhanced_error = context_info.get("enhanced_error_message", error_msg_str)
        except Exception as context_exc:
            logging.warning(f"Failed to extract error context: {context_exc}")
            # Use original error message if context extraction fails
            # For dataset evaluations, try to get code context from data as fallback
            if is_dataset_evaluation and data and isinstance(data, dict) and data.get("code_context"):
                code_context = data.get("code_context")

 
        # Use the (potentially enhanced) error message directly as the primary message.
        # For dataset evaluations with invalid solutions, use enhanced message if available
        if is_dataset_evaluation:
            # Use enhanced error message if available, otherwise fall back to original
            message_to_use = enhanced_error if enhanced_error != error_msg_str else error_msg_str
            
            # Check if message needs Error: prefix
            if message_to_use.startswith(("Error:", "Validation failed:", "Failed to", "Cannot", "Unable to", "Speedup:")):
                formatted_message = message_to_use
            else:
                formatted_message = f"Error: {message_to_use}"
            # Only show invalid examples if there are actual invalid solutions (not just timeouts)
            if data and isinstance(data, dict) and "invalid_solution_analysis" in data:
                # Check aggregate metrics to see if we have invalid solutions vs timeouts
                aggregate_metrics = data.get("aggregate_metrics", {})
                num_invalid = aggregate_metrics.get("num_invalid", 0)
                num_timeout = aggregate_metrics.get("num_timeout", 0)
                
                # Only show invalid examples if there are actual invalid solutions
                if num_invalid > 0:
                    formatted_message += "\n\n\nSnapshot not saved - invalid solutions present\n"
                    invalid_examples = data.get("invalid_solution_analysis", [])
                    # Cap at maximum 3 examples to avoid overwhelming output
                    max_examples = 3
                    for idx, snippet in enumerate(invalid_examples[:max_examples], start=1):
                        snippet_with_prefix = snippet
                        if snippet and not snippet.startswith("Error in 'is_solution': "):
                            snippet_with_prefix = f"Error in 'is_solution': {snippet}"
                        formatted_message += f"\nInvalid Example #{idx}:\n{snippet_with_prefix}\n"
                
                # Add note if there are more examples than shown
                # Summary line for additional invalid examples has been removed to reduce verbosity.
        else:
            # Use enhanced error message if available, otherwise fall back to original
            message_to_use = enhanced_error if enhanced_error != error_msg_str else error_msg_str
            
            if message_to_use.startswith(("Error:", "Validation failed:", "Failed to", "Cannot", "Unable to", "Speedup:")):
                # Message already has appropriate prefix or is an evaluation summary, use as-is
                formatted_message = message_to_use
            else:
                # Add Error: prefix for messages that don't have one
                formatted_message = f"Error: {message_to_use}"
            
        logging.error(f"Formatted Error Message: {formatted_message}")
 

        # Prepare diagnostics data: traceback + code snippet + original caller data
        merged_data: Dict[str, Any] = {
            "traceback": orig_tb if orig_tb else None, # Explicitly None if empty
            "code_context": code_context,
        }
        if data and isinstance(data, dict):
            for k, v in data.items():
                if k not in merged_data or merged_data[k] is None:
                    merged_data[k] = v
        # Remove keys with None values from merged_data for cleaner output
        merged_data = {k: v for k, v in merged_data.items() if v is not None}

 
        if code_context:
            # Check if code context is already included in the formatted message to avoid duplication
            if "Code Context:" not in formatted_message:
                formatted_message += f"\n\nCode Context:\n{code_context}"
            # Remove from data dict after appending to message (or if already present)
            if "code_context" in merged_data:
                 del merged_data["code_context"]
 

        return self._format_command_response(
            success=False,
            message=formatted_message, # The primary user-facing message
            error=error_msg if not isinstance(error_msg, str) else None,  # Optional: Error type/code for programmatic use
            status_value=status_value,
            status_field=status_field,
            data=merged_data if merged_data else None # Pass cleaned-up diagnostics dict
        )

    def _format_success_response(
        self, result: Union[CommandResult, Dict[str, Any]], status_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """Format a successful command response, preserving detailed info from dict results."""
        
        response = {}
        data_to_nest = {} # Store data to be nested under 'data' key

        if isinstance(result, dict):
            # Start by copying the entire dictionary
            response = result.copy()
            
            response["success"] = response.get("success", True)
            
            # Standardize message key
            if "message" not in response:
                response["message"] = response.get("formatted_message", "Operation successful.")
            if "formatted_message" in response and response["formatted_message"] != response["message"]:
                 response["message"] = response.pop("formatted_message")
            elif "formatted_message" in response:
                 response.pop("formatted_message")
                 
            # Set status field
            if status_field:
                if status_field not in response:
                    default_status = "success" # Generic default
                    if status_field == 'edit_status':
                        default_status = EditStatus.SUCCESS.value
                    elif status_field == 'eval_status':
                        default_status = EvalStatus.SUCCESS.value
                    elif status_field == 'profile_status':
                        default_status = ProfileStatus.SUCCESS.value
                    elif status_field == 'file_status':
                        default_status = FileStatus.SUCCESS.value
                    elif status_field == 'snapshot_status':
                        default_status = SnapshotStatus.SUCCESS.value # Assuming this exists
                    response[status_field] = default_status
            
 
            # Move everything NOT in the standard response keys into the data_to_nest dict
            standard_keys = ["success", "message", "error", status_field, "spend", "stdout", "stderr"]
            keys_to_move = [k for k in response if k not in standard_keys]
            for k in keys_to_move:
                data_to_nest[k] = response.pop(k)
     

        elif isinstance(result, CommandResult):
            response["success"] = getattr(result, "success", True)
            response_message = getattr(result, "message", "")
            # Remove redundant error line for eval_input responses
            if status_field == "eval_status" and isinstance(response_message, str):
                msg_lines = response_message.splitlines()
                msg_lines = [l for l in msg_lines if "Error: Starting evaluation..." not in l]
                response_message = "\n".join(msg_lines)
            response["message"] = response_message
            if status_field:
                response[status_field] = getattr(result, status_field, "success")
            if hasattr(result, 'error') and result.error:
                 response["error"] = result.error
            
 
            if hasattr(result, 'data') and result.data:
                data_to_nest = result.data if isinstance(result.data, dict) else {"value": result.data}
     
        else:
             logging.error(f"_format_success_response received unexpected type: {type(result)}")
             return {"success": False, "error": f"Internal error: Unexpected result type '{type(result).__name__}' during success formatting.", "message": "Operation status unclear due to internal error."}

        # Elevate stdout/stderr if they were nested in the original data
        if 'stdout' in data_to_nest and 'stdout' not in response:
            response['stdout'] = data_to_nest.pop('stdout')
        if 'stderr' in data_to_nest and 'stderr' not in response:
            response['stderr'] = data_to_nest.pop('stderr')
            
        # Add the potentially modified data_to_nest dict back under the 'data' key
        if data_to_nest:
             response['data'] = data_to_nest
                        
        # Add spend last
        try:
             response["spend"] = self.interface.state.spend
        except AttributeError:
             logging.warning("Could not retrieve spend state to add to response.")
             response["spend"] = None 
             
        logging.info(f"_format_success_response: Final response keys: {list(response.keys())}")
        return response

    def _save_snapshot_if_better(self, new_speedup: Optional[float]) -> tuple[bool, str]:
        """Saves a snapshot if the new speedup is better than the current best.
        Relies on the caller passing None for `new_speedup` if the evaluation was not `overall_valid`.
        """
        # Skip snapshot if the evaluation was deemed invalid (caller passed None for speedup)
        if new_speedup is None:
            logging.info("Evaluation result was invalid or failed. Skipping snapshot.")
            return False, "Evaluation failed or was invalid, snapshot not saved."

        current_best_speedup = self.interface.editor_state.get_best_speedup() # Assuming get_best_score will be updated -> CORRECTED
        
        logging.info(f"Comparing speedups: New={new_speedup}, Best={current_best_speedup}")
        
        # Handle initial case or if the old best was non-finite (like inf or None)
        # Need to safely handle potential None from get_best_score if not initialized
        current_best_is_finite = isinstance(current_best_speedup, (int, float)) and math.isfinite(current_best_speedup)
        should_save = not current_best_is_finite or new_speedup > current_best_speedup 
        
        if should_save:
            logging.info(f"New best speedup achieved: {new_speedup}. Saving snapshot.")
            updated = self.interface.editor_state.update_best_speedup(new_speedup)
            if updated:
                logging.info(f"Best speedup updated to {new_speedup}")
            else:
                logging.warning(f"Best speedup {new_speedup} is not better than current best {current_best_speedup}")
            # Now save snapshot with updated best_speedup in metadata
            save_result = self.interface.editor.save_snapshot()
            if save_result.get("success"):
                # Snapshot saved successfully
                return True, "Best speedup reached, state saved!\nAmong the 10+ LLMs we tested, your code did not rank in the top 3 for speed. Please use all available packages and tools to optimize its performance. Think outside the box!"
            else:
                error_msg = save_result.get("error", "Unknown error")
                logging.error(f"Error saving snapshot: {error_msg}")
                return False, f"Snapshot save failed: {error_msg}"
        else:
            logging.info(f"Speedup did not improve ({new_speedup} <= {current_best_speedup}). Snapshot not saved.")
            return False, "Speedup did not improve, snapshot not saved." 

        
        # Try to get performance_score from various locations

        mean_speedup = None
        overall_valid = False

        if "overall_valid" in eval_result:
            overall_valid = eval_result.get("overall_valid", False)
            if overall_valid:
                mean_speedup = eval_result.get("mean_speedup")
                
        # Fallback to checking data field (if structure varies)
        elif "data" in eval_result and isinstance(eval_result["data"], dict):
            eval_data = eval_result["data"]
            if "overall_valid" in eval_data:
                 overall_valid = eval_data.get("overall_valid", False)
                 if overall_valid:
                     mean_speedup = eval_data.get("mean_speedup")

        logging.info(f"After {command_source}: Overall Validity={overall_valid}, Mean Speedup={mean_speedup}")
        

        snapshot_saved, snapshot_message = self._save_snapshot_if_better(mean_speedup if overall_valid else None)

        
        # ... (rest of eval handler, potentially update response based on validity/speedup)

    def _process_edit_result_and_evaluate(
        self, 
        edit_result: CommandResult, 
        command_source: str, 
        file_name: str
    ) -> Dict[str, Any]:
        """Processes the result of an edit/delete, runs evaluation, and formats the response.

        Args:
            edit_result: The CommandResult from _handle_edit.
            command_source: The source command ('edit' or 'delete').
            file_name: The name of the file that was edited/deleted.

        Returns:
            The final formatted command response dictionary.
        """
        # If the edit/delete succeeded, return the formatted success response immediately
        if edit_result.success:
            return self._format_success_response(edit_result, status_field="edit_status")
        # Otherwise, handle the failure case below
        # If the initial edit/delete failed, format and return that error
        if not edit_result.get("success", False):
            logging.error(f"{command_source.capitalize()} failed: {edit_result.error}")
            
            # Extract potential context from the edit_result data
            data = edit_result.data if hasattr(edit_result, "data") and edit_result.data else {}
            proposed_code = data.get("proposed_code", "")
            current_code = data.get("current_code", "")
            # Build base formatted response
            formatted = self._format_command_response(
                success=False,
                message=edit_result.message,
                error=edit_result.error,
                status_value=edit_result.edit_status,
                status_field="edit_status",
                data={
                    "proposed_code": proposed_code,
                    "current_code": current_code,
                    "file_name": file_name,
                    **data
                }
            )
            # Propagate context and error details to top-level
            if "code_context" in data:
                formatted["code_context"] = data["code_context"]
            if "traceback" in data:
                formatted["traceback"] = data["traceback"]
            # Propagate diff snippets
            formatted["proposed_code"] = proposed_code
            formatted["current_code"] = current_code
            formatted["file_name"] = file_name
            return formatted

    def _execute_modify_command(
        self,
        command_str: ParsedCommand,
        source: str, # 'edit' or 'delete'
        status_field: str, # 'edit_status'
        status_value_on_failure: str, # EditStatus.FAILURE.value
        extract_new_content: Callable[[ParsedCommand], Optional[str]],
        post_process_error_context: bool # Note: This arg is not used
    ) -> Dict[str, Any]:
        """Handles file modification by calling the appropriate Editor method.

        Routes to editor.edit_file for edits and editor.delete_lines for deletes.
        Formats the response based on the editor's output.
        If edit/delete is successful, runs automatic comparative dataset evaluation.
        """
        try:
            file_name = command_str.args.get("file_name")
            if not file_name:
                return self._format_error_response("Missing file_name argument", f"handling {source} command", status_value_on_failure, status_field)
            
            # Convert file_name to Path for the editor methods
            from pathlib import Path # Ensure Path is imported
            try:
                file_path = Path(file_name)
            except TypeError:
                 return self._format_error_response(f"Invalid file_name: {file_name}", f"handling {source} command", status_value_on_failure, status_field)

            start_line = command_str.args.get("start_line")
            end_line = command_str.args.get("end_line")
            
            # Validate and convert line numbers (editor methods expect ints)
            try:
                # Convert potential string digits to int, handle None by defaulting to 0
                start_line_int = int(start_line) if isinstance(start_line, (str, int)) and str(start_line).isdigit() else (0 if start_line is None else start_line) 
                end_line_int = int(end_line) if isinstance(end_line, (str, int)) and str(end_line).isdigit() else (0 if end_line is None else end_line)
                
                # Ensure they are actually ints after conversion attempt
                if not isinstance(start_line_int, int):
                    raise ValueError(f"Start line '{start_line}' could not be converted to integer.")
                if not isinstance(end_line_int, int):
                     raise ValueError(f"End line '{end_line}' could not be converted to integer.")
                     
            except (ValueError, TypeError) as e:
                logging.error(f"Line number conversion error: {e}")
                return self._format_error_response(f"Invalid line numbers: start='{start_line}', end='{end_line}'", f"parsing {source} arguments", status_value_on_failure, status_field)

            # Call the appropriate editor method
            if source == "edit":
                new_content = extract_new_content(command_str) # Can be None or empty str
                logging.info(f"Calling editor.edit_file for '{file_path}', lines {start_line_int}-{end_line_int}")
                # edit_file handles None/empty new_content correctly for deletion/replacement
                edit_result = self.interface.editor.edit_file(
                    file_path=file_path, 
                    start_line=start_line_int, 
                    end_line=end_line_int, 
                    new_content=new_content
                )
            elif source == "delete":
                 logging.info(f"Calling editor.delete_lines for '{file_path}', lines {start_line_int}-{end_line_int}")
                 # delete_lines expects start_line >= 1
                 if start_line_int < 1:
                      return self._format_error_response(f"Start line must be >= 1 for delete (got {start_line_int})", f"handling {source} command", status_value_on_failure, status_field)
                 edit_result = self.interface.editor.delete_lines(
                    file_path=file_path, 
                    start_line=start_line_int, 
                    end_line=end_line_int
                )
            else:
                # Should not happen based on callers
                raise ValueError(f"Invalid source '{source}' for _execute_modify_command")

            # Process the result dictionary returned by the editor method
            # Ensure edit_result is a dictionary before proceeding
            if not isinstance(edit_result, dict):
                 logging.error(f"Editor method ({source}) returned non-dict result: {type(edit_result)}")
                 return self._format_error_response(
                     error_msg=f"Internal error: Editor returned unexpected result type ({type(edit_result).__name__})",
                     context=f"processing result from editor.{source}",
                     status_value=status_value_on_failure,
                     status_field=status_field
                 )
                 
            if edit_result.get("success", False):
                logging.info(f"Editor method {source} succeeded.")
                
                # Add reload of all modules after successful edit to ensure changes are recognized
                try:
                    from AlgoTuner.editor.editor_functions import reload_all_llm_src
                    logging.info(f"Reloading all modules after successful {source}")
                    reload_all_llm_src() # Call without arguments
                    logging.info(f"Successfully reloaded all modules after {source}")
                except Exception as e:
                    logging.error(f"Error reloading modules after {source}: {e}", exc_info=True)
                    # Don't let module reload failures stop the evaluation
                    logging.info(f"Continuing with evaluation despite module reload failure")
                


                # 1. Format the message about the successful edit/delete
                edit_success_msg = self.message_writer.format_edit_result(edit_result)
                if edit_success_msg is None: # Ensure it's a string
                    edit_success_msg = "(Edit/Delete operation details not available)"
                    logging.warning("edit_success_msg was None, defaulted to placeholder.")


                eval_msg = "(Evaluation did not run)" # Default message for eval_msg if try block is skipped or fails early
                eval_status_value = EvalStatus.FAILURE.value # Default status
                evaluation_details = {} # Default empty dict for data from eval
                eval_command_result_obj: Optional[CommandResult] = None # To store the CommandResult object

                # 2. Run COMPARATIVE evaluation and formatting via helper
                logging.info(f"Running evaluation after successful {source}")
                try:
                    # Run dataset evaluation for train subset after modifications
                    eval_response = self._run_and_format_dataset_eval("train", source)
                    # Merge edit context with eval response
                    merged = eval_response.copy()
                    # Prepend the edit success message
                    merged["message"] = f"{edit_success_msg}\n\n" + merged.get("message", "")
                    # Add edit_status
                    merged[status_field] = EditStatus.SUCCESS.value
                    # If eval failed, mark overall as failed
                    if not merged.get("success", False):
                        merged["success"] = False
                    return merged
                except Exception as e:
                    logging.error(f"Error during evaluation after {source}: {e}", exc_info=True)
                    # If evaluation fails, return success for the edit but note the evaluation failure
                    response = {
                        "success": True,  # Edit succeeded
                    "message": f"{edit_success_msg}\n\nWarning: Evaluation failed after edit: {str(e)}",
                    "eval_error": str(e),
                    "traceback": traceback.format_exc()
                    }
                    response[status_field] = EditStatus.SUCCESS.value
                    return response
 
            else:
                # Editor failed
                if source == "edit":
                    # Format edit failure with proposed and current code snippets
                    formatted_msg = self.message_writer.format_edit_result(edit_result)
                    # Return a response dict including the formatted message and edit status
                    response = {
                        "success": False,
                        "message": formatted_msg,
                    }
                    response[status_field] = status_value_on_failure
                    return response
                # For other sources (e.g., delete), use generic error formatting
                # Extract error details if available
                error_msg = edit_result.get("error")
                if edit_result.get("status") == "no_change":
                    error_msg = "Edit command failed: No changes were made to the file. Ensure the line range and content result in a modification."
                elif error_msg:
                    error_msg = f"Edit command failed: {error_msg}"
                return self._format_error_response(
                    error_msg=error_msg,
                    context=f"applying {source} to {file_name}",
                    status_value=status_value_on_failure,
                    status_field=status_field,
                    data={k: v for k, v in edit_result.items() if k != 'success'},
                )

        except IndexError as ie:
            # Specific handling for out-of-bounds errors
            error_msg = f"Invalid line numbers: {ie}. Please ensure the line numbers are within the file bounds (1 to {edit_result.get('file_length', '?') if 'edit_result' in locals() else '?'}) and start_line <= end_line."
            logging.error(f"IndexError during {source}: {error_msg}\n{traceback.format_exc()}")
            # Use the specific error message
            return self._format_error_response(
                error_msg=error_msg, # Pass specific message
                context=f"applying {source} to {file_name}", 
                status_value=status_value_on_failure, 
                status_field=status_field, 
                data={"traceback": traceback.format_exc()}
            )
        except Exception as e:
            # General error handling
            raw_error_msg = str(e)
            tb_str = traceback.format_exc()
            # Try to extract context for better error reporting
            try:
                temp_file_content = self.interface.editor.file_manager.read_file(file_path)
                temp_file_content = "".join(temp_file_content)
            except Exception:
                temp_file_content = "<Could not read file content>"
            
            error_data = extract_error_context(tb_str, raw_error_msg)
            refined_error_msg = error_data.get('error', raw_error_msg) # Use refined message if available
            # Clean file paths from refined error message
            cleaned_refined_error_msg = clean_build_output(refined_error_msg)
            
            logging.error(f"Error during {source}: {refined_error_msg}\n{tb_str}")
            # Format with potentially refined message and context
            return self._format_error_response(
                error_msg=cleaned_refined_error_msg, 
                context=f"applying {source} to {file_name}", 
                status_value=status_value_on_failure, 
                status_field=status_field, 
                data=error_data # Include extracted context data
            )



    def _execute_edit_command(self, command_str: ParsedCommand) -> Dict[str, Any]:
        """Handle the edit command. Modifies file contents and evaluates."""
        return self._execute_modify_command(
            command_str,
            source="edit",
            status_field="edit_status",
            status_value_on_failure=EditStatus.FAILURE.value,
            extract_new_content=lambda cmd: cmd.args.get("new_content"), # Use .get for safety
            post_process_error_context=True, # Note: This arg is not used by current _execute_modify_command
        )

    def _execute_eval_input_command(self, command_str: ParsedCommand) -> Dict[str, Any]:
        """Execute the eval_input command."""
        # Extract and normalize input (support 'input_str', fallback to 'body' and raw_text), handling None safely
        raw_in = command_str.args.get("input_str")
        if raw_in is None:
            raw_in = command_str.args.get("body") or ""
        input_str = raw_in.strip()
        if not input_str and getattr(command_str, "raw_text", None) is not None:
            raw = command_str.raw_text.strip()
            prefix = "eval_input"
            if raw.startswith(prefix):
                input_str = raw[len(prefix):].strip()
        # Missing input error
        if not input_str:
            return self._format_error_response(
                "Missing input for eval_input command",
                "handling eval_input command",
                EvalStatus.FAILURE.value,
                "eval_status"
            )
        # Delegate to unified pipeline
        return self._run_with_cast_and_format(
            input_str=input_str,
            runner=self._runner_eval_input,
            status_field="eval_status"
        )

    def _execute_revert_command(self) -> Dict[str, Any]:
        """Handle the revert command. Restores last snapshot.

        Returns:
            Dict containing success status, formatted message, and snapshot_status
        """
        try:
            logging.info("Executing revert command")
            
            # Call the editor's revert method directly
            result = self.interface.editor.revert()
            logging.info(f"Raw revert result: {result}")
            
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error during revert")
                if "No saved state to revert to" in error_msg:
                    error_msg = "No saved state to revert to. A snapshot is created automatically when you have successful test results."
                logging.error(f"Revert failed: {error_msg}")
                # Use unified error formatter for consistency
                return self._format_error_response(
                    error_msg,
                    "handling revert command",
                    SnapshotStatus.FAILURE.value,
                    "snapshot_status"
                )
            
            # If successful, force reload all modules to ensure we're using the reverted code
            try:
                from AlgoTuner.editor.editor_functions import reload_all_llm_src
                logging.info("Reloading all modules after successful revert")
                # The reload function gets code directory from environment
                reload_all_llm_src()
                logging.info("Successfully reloaded all modules after revert")
                
                # Also try to import the solver module to verify it works
                try:
                    importlib
                    if 'solver' in sys.modules:
                        importlib.reload(sys.modules['solver'])
                        logging.info("Successfully reloaded solver module")
                except Exception as e:
                    logging.error(f"Error reloading solver module: {e}")
                    
            except Exception as e:
                logging.error(f"Error reloading modules after revert: {e}")
            
            # Return success response with message
            success_msg = result.get("message", "Snapshot restored successfully")
            logging.info(f"Revert succeeded: {success_msg}")
            

            # Return the result from the revert operation directly
            # Use _format_success_response but pass the revert result dict
            # Ensure the message from the revert result is used.
            return self._format_success_response(
                 result, # Pass the revert result dictionary
                 status_field="snapshot_status" # Use snapshot_status for revert
            )

            
        except Exception as e:
            # Handle any exceptions during revert
            error_msg = str(e) or "Unknown error during revert"
            logging.error(f"Exception in revert command: {error_msg}")
            logging.error(traceback.format_exc())
            return self._format_error_response(
                error_msg,
                "handling revert command",
                SnapshotStatus.FAILURE.value,
                "snapshot_status"
            )

    def _handle_view_file(self, file_name: str, start_line: Optional[int]) -> Dict[str, Any]:
        """Handle the view_file command. Shows contents of a file.

        Args:
            file_name: The name of the file to view
            start_line: The starting line to view from

        Returns:
            Dict containing success status, formatted message, and file_status
        """
        logging.info(
            f"Executing view_file command on {file_name} from line {start_line or 1}"
        )
        try:
            # Call the editor.view_file method
            view_result = self.interface.editor.view_file(
                file_path=Path(file_name), 
                start_line=start_line or 1,
                lines_to_view=100
            )
            
            # Check if view_file was successful
            if view_result.get("success", False):
                # If the editor already provided a formatted message, use it
                if "message" in view_result:
                    return {
                        "success": True,
                        "message": view_result["message"],
                        "file_status": "success", 
                        "file_path": view_result.get("file_path", file_name)
                    }
                
                # Otherwise construct a message from formatted_content
                if "formatted_content" in view_result:
                    return {
                        "success": True,
                        "message": view_result["formatted_content"],
                        "file_status": "success",
                        "file_path": view_result.get("file_path", file_name)
                    }
                
                # Fallback if neither message nor formatted_content is present
                return {
                    "success": True,
                    "message": f"File {file_name} viewed successfully.",
                    "file_status": "success",
                    "file_path": view_result.get("file_path", file_name)
                }
            else:
                # Handle error case
                error_msg = view_result.get("error", f"Failed to view file {file_name}")
                return {
                    "success": False,
                    "error": error_msg,
                    "message": f"Error: {error_msg}",
                    "file_status": "error",
                    "file_path": view_result.get("file_path", file_name)
                }
                
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Error viewing file {file_name}: {e}"
            logging.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "message": f"Error: {error_msg}",
                "file_status": "error",
                "file_path": file_name
            }

    def _handle_ls(self, path: Optional[str]) -> Dict[str, Any]:
        """Handle the ls command. Lists contents of a directory.

        Args:
            path: The directory to list

        Returns:
            Dict containing success status, formatted message, and file_status
        """
        logging.info(f"Executing ls command on path: {path or 'root directory'}")
        try:
            if hasattr(self.interface.editor, 'list_files'):
                 # Note: list_files might not take a path argument, adjust if needed
                 list_result_raw = self.interface.editor.list_files()
                 
                 # Check if editor returned a dictionary
                 if isinstance(list_result_raw, dict):
                     # Prioritize using a 'message' key if the editor provided one
                     if list_result_raw.get('success') and 'message' in list_result_raw:
                         logging.info("Using 'message' directly from successful list_files result.")
                         return {
                             "success": True,
                             "message": list_result_raw['message'],
                             "status": list_result_raw.get('status', FileStatus.SUCCESS.value)
                         }
                     # Fallback: Try the assumed structure with 'listing' key
                     elif list_result_raw.get('success') and 'listing' in list_result_raw:
                         logging.info("Using 'listing' key from successful list_files result.")
                         listing_content = list_result_raw.get('listing', 'No listing available.')
                         return {
                             "success": True,
                             "message": f"Directory listing for '{path or '.'}':\n{listing_content}",
                             "status": FileStatus.SUCCESS.value
                         }
                     # Fallback 2: Check for a 'files' key (common pattern)
                     elif list_result_raw.get('success') and 'files' in list_result_raw:
                         logging.info("Using 'files' key from successful list_files result.")
                         file_list = list_result_raw.get('files', [])
                         if isinstance(file_list, list):
                             # Format: Just the list of files, one per line
                             formatted_listing = "\n".join(file_list)
                             # Prepend 'Files in dir:' header to the listing
                             return {
                                 "success": True,
                                 "message": f"File list:\n{formatted_listing}",
                                 "status": FileStatus.SUCCESS.value
                             }
                         else:
                             logging.warning("'files' key did not contain a list.")
                             # Fall through to generic dict string representation
                     # Handle failure dictionary from editor
                     elif not list_result_raw.get('success', True): # If success is explicitly False or missing
                         error_msg = list_result_raw.get('error', 'Unknown error listing directory')
                         return {"success": False, "error": error_msg, "status": FileStatus.FAILURE.value}
                     # Handle other successful dictionary structures (use raw dict as message?)
                     else:
                        logging.warning(f"list_files returned success dict without 'message' or 'listing': {list_result_raw}")
                        return {"success": True, "message": str(list_result_raw), "status": FileStatus.SUCCESS.value}
                 # Handle unexpected return type from editor
                 else:
                      return {"success": False, "error": f"Internal error: editor.list_files returned {type(list_result_raw)}", "status": FileStatus.FAILURE.value}
            else:
                 return {"success": False, "error": "Internal error: editor.list_files method not found", "status": FileStatus.FAILURE.value}
        except Exception as e:
             logging.error(f"Error in _handle_ls: {e}", exc_info=True)
             return {"success": False, "error": f"Error listing directory: {e}", "status": FileStatus.FAILURE.value, "traceback": traceback.format_exc()}

    def _execute_delete_command(self, command_str: ParsedCommand) -> Dict[str, Any]:
        """Handle the delete command. Deletes lines from a file."""
        # Delegate to unified modify pipeline for deletion with consistent output (no auto-evaluation)
        return self._execute_modify_command(
            command_str,
            source="delete",
            status_field="edit_status",
            status_value_on_failure=EditStatus.FAILURE.value,
            extract_new_content=lambda cmd: None,
            post_process_error_context=True,
        )

    def _execute_profile_command(self, command_str: ParsedCommand) -> Dict[str, Any]:
        """Handle the profile command. Profiles execution of code."""
        # Extract and normalize input
        input_str = command_str.args.get("input_str", "").strip()
        filename = command_str.args.get("filename")
        if not input_str:
                return self._format_error_response(
                "Missing input for profile command",
                "handling profile command",
                ProfileStatus.FAILURE.value,
                "profile_status"
            )
        # Delegate to unified pipeline
        return self._run_with_cast_and_format(
            input_str=input_str,
            runner=self._runner_profile,
            status_field="profile_status",
            filename=filename
        )

    def _execute_profile_line_command(self, command_str: ParsedCommand) -> Dict[str, Any]:
        """Execute profile_line command."""
        # Extract and normalize input
        input_str = command_str.args.get("input_str", "").strip()
        # Parse optional focus_line
        focus_line = None
        if 'focus_line' in command_str.args:
            try:
                focus_line = int(command_str.args['focus_line'])
            except (ValueError, TypeError):
                return self._format_error_response(
                    f"Invalid line number: {command_str.args['focus_line']}. Must be an integer.",
                    "parsing command",
                    ProfileStatus.FAILURE.value,
                    "profile_status"
                )
        focus_lines = [focus_line] if focus_line is not None else None
        # Delegate to unified pipeline
        return self._run_with_cast_and_format(
            input_str=input_str,
            runner=self._runner_profile_lines,
            status_field="profile_status",
            focus_lines=focus_lines
        )

    def _execute_profile_lines_command(self, command_str: ParsedCommand) -> Dict[str, Any]:
        """Execute profile_lines command."""
        # Extract and normalize input and focus_lines
        input_str = command_str.args.get("input_str", "").strip()
        focus_lines = command_str.args.get("focus_lines")
        filename = command_str.args.get("filename")
        if focus_lines is None:
                return self._format_error_response(
                "Missing focus_lines for profile_lines command",
                    "handling profile_lines command",
                    ProfileStatus.FAILURE.value,
                "profile_status"
            )
        if not input_str:
             return self._format_error_response(
                "Missing input for profile_lines command",
                 "handling profile_lines command",
                 ProfileStatus.FAILURE.value,
                "profile_status"
            )
        # Delegate to unified pipeline
        return self._run_with_cast_and_format(
            input_str=input_str,
            runner=self._runner_profile_lines,
            status_field="profile_status",
            focus_lines=focus_lines,
            filename=filename
        )

    def _run_full_cython_build(self) -> Dict[str, Any]:
        """Runs the full Cython build process for the project using pip install."""
        build_status = { # Default failure status
            "success": False,
            "message": "Full build failed before execution.",
            "error": "Build not attempted.",
            "exit_code": None,
            "stdout": "",
            "stderr": ""
        }
        logging.info("Attempting full Cython build via pip install...")

        try:
            code_dir = self.interface.editor.state.code_dir
            if not code_dir or not code_dir.exists():
                error_msg = "Code directory not found or accessible."
                logging.error(error_msg)
                build_status["error"] = error_msg
                build_status["message"] = "Build failed: Code directory missing."
                return build_status

            # Clean Cython build artifacts before full build
            try:
                for artifact in ['build', 'dist']:
                    artifact_path = code_dir / artifact
                    if artifact_path.exists():
                        shutil.rmtree(artifact_path)
                for egg in code_dir.glob('*.egg-info'):
                    shutil.rmtree(egg)
                for ext in ['*.c', '*.so', '*.pyd']:
                    for f in code_dir.rglob(ext):
                        f.unlink()
                logging.info("Cleaned Cython build artifacts before full Cython build.")
            except Exception as clean_err:
                logging.warning(f"Failed to clean build artifacts before full Cython build: {clean_err}")

            compile_cwd = str(code_dir)
            compile_timeout = 1800 # 30 minutes, same as in editor_functions
            compile_command = [
                sys.executable, # Use the current Python interpreter
                "-m",
                "pip",
                "install",
                ".", # Install from the current directory (project root)
                "--no-deps", # Don't reinstall dependencies
                "--force-reinstall", # Force reinstall to ensure recompilation
                "--no-cache-dir", # Avoid using cache
                # '--verbose', # Optional: Add verbosity for debugging
            ]

            logging.info(f"Running build command: {' '.join(compile_command)} in {compile_cwd}")

            process = subprocess.run(
                compile_command,
                cwd=compile_cwd,
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on non-zero exit
                timeout=compile_timeout,
            )

            build_status["exit_code"] = process.returncode
            build_status["stdout"] = process.stdout
            build_status["stderr"] = process.stderr

            if process.returncode == 0:
                build_status["success"] = True
                build_status["message"] = "Full Cython build successful."
                build_status["error"] = None
                logging.info(f"Full Cython build successful. Exit code: {process.returncode}")
            else:
                build_status["success"] = False
                error_msg = f"Full Cython build failed. Exit code: {process.returncode}"
                build_status["message"] = error_msg
                build_status["error"] = error_msg
                logging.error(f"{error_msg}")
                logging.error(f"Build stderr:\n{process.stderr}")

        except subprocess.TimeoutExpired as e:
            error_msg = f"Full Cython build timed out after {compile_timeout} seconds."
            logging.error(error_msg)
            build_status["success"] = False
            build_status["message"] = error_msg
            build_status["error"] = error_msg
            build_status["stdout"] = e.stdout.decode() if e.stdout else ""
            build_status["stderr"] = e.stderr.decode() if e.stderr else ""
        except Exception as e:
            error_msg = f"An unexpected error occurred during the full Cython build: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            build_status["success"] = False
            build_status["message"] = error_msg
            build_status["error"] = error_msg
            build_status["stderr"] = traceback.format_exc()

        return build_status

    def _execute_eval_command(self, command_str: ParsedCommand) -> Dict[str, Any]:
        """Execute the eval command.

        Args:
            command_str: Parsed command containing command type and arguments

        Returns:
            Dict containing success status, formatted message, and command-specific status fields
        """
        # Extract and normalize input (handling None safely)
        raw_in = command_str.args.get("input_str")
        if raw_in is None:
            raw_in = command_str.args.get("body") or ""
        input_str = raw_in.strip()
        # Fallback to raw_text only if provided (non-None)
        if not input_str and getattr(command_str, "raw_text", None) is not None:
            raw = command_str.raw_text.strip()
            parts = raw.split(None, 1)
            if len(parts) > 1:
                input_str = parts[1].strip()
        # Missing input error
        if not input_str:
            return self._format_error_response(
                "Missing input for eval command",
                "handling eval command",
                EvalStatus.FAILURE.value,
                "eval_status"
            )
        # Delegate to unified pipeline (use eval_input runner for single input evaluation)
        return self._run_with_cast_and_format(
            input_str=input_str,
            runner=self._runner_eval_input,
            status_field="eval_status"
        )

    def _execute_baseline_command(self, command_str: ParsedCommand) -> Dict[str, Any]:
        """Handle the reference command. Runs oracle evaluation on an input."""
        # Extract and normalize input (support 'input_str', fallback to 'body' and raw_text), handling None safely
        raw_in = command_str.args.get("input_str")
        if raw_in is None:
            raw_in = command_str.args.get("body") or ""
        input_str = raw_in.strip()
        if not input_str and getattr(command_str, "raw_text", None) is not None:
            raw = command_str.raw_text.strip()
            prefix = "reference"
            if raw.startswith(prefix):
                input_str = raw[len(prefix):].strip()
        # Missing input error
        if not input_str:
            return self._format_error_response(
                "Missing input for reference command",
                "handling reference command",
                EvalStatus.FAILURE.value,
                "eval_status"
            )
        # Delegate to unified pipeline with the new runner
        return self._run_with_cast_and_format(
            input_str=input_str,
            runner=self._runner_baseline,
            status_field="eval_status" # Use eval_status
        )

    def _run_and_format_dataset_eval(self, data_subset: str, command_source: str) -> CommandResult:
        """Run dataset evaluation (train subset) and return formatted response dict."""
        # Run the core evaluation runner
        result = self._runner_eval_dataset(data_subset="train", command_source=command_source)
        # Build user-facing message with optional contexts
        intro = "Starting evaluation..."
        msg = result.message or ""
        data = result.data or {}
        full_msg = f"{intro}\n\n{msg}"
        # Format according to success or failure
        if result.success:
            return self._format_success_response(
                CommandResult(
                    success=True,
                    message=full_msg,
                    error=None,
                    status=result.status,
                    data=data
                ),
                status_field="eval_status"
            )
        else:
            # On failure, report only the dataset evaluation result without the intro
            return self._format_error_response(
                error_msg=msg,
                context="running dataset evaluation",
                status_value=result.status,
                status_field="eval_status",
                data=data
            )

    def _unknown_command_error(self, command: str) -> Dict[str, Any]:
        """Handle unknown command error.

        Args:
            command: The unknown command

        Returns:
            Dict containing error status, formatted message, and command-specific status fields
        """
        error_msg = f"Unknown command: {command}"
        return self._format_error_response(
            error_msg,
            "handling unknown command",
            EvalStatus.FAILURE.value,
            "eval_status"
        )


    def _runner_eval_input(self, casted_input: Any, **runner_kwargs: Any) -> CommandResult:
        """Runner for eval_input command. Calls run_evaluation_on_input."""
        try:

            if not hasattr(self.interface, 'task_instance') or self.interface.task_instance is None:
                 raise RuntimeError("Task instance (self.interface.task_instance) not available.")
            task_instance = self.interface.task_instance

            
            command_source = runner_kwargs.get("command_source", "eval_input") # Pass command source if available

            # Call the specific eval function
            result_dict = run_evaluation_on_input(
                task_instance=task_instance, 
                problem_input=casted_input,
                command_source=command_source
            )
            

            self._log_code_dir_contents()

            # Always prepend "Starting evaluation..." for all evaluations
            original_message = result_dict.get("formatted_message", "")
            final_message = "Starting evaluation...\\n\\n" + original_message

            # Remove the specific redundant error line
            lines = final_message.splitlines()
            lines = [line for line in lines if "Error: Starting evaluation..." not in line]
            final_message = "\\n".join(lines)


            self._log_code_dir_contents()

            # Copy result data but exclude large problem objects to prevent memory issues
            data = result_dict.copy() if result_dict else {}
            # Note: problem_input removed to prevent OOM issues with large problems (e.g., 46MB SHA256 plaintexts)

            # For invalid_solution cases, treat as "success" for CommandResult so the 
            # message writer's special formatting gets applied instead of generic error formatting
            is_invalid_solution = (not result_dict.get("success", False) and 
                                 result_dict.get("error_type") == "invalid_solution")
            command_success = result_dict.get("success", False) or is_invalid_solution

            # Produce CommandResult with the enhanced message
            return CommandResult(
                success=command_success,
                message=final_message,
                error=result_dict.get("error") if not is_invalid_solution else None,
                data=data,
                status=EvalStatus.SUCCESS.value if command_success else EvalStatus.FAILURE.value
            )
            
        except Exception as e:
            logging.error(f"Error in _runner_eval_input: {e}", exc_info=True)
            tb_str = traceback.format_exc()
            # Check if we already have code context from the evaluation result
            existing_context = None
            if 'result_dict' in locals() and isinstance(result_dict, dict):
                existing_context = result_dict.get('code_context')
            
            if existing_context:
                context_info = {'code_context_snippet': existing_context, 'enhanced_error_message': str(e)}
            else:
                context_info = extract_error_context(tb_str, str(e))

            self._log_code_dir_contents()
            return CommandResult(
                success=False,
                error=f"Internal error during eval_input runner: {e}",
                status=EvalStatus.FAILURE.value,
                data={
                    "traceback": tb_str,
                    "code_context": context_info.get("code_context_snippet"),
                    "problem_input": casted_input  # Include problem input even for errors
                }
            )

    # but expecting dataset-style results. The eval command now properly uses _runner_eval_input.

    def _runner_profile_lines(self, casted_input: Any, focus_lines: Optional[List[int]] = None, **runner_kwargs: Any) -> CommandResult:
        """Runner for profile/profile_line/profile_lines. Calls TaskProfiler."""
        try:

            if not hasattr(self.interface, 'task_instance') or self.interface.task_instance is None:
                 raise RuntimeError("Task instance (self.interface.task_instance) not available.")
            task_instance = self.interface.task_instance

            
            profiler = TaskProfiler(task_instance)
            filename = runner_kwargs.get("filename")
            
            profile_result_dict = profiler.profile_solve(casted_input, focus_lines=focus_lines, filename=filename)
            
            if profile_result_dict.get("error_type") == "solver_not_found_error":
                # Special handling for solver_not_found_error from profiler
                # Ensure the message is JUST the generic message, and no data that could lead to context.
                return CommandResult(
                    success=False,
                    message=profile_result_dict.get("error"), # This is SOLVER_NOT_FOUND_GENERIC_MSG
                    error=profile_result_dict.get("error"),   # Keep error field for consistency
                    status=ProfileStatus.FAILURE.value,
                    data={
                        "error_type": "solver_not_found_error", 
                        "error": profile_result_dict.get("error"),
                        "problem_input": casted_input  # Include problem input for context
                    }
                )
            else:
                # Existing logic for other success/failure cases from profiler
                # TaskProfiler returns dict with success, profile_output, error etc.
                data = profile_result_dict.copy() if profile_result_dict else {}
                # Note: problem_input removed to prevent OOM issues with large problems (e.g., 46MB SHA256 plaintexts)
                
                return CommandResult(
                    success=profile_result_dict.get("success", False),
                    message=profile_result_dict.get("formatted_message", ""), # Uses formatter internally
                    error=profile_result_dict.get("error"),
                    data=data, # Pass full profiler dict with problem input
                    status=ProfileStatus.SUCCESS.value if profile_result_dict.get("success") else ProfileStatus.FAILURE.value
                )
            
        except Exception as e:
            logging.error(f"Error in _runner_profile_lines: {e}", exc_info=True)
            tb_str = traceback.format_exc()
            context_info = extract_error_context(tb_str, str(e))
            return CommandResult(
                success=False,
                error=f"Internal error during profile runner: {e}",
                status=ProfileStatus.FAILURE.value,
                data={
                    "traceback": tb_str,
                    "code_context": context_info.get("code_context_snippet"),
                    "problem_input": casted_input  # Include problem input even for errors
                }
            )

    # Alias for profile command (no focus lines)
    def _runner_profile(self, casted_input: Any, **runner_kwargs: Any) -> CommandResult:
        return self._runner_profile_lines(casted_input, focus_lines=None, **runner_kwargs)


    def _runner_baseline(self, casted_input: Any, **runner_kwargs: Any) -> CommandResult:
        """Runner for reference command. Calls run_oracle_evaluation."""
        try:
            # Access task instance
            if not hasattr(self.interface, 'task_instance') or self.interface.task_instance is None:
                 raise RuntimeError("Task instance (self.interface.task_instance) not available.")
            task_instance = self.interface.task_instance
            
            command_source = runner_kwargs.get("command_source", "reference")

            # Call the oracle evaluation function
            # Assuming run_oracle_evaluation takes similar args to run_evaluation_on_input
            # and returns a compatible dictionary.
            # Need to import run_oracle_evaluation if not already done.
            from AlgoTuner.utils.evaluator.runner import run_oracle_evaluation 
            
            result_dict = run_oracle_evaluation(
                task_instance=task_instance, 
                problem=casted_input, # Use the correctly casted input
            )
            

            if not isinstance(result_dict, dict):
                logging.error(f"_runner_baseline: run_oracle_evaluation did not return a dictionary (got {type(result_dict)}).")
                # Create a CommandResult indicating internal failure
                return CommandResult(
                    success=False,
                    message=f"Internal error: Baseline evaluation runner failed unexpectedly (return type: {type(result_dict)}).",
                    error=f"BaselineEvalInternalTypeError",
                    status=EvalStatus.FAILURE.value,
                    data={"raw_return_value": result_dict} # Include raw value for debugging
                )



            if not result_dict.get("success", False) and result_dict.get("error_type") in ["invalid_solution", "validation_error"]:
                logging.warning("Oracle evaluation succeeded but solution was invalid. Formatting with output/runtime/warning.")
                # Extract relevant info for the custom message
                invalid_output = result_dict.get("result")
                if invalid_output is None:
                    invalid_output = "[Output not available]"
                runtime_ms = result_dict.get("elapsed_ms")
                runtime_str = f"{runtime_ms:.5f} ms" if isinstance(runtime_ms, (int, float)) else "[Runtime not available]"
                
                # Construct the custom message
                custom_message_lines = [
                    f"Output: {invalid_output}",
                    f"Runtime: {runtime_str}",
                    "", # Blank line
                    "Warning: Solution is invalid. The input is probably improperly formatted."
                ]
                formatted_message = "\n".join(custom_message_lines)
                
                # Return success=True for the runner, but with the warning message
                # Keep the original failure details in the data field
                return CommandResult(
                    success=True, # Runner succeeded, but validation failed
                    message=formatted_message, 
                    error=None, # No runner error
                    data=result_dict, 
                    status=EvalStatus.SUCCESS.value # Indicate runner success
                )

            
            # If it wasn't an invalid solution error, format normally
            formatted_message = MessageWriter.format_oracle_result_from_raw(result_dict)
            
            return CommandResult(
                success=result_dict.get("success", False),
                message=formatted_message, # Use the explicitly formatted message
                error=result_dict.get("error"),
                data=result_dict, # Keep raw dict in data for internal use
                status=EvalStatus.SUCCESS.value if result_dict.get("success") else EvalStatus.FAILURE.value
            )
            
        except Exception as e:
            logging.error(f"Error in _runner_baseline: {e}", exc_info=True)
            tb_str = traceback.format_exc()
            context_info = extract_error_context(tb_str, str(e))
            return CommandResult(
                success=False,
                message=f"Internal error during baseline runner: {e}",
                error=f"BaselineRunnerError",
                status=EvalStatus.FAILURE.value,
                data={
                    "traceback": tb_str,
                    "code_context": context_info.get("code_context_snippet")
                }
            )



    def _runner_eval_dataset(self, data_subset: str, command_source: str) -> CommandResult:
        """Runner for eval command (no input). Calls evaluate_code_on_dataset. Always returns CommandResult."""
        try:

            code_dir_path = Path(self.interface.editor.state.code_dir)
            is_valid, validation_error_dict = validate_solver_setup(code_dir_path, command_source)
            if not is_valid:
                logging.error(f"Solver validation failed before dataset evaluation: {validation_error_dict}")
                error_msg_detail = validation_error_dict.get("error", "Solver validation failed due to an unspecified reason.")
                # Check if error already starts with "Error:" to avoid double prefixing
                if error_msg_detail.startswith("Error:"):
                    message = error_msg_detail
                else:
                    message = f"Solver validation failed: {error_msg_detail}"
                return CommandResult(
                    success=False,
                    message=message,
                    error=validation_error_dict.get("error_type", "SolverValidationError"),
                    status=EvalStatus.FAILURE.value,
                    data=validation_error_dict
                )


            # Access task instance
            if not hasattr(self.interface, 'task_instance') or self.interface.task_instance is None:
                 raise RuntimeError("Task instance (self.interface.task_instance) not available.")
            task_instance = self.interface.task_instance # This is the Task object
             
            logging.info(f"Running full dataset evaluation on '{data_subset}' subset for command '{command_source}'.")
            
            # Lazy baseline generation: generate baseline times once per subset
            baseline_attr = f"baseline_times_{data_subset}"
            baseline_times = getattr(self, baseline_attr, None)
            logging.info(f"DEBUG: baseline_attr={baseline_attr}, baseline_times cached: {baseline_times is not None}")
            if baseline_times is not None:
                # Check if cached baseline times are all zeros (indicates previous failed generation)
                sample_values = list(baseline_times.values())[:5]
                all_zeros = all(v == 0.0 for v in sample_values)
                logging.info(f"DEBUG: Cached baseline times sample: {sample_values}, all_zeros: {all_zeros}")
                if all_zeros:
                    logging.info(f"DEBUG: Forcing regeneration of baseline times due to all-zero values")
                    baseline_times = None  # Force regeneration
            
            if baseline_times is None:
                logging.info(f"Lazy baseline generation for subset '{data_subset}'")
                # Use memory-efficient approach - pass JSONL path instead of loading dataset
                data_dir = task_instance.data_dir or task_instance.get_task_directory()
                # Use a wildcard for the target time to support any cached timing (avoids NameError if variable is undefined)
                jsonl_filename = f"{task_instance.task_name}_T*ms_n*_size*_{data_subset}.jsonl"
                
                # Find the actual JSONL file
                import glob
                jsonl_pattern = os.path.join(data_dir, jsonl_filename)
                jsonl_files = glob.glob(jsonl_pattern)
                
                if jsonl_files:
                    jsonl_path = jsonl_files[0]  # Use first matching file
                    logging.info(f"Using memory-efficient JSONL path: {jsonl_path}")
                    
                    tmp_dir = os.environ.get('TEMP_DIR_STORAGE') or os.environ.get('TEMP') or tempfile.gettempdir()
                    import uuid
                    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars for brevity
                    baseline_file = os.path.join(tmp_dir, f"{task_instance.task_name}_{data_subset}_times_{unique_id}.json")
                    logging.info(f"Generating baseline times for '{data_subset}' to {baseline_file}")
                    
                    # Create a dummy iterator for compatibility, actual data will be streamed from JSONL
                    dummy_iterator = iter([])
                    # Use dev_runs for train, eval_runs for test
                    num_runs = DEV_RUNS if data_subset == "train" else EVAL_RUNS
                    # Pass test mode info to baseline evaluation
                    test_mode_baseline = hasattr(self.interface, 'max_samples') and self.interface.max_samples is not None
                    max_samples_baseline = self.interface.max_samples if test_mode_baseline else None
                    evaluate_baseline_dataset(task_instance, dummy_iterator, num_runs=num_runs, output_file=baseline_file, 
                                            jsonl_path=jsonl_path, test_mode=test_mode_baseline, max_samples=max_samples_baseline)
                else:
                    # Fallback to old method if JSONL not found
                    logging.warning(f"JSONL file not found with pattern {jsonl_pattern}, falling back to dataset loading")
                    gen_train, gen_test = task_instance.load_dataset()
                    stream_for_ids = gen_train if data_subset == "train" else gen_test
                    tmp_dir = os.environ.get('TEMP_DIR_STORAGE') or os.environ.get('TEMP') or tempfile.gettempdir()
                    import uuid
                    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars for brevity
                    baseline_file = os.path.join(tmp_dir, f"{task_instance.task_name}_{data_subset}_times_{unique_id}.json")
                    logging.info(f"Generating baseline times for '{data_subset}' to {baseline_file}")
                    # Use dev_runs for train, eval_runs for test
                    num_runs = DEV_RUNS if data_subset == "train" else EVAL_RUNS
                    # Pass test mode info to baseline evaluation
                    test_mode_baseline = hasattr(self.interface, 'max_samples') and self.interface.max_samples is not None
                    max_samples_baseline = self.interface.max_samples if test_mode_baseline else None
                    evaluate_baseline_dataset(task_instance, stream_for_ids, num_runs=num_runs, output_file=baseline_file,
                                            test_mode=test_mode_baseline, max_samples=max_samples_baseline)
                with open(baseline_file, 'r') as bf:
                    baseline_times = json.load(bf)
                logging.info(f"Loaded {len(baseline_times)} baseline times from {baseline_file}")
                # Log sample of baseline times for debugging
                sample_keys = list(baseline_times.keys())[:5]
                sample_times = {k: baseline_times[k] for k in sample_keys}
                logging.info(f"Sample baseline times: {sample_times}")
                setattr(self, baseline_attr, baseline_times)
                # Clean up temporary baseline file after loading
                try:
                    os.remove(baseline_file)
                    logging.debug(f"Cleaned up temporary baseline file: {baseline_file}")
                except Exception as e:
                    logging.warning(f"Failed to clean up baseline file {baseline_file}: {e}")
            # Load fresh dataset iterators for evaluation
            train_iter, test_iter = task_instance.load_dataset()
            dataset_to_evaluate = train_iter if data_subset == "train" else test_iter
            
            # Check if we're in test mode (moved earlier to use in baseline regeneration)
            test_mode = False
            if hasattr(self.interface, 'max_samples') and self.interface.max_samples is not None:
                test_mode = True
                logging.info(f"Test mode enabled with max_samples={self.interface.max_samples}")
            
            # Baseline completeness check
            try:
                _cache = list(dataset_to_evaluate)
                ds_len = len(_cache)
                dataset_to_evaluate = iter(_cache)
                if baseline_times is not None and len(baseline_times) < ds_len:
                    logging.warning("Incomplete baseline: %d entries vs %d items – regenerating", len(baseline_times), ds_len)
                    import glob, tempfile, uuid, json as _json
                    data_dir = task_instance.data_dir or task_instance.get_task_directory()
                    pattern = os.path.join(data_dir, f"{task_instance.task_name}_T*ms_n*_size*_{data_subset}.jsonl")
                    matches = glob.glob(pattern)
                    tmp_out = os.path.join(tempfile.gettempdir(), f"{task_instance.task_name}_{data_subset}_times_{uuid.uuid4().hex[:8]}.json")
                    regen_runs = DEV_RUNS if data_subset == "train" else EVAL_RUNS
                    if matches:
                        evaluate_baseline_dataset(task_instance, iter([]), num_runs=regen_runs, output_file=tmp_out, jsonl_path=matches[0], test_mode=test_mode)
                    else:
                        evaluate_baseline_dataset(task_instance, _cache, num_runs=regen_runs, output_file=tmp_out, test_mode=test_mode)
                    with open(tmp_out, "r") as ftmp:
                        baseline_times = _json.load(ftmp)
                    setattr(self, baseline_attr, baseline_times)
                    logging.info("Regenerated baseline – entries: %d", len(baseline_times))
            except Exception as _bc_err:
                logging.warning("Baseline completeness check failed: %s", _bc_err)
            # Use dev_runs for train, eval_runs for test
            num_runs = DEV_RUNS if data_subset == "train" else EVAL_RUNS
            
            eval_output = evaluate_code_on_dataset(
                task_obj=task_instance,
                dataset_iterable=dataset_to_evaluate,
                baseline_times=baseline_times,
                data_subset=data_subset,
                default_num_eval_runs=num_runs,
                test_mode=test_mode
            )
            
            # Check if eval stopped early due to validation error (evaluate_code_on_dataset can return a dict in this case)
            if isinstance(eval_output, dict) and eval_output.get("evaluation_stopped_early"):
                # Early exit on dataset item evaluation error: extract code context as in eval_input
                problem_id = eval_output.get("problem_id")
                err = eval_output.get('error', 'Unknown error during dataset item evaluation.')
                tb_str = eval_output.get('traceback', '')
                # Prefer code context provided by the evaluation output; fall back to re-extraction if necessary
                code_ctx = eval_output.get("code_context")
                if not code_ctx:
                    try:
                        context_info = extract_error_context(tb_str, err)
                        code_ctx = context_info.get("code_context_snippet")
                    except Exception:
                        code_ctx = None
                error_type = eval_output.get("error_type", "DatasetItemValidationError")
                # Return a CommandResult; formatting will append code context
                return CommandResult(
                    success=False,
                    message=err,
                    error=error_type,
                    status=EvalStatus.FAILURE.value,
                    data={
                        "problem_id": problem_id,
                        "traceback": tb_str,
                        "code_context": code_ctx,
                        "evaluation_type": "dataset",
                    }
                )
            
            # If it didn't stop early, eval_output is the results_list
            results_list = eval_output 
             
            if not isinstance(results_list, list):
                logging.error(f"_runner_eval_dataset expected a list from evaluate_code_on_dataset, got {type(results_list)}")
                return CommandResult(
                    success=False,
                    message="Internal error: Dataset evaluation returned unexpected data type.",
                    error="DatasetEvalTypeError",
                    status=EvalStatus.FAILURE.value,
                    data={"raw_return": results_list}
                )

            # Calculate success status and aggregated metrics
            num_total = len(results_list)
            num_success = sum(1 for item in results_list if item.get("success", False) and item.get("is_valid", False))
            overall_success = num_success == num_total # Only true if ALL problems succeeded
            aggregated_metrics = _calculate_aggregate_metrics(results_list)
            
            # Create a simple success message or get formatted stats
            if aggregated_metrics:
                dict_for_formatter = {
                    "aggregate_metrics": aggregated_metrics, 
                    "success": overall_success,
                    "evaluation_type": "dataset"
                }
                # Add invalid solution analysis if available
                invalid_ctx = getattr(results_list, "invalid_solution_analysis", None)
                if invalid_ctx:
                    dict_for_formatter["invalid_solution_analysis"] = invalid_ctx
                formatted_aggregate = self.message_writer.format_evaluation_result_from_raw(dict_for_formatter)
                summary_message = formatted_aggregate
            else:
                # Fallback to basic counting if metrics not available
                valid_count = aggregated_metrics.get("num_valid", num_success)
                total_count = aggregated_metrics.get("num_evaluated", num_total)
                summary_message = f"Dataset evaluation: {valid_count}/{total_count} problems valid."
            
            # Handle snapshot logic based on metrics
            snapshot_message = ""
            if command_source != "eval": # Only perform snapshot logic for edit/revert, not plain eval
                snapshot_saved = False
                
                # Only save snapshot if all solutions are valid
                if overall_success and aggregated_metrics:
                    mean_speedup = aggregated_metrics.get("mean_speedup")
                    
                    if mean_speedup is not None: # Only proceed if a valid speedup was determined
                        try:
                            snapshot_saved, snapshot_status_msg = self._save_snapshot_if_better(mean_speedup)
                            # Only display the status message (remove leading 'Snapshot saved ')
                            snapshot_message = snapshot_status_msg
                        except Exception as e:
                            logging.error(f"Error during snapshot check/save: {e}", exc_info=True)
                            snapshot_message = "Snapshot check/save failed due to internal error."
                else:
                    # Add a more specific message for unsuccessful evaluations
                    if aggregated_metrics:
                        num_timeout = aggregated_metrics.get("num_timeout", 0)
                        num_invalid = aggregated_metrics.get("num_invalid", 0)
                        num_errors = aggregated_metrics.get("num_errors", 0)
                        num_evaluated = aggregated_metrics.get("num_evaluated", 0)
                        
                        if num_timeout > 0 and num_timeout == num_evaluated:
                            snapshot_message = "Snapshot not saved - invalid solutions present"
                        elif num_errors > 0 and num_errors == num_evaluated:
                            snapshot_message = "Snapshot not saved - invalid solutions present"
                        elif num_invalid > 0:
                            snapshot_message = "Snapshot not saved - invalid solutions present"
                        else:
                            snapshot_message = "Snapshot not saved - invalid solutions present"
                    else:
                        snapshot_message = "Snapshot not saved - evaluation unsuccessful"
            # Skip snapshot for plain eval command

            # Combine messages
            final_message = summary_message
            if command_source != "eval" and snapshot_message:
                final_message += f"\n\n{snapshot_message}"

            self._log_code_dir_contents()
            
            # Build data payload with summary data instead of storing all per-problem results
            # This eliminates memory issues from large problem objects (e.g., 46MB SHA256 plaintexts)
            data_payload = {
                "total_problems": len(results_list),
                "successful_problems": sum(1 for r in results_list if r.get("success", False)),
                "aggregate_metrics": aggregated_metrics,
                "command_source": command_source,
                "evaluation_type": "dataset"
            }
            # Attach invalid_solution_analysis when available
            invalid_ctx = getattr(results_list, "invalid_solution_analysis", None)
            if invalid_ctx:
                data_payload["invalid_solution_analysis"] = invalid_ctx
            # Mark this as a dataset evaluation for error formatting
            data_payload["evaluation_type"] = "dataset"
            # Distinguish between evaluation process failure vs invalid solutions
            # The evaluation succeeded if problems were processed (even if solutions were invalid)
            num_processed = sum(1 for item in results_list if item.get("success", False))
            evaluation_succeeded = num_processed > 0 or num_total == 0  # Succeeded if any problems processed
            
            return CommandResult(
                success=evaluation_succeeded,
                message=final_message,
                error=None if evaluation_succeeded else "Evaluation process failed - no problems were processed successfully.",
                data=data_payload,
                status=EvalStatus.SUCCESS.value if evaluation_succeeded else EvalStatus.FAILURE.value
            )
            
        except Exception as e:
            logging.error(f"Error in _runner_eval_dataset: {e}", exc_info=True)
            tb_str = traceback.format_exc()
            code_context_snippet = "Context unavailable"
            try:
                # Check if we already have code context from the evaluation result
                existing_context = None
                if 'eval_output' in locals() and isinstance(eval_output, dict):
                    existing_context = eval_output.get('code_context')
                elif 'results_list' in locals() and isinstance(results_list, list) and results_list:
                    # Check the last failed result for code context
                    for result in reversed(results_list):
                        if isinstance(result, dict) and result.get('code_context'):
                            existing_context = result.get('code_context')
                            break
                
                if existing_context:
                    code_context_snippet = existing_context
                else:
                    context_info = extract_error_context(tb_str, str(e))
                    code_context_snippet = context_info.get("code_context_snippet", "Context unavailable")
            except NameError: 
                logging.warning("extract_error_context not available for error reporting in _runner_eval_dataset")
            except Exception as context_err:
                logging.warning(f"Failed to extract error context in _runner_eval_dataset: {context_err}")
            
            self._log_code_dir_contents()
            return CommandResult(
                success=False,
                message=f"Error: {e}",
                error="DatasetRunnerError",
                status=EvalStatus.FAILURE.value,
                data={
                    "traceback": tb_str,
                    "code_context": code_context_snippet,
                    "command_source": command_source
                }
            )


    def _handle_create_test_file(self, file_name: str, content: Optional[str] = None) -> Dict[str, Any]:
        """Handle the create_test_file command. Creates a test file for diagnostics.

        Args:
            file_name: The name/path of the test file to create
            content: Optional content to write to the file

        Returns:
            Dict containing success status, message, and other details
        """
        logging.info(f"Executing create_test_file command for file: {file_name}")
        try:
            if not file_name:
                return {
                    "success": False, 
                    "error": "Missing file name for create_test_file command", 
                    "status": FileStatus.FAILURE.value
                }
            
            content_to_write = content if content is not None else "This is a test file created for diagnostic purposes."
            
            # Call editor method
            if hasattr(self.interface.editor, "create_test_file"):
                result = self.interface.editor.create_test_file(file_name, content_to_write)
                
                if result.get("success", False):
                    return {
                        "success": True,
                        "message": result.get("message", f"Created test file at {file_name}"),
                        "status": FileStatus.SUCCESS.value,
                        "data": result
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("error", f"Failed to create test file {file_name}"),
                        "status": FileStatus.FAILURE.value,
                        "data": result
                    }
            else:
                return {
                    "success": False, 
                    "error": "Internal error: editor.create_test_file method not found", 
                    "status": FileStatus.FAILURE.value
                }
        except Exception as e:
            logging.error(f"Error in _handle_create_test_file: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Error creating test file: {e}",
                "status": FileStatus.FAILURE.value,
                "traceback": traceback.format_exc()
            }


    def _log_code_dir_contents(self):
        try:
            logging.info("Logging contents of CODE_DIR files after evaluation...")
            editor = self.interface.editor
            list_result = editor.list_files()
            
            if list_result.get('success') and 'files' in list_result:
                file_names = list_result['files']
                code_dir_path = editor.state.code_dir
                
                for filename in file_names:
                    try:
                        file_path = Path(filename) # Assume list_files gives relative path strings
                        # Read file using file_manager to handle absolute path conversion
                        lines = editor.file_manager.read_file(file_path) 
                        content = "".join(lines)
                        logging.info(f"CODE_DIR_FILE:_{filename}\n{content}")
                    except Exception as read_err:
                        logging.error(f"Could not read file {filename} for logging: {read_err}")
            else:
                logging.warning(f"Could not list files for logging. Result: {list_result}")
        except Exception as log_err:
            logging.error(f"Error during code directory content logging: {log_err}", exc_info=True)
