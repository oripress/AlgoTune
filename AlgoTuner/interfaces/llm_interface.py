import re
import logging
import os
import traceback
import numpy as np
from AlgoTuner.utils.file_helpers import load_file_content
from AlgoTuner.utils.command_helpers import find_command
from AlgoTuner.utils.code_helpers import extract_code_blocks
from AlgoTuner.editor.editor_functions import Editor, EditorState, Path, reload_all_llm_src
from AlgoTuner.utils.message_writer import MessageWriter
import litellm
from litellm import RateLimitError, APIError
from AlgoTuner.models.lite_llm_model import LiteLLMModel
from AlgoTuner.models.together_model import TogetherModel
from AlgoTuner.utils.evaluator import reload_all_llm_src, warmup_evaluator

from AlgoTuner.config.model_config import GlobalConfig, GenericAPIModelConfig
from pydantic import SecretStr
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from collections import deque
from threading import Lock

from AlgoTuner.interfaces.core import base_interface
import importlib
importlib.reload(base_interface)

from AlgoTuner.interfaces.core.message_handler import MessageHandler
from AlgoTuner.interfaces.core.spend_tracker import SpendTracker
from AlgoTuner.interfaces.commands.parser import CommandParser, ParsedCommand
from AlgoTuner.interfaces.commands.handlers import CommandHandlers
from AlgoTuner.interfaces.commands.types import CommandResult, COMMAND_PATTERNS, COMMAND_FORMATS
from AlgoTuner.interfaces.error_message import GENERIC_ERROR_MESSAGE

from AlgoTuner.config.loader import load_config
from AlgoTuner.utils.evaluator.main import run_evaluation_on_input, run_oracle_on_input, evaluate_code_on_dataset
from AlgoTuner.utils.isolated_benchmark import VALIDATION_OVERHEAD_FACTOR

import random
import time


class LLMInterface(base_interface.BaseLLMInterface):
    """
    Main LLM interface that coordinates all components.
    Inherits from BaseLLMInterface and adds command handling and message processing.
    """

    def __init__(
        self,
        model_config: GenericAPIModelConfig,
        global_config: GlobalConfig,
        model_name: str,
        task_instance,
        model_specific_config: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
    ):
        self.model_specific_config = model_specific_config if model_specific_config is not None else {}
        # Store default_params before calling super
        self.default_params = self.model_specific_config.get("default_params", {})
        self.max_samples = max_samples  # Store max_samples for test mode
        super().__init__(model_config, global_config, model_name, task_instance)

        self.message_handler = MessageHandler(self)
        self.spend_tracker = SpendTracker(self)
        self.command_handlers = CommandHandlers(self)

        if task_instance:
            def cli_eval_input(problem_input):
                cfg = load_config()
                timeout_ms = cfg.get("global", {}).get("evaluation", {}).get("timeout_ms", 10000)
                return run_evaluation_on_input(
                    self.task_instance, problem_input,
                            timeout_ms=timeout_ms,
                    command_source="eval_input"
                )
            def cli_eval_oracle(problem_input):
                cfg = load_config()
                timeout_ms = cfg.get("global", {}).get("evaluation", {}).get("timeout_ms", 10000)
                return run_oracle_on_input(
                    self.task_instance, problem_input,
                    timeout_ms=timeout_ms
                )
            def cli_eval_all(subset):
                train_iter, test_iter = self.task_instance.load_dataset()
                dataset_to_evaluate = train_iter if subset == "train" else test_iter
                
                # Check if we're in test mode based on max_samples
                test_mode = False
                if self.max_samples is not None:
                    test_mode = True
                    logging.info(f"Test mode enabled with max_samples={self.max_samples}")
                # Also check model's max_samples attribute (for DummyLLM compatibility)
                elif hasattr(self, 'model') and hasattr(self.model, 'max_samples') and self.model.max_samples is not None:
                    test_mode = True
                    logging.info(f"Test mode enabled via model.max_samples={self.model.max_samples}")
                
                logging.info(f"Calling evaluate_code_on_dataset with test_mode={test_mode}")
                
                return evaluate_code_on_dataset(
                    task_obj=self.task_instance,
                    dataset_iterable=dataset_to_evaluate,
                    data_subset=subset,
                    timeout_multiplier=1 + 1 + VALIDATION_OVERHEAD_FACTOR,  # warmup + timed + validation overhead
                    test_mode=test_mode
                )
            self.state.eval_executor = cli_eval_input
            self.state.oracle_executor = cli_eval_oracle
            self.state.eval_all_executor = cli_eval_all
            logging.info("Evaluation executors initialized and attached to state")

            from AlgoTuner.utils.profiler import TaskProfiler
            profiler = TaskProfiler(task_instance)
            profiler.profile       = profiler.profile_solve
            profiler.profile_lines = lambda focus_lines, problem_input: profiler.profile_solve(problem_input, focus_lines)
            self.state.profiler    = profiler
            logging.info("Profiler initialized and attached to state")

        if model_name != "human":
            self._setup_model()

    def _setup_model(self):
        """Set up the appropriate model based on configuration."""
        try:
            model_specific_params = self.model_specific_config.copy()
            should_drop_params = model_specific_params.pop('drop_params', False)
            model_provider = model_specific_params.pop('model_provider', None)
            model_specific_params.pop('api_key_env', None)
            model_specific_params.pop('spend_limit', None)

            init_params = self.model_params.copy()
            init_params.update(model_specific_params)

            if model_provider == "together":
                logging.info(f"Initializing TogetherModel with params: {init_params}")
                self.model = TogetherModel(**init_params)
                logging.info(
                    self.message_writer.format_system_message(
                        "TogetherModel initialized"
                    )
                )
            else:
                init_params['drop_call_params'] = should_drop_params
                
                logging.info(f"Initializing LiteLLMModel with params: {init_params}")
                if 'reasoning_effort' in model_specific_params:
                     logging.info(f"Model configured with reasoning_effort: {model_specific_params['reasoning_effort']}")
                if should_drop_params:
                     logging.info(f"Model configured to drop non-standard params for completion calls.")

                self.model = LiteLLMModel(**init_params)
                logging.info(
                    self.message_writer.format_system_message(
                        "LiteLLMModel initialized"
                    )
                )
        except Exception as e:
            error_msg = self.message_writer.format_error(str(e), "initializing model")
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def process_command(
        self,
        content,
        error_message=None,
        proposed_code=None,
        current_code=None,
        file_name=None,
        edit_status=None,
        snapshot_status=None,
        eval_status=None,
        problem_input=None,
    ) -> Dict[str, Any]:
        """
        Process a command from the LLM to update the content.

        Args:
            content: Command content
            error_message: Optional error message
            proposed_code: Optional proposed code
            current_code: Optional current code
            file_name: Optional file name
            edit_status: Optional edit status
            snapshot_status: Optional snapshot status
            eval_status: Optional eval status
            problem_input: Optional problem input that caused this evaluation

        Returns:
            Dict containing message and success state
        """
        if not content:
            logging.error("Content is empty")
            return {
                "success": False,
                "error": "Cannot process empty content",
                "message": "Cannot process empty content",
                "spend": self.state.spend,
            }

        content = str(content)

        # Check if this is a command
        parsed_command, error, total_blocks = CommandParser.parse(content)

        if error:
            # Handle structured error response
            if isinstance(error, dict) and not error.get("success", True):
                return self.command_handlers.handle_command(error)
            # Handle legacy string error format
            return self.command_handlers.handle_command(
                {"success": False, "error": error, "command": "unknown"}
            )

        if parsed_command:
            # Handle command and return full result to propagate evaluation output
            result = self.command_handlers.handle_command(parsed_command)
            # Extract code context for edit/delete propagation
            data = result.get("data", {}) or {}
            proposed_code = data.get("proposed_code", "")
            current_code = data.get("current_code", "") or data.get("old_content", "")
            file_name = data.get("file_name", "")
            # Return full result so evaluation pipeline output propagates
            return result

        # Not a command, process as regular message
        return self.message_handler.send_message(
            content,
            error_message=error_message,
            proposed_code=proposed_code,
            current_code=current_code,
            file_name=file_name,
            edit_status=edit_status,
            snapshot_status=snapshot_status,
            eval_status=eval_status,
            problem_input=problem_input,
        )

    def send_message(
        self,
        content: str,
        error_message: Optional[str] = None,
        proposed_code: Optional[str] = None,
        current_code: Optional[str] = None,
        file_name: Optional[str] = None,
        edit_status: Optional[str] = None,
        snapshot_status: Optional[str] = None,
        eval_status: Optional[str] = None,
        problem_input: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to the LLM and handle the response.

        Args:
            content: The message content. Must be a string, not a command result.
            error_message: Optional error message to include
            proposed_code: Optional code being proposed
            current_code: Optional current code state
            file_name: Optional file name for code context
            edit_status: Optional edit status from EditStatus enum
            snapshot_status: Optional snapshot status from SnapshotStatus enum
            eval_status: Optional evaluation status from EvalStatus enum
            problem_input: Optional problem input that caused this evaluation

        Returns:
            Dictionary containing response information
        """
        # Ensure content is a string
        if not isinstance(content, str):
            logging.warning(
                f"send_message received non-string content: {type(content)}. Converting to string."
            )
            content = str(content)

        # Check if this is a command
        parsed_command, error, total_blocks = CommandParser.parse(content)

        if error:
            # Handle structured error response
            if isinstance(error, dict) and not error.get("success", True):
                return self.command_handlers.handle_command(error)
            # Handle legacy string error format
            return self.command_handlers.handle_command(
                {"success": False, "error": error, "command": "unknown"}
            )

        if parsed_command:
            # Handle command and return full result to propagate evaluation output
            return self.command_handlers.handle_command(parsed_command)

        # Not a command, process as regular message
        return self.message_handler.send_message(
            content,
            error_message=error_message,
            proposed_code=proposed_code,
            current_code=current_code,
            file_name=file_name,
            edit_status=edit_status,
            snapshot_status=snapshot_status,
            eval_status=eval_status,
            problem_input=problem_input,
        )

    def update_spend(self, amount: float) -> Dict[str, Any]:
        """Update spend tracking."""
        return self.spend_tracker.update_spend(amount)

    def get_spend_status(self) -> Dict[str, float]:
        """Get current spend status."""
        return self.spend_tracker.get_spend_status()

    def format_spend_status(self) -> str:
        """Format current spend status for display."""
        return self.spend_tracker.format_spend_status()

    def check_limits(self) -> Optional[str]:
        """Check if any limits have been reached."""
        return self.spend_tracker.check_limits()

    def get_message_history(self):
        """Get the current message history."""
        return self.message_handler.get_message_history()

    def clear_message_history(self):
        """Clear the message history except for system message."""
        self.message_handler.clear_message_history()

    def get_current_code(self):
        """Get the current state of solver.py"""
        raw_view = self.editor.view_file(Path("solver.py"))
        formatted_view = self.message_writer.format_file_view_from_raw(raw_view)
        logging.info(f"[FILE VIEW] Current code view:\n{formatted_view}")
        return raw_view

    def _format_code_with_line_numbers(self, content, edited_lines=None):
        """Helper method to consistently format code with line numbers and preserve indentation."""
        return self.message_writer.format_code_with_line_numbers(content, edited_lines)

    def get_response(self):
        """Get a response from the LLM and update conversation history."""
        try:
            if limit_msg := self.check_limits():
                logging.warning(f"Spend limit reached: {limit_msg}")
                return None

            logging.debug("\nMessage History Before Truncation:")
            for i, msg in enumerate(self.state.messages):
                content = msg["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                logging.debug(f"{i+1}. {msg['role']}: {content}")
            token_limit = 4000
            # Check for context_length first (context window), then fall back to max_tokens (response limit)
            if hasattr(self, 'model_config'):
                if hasattr(self.model_config, 'context_length'):
                    config_limit = getattr(self.model_config, 'context_length', 4000)
                    if isinstance(config_limit, int) and config_limit > 0:
                        token_limit = config_limit
                        logging.debug(f"Using context_length from config: {token_limit}")
                    else:
                        logging.warning(f"Invalid context_length in config ({config_limit}), trying max_tokens.")
                        # Fall back to max_tokens if context_length is invalid
                        config_limit = getattr(self.model_config, 'max_tokens', 4000)
                        if isinstance(config_limit, int) and config_limit > 0:
                            token_limit = config_limit
                            logging.debug(f"Using max_tokens from config: {token_limit}")
                        else:
                            logging.warning(f"Invalid max_tokens in config ({config_limit}), using default {token_limit}.")
                elif hasattr(self.model_config, 'max_tokens'):
                    config_limit = getattr(self.model_config, 'max_tokens', 4000)
                    if isinstance(config_limit, int) and config_limit > 0:
                        token_limit = config_limit
                        logging.debug(f"Using max_tokens from config (no context_length): {token_limit}")
                    else:
                        logging.warning(f"Invalid max_tokens in config ({config_limit}), using default {token_limit}.")
                else:
                    logging.warning(f"No context_length or max_tokens in model_config, using default {token_limit}.")
            else:
                logging.warning(f"Could not retrieve model_config, using default {token_limit}.")

            if not hasattr(self, 'message_handler') or not hasattr(self.message_handler, '_prepare_truncated_history'):
                logging.error("Message handler or _prepare_truncated_history method not available")
                return None

            try:
                prepared_data = self.message_handler._prepare_truncated_history(
                    self.state.messages,
                    token_limit
                )
            except Exception as prep_error:
                logging.error(f"Error in _prepare_truncated_history: {prep_error}")
                raise

            relevant_messages = prepared_data['messages']
            summary_info = prepared_data['summary']

            log_summary = f"Sending {len(relevant_messages)}/{len(self.state.messages)} messages ({summary_info.get('final_token_count', 'N/A')} tokens). "
            log_summary += f"Essentials kept: {len(summary_info['kept_essential_indices'])}. "
            log_summary += f"Older included: {len(summary_info['included_older_indices'])} (truncated content: {len(summary_info['content_truncated_older_indices'])}). "
            log_summary += f"Older dropped: {len(summary_info['dropped_older_indices'])}."
            logging.info(log_summary)

            # Check if we have messages to send
            if not relevant_messages:
                logging.error("No relevant messages to send")
                return None

            if len(relevant_messages) > 1:
                logging.debug("Previous context being sent to LLM:")
                for msg in relevant_messages[:-1]:
                    content = msg["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    logging.debug(f"{msg['role']}: {content}")
            else:
                relevant_messages = self.state.messages
            try:
                last_msg = relevant_messages[-1].copy()
                if last_msg["role"] == "system":
                    logging.debug(
                        f"Sent to LLM (system message):\n{last_msg['content']}"
                    )
                else:
                    formatted_msg = last_msg["content"]
                    logging.info(f"Sent to LLM:\n{formatted_msg}")
                    
            except Exception as msg_log_error:
                logging.error(f"Error logging message details: {msg_log_error}")

            try:
                max_retries = 5
                base_delay = 2.0

                response = None
                for attempt in range(max_retries):
                    try:
                        response = self.model.query(relevant_messages)
                        break
                    except (RateLimitError, APIError) as e:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            logging.warning(
                                f"Transient LLM error ({type(e).__name__}): {e}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(delay)
                            continue
                        logging.error(
                            f"Exceeded max retries ({max_retries}) due to {type(e).__name__}: {e}"
                        )
                        raise
                    except Exception as model_error:
                        logging.error(f"Exception in model.query: {model_error}")
                        logging.error(f"Model query exception type: {type(model_error)}")
                        import traceback
                        logging.error(f"Model query full traceback:\n{traceback.format_exc()}")
                        raise

                if response is None:
                    logging.error("Failed to obtain a response from the model after retries.")
                    return None

                if not isinstance(response, dict):
                    logging.error(f"Invalid response type: {type(response)}, expected dict")
                    return None
                
                assistant_message = response.get("message", "").strip()
                cost = float(response.get("cost", 0.0))

                logging.info(f"Received from LLM:\n{assistant_message}")

                if cost > 0:
                    try:
                        spend_result = self.update_spend(cost)
                        logging.debug(f"Updated spend: {spend_result}")
                    except Exception as spend_error:
                        logging.error(f"Error updating spend: {spend_error}")

                return assistant_message

            except Exception as e:
                logging.error(f"Exception in get_response: {e}")
                import traceback
                logging.error(f"Full get_response exception traceback:\n{traceback.format_exc()}")
                if "ContentPolicyViolationError" in str(e) or "Invalid prompt: your prompt was flagged" in str(e):
                    logging.error("Content policy violation detected - marking task as failed due to policy violation")
                    self._content_policy_violation = True
                
                return None

        except Exception as e:
            logging.error(f"Exception in get_response: {e}")
            import traceback
            logging.error(f"Full get_response exception traceback:\n{traceback.format_exc()}")
            if "ContentPolicyViolationError" in str(e) or "Invalid prompt: your prompt was flagged" in str(e):
                logging.error("Content policy violation detected - marking task as failed due to policy violation")
                self._content_policy_violation = True
            
            return None

    def _get_safe_file_path(self, file_name: str) -> Path:
        """
        Safely convert a file name to a Path object with validation.
        Ensures the file path is within the workspace and properly formatted.
        """
        try:
            file_path = Path(file_name)
            abs_path = (self.editor_state.code_dir / file_path).resolve()
            if not str(abs_path).startswith(str(self.editor_state.code_dir)):
                error_msg = self.message_writer.format_error(
                    f"File path {file_name} attempts to access location outside workspace",
                    "validating file path",
                )
                logging.error(error_msg)
                raise ValueError(error_msg)
            return file_path
        except Exception as e:
            error_msg = self.message_writer.format_error(
                f"Invalid file path {file_name}: {e}", "validating file path"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _process_content(self, content: str) -> str:
        """
        Process content from code blocks to be used in edit commands.
        Preserves exact whitespace and indentation from the original content.
        """
        try:
            if not content:
                return content
            content = content.replace("\\n", "\n")
            content = content.replace('\\"', '"').replace('"', '"')
            if content.startswith("```"):
                code_blocks = extract_code_blocks(content)
                if code_blocks:
                    _, code = code_blocks[0]
                    return code
            first_non_space = content.lstrip()
            if first_non_space:
                for quote in ['"""', "'''", '"', "'"]:
                    if (
                        first_non_space.startswith(quote)
                        and content.rstrip().endswith(quote)
                        and not first_non_space[len(quote) :].lstrip().startswith(quote)
                    ):
                        leading_space = content[: content.find(first_non_space)]
                        inner_content = first_non_space[len(quote) : -len(quote)]
                        return leading_space + inner_content

            return content

        except Exception as e:
            error_msg = self.message_writer.format_error(
                f"Error processing content: {e}", "processing edit content"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _get_example_problem(self):
        """Helper to get an example problem for error messages."""
        train_data, _ = self.task_instance.load_dataset(
            t=self.task_instance.oracle_time_limit,
            train_size=1,
            test_size=1,
            random_seed=42,
        )
        try:
            first_item = next(train_data)
            return first_item["problem"]
        except StopIteration:
            logging.error("Failed to get example problem: load_dataset returned an empty generator even when requesting size 1.")
            try:
                return self.task_instance.generate_problem(n=1, random_seed=42)
            except Exception as gen_exc:
                logging.error(f"Failed to generate fallback example problem: {gen_exc}")
                return None

    def _check_spend_limit(
        self, additional_cost: float = 0.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Thread-safe check if adding additional_cost would exceed spend limit.
        Returns (can_spend, error_message).
        """
        with self.spend_lock:
            new_spend = self.state.spend + additional_cost
            if new_spend > self.spend_limit:
                msg = self.message_writer.format_budget_limit_reached()
                return False, msg
            return True, None

    def _update_spend(self, cost: float) -> bool:
        """
        Thread-safe spend update.
        Returns True if update was successful, False if it would exceed limit.
        """
        with self.spend_lock:
            new_spend = self.state.spend + cost
            if new_spend > self.spend_limit:
                return False
            self.state.spend = new_spend
            return True

    def _queue_message(self, message: dict):
        """Thread-safe message queueing."""
        with self.message_lock:
            self.message_queue.append(message)

    def _process_message_queue(self):
        """Thread-safe message queue processing."""
        with self.message_lock:
            while self.message_queue:
                message = self.message_queue.popleft()
                self.state.messages.append(message)

    def handle_function_call(self, assistant_message):
        """
        Handle a function call from the assistant.
        """
        # --- Start Re-indenting entire method body ---
        try:
            # Extract code blocks from the message
            code_blocks = extract_code_blocks(assistant_message)
            logging.debug(f"Found {len(code_blocks)} code blocks")

            # Store the original message for error reporting
            command_context = assistant_message[:100] + "..." if len(assistant_message) > 100 else assistant_message

            # Check for the special multiple commands marker
            if code_blocks and len(code_blocks) == 1 and code_blocks[0][0] == "MULTIPLE_COMMANDS":
                error_message = code_blocks[0][1]
                logging.warning(f"handle_function_call: Multiple command blocks detected")
                return {
                    "success": False,
                    "error": error_message,
                    "message": error_message,
                    "command": "multiple_commands",
                    "is_parsing_error": True,
                }

            # Check for errors in code block extraction, including our new warning about text after commands
            if code_blocks and code_blocks[0][0] == "ERROR":
                error_message = code_blocks[0][1]
                logging.error(f"Code block extraction error: {error_message}")
                result = {
                    "success": False,
                    "error": error_message,
                    "message": self.message_writer.format_command_parse_error(
                        error_message
                    ),
                }
                return result

            # If no code blocks found, return an appropriate error
            if not code_blocks:
                logging.debug("No code blocks found in message")
                error_message = GENERIC_ERROR_MESSAGE
                return {
                    "success": False,
                    "error": error_message,
                    "message": error_message,
                }

            # Get the first code block's content
            language, command = code_blocks[0]
            # Check if the command block is empty
            if not command.strip():
                logging.debug("Empty command block found")
                error_message = GENERIC_ERROR_MESSAGE
                return {
                    "success": False,
                    "error": error_message,
                    "message": error_message,
                }

            logging.debug(f"Processing command: {command[:100]}...")

            # Pass the full assistant_message to the parser
            parsed_command, error, total_blocks = CommandParser.parse(assistant_message)
            if error:
                logging.warning(f"CommandParser.parse returned error: {error}")

                # Check if this is a specific empty command or multiple command error
                # and use our consistent error message
                nested_error = error.get("error") # Get the nested error value
                is_bad_response_type = False
                if isinstance(error, dict) and nested_error:
                    if isinstance(nested_error, str):
                         is_bad_response_type = (
                            "Empty command" in nested_error or
                            "Multiple commands" in nested_error or
                            "text after command" in nested_error or
                            not nested_error.strip() # Check type before strip/in
                        )

                if is_bad_response_type:
                    error_message = GENERIC_ERROR_MESSAGE
                    return {
                        "success": False,
                        "error": error_message,
                        "message": error_message,
                    }

                # If it's a specific command validation error, we'll use the error directly
                final_error_message = "Unknown parse error" # Default
                if isinstance(error, str):
                    final_error_message = error
                elif isinstance(error, dict):
                    nested_error = error.get("error")
                    if isinstance(nested_error, str):
                        final_error_message = nested_error
                    elif isinstance(nested_error, dict):
                        # Handle the doubly nested case like the 'edit' validation error
                        final_error_message = nested_error.get("error", "Unknown nested error")
                    elif nested_error: # Handle non-str/non-dict nested errors?
                         final_error_message = str(nested_error)
                    else: # error is a dict, but has no 'error' key or it's None/empty
                         final_error_message = error.get("message", "Unknown dictionary error") # Use message as fallback

                result = {
                    "success": False,
                    "error": final_error_message, # Assign the extracted string
                    "command": parsed_command,
                }

                # Format the command message properly using the string template
                result["message"] = self.message_writer.format_command_parse_error(
                    template=final_error_message, command=command
                )

                return result

            # Dispatch the command
            try:
                # Execute the command using the handler
                result = self.command_handlers.handle_command(parsed_command)

                # --- REVISED: Return the result directly ---
                # The command handler now returns a fully formatted dictionary
                # (including the message with diffs/context on failure).
                # We should return this dictionary directly to preserve the formatting.
                if result is None:
                    logging.warning("Command handler returned None unexpectedly.")
                    # Return a generic error if the handler returns None
                    return {
                        "success": False,
                        "error": "Command handler returned None.",
                        "message": "Internal error: Command handler did not return a result.",
                    }

                # Log success or failure based on the result from the handler
                if result.get("success", False):
                    logging.debug(f"Command executed successfully: {result.get('message', '')[:100]}...")
                else:
                    # Log the *full* formatted message from the result on failure
                    log_msg = result.get('message', 'Unknown error')
                    logging.error(f"Command failed. Full formatted message:\n{log_msg}")

                # Return the complete result dictionary from the command handler
                return result
                # --- END REVISED ---
            except Exception as e:
                error = f"Error executing {command_context}: {str(e)}"
                logging.error(f"{error}\n{traceback.format_exc()}")
                return {
                    "success": False,
                    "error": error,
                    "command": command_context,
                    "is_execution_error": True,
                }

        except Exception as e:
            error = f"Error handling function call: {str(e)}"
            logging.error(f"{error}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": error,
                "message": self.message_writer.format_command_error(error),
            }
        # --- End Re-indenting entire method body ---

    def run(self):
        """Main loop for LLM interaction in code-only mode."""
        should_terminate = False

        while not should_terminate:
            # Check spend limit
            if self.check_limits():
                logging.debug(self.format_spend_status())
                break

            try:
                response = self.get_response()
                if response is None:
                    logging.warning(
                        self.message_writer.format_warning(
                            "No response received. Exiting the loop."
                        )
                    )
                    break

                # Ensure we have a string response
                if isinstance(response, dict):
                    response = response.get("message", "").strip()

                # Check for API or rate-limit errors
                if "API Error" in response or "Rate limit exceeded" in response:
                    logging.error(self.message_writer.format_api_error(response))
                    should_terminate = True
                    break

                # Add the LLM's response to history
                self.message_handler.add_message("assistant", response)

                # Handle the command and get result
                output = self.handle_function_call(response)
                if output is not None:
                    # Store the command result in history
                    self.message_handler.add_command_result(output)

            except (RateLimitError, APIError) as e:
                logging.error(self.message_writer.format_api_error(str(e)))
                should_terminate = True
                break
            except Exception as e:
                logging.error(
                    self.message_writer.format_error(
                        str(e), "unexpected error in run loop"
                    )
                )
                should_terminate = True
                break

        # After the loop, revert and do final evaluation if no critical error
        if not should_terminate:
            try:
                self.editor.restore_snapshot()
                logging.debug(
                    self.message_writer.format_system_message(
                        "Reverted changes after run completion."
                    )
                )
            except Exception as e:
                logging.error(
                    self.message_writer.format_error(str(e), "during revert_changes")
                )

    def run_human_mode(self):
        print(
            self.message_writer.format_system_message(
                "Entering human mode. Type 'exit' to quit."
            )
        )
        while True:
            user_input = input(
                self.message_writer.format_system_message("Enter command: ", "prompt")
            )
            if user_input.lower() == "exit":
                break
            output = self.handle_function_call(user_input)
            if output:
                print(output)

    def run_task(self):
        """Execute the task using the LLM model."""
        logging.info(self.message_writer.format_task_status("starting"))
        had_working_solution = False
        should_terminate = False
        api_error = None

        while not should_terminate:
            # Check spend limit
            if self.check_limits():
                logging.debug(self.format_spend_status())
                break

            try:
                response = self.get_response()
                if response is None:
                    logging.warning(
                        self.message_writer.format_warning(
                            "No response received. Exiting the loop."
                        )
                    )
                    break

                # Check for API errors
                if "API Error" in response or "Rate limit exceeded" in response:
                    api_error = response
                    logging.error(self.message_writer.format_api_error(response))
                    should_terminate = True
                    break

                # Add the LLM's response to history
                self.message_handler.add_message("assistant", response)

                # Handle the command and get result
                output = self.handle_function_call(response)
                if output is not None:
                    # Store the command result in history
                    self.message_handler.add_command_result(output)

            except (RateLimitError, APIError) as e:
                api_error = str(e)
                logging.error(self.message_writer.format_api_error(str(e)))
                should_terminate = True
                break
            except Exception as e:
                logging.error(
                    self.message_writer.format_error(str(e), "during task execution")
                )
                should_terminate = True
                break

        logging.info(self.message_writer.format_task_status("completed", "Attempting final snapshot restore and evaluation..."))

        # --- Final Evaluation Logic ---
        final_eval_success = False # Flag to track if final eval ran successfully
        snapshot_restored = False # Flag to track snapshot restoration status

        # Check if the loop terminated normally (not due to API error or other exception)
        if not should_terminate:
            logging.info(self.message_writer.format_task_status("completed: Attempting final snapshot restore and evaluation..."))
            try:
                # Always try to restore the best performing snapshot
                logging.info("Attempting to restore best performing snapshot...")
                restore_result = self.editor.restore_snapshot()
                if restore_result.get("success"):
                    snapshot_restored = True
                    logging.info(self.message_writer.format_system_message(f"Successfully restored best snapshot: {restore_result.get('message', '')}"))
                    # Reload modules AFTER successful restore
                    try:
                        from AlgoTuner.editor.editor_functions import reload_all_llm_src
                        reload_all_llm_src()
                        logging.info("Reloaded modules after successful snapshot restore.")
                    except Exception as reload_err:
                        logging.error(f"Error reloading modules after snapshot restore: {reload_err}")
                else:
                    # Handle specific "no snapshot" case vs other restore errors
                    error_msg = restore_result.get("error", "Unknown restore error")
                    if "No snapshot metadata found" in error_msg or "No snapshot directory found" in error_msg or "No saved state" in error_msg:
                        logging.info(self.message_writer.format_system_message("No best snapshot found to restore for final evaluation."))
                    else:
                        logging.error(self.message_writer.format_error(error_msg, "during final snapshot restore"))
            except Exception as e:
                logging.error(self.message_writer.format_error(str(e), "during final snapshot restore"))
                # Snapshot restoration failed, snapshot_restored remains False

            # Run final evaluation on the test dataset regardless of snapshot restore status
            try:
                logging.info("Running final evaluation on 'test' dataset...")
                # Use the specific dataset evaluation runner
                eval_result = self.command_handlers._runner_eval_dataset(
                    data_subset="test",
                    command_source="final_evaluation" # Indicate source
                )
                # Build readable message
                final_message = getattr(eval_result, 'message', None)
                if final_message is None:
                    final_message = str(eval_result)
                logging.info(f"Final Test Performance: {final_message}")

                # Determine success flag and extract metrics from various result forms
                metrics_candidate = None
                if isinstance(eval_result, dict):
                    final_eval_success = bool(eval_result.get('success', True))
                    metrics_candidate = eval_result.get('aggregate_metrics')
                elif isinstance(eval_result, list):  # AttributedList inherits list
                    # Dataset evaluation returns AttributedList; treat as success if not empty
                    final_eval_success = True
                    metrics_candidate = getattr(eval_result, 'aggregate_metrics', None)
                else:
                    final_eval_success = bool(getattr(eval_result, 'success', True))
                    if hasattr(eval_result, 'data'):
                        dat = eval_result.data
                        metrics_candidate = getattr(dat, 'aggregate_metrics', None) if not isinstance(dat, dict) else dat.get('aggregate_metrics')

                self._final_eval_metrics = metrics_candidate if metrics_candidate else None

                if self._final_eval_metrics:
                    logging.info(f"Stored final test metrics: {self._final_eval_metrics}")

                if self._final_eval_metrics:
                    mean_speedup = self._final_eval_metrics.get('mean_speedup')
                    label = "N/A" if mean_speedup is None else mean_speedup
                    logging.info(f"Final Test Speedup: {label}")

            except Exception as e:
                logging.error(self.message_writer.format_error(str(e), "during final test evaluation"))
                logging.info(
                    self.message_writer.format_system_message("Final Test Performance: Error during evaluation")
                )

        # --- FIX: Use self.task_instance.task_name ---
        task_display_name = getattr(self.task_instance, 'task_name', 'unknown')
        logging.info(f"LLM interface for task {task_display_name} executed.")
        # --- END FIX ---
        
        # Log the contents of every file in CODE_DIR
        try:
            # Get the aggregate metrics if available from final evaluation
            if final_eval_success and self._final_eval_metrics:
                mean_speedup = self._final_eval_metrics.get('mean_speedup')
                logging.info(f"Final Test Speedup (mean): {mean_speedup}")
            
            # Log contents of all files in CODE_DIR (always do this regardless of evaluation status)
            code_dir = os.environ.get("CODE_DIR", "llm_src")
            if os.path.exists(code_dir):
                for filename in os.listdir(code_dir):
                    file_path = os.path.join(code_dir, filename)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                                logging.info(f"FILE IN CODE DIR {filename}:\n{content}")
                        except Exception as e:
                            logging.error(f"Error reading file {filename}: {e}")
            else:
                logging.error(f"CODE_DIR path {code_dir} does not exist")
        except Exception as e:
            logging.error(f"Error during final file content logging: {e}", exc_info=True)
