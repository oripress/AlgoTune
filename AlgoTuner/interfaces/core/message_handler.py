import logging
import os
from typing import Dict, Any, List, Optional
from collections import deque
from threading import Lock
import litellm
import math
import traceback

from AlgoTuner.interfaces.core.base_interface import BaseLLMInterface
from AlgoTuner.utils.message_writer import MessageWriter
from AlgoTuner.utils.code_helpers import extract_code_blocks
from AlgoTuner.utils.error_helpers import get_error_messages_cached
from AlgoTuner.utils.code_diff import unified_diff


class MessageHandler:
    """Handles message processing and history management."""

    def __init__(self, interface: BaseLLMInterface):
        self.interface = interface
        self.message_queue = deque()
        self.message_lock = Lock()
        self.message_writer = MessageWriter()

    def add_message(self, role: str, content: Any) -> None:
        """Add a message to history with proper formatting.

        Args:
            role: The role of the message sender ("user", "assistant", "system")
            content: The message content (can be string or dict)
        """
        formatted_content = (
            content.get("message", str(content))
            if isinstance(content, dict)
            else str(content)
        )
        
        logging.info(
            f"Adding message - Role: {role}, Content: {formatted_content[:200]}..."
        )

        with self.message_lock:
            message_to_append = {"role": role, "content": formatted_content}
            self.interface.state.messages.append(message_to_append)

    def add_command_result(self, result: Any) -> None:
        """Add a command result to history with proper role and formatting.

        Args:
            result: The command execution result (can be string or dict)
        """
        # Increment messages_sent counter before getting budget status
        self.interface.state.messages_sent += 1

        with self.message_lock:
            # Extract the formatted message content from the result
            if isinstance(result, dict):
                message_to_send = result.get("message", str(result))
            else:
                message_to_send = str(result)

            # Get budget status and ALWAYS prepend it to ensure it's at the beginning
            budget_status = self.message_writer.get_formatted_budget_status(self.interface)
            
            # Check if this is an evaluation command that should get "Starting evaluation..." prefix
            should_add_starting_eval = (
                isinstance(result, dict) and 
                result.get("eval_status") is not None and 
                "edit_status" not in result and
                not message_to_send.startswith("Starting evaluation")
            )
            
            # Build the final message: budget first, then optional eval prefix, then content
            if should_add_starting_eval:
                final_message = f"{budget_status}\n\nStarting evaluation...\n\n{message_to_send}"
            else:
                final_message = f"{budget_status}\n\n{message_to_send}"
            
            # Debug logging
            logging.debug(f"BUDGET_DEBUG: Budget status: {budget_status}")
            logging.debug(f"BUDGET_DEBUG: Should add starting eval: {should_add_starting_eval}")
            logging.debug(f"BUDGET_DEBUG: Final message starts with: {final_message[:100]}...")

            # Decode any literal "\\n" sequences into real newlines so they render correctly
            final_message = final_message.replace('\\n', '\n')
            # Log the result structure for debugging (using the input 'result' dict)
            logging.debug(
                f"Command result structure: success={result.get('success', 'N/A')}, has_error={bool(result.get('error'))}"
            )
            if result.get("error"):
                 logging.debug(f"Error in command result: {result['error']}") # Log the raw error if present

            logging.debug(
                f"Adding command result to history: {final_message[:200]}..."
            )
            logging.debug(f"Full message being added to history: {final_message}")

            self.interface.state.messages.append(
                {"role": "user", "content": final_message}
            )

    def _format_problem_input_for_logging(self, problem_input: Any) -> str:
        """Format problem input for logging, summarizing large data structures."""
        if problem_input is None:
            return "None"
        
        # Handle different data types
        if isinstance(problem_input, (int, float, str, bool)):
            return str(problem_input)
        
        if isinstance(problem_input, list):
            if len(problem_input) <= 10:
                return str(problem_input)
            else:
                first_few = problem_input[:3]
                return f"list[{len(problem_input)}]: [{first_few[0]}, {first_few[1]}, {first_few[2]}, ...]"
        
        if isinstance(problem_input, dict):
            if len(problem_input) <= 3:
                return str(problem_input)
            else:
                keys = list(problem_input.keys())
                sample_items = []
                for i, key in enumerate(keys[:2]):
                    value = problem_input[key]
                    if isinstance(value, (list, tuple)) and len(value) > 5:
                        sample_items.append(f"'{key}': {type(value).__name__}[{len(value)}]")
                    else:
                        sample_items.append(f"'{key}': {repr(value)}")
                return f"dict[{len(problem_input)}]: {{{', '.join(sample_items)}, ...}}"
        
        if hasattr(problem_input, 'shape'):  # numpy arrays, torch tensors, etc.
            return f"{type(problem_input).__name__}{problem_input.shape}"
        
        if isinstance(problem_input, tuple):
            if len(problem_input) <= 5:
                return str(problem_input)
            else:
                first_few = problem_input[:3]
                return f"tuple[{len(problem_input)}]: ({first_few[0]}, {first_few[1]}, {first_few[2]}, ...)"
        
        # Fallback for other types
        try:
            str_repr = str(problem_input)
            if len(str_repr) <= 100:
                return str_repr
            else:
                return f"{type(problem_input).__name__}: {str_repr[:50]}..."
        except Exception:
            return f"{type(problem_input).__name__} object"

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
        """Send a message to the LLM and get its response.

        Args:
            content: The main message content
            error_message: Optional error message to include
            proposed_code: Optional proposed code changes
            current_code: Optional current code state
            file_name: Optional file name for code display
            edit_status: Optional edit operation status
            snapshot_status: Optional snapshot operation status
            eval_status: Optional evaluation status
            problem_input: Optional problem input that caused this evaluation

        Returns:
            Dict containing success status, response/error message, and spend amount
        """
        try:
            if self.interface.check_limits():
                return {
                    "success": False,
                    "error": self.interface.format_spend_status(),
                    "spend": self.interface.state.spend,
                }
            # Log problem input context if provided (before "Sent to LLM" message)
            if problem_input is not None:
                formatted_input = self._format_problem_input_for_logging(problem_input)
                logging.info(f"Input that caused this: {formatted_input}")
                # Additional prominent logging for test environments
                if os.environ.get("AGENT_MODE") == "1":
                    logging.info(f"TEST INPUT CONTEXT: The following test input caused this response: {formatted_input}")

            # Format message parts
            message_parts = []

            # 1. Format error message if present
            if error_message:
                formatted_error = self._format_error_message(error_message)
                if formatted_error:
                    # If it's already a properly formatted error with traceback, use as is
                    if error_message.startswith("Error:") and "Traceback" in error_message:
                        message_parts.append(error_message)
                    # Otherwise format according to type
                    elif "Traceback" in error_message:
                        message_parts.append(error_message)
                    else:
                        message_parts.append(formatted_error)

            # 2. Format code diff if both proposed and current codes are provided
            if proposed_code is not None and current_code is not None:
                from AlgoTuner.utils.code_diff import unified_diff
                diff_text = unified_diff(
                    current_code,
                    proposed_code,
                    fromfile=f"before {file_name or ''}",
                    tofile=f"after  {file_name or ''}",
                )
                message_parts.append(
                    self.message_writer.format_command_output(
                        stdout=diff_text,
                        command="Diff (current → proposed)",
                    )
                )
            else:
                # Fallback: show proposed and/or current code separately
                if proposed_code:
                    message_parts.append(f"Proposed code:\n```\n{proposed_code}\n```")
                if current_code:
                    if file_name:
                        message_parts.append(
                            self.message_writer.format_file_view(
                                file_name, current_code
                            )
                        )
                    else:
                        message_parts.append(f"Current code:\n```\n{current_code}\n```")
                # Add an extra newline after code display
                message_parts.append("")

            # 4. Add the main content
            if content:
                message_parts.append(content)

            # Join all parts with double newlines
            formatted_content = "\n\n".join(filter(None, message_parts))

            if not hasattr(self.interface, "get_response"):
                logging.error("Interface does not have get_response method")
                return {
                    "success": False,
                    "error": "Interface configuration error: missing get_response method",
                    "spend": self.interface.state.spend,
                }
            with self.message_lock:
                try:
                    # Increment messages_sent counter before getting budget status
                    self.interface.state.messages_sent += 1

                    # Get current budget status using the helper
                    budget_status = self.message_writer.get_formatted_budget_status(
                        self.interface
                    )

                    # Create the final message with budget status using MessageWriter
                    message_to_send = self.message_writer.format_message_with_budget(
                        budget_status, formatted_content
                    )

                    # Double check that budget status is present
                    if not message_to_send.startswith(budget_status):
                        message_to_send = f"{budget_status}\n\n{message_to_send}"

                    # Log for message construction
                    logging.debug(f"Sending formatted message: {message_to_send[:200]}...")

                    # For test environments, log the raw test input from the test file right before sending to LLM
                    if os.environ.get("AGENT_MODE") == "1":
                        if hasattr(self.interface, 'model') and hasattr(self.interface.model, 'inputs') and hasattr(self.interface.model, 'current_index'):
                            try:
                                current_idx = max(0, self.interface.model.current_index - 1)  # Use previous index since we haven't processed current yet
                                if current_idx < len(self.interface.model.inputs):
                                    raw_test_input = self.interface.model.inputs[current_idx].strip()
                                    task_name = getattr(self.interface.model, 'task_name', 'unknown')
                                    if raw_test_input:
                                        logging.info(f"[TEST INPUT] Raw test input from tests/inputs/{task_name}.txt that led to this error: {raw_test_input[:500]}")
                                    else:
                                        logging.info(f"[TEST INPUT] Raw test input from tests/inputs/{task_name}.txt that led to this error: [EMPTY]")
                            except Exception as e:
                                logging.debug(f"Could not extract raw test input context: {e}")

                    # --- LOGGING BEHAVIOUR CHANGE ---
                    # Emit a one-liner that contains the first non-empty line of the payload right after the
                    # prefix so test-parsers that split on "Sent to LLM:" reliably capture it.  We still
                    # emit the complete payload on the next line(s) for human debugging.

                    # Extract first non-empty line (after stripping leading newlines)
                    first_line = next((ln for ln in message_to_send.splitlines() if ln.strip()), "")
                    logging.info(f"Sent to LLM: {first_line}")
                    # Also dump the full message for completeness/debugging at DEBUG level
                    logging.debug(f"Full payload sent to LLM:\n{message_to_send}")

                    # Add the message WITH budget information to history
                    self.add_message("user", message_to_send)

                except Exception as format_error:
                    logging.error(f"Error during message formatting: {str(format_error) if format_error is not None else 'Unknown error'}")
                    raise

            # Get response from LLM
            try:
                response = self.interface.get_response()
                    
            except Exception as response_error:
                logging.error(f"Exception in get_response: {str(response_error) if response_error is not None else 'Unknown error'}")
                import traceback
                logging.error(f"Full traceback:\n{traceback.format_exc()}")
                raise
            if response is None:
                # Handle None response by adding empty message and returning error
                try:
                    error_msgs = get_error_messages_cached()
                    empty_response_msg = f"Empty response from model\n\n{error_msgs}"
                    logging.warning("Received empty response from LLM")
                    self.add_message("assistant", "")

                    # Format empty response message with budget
                    budget_status_after = self.message_writer.get_formatted_budget_status(
                        self.interface
                    )
                    formatted_empty_msg = self.message_writer.format_message_with_budget(
                        budget_status_after, empty_response_msg
                    )
                    if not formatted_empty_msg.startswith(budget_status_after):
                        formatted_empty_msg = f"{budget_status_after}\n\n{formatted_empty_msg}"

                    return {
                        "success": False,
                        "error": formatted_empty_msg,
                        "spend": self.interface.state.spend,
                    }
                except Exception as empty_error:
                    logging.error(f"Error handling None response: {empty_error}")
                    return {
                        "success": False,
                        "error": "Failed to handle empty response from LLM",
                        "spend": self.interface.state.spend,
                    }

            # Extract message from response
            try:
                if isinstance(response, dict):
                    message = response.get("message", "").strip()
                else:
                    message = str(response).strip()
            except Exception as extract_error:
                logging.error(f"Error extracting message: {extract_error}")
                raise

            # Handle empty message
            if not message or message.strip() == "":
                # Handle empty response by getting error messages
                error_msgs = get_error_messages_cached()
                empty_response_msg = f"Empty response from model\n\n{error_msgs}"
                logging.warning("Received empty response from LLM")
                self.add_message("assistant", "")
                
                # Format empty response message with budget
                budget_status_after = self.message_writer.get_formatted_budget_status(self.interface)
                formatted_empty_msg = self.message_writer.format_message_with_budget(
                    budget_status_after, empty_response_msg
                )
                if not formatted_empty_msg.startswith(budget_status_after):
                     formatted_empty_msg = f"{budget_status_after}\n\n{formatted_empty_msg}"

                return {
                    "success": False,
                    "error": formatted_empty_msg,
                    "spend": self.interface.state.spend,
                }

            # Log the received response
            logging.info(f"Received from LLM:\n{message}")

            # Add the assistant's response to history
            self.add_message("assistant", message)

            # Log the updated budget status after response
            logging.debug(self.interface.format_spend_status())
            return {
                "success": True,
                "message": message,
                "spend": self.interface.state.spend,
            }
            
        except Exception as e:
            logging.error(f"Exception in send_message: {e}")
            import traceback
            logging.error(f"Full exception traceback:\n{traceback.format_exc()}")
            # Handle other exceptions
            error_msgs = get_error_messages_cached()
            error_msg = f"Error sending message: {str(e)}\n\n{error_msgs}"
            
            return {
                "success": False,
                "error": error_msg,
                "spend": self.interface.state.spend,
            }

    def _format_error_message(self, error_message: str) -> Optional[str]:
        """Format error message based on type, avoiding duplication of content."""
        if not error_message:
            return None

        # Format error message based on type
        if "Invalid command format" in error_message:
            # Extract just the specific error without the generic format instructions
            error_lines = error_message.split("\n")
            specific_error = next(
                (
                    line
                    for line in error_lines
                    if "must be" in line
                    or "Invalid" in line
                    or "Expected:" in line
                    or "missing" in line
                ),
                error_lines[0],
            )
            return self.message_writer.format_error(specific_error)
        elif "File not found" in error_message:
            # Add context about file operations
            return self.message_writer.format_error(
                f"{error_message}\nNote: For new files, use edit command with lines: 0-0"
            )
        elif "line range" in error_message.lower():
            # Add context about line numbers
            return self.message_writer.format_error(
                f"{error_message}\nNote: Line numbers must be valid and end line must be >= start line"
            )
        else:
            return self.message_writer.format_error(error_message)

    def get_message_history(self) -> List[Dict[str, str]]:
        """Get the current message history."""
        return self.interface.state.messages.copy()

    def clear_message_history(self) -> None:
        """Clear the message history except for the system message."""
        with self.message_lock:
            system_message = next(
                (
                    msg
                    for msg in self.interface.state.messages
                    if msg["role"] == "system"
                ),
                None,
            )

            logging.info(
                f"Clearing message history. Preserving system message: {system_message is not None}"
            )

            self.interface.state.messages.clear()
            if system_message:
                self.interface.state.messages.append(system_message)

    # --- Helper function for custom truncation --- 
    def _prepare_truncated_history(self, full_history: List[Dict[str, str]], token_limit: int) -> Dict[str, Any]:
        """
        Prepares the message history for the LLM API call, applying custom truncation rules.

        Rules:
        1. Always include the initial system prompt (index 0).
        2. Always include the full content of the last 5 user and 5 assistant messages.
        3. Truncate older messages (between system prompt and last 10) to 100 characters.
        4. If total tokens still exceed the limit, remove oldest truncated messages until it fits.

        Args:
            full_history: The complete list of message dictionaries.
            token_limit: The maximum allowed token count for the target model.

        Returns:
            Dict containing:
                - 'messages': The final list of message dictionaries ready for the API call.
                - 'summary': Dict summarizing the truncation (indices are original).
                    - 'kept_essential_indices': Set of original indices kept (system + last 5 user/asst).
                    - 'included_older_indices': Set of original indices of older messages included (potentially truncated).
                    - 'content_truncated_older_indices': Set of original indices of older messages whose content was truncated.
                    - 'dropped_older_indices': Set of original indices of older messages dropped due to token limits.
        """
        num_messages = len(full_history)
        truncation_summary = { # <-- Initialize Summary
            'kept_essential_indices': set(),
            'included_older_indices': set(),
            'content_truncated_older_indices': set(),
            'dropped_older_indices': set()
        }
        if num_messages == 0:
            return { 'messages': [], 'summary': truncation_summary }

        # --- Step 2: Isolate Essential Messages ---
        # Handle Claude's special case: [user:".", system:"..."] vs normal [system:"..."]
        system_prompt_index = 0
        if (len(full_history) >= 2 and 
            full_history[0].get("role") == "user" and 
            full_history[0].get("content") == "." and 
            full_history[1].get("role") == "system"):
            # Claude model case: placeholder user message + system message
            system_prompt_index = 1
            system_prompt = full_history[1]
            logging.debug("Detected Claude model message structure (placeholder user + system)")
        elif full_history[0].get("role") == "system":
            # Normal case: system message first
            system_prompt = full_history[0]
            logging.debug("Detected normal message structure (system first)")
        else:
            logging.error("History does not start with a system prompt. Cannot apply custom truncation.")
            return { 'messages': full_history, 'summary': truncation_summary } # Fallback

        # Mark essential messages: Claude placeholder (if exists) + system prompt
        essential_indices_in_original = set()
        if system_prompt_index == 1:
            # Claude case: keep both placeholder user and system message
            essential_indices_in_original.update([0, 1])
        else:
            # Normal case: keep just the system message
            essential_indices_in_original.add(0)
        last_user_indices_in_original = []
        last_assistant_indices_in_original = []
        num_essential_recent = 5
        
        # Iterate backwards from the second-to-last message
        for i in range(num_messages - 1, 0, -1):
            # Stop searching if we've found enough of both roles
            if len(last_user_indices_in_original) >= num_essential_recent and \
               len(last_assistant_indices_in_original) >= num_essential_recent:
                break 

            message = full_history[i]
            role = message.get("role")

            if role == "user" and len(last_user_indices_in_original) < num_essential_recent:
                last_user_indices_in_original.append(i)
                essential_indices_in_original.add(i)
            elif role == "assistant" and len(last_assistant_indices_in_original) < num_essential_recent:
                last_assistant_indices_in_original.append(i)
                essential_indices_in_original.add(i)

        truncation_summary['kept_essential_indices'] = essential_indices_in_original.copy() # Store essential indices
        logging.debug(f"Summary: Kept essential indices: {sorted(list(truncation_summary['kept_essential_indices']))}")
        # --- End Step 2 ---

        # --- Step 3: Process Older Messages ---
        processed_older_messages_map = {} # Map original index to processed message
        original_indices_of_older = []
        for idx, original_msg in enumerate(full_history):
            if idx not in essential_indices_in_original: # Process only non-essential messages
                original_indices_of_older.append(idx) # Keep track of which indices were older
                processed_msg = original_msg.copy()
                content = processed_msg.get("content", "")
                if len(content) > 100:
                    processed_msg["content"] = content[:100] + "..."
                    truncation_summary['content_truncated_older_indices'].add(idx) # Track content truncation
                processed_older_messages_map[idx] = processed_msg
        logging.debug(f"Summary: Older indices with truncated content: {sorted(list(truncation_summary['content_truncated_older_indices']))}")
        # --- End Step 3 ---

        # --- Step 4: Assemble Candidate History ---
        candidate_history = []
        original_idx_to_candidate_idx = {}
        candidate_idx_to_original_idx = {}
        for original_idx in range(num_messages):
             current_candidate_idx = len(candidate_history)
             if original_idx in essential_indices_in_original:
                 message_to_add = full_history[original_idx]
                 candidate_history.append(message_to_add)
                 original_idx_to_candidate_idx[original_idx] = current_candidate_idx
                 candidate_idx_to_original_idx[current_candidate_idx] = original_idx
                 # Debug log essential messages
                 role = message_to_add.get('role', 'unknown')
                 content_preview = message_to_add.get('content', '')[:100]
                 logging.debug(f"Added essential message {original_idx}: role={role}, content={content_preview}...")
             elif original_idx in processed_older_messages_map:
                 message_to_add = processed_older_messages_map[original_idx]
                 candidate_history.append(message_to_add)
                 original_idx_to_candidate_idx[original_idx] = current_candidate_idx
                 candidate_idx_to_original_idx[current_candidate_idx] = original_idx
                 truncation_summary['included_older_indices'].add(original_idx) # Track initially included older messages
                 # Debug log older messages
                 role = message_to_add.get('role', 'unknown')
                 content_preview = message_to_add.get('content', '')[:100]
                 logging.debug(f"Added older message {original_idx}: role={role}, content={content_preview}...")

        logging.debug(f"Summary: Initially included older indices: {sorted(list(truncation_summary['included_older_indices']))}")
        logging.debug(f"Assembled candidate history with {len(candidate_history)} messages.")
        # --- End Step 4 ---

        # --- Step 5: Implement Token Counting ---
        current_token_count = 0
        model_name = self.interface.model_name
        if not model_name:
            logging.error("Model name not found on interface. Cannot calculate tokens.")
            return { 'messages': candidate_history, 'summary': truncation_summary } # Return candidate and partial summary

        try:
            current_token_count = litellm.token_counter(model=model_name, messages=candidate_history)
            logging.debug(f"Calculated initial token count: {current_token_count} (Limit: {token_limit})")
        except Exception as e:
            logging.error(f"Error calculating initial token count: {e}. Proceeding without token-based truncation.")
            return { 'messages': candidate_history, 'summary': truncation_summary } # Return candidate and partial summary
        # --- End Step 5 ---

        # --- Step 6: Apply Token-Based Truncation (Dropping Older Messages) ---
        removed_older_count = 0
        while current_token_count > token_limit:
            index_to_remove_in_candidate = -1
            original_idx_to_drop = -1
            # Find the first message *after* system prompt that is older
            for idx_candidate in range(1, len(candidate_history)):
                original_idx = candidate_idx_to_original_idx.get(idx_candidate)
                if original_idx is not None and original_idx in original_indices_of_older:
                    index_to_remove_in_candidate = idx_candidate
                    original_idx_to_drop = original_idx
                    break 

            if index_to_remove_in_candidate == -1:
                logging.warning(f"Cannot truncate further. No more older messages to remove. Limit {token_limit}, Count {current_token_count}")
                break

            try:
                removed_msg = candidate_history.pop(index_to_remove_in_candidate)
                removed_older_count += 1
                # Track dropped message
                truncation_summary['dropped_older_indices'].add(original_idx_to_drop)
                truncation_summary['included_older_indices'].discard(original_idx_to_drop)
                if original_idx_to_drop in truncation_summary['content_truncated_older_indices']:
                     truncation_summary['content_truncated_older_indices'].discard(original_idx_to_drop)
                
                # Update mappings (adjust indices > removed index)
                candidate_idx_to_original_idx.pop(index_to_remove_in_candidate)
                new_candidate_idx_to_original_idx = {}
                for old_cand_idx, orig_idx in candidate_idx_to_original_idx.items():
                     new_candidate_idx_to_original_idx[old_cand_idx - 1 if old_cand_idx > index_to_remove_in_candidate else old_cand_idx] = orig_idx
                candidate_idx_to_original_idx = new_candidate_idx_to_original_idx
                
                original_idx_to_candidate_idx.pop(original_idx_to_drop)
                for orig_idx, cand_idx in list(original_idx_to_candidate_idx.items()): # Iterate over copy for modification
                     if cand_idx > index_to_remove_in_candidate:
                          original_idx_to_candidate_idx[orig_idx] = cand_idx - 1

                logging.debug(f"Removed older message (Original Index: {original_idx_to_drop}, Candidate Index: {index_to_remove_in_candidate})")
                # Recalculate token count
                current_token_count = litellm.token_counter(model=model_name, messages=candidate_history)
                logging.debug(f"New token count after removal: {current_token_count}")
            except Exception as e: # Catch potential errors during removal or recount
                logging.error(f"Error during token-based truncation removal: {e}")
                break # Stop truncation if error occurs

        logging.debug(f"Summary: Final included older indices: {sorted(list(truncation_summary['included_older_indices']))}")
        logging.debug(f"Summary: Dropped older indices: {sorted(list(truncation_summary['dropped_older_indices']))}")
        logging.info(f"Final history size before placeholder: {len(candidate_history)} messages, {current_token_count} tokens. Removed {removed_older_count} older messages by token limit.")
        # --- End Step 6 ---

        # --- Step 7: Add Placeholder if Truncation Occurred & Finalize ---
        final_history = candidate_history
        placeholder_added = False
        
        if removed_older_count > 0:
            placeholder_message = {
                "role": "system", 
                "content": "[Older conversation history truncated due to context length limits]"
            }
            # Insert after the initial system prompt
            final_history.insert(1, placeholder_message)
            placeholder_added = True
            logging.info("Inserted truncation placeholder message.")

            # Recalculate token count after adding placeholder
            try:
                current_token_count = litellm.token_counter(model=model_name, messages=final_history)
                logging.debug(f"Token count after adding placeholder: {current_token_count}")

                # If placeholder pushed us over the limit, remove the *new* oldest truncated message (at index 2)
                if current_token_count > token_limit:
                    logging.warning(f"Adding placeholder exceeded limit ({current_token_count}/{token_limit}). Removing oldest truncated message to compensate.")
                    # Find the first non-essential message after the placeholder (index > 1)
                    index_to_remove_after_placeholder = -1
                    original_idx_to_remove_again = -1
                    for idx_candidate in range(2, len(final_history)):
                        original_idx = candidate_idx_to_original_idx.get(idx_candidate -1) # Adjust index due to placeholder insertion
                        # Check if this original index belonged to an older message
                        if original_idx is not None and original_idx in original_indices_of_older: 
                             index_to_remove_after_placeholder = idx_candidate
                             original_idx_to_remove_again = original_idx
                             break
                             
                    if index_to_remove_after_placeholder != -1:
                        removed_for_placeholder = final_history.pop(index_to_remove_after_placeholder)
                        # Update summary (important: this index was previously 'included')
                        truncation_summary['dropped_older_indices'].add(original_idx_to_remove_again)
                        truncation_summary['included_older_indices'].discard(original_idx_to_remove_again)
                        if original_idx_to_remove_again in truncation_summary['content_truncated_older_indices']:
                             truncation_summary['content_truncated_older_indices'].discard(original_idx_to_remove_again)
                        logging.info(f"Removed message at index {index_to_remove_after_placeholder} to make space for placeholder.")
                        # Recalculate final count one last time
                        current_token_count = litellm.token_counter(model=model_name, messages=final_history)
                    else:
                        logging.error("Could not remove an older message to make space for placeholder, limit might be exceeded.")
                        # Optionally, remove the placeholder itself if absolutely necessary
                        # final_history.pop(1)
                        # placeholder_added = False

            except Exception as e:
                logging.error(f"Error recalculating token count after adding placeholder: {e}. Placeholder might cause limit exceedance.")
        
        # Add final token count to summary for logging in send_message
        truncation_summary['final_token_count'] = current_token_count
        # --- End Step 7 ---

        return { 'messages': final_history, 'summary': truncation_summary }
    # --- END _prepare_truncated_history ---
