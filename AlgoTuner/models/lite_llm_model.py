import litellm
from litellm.exceptions import RateLimitError, APIError, InternalServerError
import logging
import time
import random
from AlgoTuner.utils.message_writer import MessageWriter
from AlgoTuner.utils.error_helpers import get_error_messages_cached


class LiteLLMModel:
    def __init__(self, model_name: str, api_key: str, drop_call_params: bool = False, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.drop_call_params = drop_call_params # Store the flag
        self.message_writer = MessageWriter()
        
        # Remove max_tokens if present, as it's often miscalculated and invalid
        kwargs.pop('max_tokens', None) 

        # Filter out configuration-only parameters that shouldn't be sent to API
        config_only_params = {'modify_params', 'drop_params'}
        self.additional_params = {k: v for k, v in kwargs.items() if k not in config_only_params}
        logging.info(f"LiteLLMModel initialized. Drop Params: {self.drop_call_params}. Additional Params: {self.additional_params}")

    def query(self, messages: list[dict[str, str]]) -> dict:
        # Retry configuration
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                return self._execute_query(messages)
            except RateLimitError as e:
                # Handle 429 rate-limit responses separately so they are always considered retryable
                retry_after = getattr(e, "retry_after", None)
                if retry_after is not None:
                    try:
                        # Some SDKs return it as a string header value
                        delay = float(retry_after)
                    except (TypeError, ValueError):
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                else:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)

                if attempt < max_retries - 1:
                    logging.warning(
                        f"Rate limit exceeded. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    continue
                # Out of attempts – propagate the error
                logging.error(self.message_writer.format_api_error("Rate limit exceeded after max retries."))
                raise e
            except (InternalServerError, APIError) as e:
                # Check if this is a retryable error (overloaded or similar)
                is_retryable = self._is_retryable_error(e)
                
                if is_retryable and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"LiteLLM API returned retryable error: {str(e)}. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    # Not retryable or max retries reached
                    if is_retryable:
                        logging.error(f"LiteLLM API retryable error after {max_retries} retries: {e}")
                    else:
                        logging.error(f"LiteLLM API non-retryable error: {e}")
                    raise e
            except Exception as e:
                # Other exceptions are not retryable
                logging.error(f"Error in litellm call: {e}")
                logging.error(f"Error in model call: {str(e)}\n\n{get_error_messages_cached()}")
                raise e
        
        # Should never reach here
        raise Exception("Exhausted all retry attempts")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable (overloaded, server errors, etc.)"""
        error_str = str(error).lower()
        
        # Check for specific overloaded error patterns
        retryable_patterns = [
            "overloaded",
            "server error",
            "503",
            "502", 
            "504",
            "500",  # Internal server error
            "timeout",
            "connection",
            "429",  # Rate limit
            "rate limit",
        ]
        
        return any(pattern in error_str for pattern in retryable_patterns)

    def _execute_query(self, messages: list[dict[str, str]]) -> dict:
        try:
            # Debug logging of context
            if len(messages) > 1:
                logging.debug("Previous context being sent to LLM:")
                for msg in messages[:-1]:
                    logging.debug(
                        self.message_writer.format_message_to_llm(
                            f"{msg['role']}: {msg['content']}"
                        )
                    )
            last_msg = messages[-1]
            logging.debug(
                self.message_writer.format_message_to_llm(
                    f"{last_msg['role']}: {last_msg['content']}"
                )
            )

            if "deepseek" in self.model_name.lower():
                if len(messages) == 1 and messages[0]["role"] == "system":
                    messages.append({"role": "user", "content": "Proceed."})
                    logging.debug("Appended dummy user message for Deepseek initial system message.")

            system_prompt_content = None
            completion_messages = messages
            
            # Debug logging for input messages
            logging.debug(f"Input messages count: {len(messages)}")
            for i, msg in enumerate(messages):
                logging.debug(f"Message {i}: role='{msg.get('role', 'MISSING')}', content_length={len(str(msg.get('content', '')))}")
            
            if self.model_name.startswith("vertex_ai/") or self.model_name.startswith("gemini/"):
                system_msgs = [m for m in messages if m["role"] == "system"]
                chat_msgs = [m for m in messages if m["role"] != "system"]

                logging.debug(f"Found {len(system_msgs)} system messages and {len(chat_msgs)} chat messages")

                if system_msgs:
                    # Concatenate system messages into a single prompt
                    system_prompt_content = "\n".join(m["content"] for m in system_msgs)
                    logging.debug(f"Extracted system prompt for Gemini model: {system_prompt_content[:100]}...")
                
                # Use only non-system messages for the main messages list if system prompt was extracted
                if system_prompt_content and chat_msgs:
                    completion_messages = chat_msgs
                    logging.debug("Using non-system messages for Gemini completion messages.")
                elif system_prompt_content and not chat_msgs:
                    # If only system message exists, send its content as the first user message
                    # along with using the system_prompt parameter.
                    completion_messages = [{'role': 'user', 'content': system_prompt_content}]
                    logging.debug("Only system prompt found. Sending its content as first user message alongside system_prompt.")
                # If no system message, completion_messages remains the original messages list
            
            # Ensure we never have empty messages
            if not completion_messages:
                logging.warning("No completion messages after processing - this will cause API error")
                # Create a minimal message to prevent empty content error
                completion_messages = [{'role': 'user', 'content': 'Hello'}]
                logging.debug("Added fallback user message to prevent empty content error")
            
            logging.debug(f"Final completion_messages count: {len(completion_messages)}")
            for i, msg in enumerate(completion_messages):
                logging.debug(f"Final message {i}: role='{msg.get('role', 'MISSING')}', content_length={len(str(msg.get('content', '')))}")

            completion_params = {
                "model": self.model_name,
                "messages": completion_messages, # Use potentially modified message list
                "api_key": self.api_key,
                "timeout": 600,  # 10 minutes for deep thinking models
            }
            # Add system prompt if extracted for Vertex AI
            if system_prompt_content:
                completion_params["system_prompt"] = system_prompt_content
            
            if self.additional_params:
                # Handle Gemini thinking parameters specially
                if self.model_name.startswith("gemini/") and any(k in self.additional_params for k in ['thinking_budget', 'include_thoughts']):
                    # Convert thinking_budget and include_thoughts to the thinking parameter format
                    thinking_budget = self.additional_params.get('thinking_budget', 32768)
                    include_thoughts = self.additional_params.get('include_thoughts', True)
                    
                    # Create the thinking parameter in the format LiteLLM expects
                    completion_params['thinking'] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget
                    }
                    
                    # Handle include_thoughts separately if needed
                    if include_thoughts:
                        completion_params['include_thoughts'] = True
                    
                    # Add other params except the ones we've handled
                    other_params = {k: v for k, v in self.additional_params.items() 
                                  if k not in ['thinking_budget', 'include_thoughts']}
                    completion_params.update(other_params)
                    
                    logging.debug(f"Converted Gemini thinking params: thinking={completion_params['thinking']}, include_thoughts={include_thoughts}")
                else:
                    # For non-Gemini models, pass params as-is
                    completion_params.update(self.additional_params)
                    
                logging.debug(f"Passing additional params to litellm: {self.additional_params}")

            # Add allowed_openai_params for thinking if present
            extra_params = {}
            if 'thinking' in completion_params and not self.drop_call_params:
                extra_params['allowed_openai_params'] = ['thinking']
            
            # Debug logging for Gemini models
            if self.model_name.startswith("gemini/"):
                logging.debug(f"Gemini API call - Model: {completion_params['model']}")
                logging.debug(f"Gemini API call - Messages count: {len(completion_messages)}")
                logging.debug(f"Gemini API call - Has system prompt: {system_prompt_content is not None}")
                if completion_messages:
                    logging.debug(f"Gemini API call - First message: {completion_messages[0]}")
            
            response = litellm.completion(
                **completion_params,
                drop_params=self.drop_call_params,
                **extra_params
            )
            cost = response._hidden_params.get("response_cost", 0.0)

            try:
                choices = response.get("choices", [])
                if not choices:
                    logging.warning(f"Empty response from model (no choices)\n\n{get_error_messages_cached()}")
                    return {"message": "", "cost": cost}

                message = choices[0].get("message", {})
                content = message.get("content")

                if content is None or not content.strip():
                    logging.warning(f"Empty response from model (content empty)\n\n{get_error_messages_cached()}")
                    return {"message": "", "cost": cost}

                return {"message": content.strip(), "cost": cost}

            except (AttributeError, IndexError, KeyError) as e:
                logging.warning(
                    self.message_writer.format_error(
                        f"Error extracting model response: {str(e)}",
                        "response extraction error",
                    )
                )
                return {"message": "", "cost": cost}

        except RateLimitError as e:
            logging.error(self.message_writer.format_api_error("Rate limit exceeded."))
            raise e
        except APIError as e:
            logging.error(self.message_writer.format_api_error(str(e)))
            raise e
        except Exception as e:
            logging.error(f"Error in litellm call: {e}")
            logging.error(f"Error in model call: {str(e)}\n\n{get_error_messages_cached()}")
            raise e
