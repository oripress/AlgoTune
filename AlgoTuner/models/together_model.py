import requests
import logging
import json
import time
import random
import signal
from typing import Dict, List, Any
from AlgoTuner.utils.message_writer import MessageWriter
from AlgoTuner.utils.error_helpers import get_error_messages_cached


class TogetherModel:
    """
    Together API client for models not available in LiteLLM.
    Specifically designed for DeepSeek-R1 (DeepSeek-R1-0528) and other reasoning models.
    """
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        # Fix model name case for Together API (they expect proper capitalization)
        if model_name.lower() == "deepseek-ai/deepseek-r1":
            self.model_name = "deepseek-ai/DeepSeek-R1"
        else:
            self.model_name = model_name
        self.api_key = api_key
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.message_writer = MessageWriter()
        
        # Remove configuration-only parameters
        config_only_params = {'modify_params', 'drop_params', 'model_provider'}
        self.additional_params = {k: v for k, v in kwargs.items() if k not in config_only_params}
        
        # Set default parameters for reasoning models (based on Together AI recommendations)
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.6),
            "top_p": kwargs.get("top_p", 0.95),
            "max_tokens": kwargs.get("max_tokens", 32000),
        }
        
        # Set up signal handlers for debugging
        self._setup_signal_handlers()
        
        logging.info(f"TogetherModel initialized for {model_name}. Additional params: {self.additional_params}")

    def _setup_signal_handlers(self):
        """Set up signal handlers to catch external termination signals."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logging.error(f"TogetherModel: Received signal {signal_name} ({signum}) - process being terminated")
            raise KeyboardInterrupt(f"Process terminated by signal {signal_name}")
        
        # Set up handlers for common termination signals
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGQUIT'):
                signal.signal(signal.SIGQUIT, signal_handler)
        except (ValueError, OSError) as e:
            logging.warning(f"TogetherModel: Could not set up signal handlers: {e}")

    def query(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Send a query to the Together API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Dictionary with 'message' and 'cost' keys
        """
        # Debug logging of context
        if len(messages) > 1:
            logging.debug("Previous context being sent to Together API:")
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

        # Handle DeepSeek-specific requirements
        if "deepseek" in self.model_name.lower():
            if len(messages) == 1 and messages[0]["role"] == "system":
                messages.append({"role": "user", "content": "Proceed."})
                logging.debug("Appended dummy user message for DeepSeek initial system message.")

        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            **self.default_params,
            **self.additional_params
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logging.debug(f"Sending request to Together API: {json.dumps(payload, indent=2)}")
        
        # Make the API request with retry logic for 503 errors
        max_retries = 5
        base_delay = 2.0
        response = None
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=600  # 10 minutes for reasoning models
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                break  # Success, exit retry loop
                
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_exception = e
                
                # Determine if this is a retryable error
                is_retryable = False
                error_description = ""
                
                if isinstance(e, requests.exceptions.HTTPError):
                    if e.response and e.response.status_code in [503, 502, 504, 429]:  # Service unavailable, bad gateway, gateway timeout, rate limit
                        is_retryable = True
                        error_description = f"HTTP {e.response.status_code}"
                elif isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                    is_retryable = True
                    error_description = "Connection/Timeout"
                
                if is_retryable and attempt < max_retries - 1:
                    # Retryable error, retry with exponential backoff
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Together API returned {error_description} error. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    
                    # Add more detailed retry logging
                    logging.info(f"Together API retry: Starting sleep for {delay:.2f}s before attempt {attempt + 2}")
                    
                    try:
                        time.sleep(delay)
                        logging.info(f"Together API retry: Sleep completed, proceeding to retry attempt {attempt + 2}")
                    except KeyboardInterrupt:
                        logging.error("Together API retry: Sleep interrupted by KeyboardInterrupt")
                        raise
                    except Exception as sleep_error:
                        logging.error(f"Together API retry: Sleep interrupted by exception: {sleep_error}")
                        raise
                    
                    logging.info(f"Together API retry: About to retry request (attempt {attempt + 2}/{max_retries})")
                    continue
                else:
                    # Not retryable or max retries reached, handle the error
                    if is_retryable:
                        # All retries exhausted for retryable error
                        error_msg = f"Together API {error_description} error after {max_retries} retries: {e}"
                        if hasattr(e, 'response') and hasattr(e.response, 'text'):
                            error_msg += f" - Response: {e.response.text}"
                        logging.error(self.message_writer.format_api_error(error_msg))
                        raise Exception(error_msg)
                    else:
                        # Not retryable, re-raise immediately
                        raise
                    
        if response is None:
            # This should never happen, but just in case
            if last_exception:
                error_msg = f"Together API HTTP error: {last_exception}"
                if hasattr(last_exception.response, 'text'):
                    error_msg += f" - Response: {last_exception.response.text}"
                logging.error(self.message_writer.format_api_error(error_msg))
                raise Exception(error_msg)
            else:
                raise Exception("Failed to get response from Together API after all retries")
        
        # Parse JSON response
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            error_msg = f"Together API JSON decode error: {e}"
            logging.error(self.message_writer.format_api_error(error_msg))
            raise Exception(error_msg)
        
        logging.debug(f"Received response from Together API: {json.dumps(response_data, indent=2)}")
        
        # Extract the message content
        try:
            choices = response_data.get("choices", [])
            if not choices:
                logging.warning(f"Empty response from Together API (no choices)\n\n{get_error_messages_cached()}")
                return {"message": "", "cost": 0.0}

            message = choices[0].get("message", {})
            content = message.get("content")

            if content is None or not content.strip():
                logging.warning(f"Empty response from Together API (content empty)\n\n{get_error_messages_cached()}")
                return {"message": "", "cost": 0.0}

            # Calculate cost if usage information is available
            cost = 0.0
            usage = response_data.get("usage", {})
            if usage:
                # Together API typically provides token usage information
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                # Cost calculation based on Together AI pricing for DeepSeek-R1
                # DeepSeek-R1: $3 input / $7 output per 1M tokens
                prompt_cost_per_token = 3.0 / 1_000_000  # $3 per 1M tokens
                completion_cost_per_token = 7.0 / 1_000_000  # $7 per 1M tokens
                
                cost = (prompt_tokens * prompt_cost_per_token + 
                       completion_tokens * completion_cost_per_token)
                
                logging.debug(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Cost: ${cost:.6f}")

            return {"message": content.strip(), "cost": cost}

        except (KeyError, IndexError, TypeError) as e:
            logging.warning(
                self.message_writer.format_error(
                    f"Error extracting Together API response: {str(e)}",
                    "response extraction error",
                )
            )
            return {"message": "", "cost": 0.0}