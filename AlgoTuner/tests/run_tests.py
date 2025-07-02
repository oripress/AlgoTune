#!/usr/bin/env python3
import sys
import logging
import argparse
import os
from pathlib import Path
from typing import List, Dict, Union
from dataclasses import dataclass
import traceback
import inspect
import time
import json
import numpy as np

 

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import required modules first
from AlgoTuneTasks.base import TASK_REGISTRY
from AlgoTuneTasks.factory import TaskFactory
from AlgoTuner.interfaces.llm_interface import LLMInterface
from AlgoTuner.config.loader import load_config
from AlgoTuner.config.model_config import GlobalConfig, GenericAPIModelConfig
from AlgoTuner.utils.file_helpers import load_file_content
from AlgoTuner.utils.logger import setup_logging
from AlgoTuner.utils.casting import cast_input
from AlgoTuner.utils.profiler import TaskProfiler

@dataclass
class DummyConfig:
    """Config for dummy model with all required fields"""

    spend_limit: float = 1000.0
    name: str = "dummy"
    api_key: str = "dummy-key"
    api_key_env: str = "DUMMY_API_KEY"
    api_base: str = "https://dummy.api/v1"
    api_version: str = "2024-01"
    deployment_name: str = "dummy-deployment"
    provider: str = "dummy"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 4096
    stop_sequences: List[str] = None
    context_length: int = 8192

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


def _format_evaluation_result_for_logging(eval_result):
    """Format evaluation result for logging, summarizing large data structures."""
    if not isinstance(eval_result, dict):
        return str(eval_result)
    
    # Create a copy to avoid modifying the original
    formatted = {}
    for key, value in eval_result.items():
        if key == 'problem_input':
            # Apply smart formatting to problem input
            formatted[key] = _format_problem_input_for_logging(value)
        elif isinstance(value, list) and len(value) > 10:
            formatted[key] = f"list[{len(value)}]: [{value[0]}, {value[1]}, ...]"
        elif isinstance(value, str) and len(value) > 500:
            formatted[key] = f"{value[:200]}... ({len(value)} chars total)"
        else:
            formatted[key] = value
    
    return formatted

def _format_problem_input_for_logging(problem_input):
    """Format problem input for logging, summarizing large data structures."""
    if problem_input is None:
        return "None"
    
    # Handle different data types
    if isinstance(problem_input, (int, float, bool)):
        return str(problem_input)
    
    if isinstance(problem_input, str):
        if len(problem_input) <= 200:
            return repr(problem_input)
        else:
            return f"str({len(problem_input)} chars): {repr(problem_input[:50])}..."
    
    if isinstance(problem_input, bytes):
        if len(problem_input) <= 50:
            return f"bytes({len(problem_input)}): {problem_input.hex()}"
        else:
            return f"bytes({len(problem_input)}): {problem_input[:20].hex()}..."
    
    if isinstance(problem_input, list):
        if len(problem_input) <= 10:
            return str(problem_input)
        else:
            first_few = problem_input[:3]
            return f"list[{len(problem_input)}]: [{first_few[0]}, {first_few[1]}, {first_few[2]}, ...]"
    
    if isinstance(problem_input, dict):
        # Special handling for dictionaries with binary data (like SHA-256 plaintext)
        formatted_items = []
        for key, value in problem_input.items():
            if isinstance(value, bytes):
                if len(value) <= 50:
                    formatted_items.append(f"'{key}': bytes({len(value)})")
                else:
                    formatted_items.append(f"'{key}': bytes({len(value)}) [LARGE_BINARY_DATA]")
            elif isinstance(value, str) and len(value) > 200:
                formatted_items.append(f"'{key}': str({len(value)} chars)")
            elif isinstance(value, (list, tuple)) and len(value) > 5:
                formatted_items.append(f"'{key}': {type(value).__name__}[{len(value)}]")
            else:
                formatted_items.append(f"'{key}': {repr(value)}")
        
        # If it's a small dict, show all items; otherwise truncate
        if len(problem_input) <= 3:
            return f"{{{', '.join(formatted_items)}}}"
        else:
            return f"dict[{len(problem_input)}]: {{{', '.join(formatted_items[:2])}, ...}}"
    
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
        if len(str_repr) <= 200:
            return str_repr
        else:
            return f"{type(problem_input).__name__}({len(str_repr)} chars): {str_repr[:50]}..."
    except Exception:
        return f"{type(problem_input).__name__} object"


class DummyModel:
    """Model that returns pre-written responses for testing"""

    def __init__(self, task_name: str, input_file: str = None, max_samples: int = None):
        # Load our pre-written responses
        if input_file and Path(input_file).exists():
            test_file = Path(input_file)
        else:
            # Fallback to constructing path from task name if input file not provided
            test_file = Path(__file__).parent / "inputs" / f"{task_name}.txt"
        
        with open(test_file) as f:
            content = f.read()

        # Split on INPUT_SEPARATOR but keep empty responses
        parts = content.split("[INPUT_SEPARATOR]")
        
        # Store both inputs and responses - they are the same in our test setup
        self.inputs = [part.strip() for part in parts[1:]]  # Skip first empty part
        self.responses = []
        for i, resp in enumerate(parts[1:], 1):
            # A response is only truly empty if it contains no non-whitespace characters
            # or only contains empty code blocks (```)
            cleaned_resp = resp.replace('```', '').strip()
            if not cleaned_resp:
                logging.info(f"Found empty response at position {i}")
                self.responses.append("")
            else:
                # Preserve original whitespace but remove any trailing whitespace
                self.responses.append(resp.rstrip())
        
        # Don't truncate responses - we want to process all responses in the test file
        # The max_samples parameter will be used to limit dataset evaluation instead
        logging.info(f"DummyModel: Processing all {len(self.responses)} responses from test file (no truncation)")
                
        logging.info(f"Loaded {len(self.responses)} responses ({sum(1 for r in self.responses if not r.replace('```', '').strip())} empty)")
        
        # Add one extra eval command at the end to trigger final evaluation
        self.responses.append("eval")
        logging.info(f"Added final eval command, total responses: {len(self.responses)}")

        self.current_index = 0
        self.task_name = task_name
        self.max_samples = max_samples  # Store max_samples for test mode
        self.config = DummyConfig()
        self.max_calls = len(self.responses) + 2  # Allow a few extra calls beyond responses
        self.call_count = 0

    def query(self, messages: List[Dict[str, str]]) -> Dict[str, Union[str, float]]:
        """Simulate a model query by returning the next pre-written response"""
        self.call_count += 1
        
        # If we've already returned all responses and are being asked for another,
        # that means all responses have been processed - time to exit
        if self.current_index >= len(self.responses):
            logging.info(f"DummyModel: All {len(self.responses)} responses have been returned and processed. Exiting test.")
            raise SystemExit(0)
        
        # Log the raw test input from the test file that corresponds to this response
        if hasattr(self, 'inputs') and self.current_index < len(self.inputs):
            raw_test_input = self.inputs[self.current_index].strip()
            if raw_test_input:
                logging.info(f"[TEST INPUT] Raw test input from tests/inputs/{self.task_name}.txt (response {self.current_index + 1}): {raw_test_input[:300]}...")
            else:
                logging.info(f"[TEST INPUT] Raw test input from tests/inputs/{self.task_name}.txt (response {self.current_index + 1}): [EMPTY]")
        
        # Only log the last message (the one we're responding to)
        if messages:
            last_msg = messages[-1]
            logging.info(f"Processing message {self.current_index + 1}/{len(self.responses)} (call {self.call_count})")
            logging.info(f"Last message role: {last_msg['role']}, content: {last_msg['content'][:200]}...")
            
            # Check if this is the initial system message
            if len(messages) == 1 and last_msg['role'] == 'system':
                logging.info(f"Responding to initial system message with response {self.current_index + 1}")
            else:
                logging.info(f"Responding to user message with response {self.current_index + 1}")

        # Get the current response
        response = self.responses[self.current_index]
        
        # A response is only empty if it contains no content after removing code blocks
        cleaned_resp = response.replace('```', '').strip()
        if not cleaned_resp:
            logging.info(f"Processing empty response at position {self.current_index + 1}")
        else:
            logging.info(f"Processing response at position {self.current_index + 1} ({len(response)} chars)")
        
        # Increment index for next call
        self.current_index += 1
        
        return {
            "message": response,
            "cost": 0.01,
            "model": "dummy-model",
            "finish_reason": "stop",
        }


class DummyLLM(LLMInterface):
    """LLM interface that uses our dummy model"""

    def __init__(self, model_name: str, task_name: str, input_file: str = None, max_samples: int = None):
        config = load_config()
        global_config = GlobalConfig(**config.get("global", {}))

        # --- Get data_dir from environment variable set by submit_test.sh ---
        task_specific_data_dir = os.environ.get("DATASET_PATH")
        if not task_specific_data_dir:
            logging.warning(f"[DummyLLM] DATASET_PATH environment variable not set. Falling back to constructing from DATA_DIR.")
            # Fallback construction (might be less reliable than direct env var)
            base_data_dir = os.environ.get("DATA_DIR")
            if base_data_dir:
                task_specific_data_dir = os.path.join(base_data_dir, task_name)
            else:
                logging.error("[DummyLLM] Cannot determine task data directory. DATASET_PATH and DATA_DIR env vars are missing.")
                # Set to None or raise error depending on desired behavior
                task_specific_data_dir = None
        else:
            logging.info(f"[DummyLLM] Using task-specific data directory from DATASET_PATH: {task_specific_data_dir}")
        # --- End data_dir retrieval ---

        # --- Debugging oracle_time_limit for DummyLLM ---
        loaded_oracle_time_limit = global_config.oracle_time_limit
        logging.info(f"[DummyLLM] Loaded global_config.oracle_time_limit: {loaded_oracle_time_limit}")

        # For now, still use the loaded value, but we could force it if needed:
        effective_oracle_time_limit = 100 # Force to 100 for testing dummy evaluations
        # effective_oracle_time_limit = loaded_oracle_time_limit
        logging.info(f"[DummyLLM] Effective oracle_time_limit for TaskFactory: {effective_oracle_time_limit} (FORCED)")
        # --- End Debugging ---

        task_instance = TaskFactory(
            task_name,
            oracle_time_limit=effective_oracle_time_limit,
            data_dir=task_specific_data_dir # Pass the retrieved task-specific data_dir
        )

        # Create our dummy model and use its config
        self.model = DummyModel(task_name, input_file, max_samples)
        super().__init__(
            model_config=self.model.config,
            global_config=global_config,
            model_name=model_name,
            task_instance=task_instance,
            max_samples=max_samples,  # Pass max_samples to parent
        )
        
        # --- Explicitly load dataset during init for testing (skipped if SKIP_DATASET_GEN=1) ---
        if os.environ.get("SKIP_DATASET_GEN") != "1":
            try:
                logging.info(f"[TEST] Explicitly calling load_dataset in DummyLLM.__init__ for {task_name}...")
                # Use default sizes/seed, matching potential implicit call later
                _train, _test = self.task_instance.load_dataset(
                    train_size=config.get('dataset', {}).get('train_size', 100),
                    test_size=config.get('dataset', {}).get('test_size', 100),
                    random_seed=42 # Or use a configured seed if available
                )
                logging.info(f"[TEST] Explicit load_dataset call completed.")
                # Consume a single item to ensure generation if needed
                # next(_train, None)
                # next(_test, None)
            except Exception as e:
                logging.error(f"[TEST] Error during explicit load_dataset call: {e}", exc_info=True)
        else:
            logging.info(f"[TEST] SKIP_DATASET_GEN set; reusing existing dataset for {task_name}")
        # --- End explicit load ---
        
        # Create and set up the profiler interface with the correct method signatures
        from AlgoTuner.utils.profiler import TaskProfiler
        
        # Create a wrapper class for TaskProfiler that has the expected interface
        class ProfilerInterface:
            def __init__(self, task_instance):
                self.task_instance = task_instance
                self.profiler = TaskProfiler(task_instance)
                
            def profile(self, problem_input, focus_lines=None):
                """Bridge to the TaskProfiler's profile_solve method."""
                import logging
                
                logging.info(f"ProfilerInterface.profile: Called with input: {_format_problem_input_for_logging(problem_input)}, focus_lines: {focus_lines}")
                
                # Process the input using cast_input
                if isinstance(problem_input, np.ndarray):
                    cast_problem = problem_input
                else:
                    cast_problem = cast_input(problem_input, self.task_instance)
                
                # Call the real profiler with the processed input
                result = self.profiler.profile_solve(cast_problem, focus_lines)
                
                # Add common fields
                result["command_source"] = "profile"
                result["problem_input"] = problem_input
                
                # Add a message field if not present
                if "message" not in result and "formatted_message" in result:
                    result["message"] = result["formatted_message"]
                elif "message" not in result and "profile_output" in result:
                    # Create a message from the profile output (without solution)
                    message_lines = []
                    message_lines.append(f"Input: {problem_input}")
                    # Don't include the solution
                    message_lines.append("Profiling results:")
                    message_lines.append(result["profile_output"])
                    if "elapsed_ms" in result:
                        message_lines.append(f"Time: {result['elapsed_ms']:.2f}ms")
                    result["message"] = "\n".join(message_lines)
                
                return result
                
            def profile_lines(self, focus_lines, problem_input):
                """Pass through to profile method with arguments swapped."""
                return self.profile(problem_input, focus_lines)
        
        # Initialize with our wrapper class that provides the expected interface
        self.profiler = ProfilerInterface(task_instance)
        
        # CRITICAL: Clear message history to ensure clean start
        logging.info("[DummyLLM] Clearing message history to ensure clean test start")
        if hasattr(self, 'state') and hasattr(self.state, 'messages'):
            logging.info(f"[DummyLLM] Initial message count before clear: {len(self.state.messages)}")
            self.clear_message_history()
            logging.info(f"[DummyLLM] Message count after clear: {len(self.state.messages)}")
        else:
            logging.warning("[DummyLLM] No state.messages found during initialization")

    def _setup_model(self):
        """Override to prevent LiteLLM initialization and use our dummy model"""
        # CRITICAL: Check if we already have a dummy model set up
        if hasattr(self, 'model') and hasattr(self.model, 'responses'):
            # We already have a DummyModel - don't let parent class overwrite it!
            logging.info("DummyLLM: Preserving existing dummy model, skipping LiteLLM model setup")
            return
            
        # If for some reason we don't have the dummy model, this is an error
        logging.error("DummyLLM: _setup_model called but no dummy model found!")
        raise RuntimeError("DummyLLM: Expected dummy model to be set up before _setup_model is called")

    def get_response(self, *args, **kwargs):
        """Get response from our dummy model, but go through parent's message handling"""
        try:
            # Call the parent get_response method which handles all the logging and message processing
            # The parent method will call self.model.query() which will use our dummy model
            result = super().get_response(*args, **kwargs)
            return result
        except SystemExit:
            raise
        except Exception as e:
            logging.error(f"Error in get_response: {str(e)}\n{traceback.format_exc()}")
            raise

    def run(self):
        """Override run to add logging"""
        try:
            super().run()
        except SystemExit:
            raise
        except Exception as e:
            logging.error(f"Error in run loop: {str(e)}\n{traceback.format_exc()}")
            raise

    # Override the command handler to add evaluation after delete and edit
    def execute_command(self, command_str: str) -> dict:
        """Execute a command and return its result - simplified for testing without heavy evaluation."""
        try:
            # Log the command being executed
            logging.info(f"DummyLLM: Executing command: {command_str}")
            
            # Execute the command using the parent class method
            result = super().execute_command(command_str)
            
            # For testing, we don't want to run heavy evaluations after each command
            # Just return the result and let the test continue to the next response
            logging.info(f"DummyLLM: Command completed with success: {result.get('success', False)}")
            
            return result
        except Exception as e:
            logging.error(f"Error in DummyLLM execute_command: {str(e)}\n{traceback.format_exc()}")
            # Fall back to parent implementation
            return super().execute_command(command_str)



def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Run tests or evaluations.")

    # Add arguments
    parser.add_argument("--model", type=str, default="codellama-7b", help="Model name to use.")
    parser.add_argument("--task", type=str, default="all", help="Task name or 'all'.")
    parser.add_argument("--input", type=str, default=None, help="Optional path to input file for dummy model.")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation, skip generation.")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index for generation.")
    parser.add_argument("--end-idx", type=int, default=-1, help="End index for generation (-1 means all).")
    parser.add_argument("--data-subset", type=str, default="train", help="Data subset (train/test).")
    parser.add_argument("--max-problems", type=int, default=-1, help="Max problems per task (-1 means all).")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for each sub-process.")
    parser.add_argument("--max-concurrent", type=int, default=1, help="Max concurrent sub-processes.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate")

    # Parse known args, allowing others to pass through (for specific test runners etc.)
    args, unknown = parser.parse_known_args()

    # Set logging level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # If a specific task is provided, run it
    if args.task != "all":
        # Set up logging with task name and model
        task_name = args.task
        # If task is not provided but input file is, extract task name from the input file
        if not task_name and args.input:
            task_name = Path(args.input).stem
        
        # Set up CODE_DIR if not already set (for tests)
        if not os.environ.get("CODE_DIR"):
            import tempfile
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            temp_code_dir = tempfile.mkdtemp(prefix=f"algotune_test_{task_name}_{unique_id}_")
            os.environ["CODE_DIR"] = temp_code_dir
            print(f"📁 Created temporary CODE_DIR for test: {temp_code_dir}")
        
        logger = setup_logging(task=task_name, model=args.model)

        interface = DummyLLM(args.model, task_name, args.input, args.max_samples)
        interface.run()
    else:
        # If task is "all", run all tasks
        # This is a placeholder implementation. You might want to implement a more robust task runner
        # for running all tasks in a distributed environment.
        pass


if __name__ == "__main__":
    main()
