import logging
import ast
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pydantic import SecretStr
import inspect
import re
import os
import sys
from pathlib import Path

from AlgoTuner.config.model_config import GlobalConfig, GenericAPIModelConfig
from AlgoTuner.utils.file_helpers import load_file_content
from editor.editor_functions import Editor, EditorState, reload_all_llm_src
from AlgoTuner.utils.message_writer import MessageWriter
from AlgoTuner.utils.error_helpers import get_error_messages_cached
from AlgoTuneTasks.base import Task
from AlgoTuner.utils.type_inspection import describe_type

@dataclass
class InterfaceState:
    """Represents the current state of the LLM interface."""

    spend: float = 0.0
    messages_sent: int = 0
    messages: List[Dict[str, str]] = None
    _state_dict: Dict[str, Any] = None
    editor_state: Optional[EditorState] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self._state_dict is None:
            self._state_dict = {}

    def get(self, key: str, default=None) -> Any:
        """Get a value from the state dictionary.
        
        Args:
            key: The key to get the value for
            default: The default value to return if the key is not found
            
        Returns:
            The value for the key, or the default if not found
        """
        return self._state_dict.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the state dictionary.
        
        Args:
            key: The key to set the value for
            value: The value to set
        """
        self._state_dict[key] = value

    def get_best_speedup(self) -> Optional[float]:
        """Get the best speedup achieved so far."""
        if self.editor_state:
            return self.editor_state.get_best_speedup()
        logging.warning("Attempted to get best speedup, but editor_state is not set in InterfaceState.")
        return None

    def update_best_speedup(self, new_speedup: Optional[float]) -> bool:
        """Update the best speedup if the new one is better."""
        if self.editor_state:
            return self.editor_state.update_best_speedup(new_speedup)
        logging.warning("Attempted to update best speedup, but editor_state is not set in InterfaceState.")
        return False


class BaseLLMInterface:
    """Base class for LLM interfaces with core functionality."""

    def __init__(
        self,
        model_config: GenericAPIModelConfig,
        global_config: GlobalConfig,
        model_name: str,
        task_instance,
    ):
        logging.info(f"BaseLLMInterface __init__ started for model: {model_name}")
        self.model_config = model_config
        self.model_name = model_name
        self.task_instance = task_instance
        self.message_writer = MessageWriter()

        # Initialize configuration
        if global_config is not None:
            self.spend_limit = model_config.spend_limit
            self.total_messages = global_config.total_messages
            self.max_messages_in_history = global_config.max_messages_in_history
        else:
            # Default values for human mode
            self.spend_limit = float("inf")
            self.total_messages = float("inf")
            self.max_messages_in_history = 5

        # Initialize editor
        try:
            self.editor_state = EditorState()
            self.editor = Editor(self.editor_state)
            logging.info(
                self.message_writer.format_system_message("Editor initialized")
            )
        except Exception as e:
            error_msg = self.message_writer.format_error(str(e), "initializing editor")
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        # Now initialize InterfaceState and pass editor_state
        self.state = InterfaceState(editor_state=self.editor_state)

        # Load error messages template
        try:
            self.error_message_template = get_error_messages_cached()
        except Exception as e:
            error_msg = self.message_writer.format_error(
                str(e), "loading error messages template"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        if model_name != "human":
            logging.info("Calling _initialize_model...")
            self._initialize_model()
            logging.info("Calling _load_initial_messages...")
            self._load_initial_messages()

        logging.info("BaseLLMInterface __init__ finished.")

    def _initialize_model(self):
        """Initialize the LLM model with appropriate configuration."""
        logging.info("_initialize_model started.")
        api_key = self.model_config.api_key
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()

        if not api_key:
            error_msg = self.message_writer.format_error(
                f"No API key found. Please set the {self.model_config.api_key_env} environment variable.",
                "API configuration",
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Initialize model parameters
        self.model_params = {
            "model_name": self.model_config.name,
            "api_key": api_key,
            "top_p": self.model_config.top_p,
            "max_tokens": self.model_config.max_tokens,
        }

        if (
            hasattr(self.model_config, "temperature")
            and self.model_config.temperature is not None
        ):
            self.model_params["temperature"] = self.model_config.temperature

        logging.info("_initialize_model finished.")

    def _load_initial_messages(self):
        """Load and set up initial system message."""
        logging.info("Entered _load_initial_messages.")
        initial_message_path = "AlgoTuner/messages/initial_system_message.txt"
        description_path = f"{self.task_instance.get_task_directory()}/description.txt"

        # Load initial template
        initial_content = load_file_content(initial_message_path)
        description_content = load_file_content(description_path)
        
        # Dynamically update the package list from pyproject.toml
        try:
            # Read project dependencies from pyproject.toml
            import tomllib as toml_lib
        except ImportError:
            import toml as toml_lib
            _using_tomllib = False # Flag that we are using the older toml library
        else:
            _using_tomllib = True # Flag that we are using the standard tomllib
            
        try:
            pyproject_path = "pyproject.toml"
            if not os.path.exists(pyproject_path):
                logging.warning(f"{pyproject_path} not found. Skipping dynamic package list update.")
            else:
                # Open in the correct mode based on the library
                file_mode = 'rb' if _using_tomllib else 'r'
                try:
                    with open(pyproject_path, file_mode) as fp:
                        proj = toml_lib.load(fp)
                except Exception as e_load:
                    logging.error(f"Failed to load {pyproject_path} using mode '{file_mode}': {e_load}", exc_info=True)
                    # Optional: Attempt fallback mode?
                    raise # Re-raise the loading error for now
                
                # Try PEP 621 `project.dependencies` first
                deps = proj.get('project', {}).get('dependencies')
                if deps is None:
                    # Fallback to Poetry's `tool.poetry.dependencies`
                    deps = proj.get('tool', {}).get('poetry', {}).get('dependencies', [])
                    logging.info("Using dependencies from [tool.poetry.dependencies]")
                else:
                     logging.info("Using dependencies from [project.dependencies]")

                if not isinstance(deps, (list, dict)): # Poetry uses a dict, PEP 621 uses list
                    logging.warning(f"Unexpected type for dependencies: {type(deps)}. Skipping dynamic package list update.")
                    deps = []

                logging.debug(f"Raw dependencies found: {deps}")

                # Exclude dev/tooling packages AND python itself
                exclude = {'litellm', 'google-generativeai', 'pylint', 'line_profiler', 'pytest', 'toml', 'python', 'orjson', 'pyaml', 'pillow'}
                
                # Normalize dependency strings/keys (strip extras, version specifiers)
                pkg_names = []
                
                if isinstance(deps, list): # PEP 621 list of strings
                    dep_iterable = deps
                elif isinstance(deps, dict): # Poetry dict {name: version/spec}
                    dep_iterable = deps.keys()
                else:
                    dep_iterable = [] # Should not happen based on earlier check

                for dep in dep_iterable:
                    # Extract package name before any bracket or comparison
                    name = dep.split('[')[0].split(' ')[0].split('=')[0].split('>')[0].split('<')[0].strip().strip('"').strip("'")
                    if name: # Avoid empty strings
                        pkg_names.append(name)
                        
                logging.debug(f"Normalized package names: {pkg_names}")
                
                extras = sorted([p for p in pkg_names if p not in exclude])
                logging.debug(f"Filtered packages (extras): {extras}")
                
                if not extras:
                     logging.warning("No extra packages found after filtering. Placeholder might remain.")
                     
                # Build new bullet list with actual newlines
                bullets = ''.join(f" - {p}\n" for p in extras) if extras else " - (None specified or all filtered)\n"
                logging.debug(f"Generated bullets string:\n{bullets}")
                
                # Regex to find the line 'additional packages:' and subsequent bullet points
                # It captures the prefix up to and including 'additional packages:\n'
                # It then matches (and discards) one or more lines starting with optional space/tab and '-' 
                pattern = re.compile(
                    r"(?P<prefix>^.*?additional packages:\s*\r?\n)(?:^[ \t]*-[^\r\n]*\r?\n?)+",
                    re.IGNORECASE | re.MULTILINE
                )
                
                original_content = initial_content # Keep a copy for comparison
                initial_content = pattern.sub(lambda m: m.group('prefix') + bullets, initial_content, count=1)
                
                if initial_content == original_content:
                    logging.warning("Regex pattern did not match or substitute in initial_system_message.txt. Placeholder may remain.")
                else:
                    logging.info("Successfully substituted placeholder with dynamic package list.")

        except Exception as e:
            # Fallback to original content on failure
            logging.error(f"Error during dynamic package list update: {e}", exc_info=True)
            # Ensure initial_content retains its original value if loaded before error
            initial_content = load_file_content(initial_message_path) # Reload original on error
            logging.warning("Fell back to original system message content due to error.")
        
        # Log loaded file contents
        logging.info(f"Loaded initial_content (type: {type(initial_content)}, len: {len(initial_content)}). Starts with:\n-------\n{str(initial_content)[:200]}\n-------")
        logging.info(f"Loaded description_content (type: {type(description_content)}, len: {len(description_content)}). Starts with:\n-------\n{str(description_content)[:200]}\n-------")

        # Get the solve function implementation using inspect
        solve_function_source = inspect.getsource(self.task_instance.solve)
        
        # Get the module containing the task class
        task_module = inspect.getmodule(self.task_instance)
        
        # --- ADDED BACK --- Define all_methods and all_module_functions
        all_methods = inspect.getmembers(self.task_instance.__class__, predicate=inspect.isfunction)
        all_module_functions = inspect.getmembers(task_module, predicate=inspect.isfunction)
        # --- END ADDED BACK ---

        # Get imports from the module's source
        module_source = ""
        source_error = None
        try:
            module_source = inspect.getsource(task_module)
            logging.info(f"Successfully retrieved task module source (length: {len(module_source)}).")
            # Log first few lines for verification
            logging.info(f"Module source starts with:\n-------\n{module_source[:200]}\n-------")
        except Exception as e:
            source_error = str(e)
            logging.error(f"Failed to retrieve task module source: {e}")
            module_source = "" # Ensure it's empty on error
            
        import_lines = []
        if not source_error:
            for line in module_source.split('\n'):
                line = line.strip()
                # Skip any line containing logging, even in comments
                if 'logging' in line:
                    continue
                # --- ADDED --- Skip tasks.base imports
                if 'tasks.base' in line.lower():
                    continue
                # --- END ADDED ---
                # Capture import statements
                if line.startswith('import ') or line.startswith('from '):
                    import_lines.append(line)
            logging.info(f"Collected {len(import_lines)} import lines. Content: {import_lines}")
        else:
            logging.warning(f"Skipping import line collection due to source retrieval error: {source_error}")

        # Extract just the function definition by finding the first def
        lines = solve_function_source.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('def solve'):
                start_idx = i
                break
        lines = lines[start_idx:]
        
        # Find the actual indentation of the function definition
        indent = len(lines[0]) - len(lines[0].lstrip())
        # Remove the indentation from all lines
        unindented_source = "\n".join(line[indent:] for line in lines)
        
        # Clean up logging but keep solve as a class method (don't remove self references)
        # NOTE: We intentionally do NOT remove self. references from the solve function
        # as it should remain a proper class method
        # Remove any logging lines from solve function
        unindented_source = '\n'.join(line for line in unindented_source.split('\n') if 'logging.' not in line)
        
        # Add a validation function notice inside the solve function docstring
        if '"""' in unindented_source:
            docstring_end = unindented_source.find('"""', unindented_source.find('"""') + 3)
            if docstring_end > 0:
                validation_note = (
                    "\n\n    NOTE: Your solution must pass validation by:"
                    "\n    1. Returning correctly formatted output"
                    "\n    2. Having no NaN or infinity values"
                    "\n    3. Matching expected results within numerical tolerance"
                    "\n    "
                )
                unindented_source = (
                    unindented_source[:docstring_end] + 
                    validation_note + 
                    unindented_source[docstring_end:]
                )
        logging.info("Added validation requirement notes inside the solve function docstring")
        logging.info("Successfully cleaned solve function source.")
        
        # Get the validation function directly using the same approach as the solve function
        validation_source = ""
        try:
            # First try to get the validation function directly from the task instance
            logging.info("Attempting to fetch validation function source code")
            validation_function = self.task_instance.is_solution
            validation_source = inspect.getsource(validation_function)
            logging.info(f"Successfully retrieved validation function from task instance: {len(validation_source)} characters")
        except (AttributeError, TypeError) as e:
            # If that fails, get it from the base Task class
            try:
                logging.info(f"Failed to get task's validation function: {e}. Trying base class...")
                validation_function = Task.is_solution
                validation_source = inspect.getsource(validation_function)
                logging.info(f"Successfully retrieved validation function from base Task: {len(validation_source)} characters")
            except Exception as e2:
                logging.error(f"Failed to get validation function source: {e2}")
                validation_source = "def is_solution():\n    pass"
        
        # Clean up the validation source similar to other functions
        lines = validation_source.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('def is_solution'):
                start_idx = i
                break
        lines = lines[start_idx:]
        
        # Find and remove indentation
        indent = len(lines[0]) - len(lines[0].lstrip())
        validation_source = "\n".join(line[indent:] for line in lines)
        
        # Clean up self references
        validation_source = validation_source.replace("self.", "")
        validation_source = validation_source.replace("def is_solution(self,", "def is_solution(")
        validation_source = validation_source.replace("def is_solution(self ,", "def is_solution(")
        validation_source = validation_source.replace("def is_solution(self)", "def is_solution()")
        validation_source = validation_source.replace("def is_solution(self):", "def is_solution():")
        validation_source = validation_source.replace("def is_solution(self, ", "def is_solution(")
        # Remove lines that start with a logging call
        validation_source = '\n'.join(line for line in validation_source.split('\n') if not line.strip().startswith('logging.'))
        # If the resulting validation_source is too short, use a fallback message
        if len([line for line in validation_source.split('\n') if line.strip()]) < 3:
            validation_source = "[Validation function source not available (possibly removed due to logging cleanup)]"

        # Add needed config imports if the task did not override is_solution
        if self.task_instance.__class__.__dict__.get('is_solution', None) is None:
            config_import = "from config.model_config import GlobalConfig, GenericAPIModelConfig"
            validation_source = config_import + "\n\n" + validation_source
        logging.info("Successfully retrieved and cleaned validation function source.")

        # Find helper functions used in solve and their dependencies
        helper_functions = set()  # Use a set to avoid duplicates
        
        def process_function_source(source_code):
            """Extract function names used in the source code"""
            used_functions = set()
            lines = source_code.split('\n')
            
            # Pattern to detect function calls: function_name(
            function_call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            
            for line in lines:
                # Skip comments and empty lines
                if line.strip().startswith('#') or not line.strip():
                    continue
                
                # Find all function calls in the line
                matches = re.findall(function_call_pattern, line)
                for match in matches:
                    func_name = match
                    
                    # Skip built-in functions and common library functions
                    builtin_functions = {
                        'len', 'max', 'min', 'sum', 'abs', 'round', 'float', 'int', 'str', 'bool',
                        'list', 'dict', 'set', 'tuple', 'range', 'enumerate', 'zip', 'map', 'filter',
                        'print', 'isinstance', 'hasattr', 'getattr', 'setattr', 'type', 'vars',
                        'super', 'property', 'staticmethod', 'classmethod', 'sorted', 'reversed',
                        'all', 'any', 'open', 'iter', 'next', 'format'
                    }
                    
                    # Skip numpy, scipy, and other library functions
                    if '.' in line and func_name in line.split('.'):
                        continue
                    
                    # Skip built-in functions
                    if func_name in builtin_functions:
                        continue
                    
                    # Skip if it's a method call on an object (contains dot before function name)
                    func_pos = line.find(func_name + '(')
                    if func_pos > 0 and line[func_pos-1] == '.':
                        continue
                    
                    # Add to used functions if it's not already excluded
                    used_functions.add(func_name)
                
                # Also look for self.function_name patterns specifically
                self_method_pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                self_matches = re.findall(self_method_pattern, line)
                for match in self_matches:
                    used_functions.add(match)
            
            return used_functions
        
        # First get all available helper functions
        available_helpers = {}
        
        # Get class methods (excluding special methods and public interface methods)
        for name, func in all_methods:
            # Include all methods except special methods (__init__, __str__, etc.) and main interface methods
            if (name != '__init__' and 
                not name.startswith('__') and 
                name not in ['solve', 'generate_problem', 'is_solution'] and 
                func.__module__ == task_module.__name__):
                try:
                    helper_source = inspect.getsource(func)
                    available_helpers[name] = helper_source
                except OSError:
                    # Some functions might not have source available (built-ins, etc.)
                    logging.debug(f"Could not get source for method {name}")
        
        # Get module-level functions
        for name, func in all_module_functions:
            if name != 'solve' and func.__module__ == task_module.__name__:
                try:
                    helper_source = inspect.getsource(func)
                    available_helpers[name] = helper_source
                except OSError:
                    # Some functions might not have source available (built-ins, etc.)
                    logging.debug(f"Could not get source for function {name}")
        logging.info(f"Found {len(available_helpers)} potential helper functions defined in the task module.")
        
        # Start with functions directly used in solve
        functions_to_process = process_function_source(solve_function_source)
        processed_functions = set()
        
        # Keep processing until we've found all dependencies
        while functions_to_process:
            func_name = functions_to_process.pop()
            if func_name in processed_functions:
                continue
                
            if func_name in available_helpers:
                # Get and clean up the helper function
                helper_source = available_helpers[func_name]
                helper_lines = helper_source.split('\n')
                helper_indent = len(helper_lines[0]) - len(helper_lines[0].lstrip())
                helper_unindented = "\n".join(line[helper_indent:] for line in helper_lines)
                
                # Clean up self references
                helper_unindented = helper_unindented.replace("self.", "")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self,", f"def {func_name}(")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self ,", f"def {func_name}(")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self)", f"def {func_name}()")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self):", f"def {func_name}():")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self, ", f"def {func_name}(")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self,", f"def {func_name}(")
                
                # Remove any logging lines
                helper_unindented = '\n'.join(line for line in helper_unindented.split('\n') if 'logging.' not in line)
                
                helper_functions.add(helper_unindented)
                processed_functions.add(func_name)
                
                # Extract imports from this helper function
                helper_imports = self._extract_imports_from_source(helper_source)
                for imp in helper_imports:
                    if imp not in import_lines:
                        import_lines.append(imp)
                
                # Find any new functions this helper uses
                new_functions = process_function_source(helper_source)
                functions_to_process.update(new_functions - processed_functions)
        logging.info(f"Finished processing helper function dependencies. Found {len(helper_functions)} relevant helper functions.")

        # --- Revert: Remove import filtering logic ---
        # Use all collected non-logging imports directly
        imports_str = "\n".join(import_lines)
        logging.info(f"Using all {len(import_lines)} non-logging imports from module.")
        # --- End Revert ---

        # Combine helper functions and the main solve function
        main_code_parts = list(helper_functions) + [unindented_source]
        main_code_str = "\n\n".join(main_code_parts)

        # Add line numbers to the main code block, starting from 1
        main_code_lines = main_code_str.split('\n')
        width = len(str(len(main_code_lines)))
        solve_numbered_source = "\n".join(f"| {str(i).zfill(width)}: {line}" for i, line in enumerate(main_code_lines, 1))

        # Prepend imports to the numbered solve source
        solve_with_imports = imports_str + "\n\n" + solve_numbered_source

        # Also include the validation function in the initial message
        # First try to get the validation function directly from the task instance
        validation_source = ""
        try:
            logging.info("Getting validation function for initial message")
            validation_function = self.task_instance.is_solution
            validation_source = inspect.getsource(validation_function)
        except (AttributeError, TypeError) as e:
            # If that fails, get it from the base Task class
            try:
                logging.info(f"Getting validation function from base Task class instead")
                validation_function = Task.is_solution
                validation_source = inspect.getsource(validation_function)
            except Exception as e2:
                logging.error(f"Failed to get validation function source: {e2}")
                validation_source = "def is_solution():\n    pass"
        
        # Clean up the validation source
        lines = validation_source.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('def is_solution'):
                start_idx = i
                break
        lines = lines[start_idx:]
        
        # Find and remove indentation
        indent = len(lines[0]) - len(lines[0].lstrip())
        validation_source = "\n".join(line[indent:] for line in lines)
        
        # Clean up self references
        validation_source = validation_source.replace("self.", "")
        validation_source = validation_source.replace("def is_solution(self,", "def is_solution(")
        validation_source = validation_source.replace("def is_solution(self ,", "def is_solution(")
        validation_source = validation_source.replace("def is_solution(self)", "def is_solution()")
        validation_source = validation_source.replace("def is_solution(self):", "def is_solution():")
        validation_source = validation_source.replace("def is_solution(self, ", "def is_solution(")
        
        # -------------------------------------------------------------
        # NEW: Include helper functions used by is_solution()
        # -------------------------------------------------------------
        validation_helper_functions = set()

        # Start with functions referenced directly inside is_solution
        validation_functions_to_process = process_function_source(validation_source)

        while validation_functions_to_process:
            func_name = validation_functions_to_process.pop()
            # Skip if we've already processed this function earlier (solve helpers)
            if func_name in processed_functions:
                continue

            if func_name in available_helpers:
                helper_source = available_helpers[func_name]
                helper_lines = helper_source.split('\n')
                helper_indent = len(helper_lines[0]) - len(helper_lines[0].lstrip())
                helper_unindented = "\n".join(line[helper_indent:] for line in helper_lines)

                # Remove self references similar to above
                helper_unindented = helper_unindented.replace("self.", "")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self,", f"def {func_name}(")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self ,", f"def {func_name}(")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self)", f"def {func_name}()")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self):", f"def {func_name}():")
                helper_unindented = helper_unindented.replace(f"def {func_name}(self, ", f"def {func_name}(")

                # Remove logging lines inside helper
                helper_unindented = '\n'.join(line for line in helper_unindented.split('\n') if 'logging.' not in line)

                validation_helper_functions.add(helper_unindented)
                processed_functions.add(func_name)

                # Extract and store additional imports used by this helper
                helper_imports = self._extract_imports_from_source(helper_source)
                for imp in helper_imports:
                    if imp not in import_lines:
                        import_lines.append(imp)

                # Queue up any further functions this helper calls
                new_funcs = process_function_source(helper_source)
                validation_functions_to_process.update(new_funcs - processed_functions)

        logging.info(f"Added {len(validation_helper_functions)} helper functions for is_solution().")

        # Combine helper functions and validation function
        validation_code_parts = list(validation_helper_functions) + [validation_source]
        validation_code_str = "\n\n".join(validation_code_parts)

        # Number lines for combined validation code
        validation_lines = validation_code_str.split('\n')
        validation_width = len(str(len(validation_lines)))
        validation_numbered_source = "\n".join(
            f"| {str(i).zfill(validation_width)}: {line}" for i, line in enumerate(validation_lines, 1)
        )

        # Prepend imports to the numbered validation source
        validation_with_imports = imports_str + "\n\n" + validation_numbered_source
        
        # Initialize validation_function_description
        validation_function_description = ""

        # Log the imports string just before combining
        logging.info(f"--- Imports string before combining ---\n{imports_str}\n-------------------------------------")

        # Combine all content with the structured code blocks
        combined_content = (
            initial_content
            + description_content
            + "\n\nBelow is the reference implementation. Your function should run much quicker.\n\n"
            + solve_with_imports # Contains imports + numbered solve
            + "\n\nThis function will be used to check if your solution is valid for a given problem. If it returns False, it means the solution is invalid:\n\n"
            + validation_with_imports # Contains imports + numbered validation
        )

        combined_content += validation_function_description

        # --- ADDED --- Log the full message content
        logging.info(f"--- Full Initial System Message Content ---\n{combined_content}\n----------------------------------------")
        # --- END ADDED ---

        logging.info(f"Validation source length: {len(validation_source)}")
        logging.info("Added validation function to combined content")
        logging.info(f"Final combined content length: {len(combined_content)}")
        
        # Check if the validation function is actually included in the message
        contains_validation = "VALIDATION FUNCTION" in combined_content and len(validation_source) > 0
        logging.info(f"Message contains validation function: {contains_validation}")
        
        if not contains_validation:
            logging.error("Validation function failed to be included in the message")

        # For Anthropic models (Claude), we need to ensure first message has user role
        if "claude" in self.model_config.name.lower():
            # Add a placeholder user message first
            self.state.messages.append({"role": "user", "content": "."})
            # Then add system message
            self.state.messages.append({"role": "system", "content": combined_content})
        else:
            # For other models, just add the system message
            self.state.messages.append({"role": "system", "content": combined_content})

    def get_current_code(self) -> str:
        """Get the current state of solver.py"""
        return self.editor.view_file(Path("solver.py"))

    def _extract_imports_from_source(self, source_code: str) -> list[str]:
        """Extract import statements from source code, excluding AlgoTuner and logging imports."""
        import_lines = []
        lines = source_code.split('\n')
        
        for line in lines:
            stripped = line.strip()
            # Check if it's an import line
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Skip AlgoTuner-related imports
                if 'AlgoTuner' in stripped or 'algotune' in stripped.lower():
                    continue
                # Skip logging imports
                if stripped.startswith('import logging') or stripped.startswith('from logging'):
                    continue
                # Skip relative imports (they won't work in standalone solver)
                if stripped.startswith('from .') or stripped.startswith('from ..'):
                    continue
                
                import_lines.append(stripped)
        
        return import_lines

    @staticmethod
    def _format_numbered_code(code_str: str, start_line: int = 1) -> str:
        """Format code with line numbers and preserve indentation."""
        lines = code_str.split('\n')
        if not lines:
            return ""

        width = len(str(len(lines) + start_line - 1))
        numbered_lines = [f"| {str(i).zfill(width)}: {line}" for i, line in enumerate(lines, start=start_line)]
        return "\n".join(numbered_lines)