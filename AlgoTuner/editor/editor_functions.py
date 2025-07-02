import os
import sys
import re
import ast
import json
import shutil
import hashlib
import tempfile
import logging
import importlib
import pkgutil
import traceback
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from pylint.lint import Run
from io import StringIO
from AlgoTuner.utils.message_writer import MessageWriter
from AlgoTuner.utils.snippet_utils import compute_centered_snippet_bounds, compute_snippet_bounds
from AlgoTuner.utils.profiler import TaskProfiler
from AlgoTuner.interfaces.commands.types import SnapshotStatus
from pylint.reporters import JSONReporter
import math
import signal
import threading
import queue



def get_code_dir() -> str:
    """
    Get the code directory path from the CODE_DIR environment variable.
    If not set, fall back to the current working directory.
    """
    code_dir = os.environ.get("CODE_DIR")
    if not code_dir:
        logging.warning("CODE_DIR environment variable not set; defaulting to current working directory")
        code_dir = os.getcwd()
    return code_dir


def reload_all_llm_src() -> None:
    """
    Dynamically reloads modules found in CODE_DIR (top-level .py files).
    Only reloads modules that are already in sys.modules.
    """
    import signal
    from typing import List
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Module reload operation timed out")
    
    code_dir = get_code_dir()
    if not os.path.exists(code_dir):
        logging.error(f"CODE_DIR {code_dir} does not exist. Reload skipped.")
        return

    logging.info(f"Attempting to reload all LLM source modules...")
    
    modules_reloaded = 0
    modules_to_reload = []
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
            logging.debug(f"Added {code_dir} to sys.path")

        logging.info(f"Scanning CODE_DIR '{code_dir}' for Python modules...")
        try:
            for f in Path(code_dir).glob("*.py"):
                module_name = f.stem
                if module_name in sys.modules:
                    modules_to_reload.append(module_name)
                    logging.debug(f"Found module to reload: {module_name}")
        except Exception as e:
            logging.error(f"Error scanning directory for modules: {e}")
            return
            
        logging.info(f"Found {len(modules_to_reload)} modules to reload: {modules_to_reload}")
        
        for i, m in enumerate(reversed(modules_to_reload)):
            try:
                logging.info(f"Reloading module {i+1}/{len(modules_to_reload)}: {m}")
                
                if m in sys.modules:
                    def reload_module():
                        return importlib.reload(sys.modules[m])
                    
                    result_queue = queue.Queue()
                    def worker():
                        try:
                            result = reload_module()
                            result_queue.put(("success", result))
                        except Exception as e:
                            result_queue.put(("error", e))
                    
                    thread = threading.Thread(target=worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=5.0)
                    
                    if thread.is_alive():
                        logging.warning(f"Module {m} reload timed out after 5 seconds - skipping")
                        continue
                    
                    try:
                        result_type, result = result_queue.get_nowait()
                        if result_type == "success":
                            modules_reloaded += 1
                            logging.debug(f"Successfully reloaded module {m}")
                        else:
                            logging.warning(f"Failed to reload module {m}: {result}")
                    except queue.Empty:
                        logging.warning(f"No result received for module {m} reload")
                else:
                    logging.debug(f"Module {m} not in sys.modules - skipping")
                    
            except Exception as e:
                logging.warning(f"Exception reloading module {m}: {e}")
                continue
                
        logging.info(f"Successfully reloaded {modules_reloaded}/{len(modules_to_reload)} modules from CODE_DIR '{code_dir}'")
        
        try:
            if 'solver' in sys.modules:
                logging.info("Attempting to reload solver module specifically...")
                
                def reload_solver():
                    return importlib.reload(sys.modules['solver'])
                
                solver_queue = queue.Queue()
                def solver_worker():
                    try:
                        result = reload_solver()
                        solver_queue.put(("success", result))
                    except Exception as e:
                        solver_queue.put(("error", e))
                
                solver_thread = threading.Thread(target=solver_worker)
                solver_thread.daemon = True
                solver_thread.start()
                solver_thread.join(timeout=5.0)
                
                if solver_thread.is_alive():
                    logging.warning("Solver module reload timed out after 5 seconds")
                else:
                    try:
                        result_type, result = solver_queue.get_nowait()
                        if result_type == "success":
                            logging.info("Successfully reloaded solver module")
                        else:
                            logging.warning(f"Failed to reload solver module: {result}")
                    except queue.Empty:
                        logging.warning("No result received for solver module reload")
            else:
                logging.debug("Solver module not in sys.modules - skipping")
        except Exception as e:
            logging.warning(f"Exception reloading solver module: {e}")
            
    except TimeoutError:
        logging.error(f"Module reload operation timed out after 30 seconds. Continuing with evaluation.")
    except Exception as e:
        logging.error(f"Error during module reload: {e}")
    finally:
        try:
            signal.alarm(0)
        except:
            pass
        
    logging.info("Module reload operation completed.")




def format_line_with_marker(
    line_num: int, line: str, marker: str, max_width: int
) -> dict:
    """
    Return raw data for line formatting with marker.
    Returns a dict with:
      - line_num: the line number
      - line: the cleaned line content
      - marker: the marker to use ('>' for changed, '|' for unchanged)
      - max_width: padding width for line numbers
    """
    return {
        "line_num": line_num,
        "line": line.rstrip("\n"),
        "marker": marker,
        "max_width": max_width,
    }



# EditorState



@dataclass
class EditorState:
    """
    Manages code directory and snapshots.
    """

    _code_dir: Optional[Path] = None
    snapshot_dir: Path = field(init=False)
    snapshot_file: Path = field(init=False)
    _initialized: bool = field(default=False, init=False)
    best_speedup: Optional[float] = None

    @property
    def code_dir(self) -> Path:
        if self._code_dir is None:
            dir_str = get_code_dir()
            code_path = Path(dir_str)
            if not code_path.exists():
                raise ValueError(f"Code directory does not exist: {code_path}")
            if not code_path.is_dir():
                raise ValueError(f"Code directory path is not a directory: {code_path}")
            self._code_dir = code_path
        return self._code_dir

    def __post_init__(self):
        _ = self.code_dir
        self.snapshot_dir = self.code_dir / ".snapshots"
        self.snapshot_file = self.code_dir / ".snapshot_metadata.json"
        self._load_initial_best_speedup()

    def _load_initial_best_speedup(self):
        """Loads the best speedup from the latest snapshot metadata, if available."""
        logging.info("Initializing best_speedup to None (loading from snapshot disabled).")
        self.best_speedup = None

    def ensure_directories(self) -> None:
        if not self._initialized:
            self.code_dir.mkdir(exist_ok=True)
            self.snapshot_dir.mkdir(exist_ok=True)
            self._initialized = True

    def get_best_speedup(self) -> Optional[float]:
        """Returns the current best recorded speedup."""
        return self.best_speedup

    def update_best_speedup(self, new_speedup: Optional[float]) -> bool:
        """Updates the best speedup if the new one is strictly better."""
        if new_speedup is None:
            return False

        current_best = self.best_speedup
        is_better = False
        
        if current_best is None:
            is_better = True
        elif new_speedup == float('inf') and current_best != float('inf'):
            is_better = True
        elif math.isfinite(new_speedup) and (current_best == float('-inf') or new_speedup > current_best):
             is_better = True
             
        if is_better:
            logging.info(f"Updating best speedup from {current_best} to {new_speedup}")
            self.best_speedup = new_speedup
            return True
        return False




class FileManager:
    """
    Handles file read/write and provides raw file view data.
    """

    def __init__(self, state: EditorState):
        self.state = state

    def read_file(self, file_path: Path) -> List[str]:
        """
        Reads a file and returns its content as a list of strings (lines).
        Handles both relative and absolute paths.
        """
        abs_path = self._make_absolute(file_path)
        logging.info(f"FileManager: Attempting to read file at absolute path: {abs_path}")

        if not abs_path.exists():
            logging.error(f"FileManager: File not found at {abs_path} during read attempt.")
            pass

        file_size = -1
        if abs_path.exists() and abs_path.is_file():
            try:
                file_size = abs_path.stat().st_size
                logging.info(f"FileManager: File {abs_path} exists. Size: {file_size} bytes before reading.")
            except Exception as e:
                logging.error(f"FileManager: Could not get size for file {abs_path}: {e}")
        elif not abs_path.exists():
            logging.info(f"FileManager: File {abs_path} does not exist before attempting read.")
        else:
            logging.info(f"FileManager: Path {abs_path} exists but is not a file (e.g., a directory).")


        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines and file_size == 0 :
                logging.info(f"FileManager: Successfully read file {abs_path}. File was empty (0 lines, 0 bytes).")
            elif not lines and file_size > 0:
                 logging.warning(f"FileManager: Successfully read file {abs_path}. File yielded 0 lines but size was {file_size} bytes. This might indicate an issue or a file with only newlines.")
            else:
                logging.info(f"FileManager: Successfully read {len(lines)} lines from {abs_path} (reported size: {file_size} bytes).")
            return lines
        except FileNotFoundError:
            logging.error(f"FileManager: FileNotFoundError when trying to open {abs_path} for reading. This should have been caught by pre-check if file truly doesn't exist.")
            # Propagate the error or return empty list based on desired strictness
            raise # Or return []
        except Exception as e:
            logging.error(f"FileManager: Error reading file {abs_path}: {e}")
            # Propagate the error or return empty list
            raise # Or return []

    def write_file(self, file_path: Path, content: Union[str, List[str]]) -> None:
        """
        Write content to a file.
        Handles both relative and absolute paths.
        If content is a list of strings, they are joined with newlines.
        Content is expected to be UTF-8 encoded.
        """
        self.state.ensure_directories()
        abs_path = self._make_absolute(file_path)
        
        # Ensure content is a single string
        if isinstance(content, list):
            content_str = "".join(content) # Assuming lines already have newlines if intended
        elif isinstance(content, str):
            content_str = content
        else:
            logging.error(f"FileManager: Invalid content type {type(content)} for writing to {abs_path}. Must be str or List[str].")
            raise TypeError("Content must be a string or list of strings")

        content_size = len(content_str.encode('utf-8'))
        logging.info(f"FileManager: Attempting to write {content_size} bytes to file at absolute path: {abs_path}")
        
        # Log if file exists and its size before overwriting
        if abs_path.exists() and abs_path.is_file():
            try:
                old_size = abs_path.stat().st_size
                logging.info(f"FileManager: File {abs_path} exists. Current size: {old_size} bytes. It will be overwritten.")
            except Exception as e:
                logging.warning(f"FileManager: Could not get size for existing file {abs_path} before overwrite: {e}")
        elif abs_path.exists() and not abs_path.is_file():
            logging.error(f"FileManager: Path {abs_path} exists but is not a file. Cannot write.")
            raise IsADirectoryError(f"Cannot write to {abs_path}, it is a directory.")

        try:
            abs_path.write_text(content_str, encoding="utf-8")
            logging.info(f"FileManager: Successfully wrote {content_size} bytes to {abs_path}.")
        except Exception as e:
            logging.error(f"FileManager: Error writing file {abs_path}: {e}")
            # Propagate the error
            raise

    def view_file(
        self,
        file_path: Path,
        changed_range: Optional[Tuple[int, int]] = None,
        start_line: int = 1,
        lines_to_view: int = 50,
        pre_context: Optional[int] = None,
        post_context: Optional[int] = None,
    ) -> dict:
        """
        Returns file contents without complex formatting.
        """
        try:
            # Get the absolute path
            abs_path = self._make_absolute(file_path)
            
            # Check if file exists
            if not abs_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path.name}",
                    "file_path": file_path.name
                }
            
            # Check if it's a file (not a directory)
            if not abs_path.is_file():
                return {
                    "success": False,
                    "error": f"Path exists but is not a file: {abs_path}",
                    "file_path": str(abs_path)
                }
            
            # Read file content
            lines = self.read_file(abs_path)
            total_lines = len(lines)
            
            if total_lines == 0:
                start_line = 0
                view_end_line = 0
                lines_to_view = 0
            else:
                start_line = max(1, min(start_line, total_lines))
                view_end_line = min(start_line + lines_to_view - 1, total_lines)
                lines_to_view = view_end_line - start_line + 1
            
            lines_slice = lines[start_line - 1 : view_end_line]

            formatted_lines = []
            line_number_width = len(str(total_lines))

            for i, line in enumerate(lines_slice, start_line):
                line_str = str(i).rjust(line_number_width)
                formatted_lines.append(f"{line_str}: {line.rstrip()}")

            formatted_content = "\n".join(formatted_lines)

            header = f"File: {abs_path.name} (lines {start_line}-{view_end_line} out of {total_lines})"
            if start_line > 1:
                header += "\n..."
            if view_end_line < total_lines:
                formatted_content += "\n..."

            return {
                "success": True,
                "file_path": str(abs_path),
                "formatted_content": formatted_content,
                "total_lines": total_lines,
                "start_line": start_line,
                "end_line": view_end_line,
                "message": f"{header}\n\n{formatted_content}",
            }
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"view_file: Error occurred: {e}")
            logging.error(f"view_file: Traceback: {tb}")
            
            return {
                "success": False,
                "error": f"Error viewing file: {str(e)}",
                "traceback": tb,
                "file_path": str(file_path)
            }

    def _make_absolute(self, file_path: Path) -> Path:
        """
        Convert a relative path to an absolute path.
        Added logging for debugging file path resolution issues.
        """
        original_type = type(file_path)
        if isinstance(file_path, str):
            file_path = Path(file_path)
            logging.info(f"FileManager._make_absolute: Converted input string '{file_path}' (original type: {original_type}) to Path object: {file_path}")
        else:
            logging.info(f"FileManager._make_absolute: Input path '{file_path}' (original type: {original_type}) is already a Path object.")
            
        logging.info(f"FileManager._make_absolute: Received path '{file_path}'. Is absolute: {file_path.is_absolute()}")
        
        if file_path.is_absolute():
            logging.info(f"FileManager._make_absolute: Path '{file_path}' is already absolute. Returning as is.")
            return file_path
            
        current_code_dir = self.state.code_dir
        if not current_code_dir.is_absolute():
            logging.warning(f"FileManager._make_absolute: code_dir '{current_code_dir}' is not absolute. Resolving it first.")
            current_code_dir = current_code_dir.resolve()
            logging.info(f"FileManager._make_absolute: Resolved code_dir to '{current_code_dir}'.")

        abs_path = (current_code_dir / file_path).resolve()
        logging.info(f"FileManager._make_absolute: Resolved relative path '{file_path}' to absolute path: {abs_path} (using code_dir: '{current_code_dir}'). Exists: {abs_path.exists()}")
        
        code_dir_env = os.environ.get("CODE_DIR", "")
        if code_dir_env:
            logging.info(f"FileManager._make_absolute: Current CODE_DIR environment variable is set to: '{code_dir_env}'")
        else:
            logging.warning("FileManager._make_absolute: CODE_DIR environment variable is not set!")
            
        return abs_path






class VariableChecker(ast.NodeVisitor):
    """AST Visitor to check for undefined variables."""

    def __init__(self):
        self.scopes = [set()]  # Stack of scopes, with globals at index 0
        self.errors = []

    def visit(self, node):
        """Override visit to handle scoping."""
        super().visit(node)

    def _add_to_current_scope(self, name):
        """Add a name to the current scope."""
        if self.scopes:
            self.scopes[-1].add(name)

    def _is_name_defined(self, name):
        """Check if name is defined in any scope or builtins."""
        return name in __builtins__ or any(name in scope for scope in self.scopes)

    def _handle_target(self, target):
        """Recursively handle target patterns in assignments and comprehensions."""
        if isinstance(target, ast.Name):
            self._add_to_current_scope(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._handle_target(elt)
        elif isinstance(target, ast.Attribute):
            self.visit(target.value)
        elif isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)

    def visit_Name(self, node):
        """Handle variable names."""
        if isinstance(node.ctx, ast.Store):
            self._add_to_current_scope(node.id)
        elif isinstance(node.ctx, ast.Load):
            if not self._is_name_defined(node.id):
                self.errors.append(
                    f"Undefined variable '{node.id}' on line {node.lineno}"
                )

    def visit_comprehension(self, node):
        """Handle a single 'for' clause in a comprehension."""
        # Handle the target (add variables to current scope)
        self._handle_target(node.target)
        # Visit the iterator and conditions
        self.visit(node.iter)
        for if_clause in node.ifs:
            self.visit(if_clause)

    def _visit_comprehension_generic(self, node):
        """Generic handler for all types of comprehensions."""
        # Create a single scope for the entire comprehension
        self.scopes.append(set())

        # Process all generators (for clauses) within this scope
        for gen in node.generators:
            self.visit(gen)  # This will handle each 'comprehension' node

        # Visit the element part (e.g., x in [x for ...])
        if hasattr(node, "elt"):  # ListComp, SetComp, GeneratorExp
            self.visit(node.elt)
        elif hasattr(node, "key"):  # DictComp
            self.visit(node.key)
            self.visit(node.value)

        # Pop the comprehension's scope
        self.scopes.pop()

    def visit_ListComp(self, node):
        self._visit_comprehension_generic(node)

    def visit_SetComp(self, node):
        self._visit_comprehension_generic(node)

    def visit_GeneratorExp(self, node):
        self._visit_comprehension_generic(node)

    def visit_DictComp(self, node):
        self._visit_comprehension_generic(node)

    def visit_FunctionDef(self, node):
        """Handle function definitions."""
        # Add function name to the current scope
        self._add_to_current_scope(node.name)

        # Create new scope for function body
        self.scopes.append(set())

        # Add arguments to function scope
        for arg in node.args.args:
            self._add_to_current_scope(arg.arg)
        if node.args.vararg:
            self._add_to_current_scope(node.args.vararg.arg)
        if node.args.kwarg:
            self._add_to_current_scope(node.args.kwarg.arg)
        for arg in node.args.kwonlyargs:
            self._add_to_current_scope(arg.arg)

        # Visit function body
        for stmt in node.body:
            self.visit(stmt)

        # Pop function scope
        self.scopes.pop()

    ###########################################################################
    # NEWLY ADDED: Handle lambda function parameters properly
    ###########################################################################
    def visit_Lambda(self, node):
        """
        Handle lambda expressions by creating a new scope for parameters
        so that variable references inside the lambda are recognized.
        """
        # Create a new scope for the lambda
        self.scopes.append(set())

        # Add lambda parameters to scope
        for arg in node.args.args:
            self._add_to_current_scope(arg.arg)
        if node.args.vararg:
            self._add_to_current_scope(node.args.vararg.arg)
        if node.args.kwarg:
            self._add_to_current_scope(node.args.kwarg.arg)
        for arg in node.args.kwonlyargs:
            self._add_to_current_scope(arg.arg)

        # Visit the body (the expression node)
        self.visit(node.body)

        # Pop the lambda scope
        self.scopes.pop()

    ###########################################################################

    def visit_ClassDef(self, node):
        """Handle class definitions."""
        # Add class name to current scope
        self._add_to_current_scope(node.name)

        # Create new scope for class body
        self.scopes.append(set())

        # Visit class body
        for stmt in node.body:
            self.visit(stmt)

        # Pop class scope
        self.scopes.pop()

    def visit_Import(self, node):
        """Handle import statements."""
        for name in node.names:
            if name.asname:
                self._add_to_current_scope(name.asname)
            else:
                self._add_to_current_scope(name.name.split(".")[0])

    def visit_ImportFrom(self, node):
        """Handle from-import statements."""
        for name in node.names:
            if name.asname:
                self._add_to_current_scope(name.asname)
            else:
                self._add_to_current_scope(name.name)

    @staticmethod
    def validate(tree):
        checker = VariableChecker()
        checker.visit(tree)
        return checker.errors


class CodeValidator:
    """
    Handles formatting, optional 'solve' function checks, etc.
    """

    @staticmethod
    def run_pylint(content: str) -> List[str]:
        """
        Run pylint on the given content and return any errors/warnings.
        Note: Array-related lint checks (which can mess up square arrays) have been disabled.
        """
        # Create a temporary file to run pylint on
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            tmp_path = tmp_file.name

        try:
            # --- RESTORED: Use JSON Reporter ---
            output_stream = StringIO()
            reporter = JSONReporter(output_stream)
            lint_output = [] # Initialize just in case

            # Run pylint with the original full set of checks
            try:
                # Removed redirect_stdout/stderr
                Run(
                    [
                        tmp_path,
                        "--rcfile=/dev/null",
                        "--disable=unused-import,unexpected-keyword-arg,redundant-keyword-arg,no-value-for-parameter,redefined-builtin,broad-exception-caught,logging-fstring-interpolation,import-error,undefined-variable,return-in-init",
                    ],
                    reporter=reporter, # Use JSON reporter
                    exit=False,
                )
            except Exception as pylint_err:
                 # Keep existing crash handling
                 logging.error(f"ERROR: Pylint execution itself failed: {pylint_err}")
                 logging.error(traceback.format_exc())
                 lint_output_str = output_stream.getvalue()
                 if lint_output_str:
                     try:
                        lint_output = json.loads(lint_output_str)
                     except json.JSONDecodeError:
                        lint_output = [{"type": "fatal", "message": f"Pylint crashed: {pylint_err}", "symbol": "pylint-crash", "line": 0}]
                 else:
                     lint_output = [{"type": "fatal", "message": f"Pylint crashed: {pylint_err}", "symbol": "pylint-crash", "line": 0}]
            else:
                # --- RESTORED: Parse JSON output ---
                lint_output_str = output_stream.getvalue()
                try:
                     lint_output = json.loads(lint_output_str)
                except json.JSONDecodeError as json_err:
                     logging.error(f"ERROR: Failed to parse Pylint JSON output: {json_err}")
                     logging.error(f"Raw Pylint output was: {lint_output_str}")
                     lint_output = [{"type": "fatal", "message": f"Pylint JSON parse error: {json_err}", "symbol": "pylint-json-error", "line": 0}]


            # --- RESTORED: Filter and format messages from JSON ---
            errors = []
            # Add safety checks for parsing
            if isinstance(lint_output, list):
                for msg in lint_output:
                    if isinstance(msg, dict) and all(k in msg for k in ['type', 'line', 'message', 'symbol']):
                        if msg["type"] in ("error", "fatal"):  # Ignore warnings—they are informational
                            errors.append(
                                f"Line {msg['line']}: {msg['message']} ({msg['symbol']})"
                            )
                    elif isinstance(msg, dict) and msg.get('type') == 'fatal':
                         errors.append(f"Fatal Pylint Error: {msg.get('message', 'Unknown fatal error')}")
                    else:
                        logging.warning(f"Skipping malformed Pylint message: {msg}")
            else:
                 logging.error(f"Pylint output was not a list as expected: {lint_output}")
                 # Optionally add a generic error if parsing failed badly
                 if lint_output: # Add error only if output wasn't empty/None
                     errors.append("Failed to parse Pylint output structure.")

            return errors

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

    @staticmethod
    def validate_python_syntax(content: str) -> Tuple[bool, Optional[str]]:
        """
        Validates Python syntax by parsing the code and checking for common issues.
        Returns (is_valid, error_message).
        """
        try:
            # First try to parse the code
            tree = ast.parse(content)

            # Run pylint for additional checks on the actual content
            lint_errors = CodeValidator.run_pylint(content)
            if lint_errors:
                return False, "\n".join(lint_errors)

            return True, None

        except SyntaxError as e:
            # Format the error manually to avoid the unwanted comma from str(e)
            error_msg = f"Syntax error: {e.msg} (line {e.lineno})"
            return False, error_msg
        except Exception as e:
            return False, f"Error validating Python syntax: {str(e)}"

    @staticmethod
    def format_python(content: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates Python syntax without formatting.
        Returns (success, error_message, content).
        """
        # Only validate syntax
        is_valid, error_msg = CodeValidator.validate_python_syntax(content)
        if not is_valid:
            return False, error_msg, None

        # Return original content without formatting
        return True, None, content

    @staticmethod
    def validate_solve_function(tree: ast.AST) -> Tuple[bool, Optional[str]]:
        solve_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "solve":
                solve_node = node
                break
        if not solve_node:
            return False, "Function 'solve' not found."
        num_args = len(solve_node.args.args)
        if num_args != 1:
            return False, "Function 'solve' must take exactly one argument."
        has_return = any(isinstance(x, ast.Return) for x in ast.walk(solve_node))
        if not has_return:
            return False, "Function 'solve' must contain a return statement."
        return True, None



# SnapshotManager



class SnapshotManager:
    """
    Manages code snapshots in .snapshots, verifying file hashes.
    """

    def __init__(self, state: EditorState):
        self.state = state

    def save_snapshot(self) -> str:
        try:
            self.state.ensure_directories()
            code_dir = self.state.code_dir
            logging.info(f"Saving snapshot: code_dir={code_dir}, snapshot_dir={self.state.snapshot_dir}")

            # Create the temporary staging directory in the parent of code_dir
            # to prevent it from being included in code_dir.rglob("*").
            staging_parent_dir = code_dir.parent
            with tempfile.TemporaryDirectory(prefix="algotune_snapshot_stage_", dir=staging_parent_dir) as tempdir_str:
                temp_path = Path(tempdir_str)
                logging.info(f"Snapshot staging directory: {temp_path}")
                files_dict = {}

                if not code_dir.exists():
                    logging.error(f"Code directory does not exist: {code_dir}")
                    raise ValueError(f"Code directory does not exist: {code_dir}")

                # Count files to be included in snapshot
                file_count = 0
                for f in code_dir.rglob("*"):
                    if not self._should_ignore_file(f):
                        file_count += 1
                
                logging.info(f"Found {file_count} files to include in snapshot")
                
                # Copy files to temporary directory and calculate hashes
                for f in code_dir.rglob("*"):
                    if self._should_ignore_file(f):
                        continue
                    rel_path = f.relative_to(code_dir)
                    dest = temp_path / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)

                    source_hash = self._calc_hash(f)
                    dest_hash = self._calc_hash(dest)
                    if source_hash != dest_hash:
                        logging.error(f"Hash mismatch copying {f}")
                        raise IOError(f"Hash mismatch copying {f}")

                    files_dict[str(rel_path)] = {
                        "hash": source_hash,
                        "size": f.stat().st_size,
                    }

                if not files_dict:
                    logging.error("No files found to snapshot in code directory.")
                    raise ValueError("No files found to snapshot in code directory.")

                meta = {"files": files_dict, "code_dir": str(code_dir)}
                snap_id = hashlib.sha256(json.dumps(meta).encode()).hexdigest()[:8]
                snap_path = self.state.snapshot_dir / f"snapshot_{snap_id}"
                
                logging.info(f"Creating snapshot with ID {snap_id} at path {snap_path}")

                if snap_path.exists():
                    logging.info(f"Removing existing snapshot at {snap_path}")
                    shutil.rmtree(snap_path)

                shutil.copytree(temp_path, snap_path)

                meta["snapshot_id"] = snap_id
                meta["snapshot_path"] = str(snap_path)
                
                # --- ADDED: Get and store current best_speedup --- 
                current_best_speedup = self.state.get_best_speedup()
                meta["best_speedup"] = current_best_speedup
                logging.info(f"Storing best_speedup in metadata: {current_best_speedup}")
                # --- END ADDED ---

                logging.info(f"Writing snapshot metadata to {self.state.snapshot_file}")
                self.state.snapshot_file.write_text(json.dumps(meta, indent=2))
                
                # --- ADDED: Write metadata inside the snapshot directory as well ---
                snapshot_meta_file = snap_path / "metadata.json"
                logging.info(f"Writing metadata inside snapshot directory: {snapshot_meta_file}")
                snapshot_meta_file.write_text(json.dumps(meta, indent=2))
                # --- END ADDED ---

                logging.info(f"Snapshot saved successfully with ID: {snap_id}")
                return f"State saved successfully with snapshot ID: {snap_id}"

        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Failed to save snapshot: {e}\n{tb}")
            return f"Failed to save snapshot:\n{tb}"

    def restore_snapshot(self) -> str:
        """
        Attempts to restore from the last snapshot. Returns a dict indicating success or error.
        """
        try:
            self.state.ensure_directories()
            
            logging.info(f"Restoring snapshot: snapshot file exists = {self.state.snapshot_file.exists()}")
            logging.info(f"Snapshot file path: {self.state.snapshot_file}")

            if not self.state.snapshot_file.exists():
                logging.error("No snapshot file exists")
                return "No saved state to revert to."

            meta = json.loads(self.state.snapshot_file.read_text())
            snap_path = Path(meta["snapshot_path"])
            
            logging.info(f"Snapshot path from metadata: {snap_path}")
            logging.info(f"Snapshot path exists = {snap_path.exists()}")

            if not snap_path.exists():
                logging.error(f"Snapshot directory not found: {snap_path}")
                return "Snapshot not found or corrupted."

            code_dir = self.state.code_dir
            stored_cd = meta.get("code_dir")
            if stored_cd and str(code_dir) != stored_cd:
                logging.error(f"Snapshot code dir ({stored_cd}) doesn't match current ({code_dir})")
                return (
                    f"Snapshot was created for a different code directory: {stored_cd}"
                )

            # verify
            for rel_path_str, info in meta["files"].items():
                sf = snap_path / rel_path_str
                if not sf.exists():
                    logging.error(f"Missing file in snapshot: {rel_path_str}")
                    return f"Snapshot verification failed: missing file {rel_path_str}"
                if self._calc_hash(sf) != info["hash"]:
                    logging.error(f"Hash mismatch for file in snapshot: {rel_path_str}")
                    return f"Snapshot verification failed: hash mismatch for {rel_path_str}"

            # backup current
            with tempfile.TemporaryDirectory() as backupdir:
                backup_path = Path(backupdir)
                # Map: original relative path string -> hashed backup filename string
                backup_map_file = backup_path / "backup_map.json"
                backup_files_map = {}

                for f in code_dir.rglob("*"):
                    if not self._should_ignore_file(f):
                        relp = f.relative_to(code_dir)
                        hashed_filename = hashlib.sha256(str(relp).encode()).hexdigest()
                        
                        # Backup to a flat structure using hashed names
                        backup_target_path = backup_path / hashed_filename
                        shutil.copy2(f, backup_target_path)
                        backup_files_map[str(relp)] = hashed_filename
                
                with open(backup_map_file, 'w') as bmf:
                    json.dump(backup_files_map, bmf)

                try:
                    # remove everything except ignored
                    for f in code_dir.rglob("*"):
                        if not self._should_ignore_file(f):
                            if f.is_file(): # Only unlink files, leave dirs for now
                                f.unlink()
                            elif f.is_dir(): # Attempt to remove empty dirs, ignore if not empty
                                try:
                                    f.rmdir() 
                                except OSError:
                                    pass # Directory not empty, will be handled if files within are removed

                    # Clear out potentially empty directory structures after file unlinking
                    # This is a bit more aggressive to ensure clean state before restore
                    for d in list(code_dir.rglob("*")): # Get a list first as rglob is a generator
                        if d.is_dir() and not self._should_ignore_file(d) and not any(d.iterdir()):
                            try:
                                shutil.rmtree(d) # remove dir and all its contents if it became empty
                            except OSError as e: # catch if it was removed by parent rmtree or other race
                                logging.debug(f"Could not remove dir {d} during cleanup: {e}")
                        elif d.is_file() and not self._should_ignore_file(d): # Should have been unlinked already
                             d.unlink(missing_ok=True)


                    # restore from the selected snapshot
                    for rel_path_str in meta["files"]:
                        src = snap_path / rel_path_str
                        tgt = code_dir / rel_path_str
                        tgt.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, tgt)

                        # verify
                        if self._calc_hash(tgt) != meta["files"][rel_path_str]["hash"]:
                            raise IOError(
                                f"Restore verification mismatch for {rel_path_str}"
                            )

                    # Load metadata associated with the reverted snapshot
                    meta_path = snap_path / "metadata.json"
                    reverted_metadata = {}
                    if meta_path.exists():
                        try:
                            with open(meta_path, 'r') as f:
                                reverted_metadata = json.load(f)
                        except Exception as e:
                            logging.warning(f"Could not load metadata for reverted snapshot {snap_path}: {e}")
                    
                    # Restore best_speedup from the metadata
                    restored_speedup = reverted_metadata.get('best_speedup')
                    if restored_speedup is not None:
                        self.state.update_best_speedup(restored_speedup)
                        logging.info(f"Restored best speedup to {self.state.best_speedup} from snapshot '{snap_path}'.")
                    else:
                        # If not in metadata, maybe reset to None or keep current?
                        # Resetting to None seems safer, requires re-evaluation to set a new best.
                        self.state.best_speedup = None
                        logging.warning(f"Could not find best_speedup in metadata for snapshot '{snap_path}'. Resetting best speedup to None.")

                    return "Successfully reverted to last saved state."

                except Exception: # This is the block where we restore from the temporary backup
                    tb = traceback.format_exc()
                    logging.error(f"Error during snapshot restore, attempting to revert from temporary backup: {tb}")
                    
                    # Clear code_dir again before restoring from backup to avoid conflicts
                    for f_to_delete in code_dir.rglob("*"):
                        if not self._should_ignore_file(f_to_delete):
                            if f_to_delete.is_file():
                                f_to_delete.unlink(missing_ok=True)
                            elif f_to_delete.is_dir():
                                shutil.rmtree(f_to_delete, ignore_errors=True)

                    # Restore from the temporary backup using the map
                    if backup_map_file.exists():
                        try:
                            with open(backup_map_file, 'r') as bmf:
                                saved_backup_files_map = json.load(bmf)
                            
                            for original_rel_path_str, hashed_filename_str in saved_backup_files_map.items():
                                backup_file_src_path = backup_path / hashed_filename_str
                                if backup_file_src_path.is_file():
                                    original_target_path = code_dir / Path(original_rel_path_str)
                                    original_target_path.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(backup_file_src_path, original_target_path)
                                else:
                                    logging.warning(f"Hashed backup file {hashed_filename_str} not found for original path {original_rel_path_str}")
                            logging.info("Successfully reverted changes from temporary backup.")
                            return f"Error during revert. Backed out changes successfully using temporary backup.\\n{tb}"
                        except Exception as backup_restore_exc:
                            logging.error(f"CRITICAL: Failed to restore from temporary backup: {backup_restore_exc}")
                            return f"Error during revert, AND FAILED TO RESTORE FROM BACKUP. Code directory may be in an inconsistent state.\\nOriginal error: {tb}\\nBackup restore error: {backup_restore_exc}"
                    else:
                        logging.error("CRITICAL: Backup map file not found. Cannot restore from temporary backup.")
                        return f"Error during revert, AND BACKUP MAP NOT FOUND. Code directory may be in an inconsistent state.\\n{tb}"

        except Exception: # Outer exception for the whole restore_snapshot
            tb = traceback.format_exc()
            logging.error(tb)
            return f"Failed to restore snapshot:\n{tb}"

    def _calc_hash(self, fp: Path) -> str:
        return hashlib.sha256(fp.read_bytes()).hexdigest()

    def _should_ignore_file(self, f: Path) -> bool:
        if f is None:
            return True
        return (
            not f.is_file()
            or f.name.startswith(".snapshot")
            or f == self.state.snapshot_file
            or ".snapshots" in f.parts
            or "__pycache__" in f.parts
            or f.suffix == ".pyc"
        )



# Main Editor



class Editor:
    """
    Main Editor orchestrating file operations, code formatting, and snapshots.
    This layer returns raw outputs and error details without extra formatting.
    The MessageWriter is responsible for converting these results into human-readable messages.
    """

    def __init__(self, state: EditorState):
        self.state = state
        self.file_manager = FileManager(state)
        self.code_validator = CodeValidator()
        self.snapshot_manager = SnapshotManager(state)
        # NOTE: We no longer call MessageWriter here; the editor now returns raw data.

    def _validate_and_format_python(
        self, content: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates Python syntax and runs pylint. Does not format.
        Returns (is_valid, error_message, original_content_if_valid).
        """
        is_valid, error_msg = self.code_validator.validate_python_syntax(content)
        if not is_valid:
            return False, error_msg, None
        
        # No formatting step needed based on current CodeValidator.format_python
        # If formatting were reintroduced, it would happen here.
        # For now, just return the original content if valid.
        return True, None, content

    def list_files(self) -> dict:
        """
        Returns a dict with the list of file names in the CODE_DIR directory.
        """
        try:
            code_dir = os.environ.get("CODE_DIR")
            if not code_dir:
                return {
                    "success": False,
                    "error": "CODE_DIR environment variable not set",
                    "context": "listing files",
                }

            code_path = Path(code_dir)
            if not code_path.exists():
                return {
                    "success": False,
                    "error": f"CODE_DIR path {code_dir} does not exist",
                    "context": "listing files",
                }

            files = [
                f.name
                for f in code_path.glob("*")
                if f.is_file()
                and not f.name.startswith(".")
                and f.name != "__init__.py"
            ]
            return {"success": True, "files": sorted(files)}
        except Exception:
            tb = traceback.format_exc()
            logging.error(tb)
            return {"success": False, "error": tb, "context": "listing files"}

    def find_string(self, file_path: Path, search_str: str) -> dict:
        """
        Returns a dict with any hits of the search string in the file.
        Each hit is a tuple of (line number, line content).
        """
        try:
            lines = self.file_manager.read_file(file_path)
            hits = [(i, ln) for i, ln in enumerate(lines, 1) if search_str in ln]
            return {"success": True, "search_str": search_str, "hits": hits}
        except Exception:
            tb = traceback.format_exc()
            logging.error(tb)
            return {"success": False, "error": tb, "context": "searching for string"}

    def save_snapshot(self) -> dict:
        """
        Attempts to save a snapshot. Returns a dict indicating success or error.
        """
        result = self.snapshot_manager.save_snapshot()
        if "Failed" in result or "Error" in result:
            return {
                "success": False,
                "error": result,
                "context": "saving snapshot",
                "snapshot_status": SnapshotStatus.FAILURE.value,
            }
        return {
            "success": True,
            "message": result,
            "snapshot_status": SnapshotStatus.SUCCESS.value,
        }

    def restore_snapshot(self) -> dict:
        """
        Attempts to restore from the last snapshot. Returns a dict indicating success or error.
        """
        result = self.snapshot_manager.restore_snapshot()
        if "Error" in result or "Failed" in result:
            return {
                "success": False,
                "error": result,
                "context": "restoring snapshot",
                "snapshot_status": SnapshotStatus.FAILURE.value,
            }
        return {
            "success": True,
            "message": result,
            "snapshot_status": SnapshotStatus.SUCCESS.value,
        }

    def revert(self) -> dict:
        """
        Alias for restore_snapshot to maintain compatibility with existing code.
        """
        return self.restore_snapshot()

    def _extract_dace_error(self, stderr_text: str) -> str:
        """
        Extract clean DaCe error message from stderr, removing Python traceback noise.
        Returns just the DaceSyntaxError and location info.
        """
        if not stderr_text:
            return "DaCe compilation failed."
        
        lines = stderr_text.strip().split('\n')
        dace_error_lines = []
        
        for line in lines:
            # Look for DaCe-specific error lines
            if ('dace.frontend.python.common.DaceSyntaxError:' in line or 
                'DaceSyntaxError:' in line or
                line.strip().startswith('encountered in File')):
                # Clean up file paths to show only filename
                if 'encountered in File' in line and '/' in line:
                    # Extract just the filename from the full path
                    import os
                    parts = line.split('File "')
                    if len(parts) > 1:
                        path_and_rest = parts[1]
                        if '"' in path_and_rest:
                            full_path = path_and_rest.split('"')[0]
                            filename = os.path.basename(full_path)
                            line = line.replace(full_path, filename)
                dace_error_lines.append(line.strip())
        
        if dace_error_lines:
            return '\n'.join(dace_error_lines)
        else:
            # Fallback: return last few non-empty lines if no DaCe error found
            non_empty = [line.strip() for line in lines if line.strip()]
            if non_empty:
                return non_empty[-1]
            return "DaCe compilation failed."

    def edit_file(
        self,
        file_path: Path,
        start_line: int,
        end_line: int,
        new_content: Optional[str],
    ) -> dict:
        """
        Edits a file with the following behavior:
          - If start_line and end_line are both 0, the given code is prepended to the file.
          - Otherwise, lines [start_line, end_line] are first deleted and then the given code is inserted starting at start_line.
          - To delete code without inserting new code, provide an empty new_content.

        After the edit, the file is validated and formatted with Black.

        Returns a dict with:
          - success (bool)
          - error (if any)
          - formatted: the new (formatted) content (if changes were applied)
          - old_content: the content prior to the edit (always present)
          - proposed_content: the content before formatting
          - changed_range: a tuple (start, end) of the actual modified lines
          - temp_file_content: the content as it appeared in the temporary file
          - temp_file_error_line: the line number where error occurred in temp file
          - file_path: str
        """
        old_content = ""
        joined_proposed = ""
        tmp_path = None # Keep tmp_path for now, though not used for python validation
        reverted_due_to_compilation = False # Initialize revert flag

        try:
            # Ensure start_line and end_line are integers and validate
            if start_line is None:
                start_line = 0
            elif isinstance(start_line, str) and start_line.isdigit():
                start_line = int(start_line)
                
            if end_line is None:
                end_line = 0
            elif isinstance(end_line, str) and end_line.isdigit():
                end_line = int(end_line)
                
            # Validate line numbers
            if start_line < 0:
                return {
                    "success": False,
                    "error": f"Start line {start_line} must be greater than or equal to 0",
                    "old_content": "",
                    "current_code": "",
                    "proposed_code": new_content or "",
                    "changed_range": None,
                    "temp_file_content": new_content or "",
                    "file_path": str(file_path),
                }
            if start_line == 0:
                # Prepend mode – start_line must be 0, end_line can be any value
                pass  # No validation needed for end_line, allow any value
            else:
                # For deletion/insertion mode, ensure end_line is not less than start_line
                if end_line < start_line:
                    return {
                    "success": False,
                    "error": f"End line ({end_line}) must be greater than or equal to start line ({start_line})",
                    "old_content": "",
                    "current_code": "",
                    "proposed_code": new_content or "",
                    "changed_range": None,
                    "temp_file_content": new_content or "",
                        "file_path": str(file_path),
                    }

            # --- NEW: Resolve path and check existence before reading ---
            abs_path = self.file_manager._make_absolute(file_path)
            original_lines = []
            old_content = ""

            if abs_path.exists():
                logging.info(f"File {abs_path} exists, reading content.")
                try:
                    original_lines = self.file_manager.read_file(file_path) # Read existing file
                    old_content = "".join(original_lines)
                except Exception as e:
                    logging.error(f"Failed to read existing file {file_path}: {e}")
                    # Return error if reading an *existing* file fails
                    return {
                        "success": False,
                        "error": f"Failed to read existing file: {str(e)}",
                        "old_content": "", # No old content available
                        "current_code": "",
                        "proposed_code": new_content or "",
                        "changed_range": None,
                        "temp_file_content": new_content or "",
                        "file_path": str(file_path),
                    }
            else:
                logging.info(f"File {abs_path} does not exist. Will attempt to create.")
                # original_lines and old_content remain empty

            # --- END: Check existence ---

            # Prepare new content lines (existing logic)
            if new_content is None:
                new_content = ""
            # If new_content is not empty, clean it up (remove any leading colons but preserve indentation)
            if new_content:
                new_content = new_content.lstrip(":")
            new_lines = [
                ln + "\n" if not ln.endswith("\n") else ln
                for ln in new_content.splitlines()
            ]

            # Determine the proposed content based on the operation mode
            if start_line == 0:
                if end_line == 0:
                    # Simple prepend mode: insert new_lines at the beginning; no deletion
                    proposed_content_lines = new_lines + original_lines
                    changed_start = 1
                    changed_end = len(new_lines)
                else:
                    # Prepend and delete mode: insert new_lines at the beginning and delete lines 1 to end_line
                    proposed_content_lines = new_lines + original_lines[end_line:]
                    changed_start = 1
                    changed_end = len(new_lines)
            else:
                total = len(original_lines)
                if start_line > total + 1:
                    return {
                    "success": False,
                    "error": f"Start line {start_line} is greater than the file length ({total}) + 1",
                    "old_content": old_content,
                    "current_code": old_content,
                    "proposed_code": new_content,
                    "changed_range": None,
                    "temp_file_content": new_content,
                    "file_path": str(file_path),
                }
                # Adjust end_line if it exceeds the current file length
                # end_line = min(end_line, total)
                # Delete lines [start_line, end_line] and insert new_lines at start_line
                proposed_content_lines = (
                    original_lines[: start_line - 1]
                    + new_lines
                    + original_lines[end_line:]
                )
                if new_lines:
                    changed_start = start_line
                    changed_end = start_line + len(new_lines) - 1
                else:
                    # Deletion only: report the deleted range
                    changed_start = start_line
                    changed_end = end_line

            joined_proposed = "".join(proposed_content_lines)


            # Determine file type and apply appropriate validation/formatting
            file_type = file_path.suffix.lower()
            content_to_write = joined_proposed
            validation_error_msg = None
            error_line = None

            if file_type == ".py":
                logging.debug(f"Processing as Python file: {file_path}")
                is_valid, py_error_msg, formatted_content = self._validate_and_format_python(
                    joined_proposed
                )
                if not is_valid:
                    validation_error_msg = py_error_msg
                else:
                    # Use validated content (currently same as proposed, as no formatting)
                    content_to_write = formatted_content 
            else:
                logging.debug(f"Processing as non-Python file: {file_path}, skipping validation.")
                # For non-Python files, use the proposed content directly
                content_to_write = joined_proposed

            # Handle validation errors
            if validation_error_msg:
                # Try to extract error line number more robustly
                error_line = None
                match = re.search(r"[Ll]ine\s*(\d+)", validation_error_msg) # Case-insensitive, allows optional space
                if match:
                    try:
                        error_line = int(match.group(1))
                    except:
                        pass # Keep error_line as None if parsing fails

                return {
                    "success": False,
                    "error": validation_error_msg,
                    "old_content": old_content,
                    "current_code": old_content,
                    "proposed_code": joined_proposed,
                    "changed_range": (changed_start, changed_end),
                    "temp_file_content": joined_proposed,  # Keep original proposed content for context
                    "temp_file_error_line": error_line,
                    "file_path": str(file_path),
                }

            # Write the final content to the file
            compilation_status = None # Initialize compilation status
            try:
                # Use content_to_write which is either validated/formatted (py) or original (pyx/other)
                self.file_manager.write_file(file_path, content_to_write)

                # --- ADDED: Conditional Cython, Pythran, and DaCe Compilation ---
                # Check if this is a Pythran file (by content)
                is_pythran_file = False
                # Check if this is a DaCe-decorated file (by content)
                is_dace_file = False
                if file_type == ".py" and "@dace.program" in content_to_write:
                    is_dace_file = True
                    logging.info(f"Detected DaCe file: {file_path}")
                if file_type == ".py":
                    # Check if content contains pythran export
                    if "pythran export" in content_to_write.lower():
                        is_pythran_file = True
                        logging.info(f"Detected Pythran file: {file_path}")

                # If we just created or updated the build script, force a Cython rebuild
                if file_path.name in ["setup.py", "pyproject.toml"]:
                    try:
                        compile_cwd = str(self.state.code_dir)
                        compile_timeout = 1800
                        logging.info("Detected build script change, running full Cython rebuild.")
                        process = subprocess.run(
                            [sys.executable, "-m", "pip", "install", ".", "--no-deps", "--force-reinstall", "--no-cache-dir"],
                            cwd=compile_cwd,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=compile_timeout,
                        )
                        if process.returncode != 0:
                            logging.error(f"Cython rebuild failed after {file_path.name} update: {process.stderr}")
                        else:
                            logging.info("Cython rebuild successful after build script update.")
                    except subprocess.TimeoutExpired as e:
                        logging.error(f"Cython rebuild timed out after {file_path.name} update: {e}")
                    # Continue, skip per-file compile for this edit
                    compilation_status = {"success": True, "error": None, "dependency_check": None}
                elif is_pythran_file:
                    logging.info(f"Detected Pythran file, attempting compilation: {file_path}")
                    # Run Pythran compile in the file's directory so artifacts stay with the source
                    abs_file_path = self.file_manager._make_absolute(file_path)
                    compile_cwd = str(abs_file_path.parent)
                    compile_timeout = 300  # Shorter timeout for Pythran
                    compilation_status = {
                        "success": False,
                        "error": "Pythran compilation not attempted.",
                        "dependency_check": None,
                    }
                    
                    # --- Pythran Dependency Check ---
                    dependency_check_cmd = [
                        sys.executable,
                        "-c",
                        "import pythran; import numpy; print('Pythran and NumPy OK')"
                    ]
                    logging.info(f"Running Pythran dependency check: {' '.join(dependency_check_cmd)}")
                    dependency_check_ok = False
                    try:
                        dep_check_process = subprocess.run(
                            dependency_check_cmd,
                            cwd=compile_cwd,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=30
                        )
                        compilation_status["dependency_check"] = {
                            "exit_code": dep_check_process.returncode,
                            "stdout": dep_check_process.stdout,
                            "stderr": dep_check_process.stderr,
                        }
                        if dep_check_process.returncode == 0:
                            dependency_check_ok = True
                            logging.info("Pythran dependency check successful.")
                        else:
                            logging.error(f"Pythran dependency check failed! Exit Code: {dep_check_process.returncode}")
                            compilation_status["success"] = False
                            compilation_status["error"] = "Pythran not found or not importable."
                            compilation_status["stdout"] = dep_check_process.stdout
                            compilation_status["stderr"] = dep_check_process.stderr
                    except Exception as dep_err:
                        logging.error(f"Error running Pythran dependency check: {dep_err}")
                        compilation_status["dependency_check"] = {"error": str(dep_err)}
                        compilation_status["success"] = False
                        compilation_status["error"] = f"Failed to run Pythran dependency check: {dep_err}"
                    
                    # Only attempt compilation if dependency check passed
                    if dependency_check_ok:
                        abs_file_path = self.file_manager._make_absolute(file_path)
                        compile_command = ["pythran", "-O3", "-march=native", str(abs_file_path)]
                        try:
                            process = subprocess.run(
                                compile_command,
                                cwd=compile_cwd,
                                capture_output=True,
                                text=True,
                                check=False,
                                timeout=compile_timeout,
                            )
                            
                            compilation_status = {
                                "success": process.returncode == 0,
                                "exit_code": process.returncode,
                                "stdout": process.stdout,
                                "stderr": process.stderr,
                                "command": " ".join(compile_command),
                                "cwd": compile_cwd,
                                "error": None,
                            }
                            
                            if process.returncode != 0:
                                logging.warning(f"Pythran compilation failed for {file_path} with exit code {process.returncode}")
                                logging.warning(f"Compilation stderr:\n{process.stderr}")
                                compilation_status["error"] = f"Pythran compilation failed with exit code {process.returncode}"
                                # Include stderr in error message, cleaning file paths to only show filenames
                                stderr_text = compilation_status.get("stderr", "")
                                if stderr_text:
                                    cleaned_lines = [l.split("/")[-1] for l in stderr_text.splitlines()]
                                    cleaned_stderr = "\n".join(cleaned_lines)
                                    compilation_status["error"] += f": {cleaned_stderr}"
                            else:
                                logging.info(f"Pythran compilation successful for {file_path}")
                                
                        except subprocess.TimeoutExpired as e:
                            logging.error(f"Pythran compilation timed out for {file_path}: {e}")
                            compilation_status = {
                                "success": False,
                                "error": f"Pythran compilation timed out after {compile_timeout} seconds.",
                                "stdout": e.stdout.decode() if e.stdout else "",
                                "stderr": e.stderr.decode() if e.stderr else "",
                                "command": " ".join(compile_command),
                                "cwd": compile_cwd,
                            }
                        except Exception as e:
                            logging.error(f"Error during Pythran compilation for {file_path}: {e}")
                            compilation_status = {
                                "success": False,
                                "error": f"An unexpected error occurred during Pythran compilation: {str(e)}",
                                "stdout": "",
                                "stderr": str(e),
                                "command": " ".join(compile_command),
                                "cwd": compile_cwd,
                            }
                    
                    # Revert on compilation failure
                    if not compilation_status.get("success"):
                        logging.warning(f"Pythran compilation failed for {file_path}. Reverting file content.")
                        try:
                            self.file_manager.write_file(file_path, old_content)
                            reverted_due_to_compilation = True
                            logging.info(f"Successfully reverted {file_path} to its previous state.")
                        except Exception as revert_err:
                            logging.error(f"Failed to revert {file_path} after Pythran compilation error: {revert_err}")
                        # Immediately return failure so UI shows old file content
                        return {
                            "success": False,
                            "error": compilation_status.get("error", "Pythran compilation failed."),
                            "old_content": old_content,
                            "current_code": old_content,
                            "proposed_code": old_content,
                            "changed_range": (changed_start, changed_end),
                            "temp_file_content": old_content,
                            "file_path": str(file_path),
                            "compilation_status": compilation_status,
                            "reverted_due_to_compilation": reverted_due_to_compilation,
                        }
                elif is_dace_file:
                    logging.info(f"Detected DaCe file, attempting import/compile: {file_path}")
                    # Run DaCe import/compilation in the file's directory so artifacts stay with the source
                    abs_file_path = self.file_manager._make_absolute(file_path)
                    compile_cwd = str(abs_file_path.parent)
                    compile_timeout = 300
                    # Prepare DaCe import command to trigger JIT
                    module_name = file_path.stem
                    import_cmd = [sys.executable, "-c", f"import {module_name}"]
                    logging.info(f"Running DaCe import: {' '.join(import_cmd)}")
                    compilation_status = {
                        "success": False,
                        "error": "DaCe compilation not attempted.",
                        "stdout": None,
                        "stderr": None,
                        "command": ' '.join(import_cmd),
                        "cwd": compile_cwd,
                    }
                    try:
                        process = subprocess.run(
                            import_cmd,
                            cwd=compile_cwd,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=compile_timeout,
                        )
                        compilation_status.update({
                            "exit_code": process.returncode,
                            "stdout": process.stdout,
                            "stderr": process.stderr,
                        })
                        if process.returncode == 0:
                            compilation_status["success"] = True
                            logging.info(f"DaCe import/compilation successful for {file_path}")
                        else:
                            compilation_status["error"] = f"DaCe import failed with exit code {process.returncode}"
                            logging.error(f"DaCe import failed for {file_path} with exit code {process.returncode}")
                    except subprocess.TimeoutExpired as e:
                        compilation_status = {
                            "success": False,
                            "error": f"DaCe import timed out after {compile_timeout} seconds.",
                            "stdout": e.stdout if hasattr(e, 'stdout') else '',
                            "stderr": e.stderr if hasattr(e, 'stderr') else '',
                            "command": ' '.join(import_cmd),
                            "cwd": compile_cwd,
                        }
                        logging.error(f"DaCe import timed out for {file_path}: {e}")
                    except Exception as e:
                        compilation_status = {
                            "success": False,
                            "error": f"Unexpected error during DaCe import: {e}",
                            "stdout": "",
                            "stderr": str(e),
                            "command": ' '.join(import_cmd),
                            "cwd": compile_cwd,
                        }
                        logging.error(f"Error during DaCe import for {file_path}: {e}")
                    # Revert on compilation failure
                    if not compilation_status.get("success"):
                        logging.warning(f"DaCe import failed for {file_path}. Reverting file content.")
                        try:
                            self.file_manager.write_file(file_path, old_content)
                            reverted_due_to_compilation = True
                            logging.info(f"Successfully reverted {file_path} to its previous state.")
                        except Exception as revert_err:
                            logging.error(f"Failed to revert {file_path} after DaCe import error: {revert_err}")
                        # Include stderr in error so user sees exception details
                        dace_error = self._extract_dace_error(compilation_status.get("stderr", ""))
                        return {
                            "success": False,
                            "error": dace_error,
                            "old_content": old_content,
                            "current_code": old_content,
                            "proposed_code": old_content,
                            "changed_range": (changed_start, changed_end),
                            "temp_file_content": old_content,
                            "file_path": str(file_path),
                            "compilation_status": compilation_status,
                            "reverted_due_to_compilation": reverted_due_to_compilation,
                        }
                elif file_type in [".pyx", ".pxd"]:
                    logging.info(f"Detected {file_type} file, attempting compilation: {file_path}")
                    # Skip compile if no build script is present
                    build_py = Path(self.state.code_dir) / "setup.py"
                    build_toml = Path(self.state.code_dir) / "pyproject.toml"
                    if not build_py.exists() and not build_toml.exists():
                        logging.info("Skipping Cython compilation: no setup.py or pyproject.toml found.")
                        compilation_status = {"success": True, "error": None, "dependency_check": None}
                    else:
                        compile_cwd = str(self.state.code_dir)
                        compile_timeout = 1800 # Timeout in seconds
                        compilation_status = { # Default status
                            "success": False,
                            "error": "Compilation not attempted.",
                            "dependency_check": None,
                        } 
                        dependency_check_ok = False

                        # --- ADDED: Dependency Check ---
                        dependency_check_cmd = [
                            sys.executable,
                            "-c",
                            "import sys; print(f'Python Executable: {sys.executable}'); print(f'sys.path: {sys.path}'); import cython; import numpy; print('Cython and NumPy OK')"
                        ]
                        logging.info(f"Running dependency check: {' '.join(dependency_check_cmd)}")
                        try:
                            dep_check_process = subprocess.run(
                                dependency_check_cmd,
                                cwd=compile_cwd,
                                capture_output=True,
                                text=True,
                                check=False, 
                                timeout=30 # Shorter timeout for quick check
                            )
                            compilation_status["dependency_check"] = {
                                "exit_code": dep_check_process.returncode,
                                "stdout": dep_check_process.stdout,
                                "stderr": dep_check_process.stderr,
                            }
                            if dep_check_process.returncode == 0:
                                dependency_check_ok = True
                                logging.info("Dependency check successful.")
                                logging.debug(f"Dependency check stdout:\n{dep_check_process.stdout}")
                            else:
                                logging.error(f"Dependency check failed! Exit Code: {dep_check_process.returncode}")
                                logging.error(f"Dependency check stderr:\n{dep_check_process.stderr}")
                                compilation_status["success"] = False
                                compilation_status["error"] = "Build dependencies (Cython/NumPy) not found or importable."
                                compilation_status["stdout"] = dep_check_process.stdout # Store check output
                                compilation_status["stderr"] = dep_check_process.stderr # Store check error

                        except Exception as dep_err:
                            logging.error(f"Error running dependency check: {dep_err}")
                            compilation_status["dependency_check"] = {"error": str(dep_err)}
                            compilation_status["success"] = False
                            compilation_status["error"] = f"Failed to run dependency check: {dep_err}"
                        # --- END Dependency Check ---
                        
                        # Only attempt pip install if dependency check passed
                        if dependency_check_ok:
                            # Clean Cython build artifacts for a fresh build
                            try:
                                code_dir = Path(self.state.code_dir)
                                for artifact in ['build', 'dist']:
                                    artifact_path = code_dir / artifact
                                    if artifact_path.exists():
                                        shutil.rmtree(artifact_path)
                                for egg in code_dir.glob('*.egg-info'):
                                    shutil.rmtree(egg)
                                for ext in ['*.c', '*.so', '*.pyd']:
                                    for f in code_dir.rglob(ext):
                                        f.unlink()
                                logging.info("Cleaned Cython build artifacts before compilation.")
                            except Exception as clean_err:
                                logging.warning(f"Failed to clean build artifacts: {clean_err}")
                            compile_command = [
                                sys.executable, # Use the current Python interpreter
                                "-m",
                                "pip",
                                "install",
                                ".", # Install from the current directory (project root)
                                "--no-deps", # Don't reinstall dependencies
                                "--force-reinstall", # Force reinstall to ensure recompilation
                                "--no-cache-dir", # Avoid using cache
                            ]
                            compile_cwd = str(self.state.code_dir)
                            compile_timeout = 1800
                            compilation_error = None

                            try:
                                process = subprocess.run(
                                    compile_command,
                                    cwd=compile_cwd,
                                    capture_output=True,
                                    text=True,
                                    check=False, # Don't raise exception on non-zero exit
                                    timeout=compile_timeout,
                                )
                                
                                compilation_status = {
                                    "success": process.returncode == 0,
                                    "exit_code": process.returncode,
                                    "stdout": process.stdout,
                                    "stderr": process.stderr,
                                    "command": " ".join(compile_command),
                                    "cwd": compile_cwd,
                                    "error": None, # No subprocess error
                                }
                                if process.returncode != 0:
                                    logging.warning(f"Cython compilation failed for {file_path} with exit code {process.returncode}")
                                    logging.warning(f"Compilation stderr:\n{process.stderr}")
                                    compilation_error = f"Compilation failed with exit code {process.returncode}"
                                else:
                                    logging.info(f"Cython compilation successful for {file_path}")

                            except subprocess.TimeoutExpired as e:
                                logging.error(f"Cython compilation timed out for {file_path}: {e}")
                                compilation_error = f"Compilation timed out after {compile_timeout} seconds."
                                compilation_status = {
                                    "success": False,
                                    "error": compilation_error,
                                    "stdout": e.stdout.decode() if e.stdout else "",
                                    "stderr": e.stderr.decode() if e.stderr else "",
                                    "command": " ".join(compile_command),
                                    "cwd": compile_cwd,
                                }
                            except Exception as e:
                                logging.error(f"Error during Cython compilation for {file_path}: {e}")
                                compilation_error = f"An unexpected error occurred during compilation: {str(e)}"
                                compilation_status = {
                                    "success": False,
                                    "error": compilation_error,
                                    "stdout": "",
                                    "stderr": str(e),
                                    "command": " ".join(compile_command),
                                    "cwd": compile_cwd,
                                }

                            # --- ADDED: Revert on Compilation Failure (adjusted condition) ---
                            # Propagate any compilation_error so error isn't None
                            if compilation_error:
                                compilation_status["error"] = compilation_error
                            # Check the final compilation_status, which includes dependency check result
                            if not compilation_status.get("success"):
                                logging.warning(f"Compilation failed for {file_path}. Reverting file content.")
                                try:
                                    self.file_manager.write_file(file_path, old_content)
                                    reverted_due_to_compilation = True
                                    logging.info(f"Successfully reverted {file_path} to its previous state.")
                                except Exception as revert_err:
                                    logging.error(f"Failed to revert {file_path} after compilation error: {revert_err}")
                                # Build detailed Cython error message without full paths
                                raw_err = compilation_status.get("error", "Cython compilation failed.")
                                raw_stderr = compilation_status.get("stderr", "") or ""
                                detailed_err = raw_err
                                if raw_stderr:
                                    cleaned_lines = []
                                    for line in raw_stderr.splitlines():
                                        # Strip directory prefixes from any path, including File "..." lines
                                        cleaned = re.sub(r'(?:[^\s,"\']+[/\\])+(?P<file>[^/\\]+)', lambda m: m.group('file'), line)
                                        cleaned_lines.append(cleaned)
                                    detailed_err += ":\n" + "\n".join(cleaned_lines)
                                # Return on compilation failure to prevent further evaluation
                                return {
                                    "success": False,
                                    "error": detailed_err,
                                    "old_content": old_content,
                                    "current_code": old_content,
                                    "proposed_code": old_content,
                                    "changed_range": (changed_start, changed_end),
                                    "temp_file_content": old_content,
                                    "file_path": str(file_path),
                                    "compilation_status": compilation_status,
                                    "reverted_due_to_compilation": reverted_due_to_compilation,
                                }
                            # --- END REVERT LOGIC ---
                else:
                    # No compilation needed for this file type
                    compilation_status = {"success": True, "error": None, "dependency_check": None}

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to write file: {str(e)}",
                    "old_content": old_content,
                    "current_code": old_content,
                    "proposed_code": joined_proposed,
                    "changed_range": (changed_start, changed_end),
                    "temp_file_content": joined_proposed,
                    "file_path": str(file_path),
                    "compilation_status": compilation_status, # Include compilation status
                    "reverted_due_to_compilation": reverted_due_to_compilation, # Include revert status
                }

            return {
                "success": True,
                "formatted": content_to_write, # Return the actual written content
                "old_content": old_content,
                "current_code": old_content,
                "changed_range": (changed_start, changed_end),
                "proposed_code": joined_proposed,
                "temp_file_content": joined_proposed,
                "file_path": str(file_path),
                "compilation_status": compilation_status, # Include compilation status
                "reverted_due_to_compilation": reverted_due_to_compilation, # Include revert status
            }

        except Exception as e:
            tb = traceback.format_exc()
            logging.error(tb)
            current = old_content
            if file_path.exists():
                try:
                    current = "".join(self.file_manager.read_file(file_path))
                except:
                    pass  # Keep the old_content if we can't read the file
            return {
                "success": False,
                "error": str(e),
                "old_content": current,
                "current_code": current,
                "proposed_code": new_content,
                "changed_range": None,
                "traceback": tb,
                "temp_file_content": joined_proposed,
                "file_path": str(file_path),
                "compilation_status": compilation_status, # Include compilation status
                "reverted_due_to_compilation": reverted_due_to_compilation, # Include revert status
            }
        finally:
            # No temporary file is used anymore
            pass 

    def delete_lines(self, file_path: Path, start_line: int, end_line: int) -> dict:
        """
        Delete lines from a file. 
        This is a wrapper around edit_file with empty content.

        Args:
            file_path: Path to the file
            start_line: First line to delete (1-indexed)
            end_line: Last line to delete (1-indexed)

        Returns:
            Dictionary with edit result
        """
        # Validate line numbers
        if isinstance(start_line, str) and start_line.isdigit():
            start_line = int(start_line)
        if isinstance(end_line, str) and end_line.isdigit():
            end_line = int(end_line)
            
        if start_line < 1:
            return {
                "success": False,
                "error": f"Start line must be at least 1 for deletion (got {start_line})",
                "file_path": str(file_path),
            }
            
        if end_line < start_line:
            return {
                "success": False,
                "error": f"End line ({end_line}) must be greater than or equal to start line ({start_line})",
                "file_path": str(file_path),
            }
            
        # Call edit_file with empty content to delete the lines
        return self.edit_file(file_path, start_line, end_line, "")

    def create_test_file(self, file_path: Path, content: str = "This is a test file") -> dict:
        """
        Create a test file for diagnostic purposes.
        This is useful for verifying that file paths can be resolved correctly.
        
        Args:
            file_path: The path to create the file at
            content: Optional content to write to the file
            
        Returns:
            Dictionary with result information
        """
        try:
            # Convert to absolute path 
            abs_path = self.file_manager._make_absolute(file_path)
            logging.info(f"Creating test file at {abs_path}")
            
            # Ensure the directory exists
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            with open(abs_path, 'w') as f:
                f.write(content)
                
            return {
                "success": True,
                "message": f"Created test file at {abs_path}",
                "file_path": str(abs_path),
            }
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Error creating test file: {e}")
            logging.error(f"Traceback: {tb}")
            return {
                "success": False,
                "error": f"Error creating test file: {str(e)}",
                "traceback": tb,
                "file_path": str(file_path),
            }

    def view_file(
        self,
        file_path: Path,
        changed_range: Optional[Tuple[int, int]] = None,
        start_line: int = 1,
        lines_to_view: int = 50,
        pre_context: Optional[int] = None,
        post_context: Optional[int] = None,
    ) -> dict:
        """
        Returns file content with optional context and change highlighting.
        """
        try:
            return self.file_manager.view_file(
                file_path,
                changed_range,
                start_line,
                lines_to_view,
                pre_context,
                post_context,
            )
        except Exception:
            tb = traceback.format_exc()
            logging.error(tb)
            return {"success": False, "error": tb, "context": "viewing file"}



# Global Instances



# <<< File Format Constants or other independent code can remain >>>
