"""
Common utility functions used throughout the codebase.
This module contains utilities that are safe to import from anywhere
and don't cause circular dependencies.
"""

import os
import sys
import logging
import importlib
import traceback
from typing import Dict, Any, Optional, List, Callable, Union, Tuple

# File and path helpers
def normalize_path(path: str) -> str:
    """Normalize a path to avoid system-specific issues."""
    return os.path.normpath(path)

def ensure_directory(directory_path: str) -> bool:
    """Ensure a directory exists, creating it if needed."""
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {directory_path}: {e}")
        return False

def get_workspace_root() -> str:
    """Get the workspace root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Module loading utilities
def safe_import(module_name: str, reload: bool = False) -> Optional[Any]:
    """Safely import a module and optionally reload it."""
    try:
        if module_name in sys.modules and reload:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
        return module
    except Exception as e:
        logging.error(f"Failed to import {module_name}: {e}")
        return None

def safe_import_from_path(module_path: str, module_name: Optional[str] = None) -> Optional[Any]:
    """Safely import a module from a file path."""
    directory = os.path.dirname(os.path.abspath(module_path))
    added_to_sys_path = False
    try:
        if module_name is None:
            module_name = os.path.basename(module_path).replace(".py", "")
        
        # Add directory to sys.path if not already present.
        if directory not in sys.path:
            sys.path.insert(0, directory)
            added_to_sys_path = True
        
        # Import or reload the module.
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
        
        return module
    except Exception as e:
        logging.error(f"Failed to import {module_path}: {e}")
        return None
    finally:
        # Remove directory only if it was added here.
        if added_to_sys_path:
            sys.path.remove(directory)

# Error handling utilities
def clean_traceback(tb_str: Optional[str]) -> str:
    """Clean a traceback by removing sensitive information."""
    if not tb_str:
        return ""
    
    cleaned = []
    for line in tb_str.split('\n'):
        if 'File "' in line:
            # Strip workspace root path.
            workspace_root = get_workspace_root()
            if workspace_root in line:
                line = line.replace(workspace_root, ".")
            
            # Strip home directory.
            home = os.path.expanduser('~')
            if home in line:
                line = line.replace(home, "~")
        cleaned.append(line)
    
    return '\n'.join(cleaned)

def extract_line_numbers(error_msg: str) -> List[int]:
    """Extract line numbers from error messages."""
    import re
    patterns = [
        r"line (\d+)",         # Standard format: "line 10"
        r"[EFW]:\s*(\d+),\d+:"  # Linter format: "E: 10,0:"
    ]
    line_numbers = set()
    for pattern in patterns:
        matches = re.findall(pattern, error_msg)
        for match in matches:
            line_numbers.add(int(match))
    return sorted(line_numbers)

# Type utilities
def get_type_name(obj: Any) -> str:
    """Get a human-readable type name for an object."""
    if obj is None:
        return "None"
    
    if hasattr(obj, "__class__"):
        if obj.__class__.__module__ == "builtins":
            return obj.__class__.__name__
        else:
            return f"{obj.__class__.__module__}.{obj.__class__.__name__}"
    
    return str(type(obj))

def format_object_shape(obj: Any) -> str:
    """Format the shape of an object in a human-readable way."""
    if obj is None:
        return "None"
    
    # For objects with a shape property (like numpy arrays).
    if hasattr(obj, "shape") and isinstance(obj.shape, tuple):
        return f"shape {obj.shape}"
    
    # For lists and tuples.
    if isinstance(obj, (list, tuple)):
        base_type = "list" if isinstance(obj, list) else "tuple"
        if not obj:
            return f"empty {base_type}"
        
        length = len(obj)
        
        # Check for a nested structure.
        if isinstance(obj[0], (list, tuple)) or hasattr(obj[0], "__len__"):
            try:
                inner_lengths = [len(x) for x in obj if hasattr(x, "__len__")]
                if inner_lengths and all(l == inner_lengths[0] for l in inner_lengths):
                    return f"{base_type} of length {length} with inner length {inner_lengths[0]}"
                else:
                    return f"{base_type} of length {length} with variable inner lengths"
            except Exception:
                return f"{base_type} of length {length} (nested)"
        else:
            return f"{base_type} of length {length}"
    
    # For dictionaries.
    if isinstance(obj, dict):
        key_count = len(obj)
        if key_count == 0:
            return "empty dict"
        return f"dict with {key_count} keys"
    
    # For sets.
    if isinstance(obj, set):
        item_count = len(obj)
        if item_count == 0:
            return "empty set"
        return f"set with {item_count} items"
    
    # For strings.
    if isinstance(obj, str):
        return f"string of length {len(obj)}"
    
    # For other types.
    return f"type {get_type_name(obj)}"

# Caching utilities
class CachedProperty:
    """A property that is calculated only once per instance."""
    def __init__(self, func: Callable[[Any], Any]):
        self.func = func
        self.__doc__ = func.__doc__
        
    def __get__(self, obj: Any, cls: Any) -> Any:
        if obj is None:
            return self
        value = self.func(obj)
        obj.__dict__[self.func.__name__] = value
        return value