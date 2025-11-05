import re
from typing import Optional

def remove_self_parameter(source_code: str, function_name: Optional[str] = None) -> str:
    """
    Remove 'self' parameter from a method definition and clean up self references.
    
    This function handles various edge cases including:
    - Whitespace and newlines around 'self'
    - 'self' as the only parameter or first parameter
    - Multiple spaces, tabs, and line breaks
    - Optional function name specification
    
    Args:
        source_code: The source code containing the method definition
        function_name: Optional function name to target. If None, processes all 'def' statements
        
    Returns:
        Cleaned source code with 'self' parameter removed
    
    Examples:
        >>> remove_self_parameter("def foo(self, x): return self.value", "foo")
        'def foo(x): return value'
        
        >>> remove_self_parameter("def bar(self): pass", "bar")
        'def bar(): pass'
    """
    # Pattern to match function definitions with various whitespace
    if function_name:
        # Target specific function
        func_pattern = re.escape(function_name)
    else:
        # Match any function name
        func_pattern = r'\w+'
    
    # Pattern to match: def function_name(self) or def function_name(self, ...)
    # Handles whitespace, newlines, and various formatting
    
    # Case 1: self is the only parameter: def func(self)
    pattern_only = rf'(\bdef\s+{func_pattern}\s*\(\s*)self(\s*\))'
    source_code = re.sub(pattern_only, r'\1\2', source_code, flags=re.MULTILINE)
    
    # Case 2: self is first parameter with others: def func(self, x, y)
    # This handles: self, / self , / self  , with various whitespace
    pattern_first = rf'(\bdef\s+{func_pattern}\s*\(\s*)self\s*,\s*'
    source_code = re.sub(pattern_first, r'\1', source_code, flags=re.MULTILINE)
    
    # Remove all self. references (member access)
    # This handles self.attribute and self.method()
    source_code = re.sub(r'\bself\.', '', source_code)
    
    return source_code