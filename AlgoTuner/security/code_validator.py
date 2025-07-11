"""
Simple code validator to detect tampering attempts in solver code.

This validator runs during the edit phase to prevent monkey-patching
of standard library functions before the code is executed.
"""

import ast
import re
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TamperingError(SyntaxError):
    """Raised when code contains tampering attempts."""
    pass


class TamperingDetector(ast.NodeVisitor):
    """AST visitor that detects attempts to monkey-patch standard library modules."""
    
    # Modules that should not be modified
    PROTECTED_MODULES = {
        'hmac', 'hashlib', 'os', 'sys', 'subprocess', 'importlib',
        'builtins', '__builtins__', 'types', 'gc', 'inspect'
    }
    
    # Specific attributes that are commonly targets for tampering
    PROTECTED_ATTRIBUTES = {
        'hmac.compare_digest',
        'hashlib.sha256',
        'hashlib.sha512',
        'hashlib.md5',
        'os.system',
        'subprocess.run',
        'sys.modules',
    }
    
    def __init__(self):
        self.violations = []
    
    def visit_Assign(self, node):
        """Detect direct assignment to module attributes."""
        for target in node.targets:
            violation = self._check_assignment_target(target, node.lineno)
            if violation:
                self.violations.append(violation)
        self.generic_visit(node)
    
    def visit_AugAssign(self, node):
        """Detect augmented assignment (+=, -=, etc.) to module attributes."""
        violation = self._check_assignment_target(node.target, node.lineno)
        if violation:
            self.violations.append(violation)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Detect setattr() calls on protected modules."""
        if isinstance(node.func, ast.Name) and node.func.id == 'setattr':
            if len(node.args) >= 2:
                # Check if first arg is a protected module
                module_name = self._get_module_name(node.args[0])
                if module_name in self.PROTECTED_MODULES:
                    attr_name = self._get_string_value(node.args[1])
                    self.violations.append({
                        'line': node.lineno,
                        'type': 'setattr',
                        'module': module_name,
                        'attribute': attr_name or '<dynamic>',
                        'code': f"setattr({module_name}, ...)"
                    })
        
        # Also check for __setattr__ calls
        elif isinstance(node.func, ast.Attribute) and node.func.attr == '__setattr__':
            module_name = self._get_module_name(node.func.value)
            if module_name in self.PROTECTED_MODULES:
                self.violations.append({
                    'line': node.lineno,
                    'type': 'setattr',
                    'module': module_name,
                    'attribute': '<unknown>',
                    'code': f"{module_name}.__setattr__(...)"
                })
        
        self.generic_visit(node)
    
    def _check_assignment_target(self, target, lineno) -> Optional[dict]:
        """Check if assignment target is a protected module attribute."""
        if isinstance(target, ast.Attribute):
            module_name = self._get_module_name(target.value)
            if module_name in self.PROTECTED_MODULES:
                full_name = f"{module_name}.{target.attr}"
                return {
                    'line': lineno,
                    'type': 'assignment',
                    'module': module_name,
                    'attribute': target.attr,
                    'code': f"{full_name} = ..."
                }
        
        # Check for module.__dict__ assignments
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute):
            if target.value.attr == '__dict__':
                module_name = self._get_module_name(target.value.value)
                if module_name in self.PROTECTED_MODULES:
                    return {
                        'line': lineno,
                        'type': 'dict_assignment',
                        'module': module_name,
                        'attribute': '<unknown>',
                        'code': f"{module_name}.__dict__[...] = ..."
                    }
        
        return None
    
    def _get_module_name(self, node) -> Optional[str]:
        """Extract module name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Handle nested attributes like os.path
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return '.'.join(reversed(parts))
        return None
    
    def _get_string_value(self, node) -> Optional[str]:
        """Extract string value from AST node if it's a constant string."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s
        return None


def validate_code(code: str, filename: str = "solver.py") -> List[dict]:
    """
    Validate code for tampering attempts.
    
    Args:
        code: The Python code to validate
        filename: Name of the file (for error reporting)
        
    Returns:
        List of violation dictionaries, empty if code is clean
    """
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as e:
        # Regular syntax errors should be handled normally
        raise
    
    detector = TamperingDetector()
    detector.visit(tree)
    
    return detector.violations


def format_tampering_error(violations: List[dict], code_lines: List[str]) -> str:
    """
    Format tampering violations into a helpful error message.
    
    Args:
        violations: List of violation dictionaries
        code_lines: Lines of the original code
        
    Returns:
        Formatted error message
    """
    if not violations:
        return ""
    
    msg_parts = ["Error: Code contains security violations"]
    
    for v in violations:
        line_num = v['line']
        line_content = code_lines[line_num - 1].strip() if line_num <= len(code_lines) else ""
        
        msg_parts.append(f"\nLine {line_num}: {line_content}")
        msg_parts.append(f"  Tampering with {v['module']}.{v['attribute']} is not allowed")
        msg_parts.append(f"  Detected: {v['code']}")
    
    msg_parts.append("\nMonkey-patching standard library functions is prohibited.")
    return '\n'.join(msg_parts)


def check_code_for_tampering(code: str, filename: str = "solver.py") -> Optional[str]:
    """
    Check code for tampering and return error message if found.
    
    Args:
        code: The Python code to check
        filename: Name of the file
        
    Returns:
        Error message string if tampering found, None if code is clean
    """
    violations = validate_code(code, filename)
    
    if violations:
        code_lines = code.split('\n')
        return format_tampering_error(violations, code_lines)
    
    return None