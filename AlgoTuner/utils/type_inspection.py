"""Utilities for type inspection and validation."""
from typing import Any # Needed for describe_type if it handles Any internally or for future expansion

def describe_type(value):
    """Get a detailed type description including nested types.
    
    Args:
        value: Any Python value
        
    Returns:
        str: A human-readable description of the value's type structure
        
    Examples:
        >>> describe_type([1, 2, 3])
        'list[int]'
        >>> describe_type({'a': 1})
        'dict{str: int}'
        >>> describe_type([])
        'list[]'
        >>> describe_type([(1, 2, True), (3, 4, False)])
        'list[tuple[int,int,bool]]'
        >>> describe_type({'a': [1, 2], 'b': [3, 4]})
        'dict{str: list[int]}'
        >>> describe_type([{'a': 1}, {'b': 2}])
        'list[dict{str: int}]'
    """
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{type(value).__name__}[]"
        
        # For lists/tuples, look at all elements to ensure consistent types
        element_types = set()
        for item in value:
            if isinstance(item, (list, tuple)):
                # For nested lists/tuples, describe their structure
                inner_types = []
                for x in item:
                    inner_type = describe_type(x)
                    inner_types.append(inner_type)
                element_types.add(f"tuple[{','.join(inner_types)}]" if isinstance(item, tuple) else f"list[{','.join(inner_types)}]")
            elif isinstance(item, dict):
                # For nested dicts, get their full type description
                element_types.add(describe_type(item))
            else:
                element_types.add(type(item).__name__)
        
        # If we have multiple different types, show them all
        type_str = "|".join(sorted(element_types)) or "any"
        return f"{type(value).__name__}[{type_str}]"
        
    elif isinstance(value, dict):
        if not value:
            return "dict{}"
        # Look at all keys and values
        key_types = set()
        val_types = set()
        for k, v in value.items():
            if isinstance(k, (list, tuple, dict)):
                key_types.add(describe_type(k))
            else:
                key_types.add(type(k).__name__)
                
            if isinstance(v, (list, tuple, dict)):
                val_types.add(describe_type(v))
            else:
                val_types.add(type(v).__name__)
                
        key_str = "|".join(sorted(key_types)) or "any"
        val_str = "|".join(sorted(val_types)) or "any"
        return f"dict{{{key_str}: {val_str}}}"
        
    return type(value).__name__ 

def describe_annotation(annotation):
    """Get a human-readable description of a typing annotation."""
    from typing import get_origin, get_args
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        return repr(annotation)
    origin_name = origin.__name__ if hasattr(origin, "__name__") else str(origin)
    if not args:
        return origin_name
    arg_strs = [describe_annotation(arg) for arg in args]
    return f"{origin_name}[{', '.join(arg_strs)}]" 