"""
Smart result handling with progressive degradation.

Handles large results by trying multiple strategies in order:
1. Send full result (works for most cases)
2. Send result with metadata preservation (keeps structure, strips large arrays)
3. Send minimal placeholder (last resort)

Always preserves enough information for user debugging and validation.
"""

import logging
import numpy as np
import pickle
import sys
import traceback
from typing import Any, Dict, Optional, Tuple, Union
import tempfile
import os


class ResultMetadata:
    """Rich metadata object that preserves result information without large data."""
    
    def __init__(self, original_obj: Any, stripped_data: Optional[bytes] = None):
        self.original_type = type(original_obj).__name__
        self.original_module = getattr(type(original_obj), '__module__', None)
        self.stripped_data = stripped_data
        
        # Extract as much metadata as possible
        self.metadata = {}
        
        try:
            # For numpy arrays
            if hasattr(original_obj, 'dtype') and hasattr(original_obj, 'shape'):
                self.metadata.update({
                    'dtype': str(original_obj.dtype),
                    'shape': original_obj.shape,
                    'size': original_obj.size,
                    'nbytes': original_obj.nbytes,
                    'ndim': original_obj.ndim
                })
                
                # Store small arrays completely, larger ones as samples
                if original_obj.nbytes < 1024:  # 1KB threshold
                    self.metadata['data'] = original_obj.tolist()
                    self.metadata['complete_data'] = True
                else:
                    # Store samples from different parts of the array
                    try:
                        flat = original_obj.flatten()
                        sample_size = min(100, len(flat))
                        if sample_size > 0:
                            indices = np.linspace(0, len(flat)-1, sample_size, dtype=int)
                            self.metadata['data_sample'] = flat[indices].tolist()
                            self.metadata['sample_indices'] = indices.tolist()
                            self.metadata['complete_data'] = False
                    except:
                        pass
            
            # For sequences (lists, tuples, etc.)
            elif hasattr(original_obj, '__len__'):
                self.metadata['length'] = len(original_obj)
                self.metadata['is_sequence'] = True
                
                # Store small sequences completely
                if len(original_obj) < 100:
                    try:
                        self.metadata['data'] = list(original_obj)
                        self.metadata['complete_data'] = True
                    except:
                        pass
                else:
                    # Store samples
                    try:
                        sample_indices = [0, len(original_obj)//4, len(original_obj)//2, 
                                        3*len(original_obj)//4, len(original_obj)-1]
                        self.metadata['data_sample'] = [original_obj[i] for i in sample_indices]
                        self.metadata['sample_indices'] = sample_indices
                        self.metadata['complete_data'] = False
                    except:
                        pass
                        
                # Type information for sequence elements
                if len(original_obj) > 0:
                    try:
                        first_type = type(original_obj[0]).__name__
                        self.metadata['element_type'] = first_type
                        
                        # Check if all elements are the same type
                        if len(original_obj) > 1:
                            all_same_type = all(type(x).__name__ == first_type for x in original_obj[:10])
                            self.metadata['homogeneous'] = all_same_type
                    except:
                        pass
            
            # For other objects
            else:
                try:
                    obj_str = str(original_obj)
                    if len(obj_str) < 1000:
                        self.metadata['string_repr'] = obj_str
                        self.metadata['complete_data'] = True
                    else:
                        self.metadata['string_repr'] = obj_str[:500] + "... [truncated]"
                        self.metadata['complete_data'] = False
                except:
                    pass
            
            # Always try to get basic info
            self.metadata['str_len'] = len(str(original_obj)) if hasattr(original_obj, '__str__') else None
            self.metadata['memory_size_bytes'] = sys.getsizeof(original_obj)
            
        except Exception as e:
            logging.warning(f"ResultMetadata: Error extracting metadata: {e}")
            self.metadata['extraction_error'] = str(e)
    
    def __str__(self):
        """Human-readable representation."""
        if self.metadata.get('complete_data'):
            return f"<Complete {self.original_type}: {self.metadata}>"
        else:
            return f"<Stripped {self.original_type}: shape={self.metadata.get('shape', 'unknown')}, " \
                   f"size={self.metadata.get('memory_size_bytes', 'unknown')} bytes>"
    
    def __repr__(self):
        return self.__str__()
    
    def get_summary(self) -> str:
        """Get a summary string for logging/debugging."""
        summary_parts = [f"Type: {self.original_type}"]
        
        if 'shape' in self.metadata:
            summary_parts.append(f"Shape: {self.metadata['shape']}")
        if 'dtype' in self.metadata:
            summary_parts.append(f"DType: {self.metadata['dtype']}")
        if 'length' in self.metadata:
            summary_parts.append(f"Length: {self.metadata['length']}")
        if 'memory_size_bytes' in self.metadata:
            mb = self.metadata['memory_size_bytes'] / (1024*1024)
            summary_parts.append(f"Size: {mb:.2f}MB")
        
        return ", ".join(summary_parts)
    
    # Basic operations for compatibility with validation
    @property 
    def shape(self):
        """Provide shape for numpy-like compatibility."""
        return self.metadata.get('shape')
    
    @property
    def dtype(self):
        """Provide dtype for numpy-like compatibility."""
        dtype_str = self.metadata.get('dtype')
        if dtype_str:
            try:
                return np.dtype(dtype_str)
            except:
                return dtype_str
        return None
    
    def __len__(self):
        """Provide length for sequence-like compatibility."""
        if 'length' in self.metadata:
            return self.metadata['length']
        elif 'size' in self.metadata:
            return self.metadata['size']
        else:
            raise TypeError(f"ResultMetadata for {self.original_type} has no length")


def estimate_pickle_size(obj: Any) -> int:
    """Estimate the pickle size of an object without full serialization."""
    try:
        # For numpy arrays, estimate based on nbytes
        if hasattr(obj, 'nbytes'):
            # Pickle overhead is usually small compared to data
            return obj.nbytes + 1000  # Add some overhead
        
        # For other objects, use sys.getsizeof as approximation
        base_size = sys.getsizeof(obj)
        
        # Recursively estimate for containers
        if hasattr(obj, '__len__') and hasattr(obj, '__iter__'):
            try:
                if len(obj) > 0:
                    # Sample first few elements
                    sample_size = min(10, len(obj))
                    if hasattr(obj, '__getitem__'):
                        sample = [obj[i] for i in range(sample_size)]
                    else:
                        sample = list(obj)[:sample_size]
                    
                    avg_element_size = sum(sys.getsizeof(x) for x in sample) / len(sample)
                    estimated_total = base_size + (avg_element_size * len(obj))
                    return int(estimated_total)
            except:
                pass
        
        return base_size
        
    except Exception as e:
        logging.debug(f"Error estimating pickle size: {e}")
        return sys.getsizeof(obj) if hasattr(obj, '__sizeof__') else 10000


def try_pickle_size_check(obj: Any, max_size_mb: float = 100.0) -> Tuple[bool, int]:
    """
    Check if an object is likely to be too large for pickling.
    
    Returns:
        (can_pickle, estimated_size_bytes)
    """
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    
    estimated_size = estimate_pickle_size(obj)
    
    if estimated_size > max_size_bytes:
        return False, estimated_size
    
    # For borderline cases, try actual pickle test with size limit
    if estimated_size > max_size_bytes * 0.5:  # 50MB threshold for actual test
        try:
            pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            actual_size = len(pickled)
            return actual_size <= max_size_bytes, actual_size
        except Exception as e:
            logging.debug(f"Pickle test failed: {e}")
            return False, estimated_size
    
    return True, estimated_size


def create_smart_result(obj: Any, max_size_mb: float = 100.0) -> Tuple[Any, Dict[str, Any]]:
    """
    Create a smart result that can be safely sent via IPC.
    
    Args:
        obj: Original object
        max_size_mb: Maximum size in MB before stripping
        
    Returns:
        (processed_object, metadata_dict)
    """
    metadata = {
        'original_type': type(obj).__name__,
        'processing_applied': 'none',
        'estimated_size_mb': 0,
        'can_pickle': True
    }
    
    # First check if object is small enough
    can_pickle, estimated_size = try_pickle_size_check(obj, max_size_mb)
    metadata['estimated_size_mb'] = estimated_size / (1024*1024)
    metadata['can_pickle'] = can_pickle
    
    if can_pickle:
        # Object is small enough, send as-is
        metadata['processing_applied'] = 'none'
        return obj, metadata
    
    # Object is too large, create metadata version
    logging.info(f"Result too large ({estimated_size/(1024*1024):.2f}MB), creating metadata version")
    
    try:
        result_metadata = ResultMetadata(obj)
        metadata['processing_applied'] = 'metadata_extraction'
        metadata['preserved_info'] = result_metadata.get_summary()
        
        return result_metadata, metadata
        
    except Exception as e:
        logging.error(f"Failed to create result metadata: {e}")
        
        # Last resort: minimal placeholder
        placeholder = {
            'type': type(obj).__name__,
            'str_repr': str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj),
            'size_estimate_mb': estimated_size / (1024*1024),
            'error': 'Result too large for transmission'
        }
        
        metadata['processing_applied'] = 'minimal_placeholder'
        metadata['error'] = str(e)
        
        return placeholder, metadata


def handle_ipc_result_with_fallback(result_dict: Dict[str, Any], max_size_mb: float = 100.0) -> Dict[str, Any]:
    """
    Handle result transmission with progressive fallback strategies.
    
    Args:
        result_dict: Dictionary containing 'result' and other fields
        max_size_mb: Size threshold for applying smart handling
        
    Returns:
        Modified result_dict safe for IPC transmission
    """
    if 'result' not in result_dict or result_dict['result'] is None:
        return result_dict
    
    original_result = result_dict['result']
    
    # Try smart result creation
    processed_result, processing_metadata = create_smart_result(original_result, max_size_mb)
    
    # Update result dict
    result_dict = result_dict.copy()  # Don't modify original
    result_dict['result'] = processed_result
    result_dict['result_processing'] = processing_metadata
    
    # Log what happened
    if processing_metadata['processing_applied'] != 'none':
        logging.info(f"Applied {processing_metadata['processing_applied']} to result: "
                    f"{processing_metadata.get('preserved_info', 'unknown')}")
    
    return result_dict


def extract_original_error_from_ipc_failure(error_message: str) -> Optional[str]:
    """
    Extract the original user error from IPC failure messages.
    
    When multiprocessing fails due to large results, try to find the real
    underlying error (like MemoryError) that the user should see.
    """
    if not error_message:
        return None
    
    # Pattern 1: "Error sending result: {...}" - extract from the dict representation
    if "Error sending result:" in error_message:
        try:
            # Look for error field in the result dict representation
            if "'error':" in error_message:
                # Extract the error value from the string representation
                import re
                error_match = re.search(r"'error':\s*'([^']*)'", error_message)
                if error_match:
                    return error_match.group(1)
                    
                # Try without quotes
                error_match = re.search(r"'error':\s*([^,}]*)", error_message)
                if error_match:
                    return error_match.group(1).strip()
            
            # Look for exception types in the message
            common_errors = ['MemoryError', 'ValueError', 'TypeError', 'RuntimeError', 
                           'AttributeError', 'IndexError', 'KeyError']
            
            for error_type in common_errors:
                if error_type in error_message:
                    # Try to extract the full error message
                    pattern = rf"{error_type}[:\(]([^')]*)"
                    match = re.search(pattern, error_message)
                    if match:
                        return f"{error_type}: {match.group(1).strip()}"
                    else:
                        return f"{error_type} (extracted from IPC failure)"
                        
        except Exception as e:
            logging.debug(f"Error extracting original error: {e}")
    
    # Pattern 2: PicklingError messages
    if "PicklingError" in error_message:
        return f"Pickling failed - likely due to large result size or unsupported object type"
    
    # Pattern 3: Other common IPC failures
    if "too large" in error_message.lower() or "memory" in error_message.lower():
        return "Result too large - consider returning smaller data or using generators"
    
    return None