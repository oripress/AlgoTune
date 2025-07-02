"""
DaCe Configuration Module

This module provides centralized configuration for DaCe.
It should be imported and initialized before any other imports of DaCe.
"""

import os
import logging
from pathlib import Path
from typing import Union

def configure_dace_cache(cache_dir: Union[str, Path]) -> None:
    """
    Configure DaCe cache directory. This should be called before importing DaCe.
    
    Args:
        cache_dir: Directory to use for DaCe cache
    """
    cache_dir_str = str(cache_dir)
    
    # Set environment variables first (these affect DaCe on import)
    os.environ['DACE_CACHE'] = cache_dir_str
    os.environ['DACE_CACHE_ROOT'] = cache_dir_str
    os.environ['DACE_default_build_folder'] = cache_dir_str
    
    # Try to configure DaCe directly if it's already imported
    try:
        import sys
        if 'dace' in sys.modules:
            import dace
            dace.config.Config.set('cache', 'dir', cache_dir_str)
            dace.config.Config.set('cache', 'default_build_folder', cache_dir_str)
            logging.debug(f"Configured existing DaCe instance cache directory to: {cache_dir_str}")
    except Exception as e:
        logging.debug(f"Could not configure DaCe cache (this is normal if DaCe not imported yet): {e}")
    
    logging.debug(f"DaCe cache configured to: {cache_dir_str}")

def initialize_dace_for_process() -> None:
    """
    Initialize DaCe configuration for the current process.
    This should be called at the start of any new process that might use DaCe.
    """
    code_dir = os.environ.get('CODE_DIR')
    if code_dir:
        configure_dace_cache(code_dir)
        logging.info(f"Initialized DaCe configuration for process {os.getpid()} with cache dir: {code_dir}")
    else:
        logging.warning(f"CODE_DIR not set when initializing DaCe for process {os.getpid()}") 