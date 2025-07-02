"""
DaCe Initialization Module

This module MUST be imported before any other imports that might use DaCe.
It sets up the DaCe environment variables before DaCe is imported anywhere.
"""

import os
import logging
import tempfile, uuid, pathlib

def _ensure_dace_config():
    """
    Ensure DaCe configuration is set through environment variables.
    This must run before DaCe is imported anywhere.
    """
    temp_base = pathlib.Path(tempfile.gettempdir()) / f"dace_cache_{uuid.uuid4().hex[:8]}"
    try:
        temp_base.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot create, fall back to system temp dir itself
        temp_base = pathlib.Path(tempfile.gettempdir())

    temp_dir_str = str(temp_base)

    os.environ['DACE_CACHE'] = temp_dir_str
    os.environ['DACE_CACHE_ROOT'] = temp_dir_str
    os.environ['DACE_default_build_folder'] = temp_dir_str

    logging.debug(f"DaCe default build folder set to temp dir: {temp_dir_str}")

    # Redirect OS temp dirs to the same folder for isolation
    os.environ['TMPDIR'] = temp_dir_str
    os.environ['TEMP'] = temp_dir_str
    os.environ['TMP'] = temp_dir_str
    os.environ['NUMBA_CACHE_DIR'] = temp_dir_str

    try:
        import tempfile as _tf
        _tf.tempdir = temp_dir_str
        logging.debug(f"Python tempfile.tempdir set to temp dir: {temp_dir_str}")
    except Exception as e:
        logging.debug(f"Could not set tempfile.tempdir: {e}")

    logging.debug(f"DaCe cache configured to temp dir: {temp_dir_str}")

    # If DaCe is already imported, apply API override for default_build_folder
    try:
        import sys
        if 'dace' in sys.modules:
            import dace
            try:
                dace.config.Config.set('cache', 'default_build_folder', temp_dir_str)
                logging.debug(f"Configured existing DaCe default_build_folder via API: {temp_dir_str}")
            except Exception as e:
                logging.debug(f"Could not configure DaCe API default_build_folder: {e}")
    except Exception:
        pass

# Run configuration immediately on module import
_ensure_dace_config() 