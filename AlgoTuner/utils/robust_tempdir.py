"""
Robust temporary directory management for HPC filesystems.

This module provides a temporary directory context manager that handles
cleanup issues common on parallel filesystems like Lustre.
"""

import os
import shutil
import tempfile
import time
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


class RobustTemporaryDirectory:
    """A temporary directory that handles cleanup robustly on HPC filesystems."""
    
    def __init__(
        self, 
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[str] = None,
        cleanup_retries: int = 3,
        cleanup_delays: tuple = (0.5, 1.0, 2.0)
    ):
        """
        Initialize a robust temporary directory.
        
        Args:
            suffix: Suffix for directory name
            prefix: Prefix for directory name
            dir: Parent directory for the temp directory
            cleanup_retries: Number of cleanup attempts
            cleanup_delays: Delays between cleanup attempts (seconds)
        """
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir
        self.cleanup_retries = cleanup_retries
        self.cleanup_delays = cleanup_delays
        self.name = None
        self._finalizer = None
    
    def __enter__(self):
        """Create temporary directory on entry."""
        self.name = tempfile.mkdtemp(suffix=self.suffix, prefix=self.prefix, dir=self.dir)
        return self.name
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory on exit with retry logic."""
        if self.name is None:
            return
        
        # Check if directory still exists
        if not os.path.exists(self.name):
            logger.debug(f"Temporary directory {self.name} already deleted")
            return
        
        # Attempt cleanup with retries
        for attempt in range(self.cleanup_retries):
            try:
                shutil.rmtree(self.name)
                return  # Success
            except OSError as e:
                # Common retryable errors
                if e.errno in (2, 39):  # ENOENT or ENOTEMPTY
                    if e.errno == 2 and not os.path.exists(self.name):
                        # Directory was successfully deleted (possibly by another process)
                        logger.debug(f"Temporary directory {self.name} was deleted during cleanup")
                        return
                    
                    if attempt < self.cleanup_retries - 1:
                        # Not final attempt - retry
                        delay = self.cleanup_delays[min(attempt, len(self.cleanup_delays) - 1)]
                        logger.debug(
                            f"Temporary directory cleanup failed (attempt {attempt + 1}/{self.cleanup_retries}), "
                            f"errno={e.errno}, retrying in {delay}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        # Final attempt failed - log but don't raise
                        if e.errno == 39:
                            logger.warning(
                                f"Failed to clean up temporary directory after {self.cleanup_retries} attempts: {self.name} (directory not empty)"
                            )
                        else:
                            logger.warning(
                                f"Partial cleanup of temporary directory {self.name}: some files were already deleted"
                            )
                        # Don't raise - allow program to continue
                        return
                else:
                    # Non-retryable error
                    logger.error(f"Unexpected error cleaning up temporary directory: {e}")
                    raise


@contextmanager
def robust_tempdir(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[str] = None,
    cleanup_retries: int = 3,
    cleanup_delays: tuple = (0.5, 1.0, 2.0)
):
    """
    Context manager for robust temporary directory creation and cleanup.
    
    This is a convenience function that creates a RobustTemporaryDirectory
    and yields the directory path.
    
    Args:
        suffix: Suffix for directory name
        prefix: Prefix for directory name
        dir: Parent directory for the temp directory
        cleanup_retries: Number of cleanup attempts
        cleanup_delays: Delays between cleanup attempts (seconds)
        
    Yields:
        str: Path to the temporary directory
    """
    with RobustTemporaryDirectory(
        suffix=suffix,
        prefix=prefix,
        dir=dir,
        cleanup_retries=cleanup_retries,
        cleanup_delays=cleanup_delays
    ) as tmpdir:
        yield tmpdir