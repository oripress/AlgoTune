import numpy as np
from typing import Any
import os
import atexit

# Use pyfftw for performance if available, but do not fail if it's not.
try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False

# Always import scipy as a fallback.
from scipy.fft import dstn as scipy_dstn

# Define a path for storing FFTW wisdom. Using /tmp is standard.
WISDOM_FILE_PATH = '/tmp/solver_fftw_wisdom.dat'

class Solver:
    _wisdom_registered = False # Class-level flag to ensure atexit is registered once.

    def __init__(self):
        """
        Initializes the solver.
        - Detects if pyfftw is available.
        - If so, loads wisdom, prepares a cache, and registers wisdom saving.
        - Caches the scipy fallback function.
        """
        if PYFFTW_AVAILABLE:
            self._plan_cache = {}
            self._num_threads = os.cpu_count() or 1
            self._planner_effort = 'FFTW_MEASURE'
            
            # Load wisdom from previous runs to accelerate planning.
            try:
                with open(WISDOM_FILE_PATH, 'rb') as f:
                    pyfftw.import_wisdom(f.read())
            except FileNotFoundError:
                pass # It's okay if wisdom doesn't exist yet.

            # Register the wisdom saving function to run at exit, but only once.
            if not Solver._wisdom_registered:
                atexit.register(Solver._save_wisdom)
                Solver._wisdom_registered = True
        
        self._scipy_dstn = scipy_dstn

    @staticmethod
    def _save_wisdom():
        """Saves the accumulated pyfftw wisdom to a file."""
        if PYFFTW_AVAILABLE:
            try:
                with open(WISDOM_FILE_PATH, 'wb') as f:
                    f.write(pyfftw.export_wisdom())
            except (IOError, OSError):
                pass # Don't crash if we can't write the file.

    def solve(self, problem: list, **kwargs) -> Any:
        """
        Computes a fast 2D DST-II using a robust, wisdom-accelerated strategy.

        The strategy is:
        1.  Wisdom Management: On startup, load FFTW "wisdom" from a file.
            On exit, save the wisdom, which includes any newly generated plans.
            This makes the high cost of 'FFTW_MEASURE' a one-time investment
            across multiple runs of the program.
        2.  If `pyfftw` is available, use the highly-optimized path:
            a. Aggressive Plan Optimization: Use 'FFTW_MEASURE'. With wisdom,
               this is nearly instantaneous for known shapes.
            b. Optimized Data Copy: Use `np.copyto()` for a fast, C-level
               copy from the input list to the plan's aligned buffer.
        3.  If `pyfftw` is not available, fall back to a multi-threaded `scipy`
            implementation for correctness and reasonable performance.
        """
        # --- Path 1: High-performance pyfftw implementation ---
        if PYFFTW_AVAILABLE:
            rows = len(problem)
            if rows == 0:
                return []
            cols = len(problem[0])
            shape = (rows, cols)

            if shape not in self._plan_cache:
                # Planning is slow only if the shape is new AND not in wisdom.
                a = pyfftw.empty_aligned(shape, dtype='float64')
                plan = pyfftw.builders.dstn(
                    a, type=2, axes=(0, 1),
                    threads=self._num_threads,
                    planner_effort=self._planner_effort
                )
                self._plan_cache[shape] = plan
            
            plan = self._plan_cache[shape]
            
            np.copyto(plan.input_array, problem, casting='unsafe')
            
            result_array = plan()
            
            return result_array.tolist()

        # --- Path 2: Fallback implementation using scipy ---
        else:
            input_array = np.asarray(problem, dtype='float64')
            result_array = np.asarray(self._scipy_dstn(input_array, type=2, workers=-1))
            return result_array.tolist()