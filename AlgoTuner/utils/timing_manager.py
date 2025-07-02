"""
TimingManager: central service for calibrated phase timing.
"""

import time
import threading
import contextlib
import statistics
import gc
from enum import Enum

from AlgoTuner.utils.precise_timing import _initialize_timing_system, _timing_overhead_ns, _capture_overhead_ns

class Phase(Enum):
    TASK_LOADING = "task_loading"
    SOLVER_SETUP_VALIDATION = "solver_setup_validation"
    CODE_RELOAD = "code_reload"
    SOLVER_IMPORT = "solver_import"
    CONFIG_LOAD = "config_load"
    DATASET_LOADING = "dataset_loading"
    SOLVE_LOOP = "solve_loop"
    SOLVER_RUN = "solver_run"
    ORACLE_RUN = "oracle_run"
    AGGREGATION = "aggregation"

class TimingManager:
    """Singleton service to manage timing calibration and per-phase statistics."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # Ensure only one instance exists
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TimingManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Initialize the precise timing system (calibrates overheads)
        _initialize_timing_system()
        self._overhead_ns = _timing_overhead_ns or 0
        self._capture_overhead_ns = _capture_overhead_ns or 0
        # Storage for durations per phase (in nanoseconds)
        self._phases = {}
        self._initialized = True

    @contextlib.contextmanager
    def phase(self, name, capture_output: bool = False):  # name: str or Phase
        """Context manager to time a named phase (string or Phase), subtracting overhead."""
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            elapsed_ns = end_ns - start_ns - self._overhead_ns
            if capture_output:
                elapsed_ns = max(0, elapsed_ns - self._capture_overhead_ns)
            else:
                elapsed_ns = max(0, elapsed_ns)
            key = name.value if isinstance(name, Phase) else name
            self._phases.setdefault(key, []).append(elapsed_ns)

    def median_ms(self, name: str) -> float:
        """Return the median duration for a given phase, in milliseconds."""
        times = self._phases.get(name, [])
        if not times:
            return 0.0
        return statistics.median(times) / 1e6

    def report(self) -> str:
        """Generate a simple report of median times per phase."""
        lines = ["TimingManager Report:"]
        for name, times in self._phases.items():
            med_ms = statistics.median(times) / 1e6
            lines.append(f"  {name}: median {med_ms:.2f}ms over {len(times)} runs")
        return "\n".join(lines)

    @contextlib.contextmanager
    def gc_paused(self):
        """Context manager to disable GC during timing-sensitive phases."""
        was_enabled = gc.isenabled()
        if was_enabled:
            gc.disable()
        try:
            yield
        finally:
            if was_enabled:
                gc.enable() 