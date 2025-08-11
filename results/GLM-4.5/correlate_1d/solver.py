import numpy as np
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class Solver:
    def __init__(self, mode="full"):
        self.mode = mode

    def _correlate_pair(self, a, b):
        if self.mode == "valid" and len(b) > len(a):
            return None
        return signal.correlate(a, b, mode=self.mode)

    def solve(self, problem, **kwargs):
        # Convert all inputs to numpy arrays first
        processed_problem = []
        for a, b in problem:
            if not isinstance(a, np.ndarray):
                a = np.array(a, dtype=np.float64)
            if not isinstance(b, np.ndarray):
                b = np.array(b, dtype=np.float64)
            processed_problem.append((a, b))
        
        # Use parallel processing for multiple pairs
        num_workers = min(multiprocessing.cpu_count(), len(processed_problem))
        
        if num_workers <= 1 or len(processed_problem) <= 1:
            # Sequential processing for single pair or when parallelism isn't beneficial
            results = []
            for a, b in processed_problem:
                res = self._correlate_pair(a, b)
                if res is not None:
                    results.append(res)
            return results
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._correlate_pair, a, b) for a, b in processed_problem]
                results = [future.result() for future in futures if future.result() is not None]
            return results