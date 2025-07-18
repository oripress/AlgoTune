import os
import numpy as np
from scipy import signal
from concurrent.futures import ThreadPoolExecutor

# Set environment variables to prevent thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

class Solver:
    def __init__(self, mode='full', **kwargs):
        self.mode = mode
        
    def solve(self, problem, **kwargs):
        if not problem:
            return []
            
        # Pre-filter valid pairs and pre-convert arrays
        tasks = []
        for a, b in problem:
            if self.mode == "valid" and len(b) > len(a):
                continue
                
            # Convert to float arrays once
            a_arr = np.asfarray(a, dtype=np.float64)
            b_arr = np.asfarray(b, dtype=np.float64)
            tasks.append((a_arr, b_arr))
            
        if not tasks:
            return []
            
        # Process correlations in parallel while preserving order
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda task: signal.correlate(task[0], task[1], mode=self.mode, method='fft'),
                tasks
            ))
            
        return results