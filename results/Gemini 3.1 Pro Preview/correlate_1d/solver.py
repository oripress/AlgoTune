import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
import concurrent.futures

class Solver:
    def solve(self, problem: list) -> list:
        mode = getattr(self, 'mode', 'full')
        
        def process_pair(pair):
            idx, (a, b) = pair
            n1, n2 = len(a), len(b)
            if mode == "valid" and n2 > n1:
                return idx, None
            
            n = n1 + n2 - 1
            fast_n = next_fast_len(n, real=True)
            
            A = rfft(a, fast_n)
            B = rfft(b[::-1], fast_n)
            A *= B
            
            res = irfft(A, fast_n)[:n]
            
            if mode == "valid":
                res = res[n2 - 1 : n1]
                
            return idx, res
            
        indexed_problem = list(enumerate(problem))
        indexed_problem.sort(key=lambda x: len(x[1][0]) + len(x[1][1]), reverse=True)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_pair, indexed_problem))
            
        results.sort(key=lambda x: x[0])
        return [r for idx, r in results if r is not None]