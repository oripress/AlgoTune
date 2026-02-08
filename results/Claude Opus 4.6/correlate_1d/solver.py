import numpy as np
from scipy.fft import next_fast_len
from numpy.fft import rfft, irfft
from numba import njit, prange

@njit(cache=True, parallel=True)
def batch_direct_correlate(pairs_a, pairs_b, pairs_na, pairs_nb, max_out_len, count):
    """Direct correlation for multiple small pairs."""
    results = np.zeros((count, max_out_len))
    out_lens = np.empty(count, dtype=np.int64)
    for i in prange(count):
        na = pairs_na[i]
        nb = pairs_nb[i]
        out_len = na + nb - 1
        out_lens[i] = out_len
        for k in range(out_len):
            s = 0.0
            # correlate(a, b, 'full')[k] = sum_j a[j] * b[j - k + nb - 1]
            for j in range(na):
                bj = j - k + nb - 1
                if 0 <= bj < nb:
                    s += pairs_a[i, j] * pairs_b[i, bj]
            results[i, k] = s
    return results, out_lens

class Solver:
    def __init__(self):
        # Warm up numba
        a = np.zeros((1, 4))
        a[0, :3] = [1.0, 2.0, 3.0]
        b = np.zeros((1, 4))
        b[0, :2] = [1.0, 2.0]
        na = np.array([3], dtype=np.int64)
        nb = np.array([2], dtype=np.int64)
        batch_direct_correlate(a, b, na, nb, 4, 1)
    
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        
        results = [None] * n
        fft_groups = {}
        small_pairs = []
        
        for i in range(n):
            a, b = problem[i]
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            na = len(a)
            nb = len(b)
            out_len = na + nb - 1
            
            if out_len <= 128:
                small_pairs.append((i, a, b, na, nb))
            else:
                fft_len = next_fast_len(out_len)
                if fft_len not in fft_groups:
                    fft_groups[fft_len] = []
                fft_groups[fft_len].append((i, a, b, out_len))
        
        # Process small pairs
        if small_pairs:
            count = len(small_pairs)
            if count <= 3:
                for i, a, b, na, nb in small_pairs:
                    results[i] = np.correlate(a, b, mode='full')
            else:
                max_na = max(sp[3] for sp in small_pairs)
                max_nb = max(sp[4] for sp in small_pairs)
                max_out_len = max_na + max_nb - 1
                
                a_batch = np.zeros((count, max_na))
                b_batch = np.zeros((count, max_nb))
                na_arr = np.empty(count, dtype=np.int64)
                nb_arr = np.empty(count, dtype=np.int64)
                idx_arr = [0] * count
                
                for j, (idx, a, b, na, nb) in enumerate(small_pairs):
                    a_batch[j, :na] = a
                    b_batch[j, :nb] = b
                    na_arr[j] = na
                    nb_arr[j] = nb
                    idx_arr[j] = idx
                
                res_batch, out_lens = batch_direct_correlate(
                    a_batch, b_batch, na_arr, nb_arr, max_out_len, count
                )
                
                for j in range(count):
                    results[idx_arr[j]] = res_batch[j, :out_lens[j]].copy()
        
        # Process FFT groups
        for fft_len, group in fft_groups.items():
            batch_size = len(group)
            if batch_size == 1:
                idx, a, b, out_len = group[0]
                fa = rfft(a, n=fft_len)
                fb = rfft(b[::-1], n=fft_len)
                fa *= fb
                results[idx] = irfft(fa, n=fft_len)[:out_len]
            else:
                a_padded = np.zeros((batch_size, fft_len))
                b_padded = np.zeros((batch_size, fft_len))
                
                for j in range(batch_size):
                    _, a, b, _ = group[j]
                    a_padded[j, :len(a)] = a
                    b_padded[j, :len(b)] = b[::-1]
                
                fa = rfft(a_padded, axis=1)
                fb = rfft(b_padded, axis=1)
                fa *= fb
                res_all = irfft(fa, n=fft_len, axis=1)
                
                for j in range(batch_size):
                    idx = group[j][0]
                    out_len = group[j][3]
                    results[idx] = res_all[j, :out_len].copy()
        
        return results