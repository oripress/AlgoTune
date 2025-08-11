import numpy as np
from typing import Any, List, Dict

# FFT helpers: prefer scipy.fft (has next_fast_len), fall back to numpy.fft
try:
    from scipy.fft import rfft as _rfft, irfft as _irfft, next_fast_len as _next_fast_len  # type: ignore
except Exception:
    from numpy.fft import rfft as _rfft, irfft as _irfft  # type: ignore
    _next_fast_len = None  # type: ignore

# Optional: use scipy.signal.correlate for small/direct fallback
try:
    from scipy import signal as _signal  # type: ignore
except Exception:
    _signal = None

class Solver:
    def solve(self, problem: list, **kwargs) -> Any:
        """
        Compute 1D correlations for a list of (a, b) pairs.

        Parameters
        - problem: list of pairs (a, b), each array-like 1D
        - kwargs: may contain 'mode' ('full', 'same', 'valid')

        Returns
        - list of numpy.ndarray correlation results (float64)
        """
        mode = kwargs.get("mode", getattr(self, "mode", "full"))
        try:
            mode = str(mode).lower()
        except Exception:
            mode = "full"
        if mode not in ("full", "same", "valid"):
            mode = "full"

        n_items = len(problem)
        results: List[np.ndarray] = [None] * n_items

        # small-task direct threshold (product of lengths). Very small products are faster direct.
        DIRECT_PRODUCT_THRESHOLD = 512

        # memory budget in floats for chunking batched FFTs (keeps memory moderate)
        memory_limit_floats = 2_000_000  # ~16MB for float64 per chunk (tunable)

        def next_fast_len_local(n: int) -> int:
            if _next_fast_len is not None:
                return int(_next_fast_len(n))
            if n <= 1:
                return 1
            # fallback: next power of two
            return 1 << ((n - 1).bit_length())

        ascontig = np.ascontiguousarray

        # collect tasks that will use FFT batching
        fft_groups: Dict[int, List[Dict]] = {}
        pending_direct_indices: List[int] = []

        for idx, pair in enumerate(problem):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                results[idx] = np.zeros(0, dtype=np.float64)
                continue

            a = np.asarray(pair[0], dtype=np.float64).ravel()
            b = np.asarray(pair[1], dtype=np.float64).ravel()
            la = int(a.size)
            lb = int(b.size)

            # empty arrays -> empty result
            if la == 0 or lb == 0:
                results[idx] = np.zeros(0, dtype=np.float64)
                continue

            # For 'valid' mode, if b longer than a there is no valid overlap -> empty
            if mode == "valid" and lb > la:
                results[idx] = np.zeros(0, dtype=np.float64)
                continue

            # Small tasks: compute directly to avoid FFT overhead
            if la * lb <= DIRECT_PRODUCT_THRESHOLD:
                # compute immediately and store
                if _signal is not None:
                    res = _signal.correlate(a, b, mode=mode)
                else:
                    res = np.correlate(a, b, mode=mode)
                results[idx] = np.asarray(res, dtype=np.float64)
                continue

            # Prepare FFT task: convolve a with reversed b to get correlation
            n_full = la + lb - 1
            br = b[::-1]
            nfft = next_fast_len_local(n_full)
            fft_groups.setdefault(nfft, []).append(
                {"a": a, "br": br, "la": la, "lb": lb, "n_full": n_full, "pos": idx}
            )

        # If no FFT tasks, just return the results we computed directly
        if not fft_groups:
            return results

        # Process each FFT group in chunks to limit memory usage
        for nfft, group in fft_groups.items():
            M = len(group)
            # chunk size so that chunk_size * nfft <= memory_limit_floats
            chunk_size = max(1, memory_limit_floats // max(1, nfft))
            if chunk_size > M:
                chunk_size = M

            # number of frequency bins for rfft (not needed explicitly; rfft handles it)
            for start in range(0, M, chunk_size):
                end = min(M, start + chunk_size)
                chunk = group[start:end]
                k = len(chunk)

                # Allocate padded arrays (rows = tasks in chunk)
                A = np.zeros((k, nfft), dtype=np.float64)
                B = np.zeros((k, nfft), dtype=np.float64)

                for i, t in enumerate(chunk):
                    la = t["la"]
                    lb = t["lb"]
                    A[i, :la] = t["a"]
                    B[i, :lb] = t["br"]

                # Batched FFTs
                RA = _rfft(A, axis=1)
                RB = _rfft(B, axis=1)
                RA *= RB  # elementwise multiply each row
                # inverse real FFT to get linear convolution result
                # request n=nfft to ensure correct output length
                conv = _irfft(RA, n=nfft, axis=1)

                # Extract appropriate slices and place results
                for i, t in enumerate(chunk):
                    n_full_i = int(t["n_full"])
                    conv_full = conv[i, :n_full_i]
                    la = int(t["la"])
                    lb = int(t["lb"])

                    if mode == "full":
                        res = conv_full
                    elif mode == "same":
                        # 'same' should return output length equal to the first input (a)
                        out_len = la
                        start_idx = (n_full_i - out_len) // 2
                        if start_idx < 0:
                            start_idx = 0
                        res = conv_full[start_idx : start_idx + out_len]
                    else:  # valid
                        # by construction here lb <= la (we filtered lb > la earlier in 'valid')
                        start_idx = lb - 1
                        end_idx = start_idx + (la - lb + 1)
                        if end_idx <= start_idx:
                            res = np.zeros(0, dtype=np.float64)
                        else:
                            res = conv_full[start_idx:end_idx]

                    results[t["pos"]] = ascontig(np.asarray(res, dtype=np.float64))

        # Final safety: fill any remaining None entries with direct correlate
        for idx, r in enumerate(results):
            if r is None:
                pair = problem[idx]
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    results[idx] = np.zeros(0, dtype=np.float64)
                    continue
                a = np.asarray(pair[0], dtype=np.float64).ravel()
                b = np.asarray(pair[1], dtype=np.float64).ravel()
                if a.size == 0 or b.size == 0:
                    results[idx] = np.zeros(0, dtype=np.float64)
                    continue
                if _signal is not None:
                    res = _signal.correlate(a, b, mode=mode)
                else:
                    res = np.correlate(a, b, mode=mode)
                results[idx] = np.asarray(res, dtype=np.float64)

        return results