import numpy as np
import numba
from typing import Any
import itertools

@numba.njit(cache=True, fastmath=True)
def _gale_shapley_core(proposer_prefs: np.ndarray, receiver_ranks: np.ndarray) -> np.ndarray:
    """
    Numba-accelerated core of the Gale-Shapley algorithm, using an efficient
    "chain of proposals" implementation and memory-optimized data types.
    """
    n = proposer_prefs.shape[0]
    dtype = proposer_prefs.dtype

    receiver_match = np.full(n, -1, dtype=dtype)
    next_proposal_idx = np.zeros(n, dtype=dtype)

    for i in range(n):
        p = i
        while True:
            r = proposer_prefs[p, next_proposal_idx[p]]
            next_proposal_idx[p] += 1
            current_partner = receiver_match[r]
            
            if current_partner == -1:
                receiver_match[r] = p
                break
            
            elif receiver_ranks[r, p] < receiver_ranks[r, current_partner]:
                receiver_match[r] = p
                p = current_partner
            else:
                pass
    
    proposer_match = np.empty(n, dtype=dtype)
    proposer_match[receiver_match] = np.arange(n, dtype=dtype)
    
    return proposer_match

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Solves the Stable Matching Problem using a Numba-accelerated
        Gale-Shapley algorithm with highly optimized data preparation using
        iterators to minimize memory allocation overhead.
        """
        proposer_prefs_dict = problem["proposer_prefs"]
        receiver_prefs_dict = problem["receiver_prefs"]
        n = len(proposer_prefs_dict)
        
        dtype = np.int16

        # --- Optimized Data Preparation using fromiter ---
        # This avoids creating large intermediate lists of lists.
        # itertools.chain.from_iterable is a fast C implementation for flattening.
        # np.fromiter with a count pre-allocates the array for maximum speed.
        flat_proposer_iter = itertools.chain.from_iterable(proposer_prefs_dict.values())
        proposer_prefs = np.fromiter(flat_proposer_iter, dtype=dtype, count=n*n).reshape(n, n)

        flat_receiver_iter = itertools.chain.from_iterable(receiver_prefs_dict.values())
        receiver_prefs = np.fromiter(flat_receiver_iter, dtype=dtype, count=n*n).reshape(n, n)
        
        # The vectorized rank creation is already very fast.
        receiver_ranks = np.empty_like(receiver_prefs)
        rows = np.arange(n, dtype=dtype).reshape(-1, 1)
        ranks = np.arange(n, dtype=dtype)
        receiver_ranks[rows, receiver_prefs] = ranks

        # --- Core Algorithm ---
        proposer_match_np = _gale_shapley_core(proposer_prefs, receiver_ranks)

        # --- Format Output ---
        return {"matching": proposer_match_np.tolist()}