from typing import Any, List, Tuple
import itertools
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> List[Tuple[int, ...]]:
        """
        Optimized solver reproducing the reference greedy construction via a closed-form
        computation of the priority scores and a greedy blocking procedure.

        It constructs:
        - All candidates (m-ary n-tuples)
        - Priority scores using an exact combinatorial formula equivalent to the reference
          _priority function, avoiding enumerating the exponential auxiliary 'values' array.
        - Greedy selection in descending score order, blocking neighbors under the strong product.
        """
        num_nodes, n = problem
        m = int(num_nodes)

        if n < 0:
            raise ValueError("Exponent n must be non-negative.")
        if n == 0:
            return [tuple()]

        # Special small cases fallback (should not occur for the intended task with m=7)
        s = m - 2
        if s <= 0:
            # Basic valid independent set as a fallback
            A = list(range(0, m if m % 2 == 0 else m - 1, 2))
            return list(itertools.product(A, repeat=n))

        # Precompute base-m powers for indexing and their residues modulo s
        powers = np.array([m ** i for i in range(n - 1, -1, -1)], dtype=np.int64)  # length n
        mw_mod = np.array([pow(m, i, s) for i in range(n - 1, -1, -1)], dtype=np.int64)  # length n

        # Compute the distribution of B_j modulo s, where
        #   B_j = sum_i (2 * mw_i) * t_i,   t_i in {1, ..., n-1}
        # This equals the circular convolution (mod s) over dimensions of the per-dimension
        # residue distributions g_i[r] = |{t in [1..n-1] : (2*mw_mod[i]*t) % s = r}|.
        c = (2 * mw_mod) % s
        dist = np.zeros(s, dtype=np.int64)

        if n == 1:
            # values = product(range(1, 1), repeat=1) -> empty set; distribution is all zeros
            dist[:] = 0
        else:
            dist[0] = 1  # start with delta at 0 for convolution identity
            for ci in c:
                g = np.zeros(s, dtype=np.int64)
                for t in range(1, n):
                    g[(ci * t) % s] += 1
                # Circular convolution dist (*) g modulo s
                new_dist = np.zeros(s, dtype=np.int64)
                for r in range(s):
                    total = 0
                    for k in range(s):
                        total += dist[k] * g[(r - k) % s]
                    new_dist[r] = total
                dist = new_dist

        S_total = (n - 1) ** n  # total number of 'values' combinations
        # Constant part: sum_r dist[r] * r
        idxs = np.arange(s, dtype=np.int64)
        C_B = int(dist @ idxs)
        # Tail counts: tail[t] = sum_{r >= t} dist[r], for t in 0..s; define tail[s] = 0
        tail = np.zeros(s + 1, dtype=np.int64)
        acc = 0
        for t in range(s - 1, -1, -1):
            acc += dist[t]
            tail[t] = acc

        # Enumerate all candidates (children) as m-ary n-tuples
        # Use small dtype to save memory; cast as needed during computations
        children = np.array(list(itertools.product(range(m), repeat=n)), dtype=np.int16)
        M = children.shape[0]

        # Compute A_mod for each candidate efficiently:
        #   A = sum_i (1 + min(val_i, m-3)) * mw_i
        # Take modulo s: A_mod = A % s
        clipmax = m - 3
        baseval = (1 + np.minimum(np.arange(m), clipmax)) % s  # shape (m,)
        baseval = baseval.astype(np.int64)

        A_mods = np.zeros(M, dtype=np.int64)
        for i in range(n):
            A_mods += baseval[children[:, i]] * (mw_mod[i] % s)
        A_mods %= s

        # Priority scores (exact integer form of the reference computation):
        # score = sum_j ((A_mod + B_j_mod) % s) over all j
        #       = S_total*A_mod + C_B - s * sum_{r >= s - A_mod} dist[r]
        if S_total == 0:
            scores = np.zeros(M, dtype=np.int64)
        else:
            thr = (s - A_mods).astype(np.int64)  # in [1..s], with possible s; tail[s] = 0
            scores = S_total * A_mods + C_B - s * tail[thr]

        # Greedy selection: pick in descending score order, block neighbors in strong product
        order = np.argsort(-scores, kind="mergesort")  # stable for tie-breaking
        blocked = np.zeros(M, dtype=bool)
        selected_idx: List[int] = []

        to_block = np.array(list(itertools.product((-1, 0, 1), repeat=n)), dtype=np.int16) if n > 0 else np.zeros((1, 0), dtype=np.int16)

        for idx in order:
            if blocked[idx]:
                continue
            selected_idx.append(int(idx))
            if n == 0:
                continue
            center = children[idx].astype(np.int16)
            neighbors = (center + to_block) % m
            neighbor_indices = (neighbors @ powers).astype(np.int64)
            blocked[neighbor_indices] = True

        # Build final solution as list of tuples
        solution = [tuple(int(v) for v in children[i]) for i in selected_idx]
        return solution