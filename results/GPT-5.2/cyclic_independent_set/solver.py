from typing import Any, Dict, List, Tuple

import numpy as np

def _index_to_tuple(idx: int, powers: np.ndarray) -> Tuple[int, ...]:
    """Convert linear index to base digits (most significant first)."""
    coords: List[int] = []
    for p in powers:
        q, idx = divmod(idx, int(p))
        coords.append(int(q))
    return tuple(coords)

def _compute_priority_map(num_nodes: int, n: int) -> np.ndarray:
    """
    Reference priority(el):
      el_clipped = clip(el, max=num_nodes-3)
      values = 2*product(range(1,n), repeat=n)
      multipliers = num_nodes**[n-1..0]
      x = sum((1+values+el_clipped)*multipliers)
      priority = sum(x % (num_nodes-2))

    For fixed el, only A = sum_j (1+el_clipped[j]) * num_nodes^(n-1-j) mod (num_nodes-2)
    matters. This returns f[A] for A in 0..mod-1.
    """
    mod = num_nodes - 2  # 5 when num_nodes=7

    # When n==1, reference uses an empty product -> sum over empty = 0.
    if n == 1:
        return np.zeros(mod, dtype=np.int64)

    # multipliers mod mod: num_nodes^(n-1-j) mod mod
    m = [pow(num_nodes % mod, n - 1 - j, mod) for j in range(n)]

    # Distribution of S(v) = sum_j (2*v_j*m_j) mod mod, with v_j in {1..n-1}
    dist = [1] + [0] * (mod - 1)
    for mj in m:
        term_counts = [0] * mod
        for v in range(1, n):
            term_counts[(2 * v * mj) % mod] += 1

        new_dist = [0] * mod
        for a, da in enumerate(dist):
            if not da:
                continue
            for t, ct in enumerate(term_counts):
                if ct:
                    new_dist[(a + t) % mod] += da * ct
        dist = new_dist

    # f[A] = sum_s dist[s] * ((A+s) mod mod)
    f = np.empty(mod, dtype=np.int64)
    for A in range(mod):
        total = 0
        for s in range(mod):
            total += dist[s] * ((A + s) % mod)
        f[A] = total
    return f

class Solver:
    """
    Fast greedy solver matching the reference output.

    Key speedup: only (num_nodes-2) distinct priority values exist, so we can precompute the
    priority for all vertices in vectorized numpy and iterate candidates in descending score.
    """

    def __init__(self) -> None:
        # (num_nodes, n) -> (powers, shifts, order)
        self._cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _build_cache(self, num_nodes: int, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = num_nodes**n
        powers = (num_nodes ** np.arange(n - 1, -1, -1)).astype(np.int64)

        # Vectorized shifts in [-1,0,1]^n: shape (3^n, n)
        # (avoids Python-level itertools.product)
        shifts = (np.indices((3,) * n, dtype=np.int16).reshape(n, -1).T - 1).astype(np.int16)

        # Priority values depend on A mod (num_nodes-2).
        f = _compute_priority_map(num_nodes, n)
        mod = num_nodes - 2

        # multipliers mod mod
        m = np.array([pow(num_nodes % mod, n - 1 - j, mod) for j in range(n)], dtype=np.int16)

        idx = np.arange(N, dtype=np.int64)
        A = np.zeros(N, dtype=np.int16)
        tmp64 = np.empty(N, dtype=np.int64)
        tmp16 = np.empty(N, dtype=np.int16)

        clip_max = num_nodes - 3  # 4 for C7
        for j, p in enumerate(powers):
            # digit at position j: floor(idx/p) % num_nodes
            np.floor_divide(idx, int(p), out=tmp64)
            np.remainder(tmp64, num_nodes, out=tmp64)
            tmp16[:] = tmp64  # cast
            np.minimum(tmp16, clip_max, out=tmp16)  # clip
            # A += (1+clip)*m[j] (mod mod at the end)
            A += (tmp16 + 1) * m[j]

        # Reduce mod and map to actual score values.
        if mod != 0:
            A %= mod
        score = f[A]  # int64

        # Candidate order: descending score; within equal score, ascending index (lex order).
        # Only a handful of distinct scores exist, so build by concatenating buckets.
        uniq = np.unique(score)
        uniq.sort()  # ascending
        order_parts = [np.flatnonzero(score == s) for s in uniq[::-1]]
        order = np.concatenate(order_parts).astype(np.int32, copy=False)

        return powers, shifts, order

    def solve(self, problem, **kwargs) -> Any:
        num_nodes, n = problem
        key = (num_nodes, n)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._build_cache(num_nodes, n)
            self._cache[key] = cached

        powers, shifts, order = cached
        N = num_nodes**n

        blocked = np.zeros(N, dtype=np.bool_)
        selected_tuples: List[Tuple[int, ...]] = []

        for idx in order:
            i = int(idx)
            if blocked[i]:
                continue
            tup = _index_to_tuple(i, powers)
            selected_tuples.append(tup)

            coords = np.asarray(tup, dtype=np.int16)
            neigh_coords = (coords + shifts) % num_nodes  # (3^n, n)
            neigh_idx = neigh_coords @ powers  # (3^n,)
            blocked[neigh_idx] = True

        return selected_tuples