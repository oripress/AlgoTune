from __future__ import annotations

from typing import Any, Dict, List, Tuple

def _lsb_index(x: int) -> int:
    """Index of least-significant set bit; x must be nonzero."""
    return (x & -x).bit_length() - 1

class _Instance:
    __slots__ = (
        "n",
        "cand_orig",
        "cand_masks",
        "coverers",
        "cover_len",
        "forced_vs",
        "forced_coverer",
    )

    def __init__(
        self,
        n: int,
        cand_orig: List[int],
        cand_masks: List[int],
        coverers: List[List[int]],
        cover_len: List[int],
        forced_vs: List[int],
        forced_coverer: List[int],
    ) -> None:
        self.n = n
        self.cand_orig = cand_orig
        self.cand_masks = cand_masks
        self.coverers = coverers
        self.cover_len = cover_len
        self.forced_vs = forced_vs
        self.forced_coverer = forced_coverer

class Solver:
    """
    Exact minimum dominating set using bitmasks + branch & bound.

    Key ideas:
      - Model as minimum set cover on universe V with sets N[v] (closed neighborhoods).
      - Remove dominated sets: if N[a] ⊆ N[b], never need a.
      - Greedy to get strong initial upper bound.
      - DFS branch on an uncovered vertex with smallest #coverers.
      - Prune with a simple cover-size lower bound + memoization (state=uncovered mask).
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _build_instance(problem: List[List[int]]) -> _Instance:
        n = len(problem)
        if n == 0:
            return _Instance(
                n=0,
                cand_orig=[],
                cand_masks=[],
                coverers=[],
                cover_len=[],
                forced_vs=[],
                forced_coverer=[],
            )

        # Build adjacency lists + closed-neighborhood bitmasks.
        masks: List[int] = [0] * n
        neigh: List[List[int]] = [[] for _ in range(n)]
        for i, row in enumerate(problem):
            mi = 1 << i
            for j, v in enumerate(row):
                if v:
                    mi |= 1 << j
                    neigh[i].append(j)
            masks[i] = mi

        # Dominance reduction: if mask[i] subset of some larger/equal mask, drop i.
        order = sorted(range(n), key=lambda i: masks[i].bit_count(), reverse=True)
        kept: List[int] = []
        kept_masks: List[int] = []
        for i in order:
            mi = masks[i]
            dominated = False
            for mk in kept_masks:
                if mi & ~mk == 0:
                    dominated = True
                    break
            if not dominated:
                kept.append(i)
                kept_masks.append(mi)

        cand_orig = kept
        cand_masks = [masks[i] for i in cand_orig]
        pos_of: Dict[int, int] = {v: p for p, v in enumerate(cand_orig)}

        # Build coverers lists using neighbor lists.
        coverers: List[List[int]] = [[] for _ in range(n)]
        for u in range(n):
            lst: List[int] = []
            pu = pos_of.get(u)
            if pu is not None:
                lst.append(pu)
            for v in neigh[u]:
                pv = pos_of.get(v)
                if pv is not None:
                    lst.append(pv)
            coverers[u] = lst

        cover_len = [len(coverers[u]) for u in range(n)]
        forced_coverer = [-1] * n
        forced_vs: List[int] = []
        for u in range(n):
            if cover_len[u] == 1:
                forced_vs.append(u)
                forced_coverer[u] = coverers[u][0]

        return _Instance(
            n=n,
            cand_orig=cand_orig,
            cand_masks=cand_masks,
            coverers=coverers,
            cover_len=cover_len,
            forced_vs=forced_vs,
            forced_coverer=forced_coverer,
        )

    def solve(self, problem: List[List[int]], **kwargs: Any) -> Any:
        inst = self._build_instance(problem)
        n = inst.n
        if n == 0:
            return []

        cand_masks = inst.cand_masks
        coverers = inst.coverers
        cover_len = inst.cover_len
        forced_vs = inst.forced_vs
        forced_coverer = inst.forced_coverer

        all_uncovered = (1 << n) - 1

        bit_count = int.bit_count

        # Candidate ordering by set size (static).
        k = len(cand_masks)
        cand_sizes = [bit_count(m) for m in cand_masks]
        cand_order = list(range(k))
        cand_order.sort(key=cand_sizes.__getitem__, reverse=True)

        # Sort coverers[u] once by candidate set size (static ordering).
        for u in range(n):
            cu = coverers[u]
            if len(cu) > 1:
                cu.sort(key=cand_sizes.__getitem__, reverse=True)

        # Forced vertices mask (globally unique coverer).
        forced_mask = 0
        for u in forced_vs:
            forced_mask |= 1 << u

        # Precompute dom_union[u] = OR of candidate masks that can cover u.
        dom_union = [0] * n
        for u in range(n):
            du = 0
            for p in coverers[u]:
                du |= cand_masks[p]
            dom_union[u] = du

        # Buckets for fast pivot choice (cover_len fixed).
        max_len = max(cover_len)
        bucket_masks = [0] * (max_len + 1)
        for u, l in enumerate(cover_len):
            if l >= 0:
                bucket_masks[l] |= 1 << u
        bucket_lens = [l for l in range(1, max_len + 1) if bucket_masks[l]]

        def apply_forced(U: int, sel: int, depth: int) -> Tuple[int, int, int]:
            t = U & forced_mask
            while t:
                b = t & -t
                u = b.bit_length() - 1
                p = forced_coverer[u]
                if (sel >> p) & 1 == 0:
                    sel |= 1 << p
                    depth += 1
                U &= ~cand_masks[p]
                t &= ~b
            return U, sel, depth

        def greedy_upper_bound(U: int) -> Tuple[int, int]:
            sel = 0
            depth = 0
            while U:
                unc = bit_count(U)
                best_p = -1
                best_cov = 0
                for p in cand_order:
                    cov = bit_count(cand_masks[p] & U)
                    if cov > best_cov:
                        best_cov = cov
                        best_p = p
                        if cov == unc:
                            break
                if best_p < 0:
                    break
                sel |= 1 << best_p
                depth += 1
                U &= ~cand_masks[best_p]
            return depth, sel

        def choose_pivot(U: int) -> int:
            # Pick uncovered vertex with minimum global coverer count via buckets.
            for l in bucket_lens:
                t = U & bucket_masks[l]
                if t:
                    return (t & -t).bit_length() - 1
            return (U & -U).bit_length() - 1

        def lower_bound(U: int, depth: int, best_size: int) -> int:
            # Greedy optimistic grouping using dom_union; yields valid LB.
            t = U
            lb = 0
            while t:
                lb += 1
                if depth + lb >= best_size:
                    return lb
                b = t & -t
                u = b.bit_length() - 1
                t &= ~dom_union[u]
            return lb

        # Root forced moves.
        U0, sel0, depth0 = apply_forced(all_uncovered, 0, 0)

        greedy_depth, greedy_sel = greedy_upper_bound(U0)
        best_size = depth0 + greedy_depth
        best_sel = sel0 | greedy_sel

        memo: Dict[int, int] = {}

        def dfs(U: int, sel: int, depth: int) -> None:
            nonlocal best_size, best_sel

            if U == 0:
                if depth < best_size:
                    best_size = depth
                    best_sel = sel
                return
            if depth >= best_size:
                return

            # Forced moves (cheap with forced_mask)
            U, sel, depth = apply_forced(U, sel, depth)
            if U == 0:
                if depth < best_size:
                    best_size = depth
                    best_sel = sel
                return
            if depth >= best_size:
                return

            prev = memo.get(U)
            if prev is not None and depth >= prev:
                return
            memo[U] = depth

            lb = lower_bound(U, depth, best_size)
            if depth + lb >= best_size:
                return

            u = choose_pivot(U)
            opts = coverers[u]

            for p in opts:
                nd = depth + 1
                if nd >= best_size:
                    return
                dfs(U & ~cand_masks[p], sel | (1 << p), nd)

        dfs(U0, sel0, depth0)

        # Decode selected candidates -> original vertex indices
        res: List[int] = []
        sel = best_sel
        cand_orig = inst.cand_orig
        while sel:
            b = sel & -sel
            p = b.bit_length() - 1
            res.append(cand_orig[p])
            sel ^= b
        res.sort()
        return res