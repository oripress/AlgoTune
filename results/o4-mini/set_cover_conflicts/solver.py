import sys
from typing import Any, List

# Use bitmask B&B when objects are few; otherwise fall back to RC2 MaxSAT.
BB_N_THRESHOLD = 20

class Solver:
    def solve(self, problem: Any, **kwargs) -> List[int]:
        n, sets_list, conflicts = problem
        if n <= BB_N_THRESHOLD:
            return self._bb_solve(n, sets_list, conflicts)
        else:
            return self._rc2_solve(n, sets_list, conflicts)

    def _bb_solve(
        self,
        n: int,
        sets_list: List[List[int]],
        conflicts: List[List[int]],
    ) -> List[int]:
        sys.setrecursionlimit(1000000)
        m = len(sets_list)
        # Precompute coverage masks for each set
        sets_mask = [0] * m
        for i, s in enumerate(sets_list):
            mask = 0
            for o in s:
                mask |= 1 << o
            sets_mask[i] = mask
        universe = (1 << n) - 1

        # Precompute conflict masks
        conflict_mask = [1 << i for i in range(m)]
        for C in conflicts:
            for a in C:
                for b in C:
                    if a != b:
                        conflict_mask[a] |= 1 << b

        # Precompute which sets cover each object
        cover_list = [[] for _ in range(n)]
        for i, s in enumerate(sets_list):
            for o in s:
                cover_list[o].append(i)

        # Start with trivial cover of singletons (always exists)
        best_mask = 0
        for i, s in enumerate(sets_list):
            if len(s) == 1:
                best_mask |= 1 << i
        best_len = best_mask.bit_count()

        # Greedy improve upper bound
        covered = 0
        banned = 0
        sel_mask = 0
        while covered != universe:
            rem = universe & ~covered
            best_i = -1
            best_cov = 0
            for i in range(m):
                bit = 1 << i
                if (sel_mask & bit) or (banned & bit):
                    continue
                cov = (sets_mask[i] & rem).bit_count()
                if cov > best_cov:
                    best_cov = cov
                    best_i = i
            if best_i < 0 or best_cov == 0:
                break
            sel_mask |= 1 << best_i
            covered |= sets_mask[best_i]
            banned |= conflict_mask[best_i]
        if covered == universe:
            glen = sel_mask.bit_count()
            if glen < best_len:
                best_len = glen
                best_mask = sel_mask

        # A simple lower‐bound: largest coverage per set
        max_cs = 0
        for mask in sets_mask:
            c = mask.bit_count()
            if c > max_cs:
                max_cs = c
        if max_cs == 0:
            return []

        # Branch‐and‐Bound DFS
        def dfs(cur_sel: int, cur_cov: int, cur_ban: int, cur_len: int):
            nonlocal best_len, best_mask
            # prune by current best
            if cur_len >= best_len:
                return
            # if covered all, record
            if cur_cov == universe:
                best_len = cur_len
                best_mask = cur_sel
                return
            # lower bound by coverage
            rem = universe & ~cur_cov
            rem_cnt = rem.bit_count()
            lb = (rem_cnt + max_cs - 1) // max_cs
            if cur_len + lb >= best_len:
                return
            # pick next uncovered object by lowest‐set bit
            obj = (rem & -rem).bit_length() - 1
            for s in cover_list[obj]:
                bit = 1 << s
                if (cur_sel & bit) or (cur_ban & bit):
                    continue
                dfs(
                    cur_sel | bit,
                    cur_cov | sets_mask[s],
                    cur_ban | conflict_mask[s],
                    cur_len + 1,
                )

        dfs(0, 0, 0, 0)
        # extract solution indices
        return [i for i in range(m) if (best_mask >> i) & 1]

    def _rc2_solve(
        self,
        n: int,
        sets_list: List[List[int]],
        conflicts: List[List[int]],
    ) -> List[int]:
        # delayed import
        from pysat.formula import WCNF
        from pysat.examples.rc2 import RC2

        m = len(sets_list)
        # coverage clauses: each object must be in at least one selected set
        cover_list = [[] for _ in range(n)]
        for i, s in enumerate(sets_list):
            for o in s:
                cover_list[o].append(i + 1)
        wcnf = WCNF()
        for lits in cover_list:
            wcnf.append(lits)
        # conflict: at most one per group
        for C in conflicts:
            idxs = [i + 1 for i in C]
            for i in range(len(idxs)):
                ii = idxs[i]
                for jj in idxs[i + 1 :]:
                    wcnf.append([-ii, -jj])
        # soft: minimize number of sets
        for i in range(1, m + 1):
            wcnf.append([-i], weight=1)
        solver = RC2(wcnf)
        model = solver.compute()
        if model is None:
            return list(range(n))
        return [lit - 1 for lit in model if lit > 0]