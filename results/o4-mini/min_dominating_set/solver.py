import sys
sys.setrecursionlimit(10000)

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        # Precompute closed-neighborhood bitmasks
        neighbor_masks = [0] * n
        for u in range(n):
            m = 1 << u
            row = problem[u]
            for v, e in enumerate(row):
                if e:
                    m |= 1 << v
            neighbor_masks[u] = m
        all_mask = (1 << n) - 1
        # Greedy initial solution for upper bound
        covered = 0
        picks = []
        while covered != all_mask:
            rem = all_mask & ~covered
            best_u = -1
            best_cov = -1
            for u in range(n):
                cov = (neighbor_masks[u] & rem).bit_count()
                if cov > best_cov:
                    best_cov = cov
                    best_u = u
            picks.append(best_u)
            covered |= neighbor_masks[best_u]
        best_k = len(picks)
        best_sol = picks.copy()
        # Precompute static bound parameters
        max_cov = max(m.bit_count() for m in neighbor_masks)
        # Coverers for each vertex v
        coverers_by_v = [[] for _ in range(n)]
        for u, m in enumerate(neighbor_masks):
            mm = m
            while mm:
                lb = mm & -mm
                v = lb.bit_length() - 1
                coverers_by_v[v].append(u)
                mm ^= lb
        # Sort coverers to try larger coverage first
        sizes = [m.bit_count() for m in neighbor_masks]
        for v in range(n):
            coverers_by_v[v].sort(key=lambda u: sizes[u], reverse=True)
        memo = {}
        # Branch-and-bound DFS
        def dfs(covered_mask, depth, cur_picks):
            nonlocal best_k, best_sol
            if depth >= best_k:
                return
            if covered_mask == all_mask:
                best_k = depth
                best_sol = cur_picks.copy()
                return
            # static lower bound
            rem_mask = all_mask & ~covered_mask
            rem_cnt = rem_mask.bit_count()
            lb = (rem_cnt + max_cov - 1) // max_cov
            if depth + lb >= best_k:
                return
            # pick next vertex to cover: uncovered v with fewest coverers
            mask = rem_mask
            best_v = -1
            min_c = None
            while mask:
                lbm = mask & -mask
                v = lbm.bit_length() - 1
                c_len = len(coverers_by_v[v])
                if min_c is None or c_len < min_c:
                    min_c = c_len
                    best_v = v
                mask ^= lbm
            # branch on all coverers of best_v
            for u in coverers_by_v[best_v]:
                new_cov = covered_mask | neighbor_masks[u]
                if new_cov == covered_mask:
                    continue
                nd = depth + 1
                prev = memo.get(new_cov)
                if prev is not None and prev <= nd:
                    continue
                memo[new_cov] = nd
                cur_picks.append(u)
                dfs(new_cov, nd, cur_picks)
                cur_picks.pop()
        dfs(0, 0, [])
        return sorted(best_sol)