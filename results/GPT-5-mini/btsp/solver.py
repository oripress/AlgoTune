from typing import Any, List, Optional
import itertools
import bisect

class Solver:
    def solve(self, problem: List[List[float]], **kwargs) -> Any:
        """
        Solve the Bottleneck Traveling Salesman Problem (BTSP).

        Approach:
        - For very small n (<=9) use brute-force permutations with pruning.
        - Otherwise binary-search the bottleneck threshold over unique edge weights.
          For each threshold build the graph of allowed edges and check for a
          Hamiltonian cycle via a bitmask DP optimized with:
            * iterating only masks that include vertex 0 (odd masks)
            * iterating missing vertices by their set bits (avoids per-vertex branches)
            * local variable binding to reduce attribute lookups
            * a cheap greedy upper bound to narrow binary search
        """
        n = len(problem)
        if n <= 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]

        rows = problem  # local alias

        # Small n: brute force with pruning
        if n <= 9:
            best_b = float("inf")
            best_tour: Optional[List[int]] = None
            nodes = list(range(1, n))
            for perm in itertools.permutations(nodes):
                cur_max = 0.0
                prev = 0
                skip = False
                for v in perm:
                    w = rows[prev][v]
                    if w > cur_max:
                        cur_max = w
                        if cur_max >= best_b:
                            skip = True
                            break
                    prev = v
                if skip:
                    continue
                # closing edge
                w = rows[prev][0]
                if w > cur_max:
                    cur_max = w
                if cur_max < best_b:
                    best_b = cur_max
                    best_tour = [0] + list(perm) + [0]
            if best_tour is not None:
                return best_tour
            return list(range(n)) + [0]

        # Precompute bit masks
        bits = [1 << i for i in range(n)]
        full_mask = (1 << n) - 1
        full_mask_no_zero = full_mask ^ 1  # all bits except bit 0

        # Collect unique weights for binary search
        wset = set()
        for i in range(n):
            ri = rows[i]
            for j in range(i + 1, n):
                wset.add(ri[j])
        weights = sorted(wset)
        if not weights:
            return list(range(n)) + [0]

        # Cheap greedy tour to obtain an upper bound on bottleneck (helps narrow search)
        greedy_max = 0.0
        visited = [False] * n
        visited[0] = True
        cur = 0
        for _ in range(n - 1):
            ri = rows[cur]
            best_v = -1
            best_w = float("inf")
            for v in range(n):
                if not visited[v] and ri[v] < best_w:
                    best_w = ri[v]
                    best_v = v
            visited[best_v] = True
            greedy_max = max(greedy_max, best_w)
            cur = best_v
        greedy_max = max(greedy_max, rows[cur][0])

        hi_guess = bisect.bisect_right(weights, greedy_max) - 1
        if hi_guess < 0:
            hi_guess = len(weights) - 1

        # Core check: is there a Hamiltonian cycle using only edges <= threshold?
        def hamiltonian_cycle_with_threshold(threshold: float) -> Optional[List[int]]:
            n_local = n
            rows_local = rows
            bits_local = bits

            # Build adjacency bitmasks
            adj = [0] * n_local
            for i in range(n_local):
                row = rows_local[i]
                m = 0
                # avoid creating temporaries; test j != i inline
                for j in range(n_local):
                    if j != i and row[j] <= threshold:
                        m |= bits_local[j]
                adj[i] = m

            # Quick degree checks: every vertex must have degree >= 2 (for n>2)
            for a in adj:
                if a == 0:
                    return None
                if n_local > 2 and (a & (a - 1)) == 0:
                    return None

            # Connectivity check (undirected) from node 0 using bitmasks
            visited_mask = 1
            stack = [0]
            bl = int.bit_length
            while stack:
                u = stack.pop()
                nb = adj[u] & ~visited_mask
                while nb:
                    vbit = nb & -nb
                    nb ^= vbit
                    visited_mask |= vbit
                    stack.append(bl(vbit) - 1)
            if visited_mask != full_mask:
                return None

            size = 1 << n_local
            endpoints = [0] * size
            endpoints[1] = 1  # mask with only vertex 0 -> endpoint 0

            # Local bindings for speed
            endpoints_local = endpoints
            adj_local = adj
            full_mask_local = full_mask
            full_mask_no_zero_local = full_mask_no_zero

            # Iterate only masks that include vertex 0 (odd masks)
            for mask in range(1, size, 2):
                ep = endpoints_local[mask]
                if ep == 0:
                    continue
                # iterate vertices not in mask (exclude vertex 0)
                rem = full_mask_no_zero_local & ~mask
                while rem:
                    vbit = rem & -rem
                    rem ^= vbit
                    v = bl(vbit) - 1
                    # if some endpoint in ep is adjacent to v, then v becomes an endpoint
                    if ep & adj_local[v]:
                        endpoints_local[mask | vbit] |= vbit

            ep_full = endpoints_local[full_mask_local]
            if ep_full == 0:
                return None

            # Need an endpoint that connects back to 0
            if (ep_full & adj_local[0]) == 0:
                return None

            # pick a candidate last vertex (lowest set bit that is adjacent to 0)
            last_candidates = ep_full & adj_local[0]
            last_bit = last_candidates & -last_candidates
            last = bl(last_bit) - 1

            # Reconstruct path in reverse: fill positions n-1..1
            path: List[int] = [0] * (n_local + 1)
            path[0] = 0
            path[n_local] = 0
            mask = full_mask_local
            curv = last
            for pos in range(n_local - 1, 0, -1):
                path[pos] = curv
                prev_mask = mask ^ (1 << curv)
                cand = endpoints_local[prev_mask] & adj_local[curv]
                if cand == 0:
                    return None
                ubit = cand & -cand
                u = bl(ubit) - 1
                curv = u
                mask = prev_mask

            if curv != 0:
                return None
            return path

        # Binary search over sorted unique weights
        lo = 0
        hi = min(len(weights) - 1, hi_guess)
        best_path: Optional[List[int]] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            t = weights[mid]
            path = hamiltonian_cycle_with_threshold(t)
            if path is not None:
                best_path = path
                hi = mid - 1
            else:
                lo = mid + 1

        # Fallback: try with the maximum weight
        if best_path is None:
            best_path = hamiltonian_cycle_with_threshold(weights[-1])
            if best_path is None:
                return list(range(n)) + [0]

        return best_path