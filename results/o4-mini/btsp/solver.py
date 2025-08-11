class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        # Handle trivial cases
        if n <= 1:
            return [0, 0] if n == 1 else []
        weights = problem
        # Extract sorted unique edge weights
        uniq = sorted({weights[i][j] for i in range(n) for j in range(i + 1, n)})
        # Feasibility check: is there a Hamiltonian cycle using only edges <= d?
        def feasible(d):
            # Build adjacency bitmasks for threshold d
            neighbor_mask = [0] * n
            for i in range(n):
                m = 0
                for j in range(n):
                    if i != j and weights[i][j] <= d:
                        m |= 1 << j
                neighbor_mask[i] = m
            full_mask = (1 << n) - 1
            # DP[mask]: bitset of end vertices reachable for subset mask, starting at 0
            DP = [0] * (1 << n)
            DP[1] = 1  # only city 0 in mask => end at 0
            for mask in range(1, full_mask + 1):
                if not (mask & 1):
                    continue
                if mask == 1:
                    continue
                m_end = 0
                sub = mask
                while sub:
                    v_bit = sub & -sub
                    sub -= v_bit
                    v = v_bit.bit_length() - 1
                    if v == 0:
                        continue
                    prev = mask ^ v_bit
                    # if there's a route finishing at u in prev and edge u->v exists
                    if DP[prev] & neighbor_mask[v]:
                        m_end |= v_bit
                DP[mask] = m_end
            # Check if any end vertex can return to start
            return (DP[full_mask] & neighbor_mask[0]) != 0

        # Binary search on sorted unique weights
        lo, hi = 0, len(uniq) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(uniq[mid]):
                hi = mid
            else:
                lo = mid + 1
        thr = uniq[lo]
        # Build adjacency bitmasks for the optimal threshold
        neighbor_mask = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if i != j and weights[i][j] <= thr:
                    m |= 1 << j
            neighbor_mask[i] = m
        full_mask = (1 << n) - 1
        # Build DP table for threshold thr
        DP = [0] * (1 << n)
        DP[1] = 1
        for mask in range(1, full_mask + 1):
            if not (mask & 1):
                continue
            if mask == 1:
                continue
            m_end = 0
            sub = mask
            while sub:
                v_bit = sub & -sub
                sub -= v_bit
                v = v_bit.bit_length() - 1
                if v == 0:
                    continue
                prev = mask ^ v_bit
                if DP[prev] & neighbor_mask[v]:
                    m_end |= v_bit
            DP[mask] = m_end
        # Identify an end vertex that can return to start
        last = DP[full_mask]
        valid_ends = last & neighbor_mask[0]
        e_bit = valid_ends & -valid_ends
        end = e_bit.bit_length() - 1
        # Reconstruct reverse path from end back to 0
        path_rev = [end]
        mask = full_mask
        curr = end
        while mask != 1:
            v_bit = 1 << curr
            prev_mask = mask ^ v_bit
            # pick a predecessor u that reaches curr
            cands = DP[prev_mask] & neighbor_mask[curr]
            u_bit = cands & -cands
            u = u_bit.bit_length() - 1
            path_rev.append(u)
            curr = u
            mask = prev_mask
        # Reverse and complete cycle
        path = list(reversed(path_rev))
        path.append(0)
        return path