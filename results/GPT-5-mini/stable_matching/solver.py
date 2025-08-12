from typing import Any

# Try to use the compiled Cython implementation for speed.
_have_cy = False
_cy_match = None
try:
    from cy_gs import match as _cy_match  # compiled Cython routine
    _have_cy = True
except Exception:
    _have_cy = False
    _cy_match = None

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Gale-Shapley stable matching (proposer-optimal).
        Prefer the Cython implementation (cy_gs.match) for speed; fall back to
        a fast pure-Python implementation if the extension is unavailable.
        Returns {"matching": matching} where matching[p] = r.
        """
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # If C extension available, delegate to it (very fast C loops).
        if _have_cy and _cy_match is not None:
            matching = _cy_match(prop_raw, recv_raw)
            return {"matching": matching}

        # Pure-Python fallback (optimized)
        # normalize proposer preferences to list-of-lists
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = list(prop_raw)
            n = len(proposer_prefs)

        # normalize receiver preferences to list-of-lists
        if isinstance(recv_raw, dict):
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = list(recv_raw)

        if n == 0:
            return {"matching": []}

        total = n * n

        # flatten proposer preferences: pf[p * n + i] = receiver index
        pf = [0] * total
        for p in range(n):
            base = p * n
            prefs = proposer_prefs[p]
            for i in range(n):
                pf[base + i] = prefs[i]

        # flatten receiver ranking: rr[r * n + p] = rank (lower is better)
        rr = [0] * total
        for r in range(n):
            base = r * n
            prefs = receiver_prefs[r]
            for rank in range(n):
                p = prefs[rank]
                rr[base + p] = rank

        # Gale-Shapley algorithm (proposers propose) using LIFO stack
        nxt = [0] * n             # next index in proposer's pref list to propose
        rmatch = [-1] * n         # rmatch[r] = proposer matched to receiver r, -1 if free
        stack = list(range(n))    # LIFO stack of free proposers

        # local aliases for speed
        pf_local = pf
        rr_local = rr
        nxt_local = nxt
        rmatch_local = rmatch
        n_local = n
        pop = stack.pop
        push = stack.append

        while stack:
            p = pop()
            i = nxt_local[p]
            idx = p * n_local + i
            r = pf_local[idx]
            nxt_local[p] = i + 1

            cur = rmatch_local[r]
            if cur == -1:
                rmatch_local[r] = p
            else:
                base_r = r * n_local
                # receiver prefers lower rank value
                if rr_local[base_r + p] < rr_local[base_r + cur]:
                    rmatch_local[r] = p
                    push(cur)
                else:
                    push(p)

        # build proposer->receiver matching
        matching = [-1] * n_local
        for r in range(n_local):
            p = rmatch_local[r]
            matching[p] = r

        return {"matching": matching}