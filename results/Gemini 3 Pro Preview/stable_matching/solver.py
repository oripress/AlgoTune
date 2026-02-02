from typing import Any
try:
    import solver_cython
except ImportError:
    solver_cython = None

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # Normalize inputs
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = list(prop_raw)
            n = len(proposer_prefs)

        if isinstance(recv_raw, dict):
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = list(recv_raw)
            
        if solver_cython:
            # Ensure elements are lists for Cython typed variables
            # If they are not lists (e.g. tuples), we might need to convert them.
            # However, checking every element is slow.
            # We'll assume they are lists as per spec. If not, Cython might raise TypeError.
            # To be safe against tuples, we can try to catch TypeError and fallback or convert.
            try:
                matching = solver_cython.solve_cython(proposer_prefs, receiver_prefs, n)
                return {"matching": matching}
            except TypeError:
                # Fallback or conversion logic could go here, but for now let's assume lists
                # If it failed because rows are not lists, we can convert them.
                proposer_prefs = [list(p) for p in proposer_prefs]
                receiver_prefs = [list(r) for r in receiver_prefs]
                matching = solver_cython.solve_cython(proposer_prefs, receiver_prefs, n)
                return {"matching": matching}

        # Fallback Python implementation (should not be reached if compilation succeeds)
        recv_rank = [[0] * n for _ in range(n)]
        for r in range(n):
            prefs = receiver_prefs[r]
            for rank, p in enumerate(prefs):
                recv_rank[r][p] = rank

        next_prop_idx = [0] * n
        recv_match = [-1] * n
        free_proposers = list(range(n))

        while free_proposers:
            p = free_proposers.pop()
            r = proposer_prefs[p][next_prop_idx[p]]
            next_prop_idx[p] += 1
            current_partner = recv_match[r]
            
            if current_partner == -1:
                recv_match[r] = p
            else:
                if recv_rank[r][p] < recv_rank[r][current_partner]:
                    recv_match[r] = p
                    free_proposers.append(current_partner)
                else:
                    free_proposers.append(p)
        
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}
        return {"matching": matching}