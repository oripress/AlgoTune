from __future__ import annotations
from typing import Any, List, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Compute a stable matching using the Gale‑Shapley algorithm.
        Supports preference lists given as lists or dicts.
        Returns a dict with key "matching" mapping each proposer index to its matched receiver.
        """
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        # Normalise proposer preferences
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = list(prop_raw)
            n = len(proposer_prefs)

        # Normalise receiver preferences
        if isinstance(recv_raw, dict):
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = list(recv_raw)

        # Build ranking table for receivers: rank[r][p] = position of proposer p in receiver r's list
        rank = [[0] * n for _ in range(n)]
        for r, prefs in enumerate(receiver_prefs):
            for pos, p in enumerate(prefs):
                rank[r][p] = pos

        # Gale‑Shapley algorithm
        next_prop = [0] * n               # next proposal index for each proposer
        recv_match = [-1] * n              # current matched proposer for each receiver
        free = list(range(n))              # stack of free proposers

        while free:
            p = free.pop()                 # take a free proposer
            r = proposer_prefs[p][next_prop[p]]
            next_prop[p] += 1

            current = recv_match[r]
            if current == -1:
                # receiver is free
                recv_match[r] = p
            else:
                # receiver chooses the better proposer
                if rank[r][p] < rank[r][current]:
                    recv_match[r] = p
                    free.append(current)   # displaced proposer becomes free
                else:
                    free.append(p)         # proposer stays free

        # Convert receiver‑centric matches to proposer‑centric list
        matching = [0] * n
        for r, p in enumerate(recv_match):
            matching[p] = r

        return {"matching": matching}