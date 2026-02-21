from typing import Any
from fast_solver import solve_cython

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        if type(prop_raw) is dict:
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = prop_raw
            n = len(proposer_prefs)

        if type(recv_raw) is dict:
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = recv_raw

        matching = solve_cython(proposer_prefs, receiver_prefs, n)
        return {"matching": matching}