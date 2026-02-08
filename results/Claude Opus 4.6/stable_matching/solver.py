from typing import Any
import numpy as np
from gale_shapley_cy import gale_shapley

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]
        
        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = prop_raw if isinstance(prop_raw, list) else list(prop_raw)
            n = len(proposer_prefs)
        
        if isinstance(recv_raw, dict):
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = recv_raw if isinstance(recv_raw, list) else list(recv_raw)
        
        matching = gale_shapley(proposer_prefs, receiver_prefs, n)
        return {"matching": matching}