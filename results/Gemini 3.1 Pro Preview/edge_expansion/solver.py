from typing import Any
from solver_cy import compute_edge_expansion

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, float]:
        adj_list = problem["adjacency_list"]
        nodes_S_list = problem["nodes_S"]
        n = len(adj_list)
        
        if n == 0 or not nodes_S_list or len(nodes_S_list) == n:
            return {"edge_expansion": 0.0}
            
        return {"edge_expansion": compute_edge_expansion(adj_list, nodes_S_list, n)}