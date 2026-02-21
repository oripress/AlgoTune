from solver_cy import compute_comm_cy
from typing import Any

class Solver:
    def solve(self, problem: dict[str, list[list[int]]], **kwargs) -> dict[str, dict[int, dict[int, float]]]:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n == 0:
            return {"communicability": {}}
        
        result_comm_dict = compute_comm_cy(adj_list, n)
        
        return {"communicability": result_comm_dict}