from typing import Any
from solver_cython import count_components

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, int]:
        n = problem.get("num_nodes", 0)
        if n == 0:
            return {"number_connected_components": 0}
        
        edges = problem.get("edges", [])
        if not edges:
            return {"number_connected_components": n}
            
        components = count_components(n, edges)
        
        return {"number_connected_components": components}