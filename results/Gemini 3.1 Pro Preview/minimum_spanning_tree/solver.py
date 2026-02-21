from typing import Any
from kruskal_cython import solve_kruskal

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[list[float]]]:
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        mst_edges = solve_kruskal(num_nodes, edges)
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        
        return {"mst_edges": mst_edges}