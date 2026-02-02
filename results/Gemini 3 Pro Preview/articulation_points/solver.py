from typing import Any
import sys
try:
    from graph_utils import find_articulation_points
except ImportError:
    # Fallback or handle the error appropriately if compilation failed
    pass

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[int]]:
        num_nodes = int(problem["num_nodes"])
        edges = problem["edges"]
        
        ap_list = find_articulation_points(num_nodes, edges)
                
        return {"articulation_points": ap_list}