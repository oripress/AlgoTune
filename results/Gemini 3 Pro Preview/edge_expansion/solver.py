from typing import Any
try:
    import solver_cython
except ImportError:
    solver_cython = None

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, float]:
        # if solver_cython:
        #     try:
        #         val = solver_cython.solve_cython(problem["adjacency_list"], problem["nodes_S"])
        #         return {"edge_expansion": val}
        #     except Exception:
        #         pass # Fallback to python if cython fails for some reason

        adj_list = problem["adjacency_list"]
        nodes_S = problem["nodes_S"]
        
        len_S = len(nodes_S)
        n = len(adj_list)
        
        if len_S == 0 or len_S == n:
            return {"edge_expansion": 0.0}
            
        S_set = set(nodes_S)
        
        cut_edges = 0
        for u in nodes_S:
            # Use set to handle potential unsorted duplicates
            for v in set(adj_list[u]):
                if v not in S_set:
                    cut_edges += 1
                    
        return {"edge_expansion": float(cut_edges) / len_S}