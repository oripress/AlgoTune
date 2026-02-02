import numpy as np
try:
    from solver_cython import solve_cython
except ImportError:
    solve_cython = None

class Solver:
    def solve(self, problem, **kwargs):
        n = problem.get("num_nodes", 0)
        edges = problem.get("edges", [])
        
        if solve_cython:
            return {"number_connected_components": solve_cython(n, edges)}
            
        if n == 0:
            return {"number_connected_components": 0}
        if not edges:
            return {"number_connected_components": n}
        
        # Fallback if cython module not available (should not happen in this env if compiled correctly)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        
        rows, cols = zip(*edges)
        data = np.ones(len(rows), dtype=np.int32)
        adj = coo_matrix((data, (rows, cols)), shape=(n, n))
        n_components = connected_components(csgraph=adj, directed=False, return_labels=False)
        
        return {"number_connected_components": n_components}