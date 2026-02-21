from pagerank import compute_pagerank

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}
            
        alpha = getattr(self, 'alpha', 0.85)
        max_iter = getattr(self, 'max_iter', 100)
        tol = getattr(self, 'tol', 1e-06)
        
        r = compute_pagerank(adj_list, n, alpha, max_iter, tol)
        return {"pagerank_scores": r}