from typing import Any, Dict, List
import networkx as nx

class Solver:
    def __init__(self, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 100):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: Dict[str, List[List[int]]], **kwargs: Any) -> Dict[str, List[float]]:
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        # Handle trivial cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Build directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                G.add_edge(u, v)

        # Compute PageRank using networkx
        try:
            pagerank_dict = nx.pagerank(G, alpha=self.alpha, tol=self.tol, max_iter=self.max_iter)
            # Convert to list ordered by node index
            pagerank_list = [0.0] * n
            for node, score in pagerank_dict.items():
                if 0 <= node < n:
                    pagerank_list[node] = float(score)
        except nx.PowerIterationFailedConvergence:
            pagerank_list = [0.0] * n
        except Exception:
            pagerank_list = [0.0] * n

        return {"pagerank_scores": pagerank_list}