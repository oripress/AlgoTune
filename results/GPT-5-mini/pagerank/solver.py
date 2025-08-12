from typing import Any, Dict, List
import networkx as nx

class Solver:
    """
    PageRank solver that delegates to networkx.pagerank to match the reference behavior.
    Uses the same defaults as the reference: alpha=0.85, tol=1e-6, max_iter=100.
    """

    def __init__(self) -> None:
        self.alpha = 0.85
        self.tol = 1.0e-6
        self.max_iter = 100

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        adj_list = problem.get("adjacency_list")
        if adj_list is None:
            return {"pagerank_scores": []}

        n = len(adj_list)
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        alpha = float(kwargs.get("alpha", self.alpha))
        tol = float(kwargs.get("tol", self.tol))
        max_iter = int(kwargs.get("max_iter", self.max_iter))

        # Build the DiGraph exactly as the reference does
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj_list):
            # If neighbors is falsy (e.g., empty list), skip (consistent with reference)
            if not neighbors:
                continue
            for v in neighbors:
                # Let networkx handle node additions; match reference by not filtering targets here
                try:
                    G.add_edge(u, v)
                except Exception:
                    # Ignore any problematic entries
                    continue

        try:
            pagerank_dict = nx.pagerank(G, alpha=alpha, tol=tol, max_iter=max_iter)
            pagerank_list = [0.0] * n
            for node, score in pagerank_dict.items():
                try:
                    idx = int(node)
                except Exception:
                    continue
                if 0 <= idx < n:
                    pagerank_list[idx] = float(score)
        except nx.PowerIterationFailedConvergence:
            pagerank_list = [0.0] * n
        except Exception:
            pagerank_list = [0.0] * n

        return {"pagerank_scores": pagerank_list}