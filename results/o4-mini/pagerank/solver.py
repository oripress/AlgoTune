import networkx as nx
from typing import Any, Dict, List

class Solver:
    def __init__(self, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 100) -> None:
        """
        Initialize the PageRank solver.

        alpha: damping factor
        tol: L1 convergence tolerance
        max_iter: maximum number of iterations
        """
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, problem: Dict[str, List[List[int]]], **kwargs: Any) -> Dict[str, List[float]]:
        """
        Compute PageRank scores for the given directed graph adjacency list.
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)
        # Edge cases
        if n == 0:
            return {"pagerank_scores": []}
        if n == 1:
            return {"pagerank_scores": [1.0]}

        # Build directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        # Add edges from adjacency list
        G.add_edges_from((i, j) for i, neigh in enumerate(adj_list) for j in neigh)

        # Compute PageRank using NetworkX's implementation
        pagerank_dict = nx.pagerank(G, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol)
        # Convert to list ordered by node index
        pagerank_scores = [float(pagerank_dict.get(i, 0.0)) for i in range(n)]
        return {"pagerank_scores": pagerank_scores}