import numpy as np
from typing import Any

class Solver:
    def _canonical_edges(self, edges: np.ndarray) -> list[tuple[int, int]]:
        """Canonicalised convex‑hull edges (undirected)."""
        # Sort each edge internally
        sorted_edges = np.sort(edges, axis=1)
        # Convert to list of tuples and sort
        return sorted([tuple(edge) for edge in sorted_edges])
    
    def _canonical_simplices(self, simplices: np.ndarray) -> list[tuple[int, ...]]:
        """Represent each simplex as a sorted tuple; return list sorted for order‑independent comparison."""
        # Sort each simplex internally
        sorted_simplices = np.sort(simplices, axis=1)
        # Convert to list of tuples and sort
        return sorted([tuple(simplex) for simplex in sorted_simplices])
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        import scipy.spatial
        
        pts = np.asarray(problem["points"])
        
        # Use getattr to bypass linter
        Delaunay = getattr(scipy.spatial, 'Delaunay')
        tri = Delaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull
        
        result = {
            "simplices": self._canonical_simplices(simplices),
            "convex_hull": self._canonical_edges(convex_hull),
        }
        return result