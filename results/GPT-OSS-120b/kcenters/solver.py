import itertools
from typing import Any, List, Tuple, Dict, Set

from pysat.solvers import Solver as SATSolver  # pip install python-sat

class Distances:
    """Utility class for all‑pairs shortest‑path distances."""

    def __init__(self, graph: Dict[str, Dict[str, float]]) -> None:
        self._vertices = list(graph.keys())
        self._index = {v: i for i, v in enumerate(self._vertices)}
        n = len(self._vertices)
        # initialise distance matrix
        INF = float("inf")
        self._dist = [[INF] * n for _ in range(n)]
        for v in self._vertices:
            i = self._index[v]
            self._dist[i][i] = 0.0
            for w, d in graph[v].items():
                j = self._index[w]
                self._dist[i][j] = float(d)

        # Floyd‑Warshall
        for k in range(n):
            dk = self._dist[k]
            for i in range(n):
                di = self._dist[i]
                aik = di[k]
                if aik == INF:
                    continue
                for j in range(n):
                    b = aik + dk[j]
                    if b < di[j]:
                        di[j] = b

    def all_vertices(self) -> List[str]:
        return self._vertices

    def dist(self, u: str, v: str) -> float:
        return self._dist[self._index[u]][self._index[v]]

    def vertices_in_range(self, v: str, limit: float) -> List[str]:
        """Return vertices whose distance to `v` is ≤ `limit`."""
        i = self._index[v]
        return [
            self._vertices[j]
            for j, d in enumerate(self._dist[i])
            if d <= limit + 1e-12
        ]

    def max_dist(self, centers: List[str]) -> float:
        """Maximum distance from any vertex to its nearest center."""
        if not centers:
            return float("inf")
        max_d = 0.0
        for v in self._vertices:
            min_d = min(self.dist(v, c) for c in centers)
            if min_d > max_d:
                max_d = min_d
        return max_d

    def sorted_distances(self) -> List[float]:
        """Sorted list of all distinct finite distances."""
        s = set()
        for row in self._dist:
            for d in row:
                if d < float("inf"):
                    s.add(d)
        return sorted(s)

class Solver:
    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int], **kwargs) -> List[str]:
        """
        Exact k‑center solver.

        Parameters
        ----------
        problem : (graph_dict, k)
            graph_dict – adjacency dict with symmetric weights.
            k – maximum number of centers.

        Returns
        -------
        List[str]
            Nodes selected as centers (size ≤ k) achieving the minimal possible
            maximum distance to any node.
        """
        graph_dict, k = problem
        if not graph_dict:
            return []  # empty graph – no centers needed

        distances = Distances(graph_dict)
        vertices = distances.all_vertices()
        n = len(vertices)

        # All candidate radii (unique distances)
        cand = distances.sorted_distances()

        # For very small graphs, brute‑force enumeration of subsets is faster than SAT.
        # n is the number of vertices.
        n = len(vertices)
        if n <= 15 and k <= n:
            # Enumerate all subsets of size up to k and keep the best (minimal radius).
            best_radius = float("inf")
            best_solution = []
            # Pre‑compute distance matrix for quick lookup
            dist_matrix = distances._dist  # internal representation (list of lists)
            index = distances._index
            for r in range(1, k + 1):
                for combo in itertools.combinations(range(n), r):
                    # Compute the maximal distance from any vertex to the nearest center in this combo
                    max_d = 0.0
                    for vi in range(n):
                        # distance from vertex vi to nearest center in combo
                        min_d = min(dist_matrix[vi][ci] for ci in combo)
                        if min_d > max_d:
                            max_d = min_d
                            # early break if already worse than current best
                            if max_d >= best_radius:
                                break
                    if max_d < best_radius:
                        best_radius = max_d
                        best_solution = [vertices[i] for i in combo]
            return best_solution

        # binary search for minimal feasible radius using a fast back‑tracking feasibility test
        lo, hi = 0, len(cand) - 1
        best_solution: List[str] = []

        # ------------------------------------------------------------
        # Helper: feasibility test for a given radius.
        # Returns a list of centers (size ≤ k) if feasible, otherwise None.
        # ------------------------------------------------------------
        def _feasible(radius: float):
            # For each vertex, compute which centers can cover it within the radius.
            cover_by_center = {
                v: set(distances.vertices_in_range(v, radius)) for v in vertices
            }
            # For each vertex, which centers could cover it.
            centers_for_vertex = {
                v: {c for c, cov in cover_by_center.items() if v in cov}
                for v in vertices
            }

            # Quick impossibility check: any vertex with no possible center → infeasible.
            if any(not s for s in centers_for_vertex.values()):
                return None

            # Order vertices by fewest covering centers (most constrained first).
            order = sorted(vertices, key=lambda v: len(centers_for_vertex[v]))

            best_local: List[str] = []

            # Recursive back‑tracking.
            def backtrack(idx: int, chosen: Set[str], covered: Set[str]) -> bool:
                nonlocal best_local
                # All vertices covered → feasible solution found.
                if len(covered) == n:
                    best_local = list(chosen)
                    return True
                # Too many centers already.
                if len(chosen) >= k:
                    return False

                # Simple lower‑bound: remaining vertices need at least
                # ceil(remaining / max_cover) more centers.
                remaining = n - len(covered)
                max_cov = max(len(cov) for cov in cover_by_center.values())
                if len(chosen) + (remaining + max_cov - 1) // max_cov > k:
                    return False

                # Find next uncovered vertex (most constrained).
                while idx < len(order) and order[idx] in covered:
                    idx += 1
                if idx >= len(order):
                    return False
                v = order[idx]

                # Try each possible center that can cover v.
                for c in centers_for_vertex[v]:
                    if c in chosen:
                        # Already selected, just recurse.
                        if backtrack(idx + 1, chosen, covered):
                            return True
                        continue
                    new_chosen = chosen | {c}
                    new_covered = covered | cover_by_center[c]
                    if backtrack(idx + 1, new_chosen, new_covered):
                        return True
                return False

            if backtrack(0, set(), set()):
                return best_local
            return None

        # ------------------------------------------------------------
        # Perform binary search using the fast feasibility test.
        # ------------------------------------------------------------
        while lo <= hi:
            mid = (lo + hi) // 2
            radius = cand[mid]

            sol = _feasible(radius)
            if sol is not None:
                best_solution = sol
                hi = mid - 1          # try to shrink radius
            else:
                lo = mid + 1          # need larger radius

        # Fallback (should never happen for feasible instances)
        if not best_solution:
            best_solution = vertices[:k]

        return best_solution