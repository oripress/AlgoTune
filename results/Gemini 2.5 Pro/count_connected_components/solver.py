from typing import Any
import numpy as np
# Numba is a powerful JIT compiler for Python, ideal for accelerating
# numerical algorithms like the one used here.
from numba import njit

# By separating the find operation, we make the code cleaner.
# Numba is smart enough to inline this function call, so there's no performance penalty.
# cache=True speeds up subsequent runs by caching the compiled code.
@njit(cache=True)
def _find_root(i: int, parent: np.ndarray) -> int:
    """
    Finds the root of the set containing element `i` with path compression.
    This is a helper function for the DSU algorithm, accelerated with Numba.
    """
    # Find the root of the tree
    root = i
    while parent[root] != root:
        root = parent[root]
    
    # Apply path compression by making all nodes on the path point to the root.
    while i != root:
        next_i = parent[i]
        parent[i] = root
        i = next_i
    return root

@njit(cache=True)
def _compute_components(num_nodes: int, edges: np.ndarray) -> int:
    """
    Computes the number of connected components using a Numba-accelerated
    Disjoint Set Union (DSU) algorithm with union-by-size and path compression.
    """
    if num_nodes == 0:
        return 0

    # Initialize DSU data structures using NumPy arrays for Numba compatibility.
    # Using np.int64 is a safe choice for large numbers of nodes.
    parent = np.arange(num_nodes, dtype=np.int64)
    size = np.ones(num_nodes, dtype=np.int64)
    num_components = num_nodes

    # Process each edge to union the sets of the two nodes.
    for i in range(edges.shape[0]):
        u, v = edges[i, 0], edges[i, 1]

        # Find the roots of the sets for u and v.
        root_u = _find_root(u, parent)
        root_v = _find_root(v, parent)

        # If they are not already in the same set, union them.
        if root_u != root_v:
            # Union by size: attach the smaller tree to the root of the larger tree.
            if size[root_u] < size[root_v]:
                root_u, root_v = root_v, root_u
            
            parent[root_v] = root_u
            size[root_u] += size[root_v]
            num_components -= 1
            
    return num_components

class Solver:
    """
    Solves the Count Connected Components problem using a Numba-accelerated
    Disjoint Set Union (DSU) data structure. This approach is significantly
    faster than pure Python implementations by leveraging Just-In-Time (JIT)
    compilation.
    """
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Computes the number of connected components in an undirected graph.

        The core logic is implemented in a Numba-jitted function for high performance.
        It uses a Disjoint Set Union (DSU) data structure with path compression
        and union by size, which is an asymptotically optimal algorithm for this problem.

        Args:
            problem: A dictionary containing:
                - "num_nodes": The total number of nodes in the graph.
                - "edges": A list of tuples (u, v) representing undirected edges.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            A dictionary with the key "number_connected_components" and an integer value
            representing the number of connected components.
        """
        num_nodes = problem.get("num_nodes", 0)
        edges = problem.get("edges", [])

        # Handle edge cases efficiently in Python before calling the Numba function.
        if num_nodes == 0:
            return {"number_connected_components": 0}
        
        if not edges:
            return {"number_connected_components": num_nodes}

        # Convert the list of edges to a NumPy array, which is required by Numba.
        # Using a specific dtype for performance and consistency.
        edges_np = np.array(edges, dtype=np.int64)

        # Call the high-performance JIT-compiled function.
        num_components = _compute_components(num_nodes, edges_np)

        return {"number_connected_components": num_components}