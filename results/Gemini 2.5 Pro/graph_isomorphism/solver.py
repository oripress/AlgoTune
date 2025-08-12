import collections
from typing import Any, List, Dict, Tuple, Set

class Solver:
    """
    Solves the graph isomorphism problem using a pruned backtracking algorithm.
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Finds an isomorphism between two graphs, G1 and G2.

        The method uses a backtracking search, heavily pruned by node invariants.
        The main steps are:
        1. Represent graphs using adjacency lists for efficiency.
        2. Compute a 'signature' for each node based on its degree and the
           degrees of its neighbors. This serves as a powerful invariant.
        3. Group nodes in G2 by their signature to create candidate pools for
           each node in G1.
        4. Order G1 nodes for the search, prioritizing those with the smallest
           number of candidates to fail faster and prune the search tree more
           effectively.
        5. Perform a recursive backtracking search to find a consistent mapping.
           The first valid mapping found is returned, as the problem guarantees
           an isomorphism exists.
        """
        n: int = problem["num_nodes"]
        edges_g1: List[List[int]] = problem["edges_g1"]
        edges_g2: List[List[int]] = problem["edges_g2"]

        if n == 0:
            return {"mapping": []}

        adj1: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges_g1:
            adj1[u].append(v)
            adj1[v].append(u)

        adj2: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges_g2:
            adj2[u].append(v)
            adj2[v].append(u)
        
        adj2_sets: List[Set[int]] = [set(neighbors) for neighbors in adj2]

        deg1: List[int] = [len(adj1[i]) for i in range(n)]
        deg2: List[int] = [len(adj2[i]) for i in range(n)]

        sig1: List[Tuple[int, ...]] = [tuple(sorted(deg1[v] for v in adj1[u])) for u in range(n)]
        sig2: List[Tuple[int, ...]] = [tuple(sorted(deg2[v] for v in adj2[u])) for u in range(n)]

        candidates_map = collections.defaultdict(list)
        for i in range(n):
            # A node's full signature includes its own degree
            full_sig = (deg2[i], sig2[i])
            candidates_map[full_sig].append(i)

        g1_nodes_ordered: List[int] = sorted(
            range(n), 
            key=lambda u: len(candidates_map.get((deg1[u], sig1[u]), []))
        )

        mapping: List[int] = [-1] * n
        reverse_mapping: Dict[int, int] = {}

        def is_consistent(u: int, v: int) -> bool:
            for u_neighbor in adj1[u]:
                if mapping[u_neighbor] != -1:
                    if mapping[u_neighbor] not in adj2_sets[v]:
                        return False
            return True

        def backtrack(g1_node_idx: int) -> bool:
            if g1_node_idx == n:
                return True

            u = g1_nodes_ordered[g1_node_idx]
            u_sig = (deg1[u], sig1[u])
            
            for v in candidates_map.get(u_sig, []):
                if v in reverse_mapping:
                    continue

                if is_consistent(u, v):
                    mapping[u] = v
                    reverse_mapping[v] = u
                    
                    if backtrack(g1_node_idx + 1):
                        return True
                        
                    del reverse_mapping[v]
            
            mapping[u] = -1 # Should not be strictly necessary but good practice
            return False

        backtrack(0)
        return {"mapping": mapping}