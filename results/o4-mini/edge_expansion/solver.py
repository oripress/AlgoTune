from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, float]:
        """
        Calculate the directed edge expansion of subset S in G:
        expansion = |E(S, V-S)| / min(|S|, |V-S|)
        """

        # Extract graph data
        adj_list: List[List[int]] = problem.get("adjacency_list", [])
        nodes_S_list: List[int] = problem.get("nodes_S", [])

        # Basic edge cases
        n = len(adj_list)
        k = len(nodes_S_list)
        if n == 0 or k == 0 or k == n:
            return {"edge_expansion": 0.0}

        # Create a fast membership mask for S
        mask = bytearray(n)
        for u in nodes_S_list:
            if 0 <= u < n:
                mask[u] = 1

        # Count edges crossing the cut in directed graph (both directions)
        cut_edges = 0
        m = mask  # local alias
        for u, neighbors in enumerate(adj_list):
            if m[u]:
                # u in S, count edges to V-S
                for v in neighbors:
                    if not m[v]:
                        cut_edges += 1
            else:
                # u not in S, count edges to S
                for v in neighbors:
                    if m[v]:
                        cut_edges += 1

        # Compute expansion
        denom = k if k <= n - k else n - k
        return {"edge_expansion": cut_edges / denom}