import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[int]]:
        """
        Finds articulation points in an undirected graph using a non-recursive
        version of Tarjan's algorithm.
        """
        num_nodes = problem["num_nodes"]
        edges = problem.get("edges", [])

        if num_nodes == 0:
            return {"articulation_points": []}

        adj = [[] for _ in range(num_nodes)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        tin = np.full(num_nodes, -1, dtype=np.int32)
        low = np.full(num_nodes, -1, dtype=np.int32)
        
        articulation_points = set()
        timer = 0
        
        # visited array for the main loop
        visited_nodes = np.zeros(num_nodes, dtype=bool)

        for i in range(num_nodes):
            if not visited_nodes[i]:
                # Stack for non-recursive DFS: (node, parent, iterator over neighbours)
                # Using an iterator allows resuming the loop over neighbours
                stack = [(i, -1, iter(adj[i]))]
                visited_nodes[i] = True
                tin[i] = low[i] = timer
                timer += 1
                
                # For the root of the DFS tree
                root_children_count = 0

                while stack:
                    u, p, neighbours = stack[-1]
                    
                    try:
                        v = next(neighbours)
                        
                        if v == p:
                            continue

                        if tin[v] != -1: # v is visited (back edge)
                            low[u] = min(low[u], tin[v])
                        else: # v is not visited (tree edge)
                            # If u is the root of the DFS tree, we've found a new child
                            if u == i:
                                root_children_count += 1

                            visited_nodes[v] = True
                            tin[v] = low[v] = timer
                            timer += 1
                            stack.append((v, u, iter(adj[v])))

                    except StopIteration:
                        # Post-order for u: finished all its neighbours.
                        u, p, _ = stack.pop()
                        
                        # Update parent's low-link value.
                        # If p is -1, u is the root, so there's no parent to update.
                        if p != -1:
                            low[p] = min(low[p], low[u])
                            # AP Check for non-root nodes.
                            # If p is the root, it's handled by root_children_count.
                            # Otherwise, p is an AP if low[u] >= tin[p].
                            if p != i and low[u] >= tin[p]:
                                articulation_points.add(p)
                
                # AP Check for the root of the DFS tree.
                if root_children_count > 1:
                    articulation_points.add(i)

        return {"articulation_points": sorted(list(articulation_points))}