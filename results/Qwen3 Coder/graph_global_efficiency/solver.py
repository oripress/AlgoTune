class Solver:
    def solve(self, problem):
        """
        Calculates the global efficiency of the graph using optimized algorithms.
        """
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        
        adj_list = problem["adjacency_list"]
        n = len(adj_list)
        
        # Handle edge case: efficiency is 0 for graphs with 0 or 1 node.
        if n <= 1:
            return {"global_efficiency": 0.0}
        
        # For very small graphs, use a direct approach
        if n <= 3:
            return self._compute_efficiency_small(n, adj_list)
        
        # Count edges
        edge_count = sum(len(neighbors) for neighbors in adj_list) // 2
        if edge_count == 0:
            return {"global_efficiency": 0.0}
        
        # Create edge lists more efficiently
        edges = []
        for i, neighbors in enumerate(adj_list):
            for j in neighbors:
                if i < j:  # Only add each edge once
                    edges.append((i, j))
        
        if not edges:
            return {"global_efficiency": 0.0}
        
        # Create sparse matrix more efficiently
        row_ind, col_ind = zip(*edges)
        row_ind = np.array(row_ind, dtype=np.int32)
        col_ind = np.array(col_ind, dtype=np.int32)
        
        # Create full edge lists
        row_full = np.concatenate([row_ind, col_ind])
        col_full = np.concatenate([col_ind, row_ind])
        
        # Create sparse matrix
        data = np.ones(len(row_full), dtype=np.int8)
        adj_matrix = csr_matrix((data, (row_full, col_full)), shape=(n, n))
        
        # Compute all-pairs shortest paths
        distances = shortest_path(adj_matrix, directed=False, unweighted=True, method='D')
        
        # Calculate global efficiency using optimized operations
        np.fill_diagonal(distances, 0)
        
        # Use reciprocal for better performance
        valid_distances = distances[distances > 0]
        if len(valid_distances) == 0:
            return {"global_efficiency": 0.0}
            
        total_efficiency = np.sum(np.reciprocal(valid_distances, dtype=np.float64))
        global_efficiency = total_efficiency / (n * (n - 1))
        
        return {"global_efficiency": float(global_efficiency)}
    
    def _compute_efficiency_small(self, n, adj_list):
        """Compute efficiency for small graphs using a direct approach."""
        import numpy as np
        from collections import deque
        
        total_efficiency = 0.0
        
        # For small graphs, use BFS from each node
        for s in range(n):
            # BFS from node s
            distances = [-1] * n
            distances[s] = 0
            queue = deque([s])
            
            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if distances[v] == -1:  # Not visited
                        distances[v] = distances[u] + 1
                        queue.append(v)
            
            # Add contribution of all nodes reachable from s
            for t in range(n):
                if t != s and distances[t] > 0:  # Skip unreachable nodes and self
                    total_efficiency += 1.0 / distances[t]
        
        # Normalize by the number of node pairs
        global_efficiency = total_efficiency / (n * (n - 1))
        return {"global_efficiency": global_efficiency}