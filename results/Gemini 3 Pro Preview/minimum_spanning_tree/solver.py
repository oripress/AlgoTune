import numpy as np
try:
    import fast_solver
except ImportError:
    fast_solver = None

class Solver:
    def solve(self, problem, **kwargs):
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        if fast_solver:
            if len(edges) == 0:
                return {"mst_edges": []}
            
            # Convert to numpy arrays
            # Assuming edges is a list of [u, v, w]
            # We can use np.array but it might be slow for large lists.
            # However, we need contiguous arrays for Cython.
            
            # Fast parsing if edges is a list of lists
            # We can transpose
            edges_np = np.array(edges, dtype=np.float64)
            
            u = edges_np[:, 0].astype(np.int32)
            v = edges_np[:, 1].astype(np.int32)
            w = edges_np[:, 2]
            
            mst_edges = fast_solver.solve_cython(num_nodes, u, v, w)
            
            # Sort output by (u, v)
            mst_edges.sort(key=lambda x: (x[0], x[1]))
            
            return {"mst_edges": mst_edges}
        
        # Fallback
        adj = [[] for _ in range(num_nodes)]
        for u, v, w in edges:
            u = int(u)
            v = int(v)
            adj[u].append((v, w))
            adj[v].append((u, w))
            
        ordered_edges = []
        for u in range(num_nodes):
            for v, w in adj[u]:
                if u < v:
                    ordered_edges.append((w, u, v))
        
        ordered_edges.sort(key=lambda x: x[0])
        
        parent = list(range(num_nodes))
        rank = [0] * num_nodes
        
        def find(i):
            path = []
            root = i
            while root != parent[root]:
                path.append(root)
                root = parent[root]
            for node in path:
                parent[node] = root
            return root
        
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if rank[root_i] < rank[root_j]:
                    parent[root_i] = root_j
                elif rank[root_i] > rank[root_j]:
                    parent[root_j] = root_i
                else:
                    parent[root_i] = root_j
                    rank[root_j] += 1
                return True
            return False
            
        mst_edges = []
        edges_count = 0
        for w, u, v in ordered_edges:
            if union(u, v):
                mst_edges.append([u, v, w])
                edges_count += 1
                if edges_count == num_nodes - 1:
                    break
                    
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        
        return {"mst_edges": mst_edges}
        edges_count = 0
        for w, u, v in ordered_edges:
            if union(u, v):
                mst_edges.append([u, v, w])
                edges_count += 1
                if edges_count == num_nodes - 1:
                    break
                    
        # Sort output by (u, v) as required
        mst_edges.sort(key=lambda x: (x[0], x[1]))
        
        return {"mst_edges": mst_edges}