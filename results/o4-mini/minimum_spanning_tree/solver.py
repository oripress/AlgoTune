class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute MST using Kruskal's algorithm with union-find.
        """
        n = problem["num_nodes"]
        edges = problem["edges"]
        # Sort edges by weight ascending
        edges.sort(key=lambda x: x[2])
        parent = list(range(n))
        rank = [0] * n

        def find(u):
            # Path compression
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u, v):
            ru = find(u)
            rv = find(v)
            if ru == rv:
                return False
            # Union by rank
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            elif rank[rv] < rank[ru]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rank[ru] += 1
            return True

        mst = []
        # Kruskal main loop
        for u, v, w in edges:
            if union(u, v):
                # ensure u < v for consistency
                if u > v:
                    u, v = v, u
                mst.append([u, v, w])
                if len(mst) == n - 1:
                    break

        # final sort by (u, v)
        mst.sort(key=lambda x: (x[0], x[1]))
        return {"mst_edges": mst}