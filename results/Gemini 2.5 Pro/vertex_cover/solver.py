class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the vertex cover problem using a branch-and-bound algorithm
        with reduction rules.
        """
        self.n = len(problem)
        if self.n == 0:
            return []
        
        self.adj_list = [[] for _ in range(self.n)]
        has_edges = False
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if problem[i][j] == 1:
                    has_edges = True
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)
        
        if not has_edges:
            return []

        # Get an initial upper bound using a greedy algorithm.
        self.best_cover = self.get_greedy_cover()
        
        # The set of vertices in the graph we are currently considering.
        V = set(range(self.n))
        
        # Start the recursive search.
        self.vc_recursive(set(), V)
        
        return sorted(list(self.best_cover))

    def get_greedy_cover(self):
        """
        Finds an approximate vertex cover using a greedy approach.
        Repeatedly picks the vertex with the highest degree.
        """
        cover = set()
        temp_degrees = [len(neighbors) for neighbors in self.adj_list]
        
        for _ in range(self.n):
            u = -1
            max_d = -1
            for i in range(self.n):
                if temp_degrees[i] > max_d:
                    max_d = temp_degrees[i]
                    u = i
            
            if max_d <= 0:
                break

            cover.add(u)
            
            for v in self.adj_list[u]:
                if temp_degrees[v] > 0:
                    temp_degrees[v] -= 1
            
            temp_degrees[u] = -1
            
        return cover

    def vc_recursive(self, cover, V):
        """
        Recursive branch-and-bound function with reduction rules.
        - `cover`: The set of vertices in the current partial vertex cover.
        - `V`: The set of vertices in the remaining subgraph to be covered.
        """
        # Pruning Step 1: If current cover is already too large, backtrack.
        if len(cover) >= len(self.best_cover):
            return

        # If no vertices are left in the subgraph, we have a potential solution.
        if not V:
            if len(cover) < len(self.best_cover):
                self.best_cover = cover.copy()
            return

        # --- Reduction and Branching Strategy ---
        
        # Find degrees in the current subgraph G[V] to decide the next step.
        subgraph_degrees = {v: 0 for v in V}
        has_edges = False
        for v_node in V:
            deg = sum(1 for neighbor in self.adj_list[v_node] if neighbor in V)
            if deg > 0:
                has_edges = True
            subgraph_degrees[v_node] = deg

        # Base Case: If no edges are left, V is an independent set.
        if not has_edges:
            if len(cover) < len(self.best_cover):
                self.best_cover = cover.copy()
            return
        
        # Reduction Rule: Prioritize degree-1 vertices (pendant nodes).
        # If a vertex `v` has degree 1, its neighbor `u` MUST be in the cover.
        # This is a forced move, not a branch.
        pendant_v = -1
        for v_node, d in subgraph_degrees.items():
            if d == 1:
                pendant_v = v_node
                break
        
        if pendant_v != -1:
            neighbor_u = next(n for n in self.adj_list[pendant_v] if n in V)
            # Recurse on the smaller graph after the forced move.
            self.vc_recursive(cover | {neighbor_u}, V - {pendant_v, neighbor_u})
            return

        # Branching Step: If no reductions apply, branch on the highest-degree vertex.
        u = max(subgraph_degrees, key=subgraph_degrees.get)

        # Branch 1: Include `u` in the vertex cover.
        self.vc_recursive(cover | {u}, V - {u})

        # Branch 2: Do NOT include `u`. This forces all its neighbors into the cover.
        neighbors_of_u_in_V = {v for v in self.adj_list[u] if v in V}
        new_cover = cover | neighbors_of_u_in_V
        
        # Pruning Step 2: Check the second branch before recursing.
        if len(new_cover) < len(self.best_cover):
             self.vc_recursive(new_cover, V - {u} - neighbors_of_u_in_V)