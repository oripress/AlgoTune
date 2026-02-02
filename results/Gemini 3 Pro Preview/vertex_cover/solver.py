import sys

# Increase recursion depth just in case
sys.setrecursionlimit(20000)

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        n = len(problem)
        if n == 0:
            return []
        
        # Build complement graph as bitmasks
        # adj[i] will contain the neighbors of i in the COMPLEMENT graph
        # We want to find the Maximum Clique in the complement graph (which is the Max Independent Set in original)
        # MVC = V - MIS
        
        adj = [0] * n
        for i in range(n):
            row = problem[i]
            mask = 0
            for j in range(n):
                if i != j and row[j] == 0:
                    mask |= (1 << j)
            adj[i] = mask
            
        self.max_clique_size = 0
        self.best_clique = 0
        
        # Precompute bit counts if needed, but int.bit_count() is fast in Python 3.10+
        if not hasattr(int, "bit_count"):
            def bit_count(n):
                return bin(n).count('1')
        else:
            def bit_count(n):
                return n.bit_count()

        # Order vertices by degree in the complement graph to improve pruning
        # Higher degree nodes first often helps find large cliques earlier
        degrees = [(bit_count(adj[i]), i) for i in range(n)]
        degrees.sort(key=lambda x: x[0], reverse=True)
        sorted_nodes = [x[1] for x in degrees]
        
        # Remapping
        # Map sorted_nodes[0] -> 0, sorted_nodes[1] -> 1, ...
        old_to_new = {node: i for i, node in enumerate(sorted_nodes)}
        new_to_old = sorted_nodes
        
        new_adj = [0] * n
        for i in range(n):
            original_node = new_to_old[i]
            original_mask = adj[original_node]
            new_mask = 0
            # Remap the mask
            for bit in range(n):
                if (original_mask >> bit) & 1:
                    new_mask |= (1 << old_to_new[bit])
            new_adj[i] = new_mask
            
        # Now we can process bits from 0 to n-1 (LSB to MSB) which corresponds to high degree to low degree
        
        def expand_remapped(candidates, current_sz, current_mask):
            if current_sz + bit_count(candidates) <= self.max_clique_size:
                return
            
            if candidates == 0:
                if current_sz > self.max_clique_size:
                    self.max_clique_size = current_sz
                    self.best_clique = current_mask
                return

            while candidates:
                # Get LSB (which is the highest degree node available due to remapping)
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                
                # Recurse
                # Add v to clique
                # New candidates are neighbors of v inside current candidates
                if current_sz + 1 + bit_count(candidates & new_adj[v]) > self.max_clique_size:
                    expand_remapped(candidates & new_adj[v], current_sz + 1, current_mask | (1 << v))
                
                # Remove v from candidates
                candidates ^= lsb
                
        expand_remapped((1 << n) - 1, 0, 0)
        
        # Reconstruct solution
        # self.best_clique is a mask in the NEW mapping
        # We need to map back to original indices
        
        mis_set = set()
        for i in range(n):
            if (self.best_clique >> i) & 1:
                mis_set.add(new_to_old[i])
                
        # VC = V - MIS
        vc = [i for i in range(n) if i not in mis_set]
        return sorted(vc)