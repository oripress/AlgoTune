import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components

class Solver:
    def solve(self, problem, **kwargs):
        G_dict, k = problem
        nodes = list(G_dict.keys())
        n = len(nodes)
        
        if k == 0:
            return []
        
        if n <= k:
            return set(nodes)
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build adjacency matrix
        data = []
        row_ind = []
        col_ind = []
        
        for u, neighbors in G_dict.items():
            u_idx = node_to_idx[u]
            for v, w in neighbors.items():
                v_idx = node_to_idx[v]
                data.append(w)
                row_ind.append(u_idx)
                col_ind.append(v_idx)
        
        if n > 0:
            adj_mat = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
            dist_matrix = shortest_path(adj_mat, directed=False)
        else:
            return []

        # Check connectivity
        n_comps, _ = connected_components(adj_mat, directed=False)
        if n_comps > k:
            return set(nodes[:k])

        # Gonzalez Algorithm
        centers_idx = [0]
        min_dist_to_center = dist_matrix[0].copy()
        
        for _ in range(1, k):
            farthest_node = np.argmax(min_dist_to_center)
            if min_dist_to_center[farthest_node] == 0:
                break
            centers_idx.append(farthest_node)
            min_dist_to_center = np.minimum(min_dist_to_center, dist_matrix[farthest_node])
        
        upper_bound = np.max(min_dist_to_center)
        heuristic_centers = [nodes[i] for i in centers_idx]
        
        if upper_bound == 0:
            final_set = set(heuristic_centers)
            if len(final_set) < k:
                for node in nodes:
                    if node not in final_set:
                        final_set.add(node)
                        if len(final_set) == k: break
            return final_set

        # Candidates
        unique_dists = np.unique(dist_matrix)
        if np.isinf(upper_bound):
            candidates = unique_dists[~np.isinf(unique_dists)]
        else:
            candidates = unique_dists[(unique_dists >= upper_bound / 2) & (unique_dists < upper_bound)]
        
        if len(candidates) == 0:
            final_set = set(heuristic_centers)
            if len(final_set) < k:
                for node in nodes:
                    if node not in final_set:
                        final_set.add(node)
                        if len(final_set) == k: break
            return final_set

        # Binary Search with Bitmask Backtracking
        low = 0
        high = len(candidates) - 1
        best_centers = heuristic_centers
        
        # Precompute bitmasks for all distances? No, too much memory.
        # Compute on the fly or precompute for current limit.
        
        def get_dominating_set(limit):
            # 1. Compute degrees in G_R to reorder nodes
            degrees = np.sum(dist_matrix <= limit, axis=1)
            perm = np.argsort(degrees)
            
            masks = []
            neighbors = []
            
            dist_perm = dist_matrix[perm][:, perm]
            
            # Precompute masks and neighbors
            for i in range(n):
                cov_indices = np.where(dist_perm[i] <= limit)[0]
                m = 0
                for c in cov_indices:
                    m |= (1 << int(c))
                masks.append(m)
                neighbors.append(cov_indices)

            # Dominance Reduction
            # If masks[u] is a subset of masks[v], then v dominates u.
            # We can remove u from being a candidate center.
            # Note: u still needs to be covered!
            
            # Identify dominated nodes
            # O(N^2)
            dominated = np.zeros(n, dtype=bool)
            
            # Sort by popcount to optimize checks (check smaller against larger)
            try:
                popcounts = [m.bit_count() for m in masks]
            except AttributeError:
                popcounts = [bin(m).count('1') for m in masks]
            
            # Indices sorted by popcount descending
            sorted_indices = np.argsort(popcounts)[::-1]
            
            # We only need to keep 'non-dominated' nodes as candidates.
            # A node u is dominated if exists v s.t. masks[u] | masks[v] == masks[v] (u subset v)
            # and popcount(v) >= popcount(u).
            # If popcounts equal, we can drop one of them (e.g. smaller index).
            
            valid_centers = np.ones(n, dtype=bool)
            
            for i in range(n):
                u = sorted_indices[i]
                if not valid_centers[u]:
                    continue
                for j in range(i + 1, n):
                    v = sorted_indices[j]
                    if not valid_centers[v]:
                        continue
                    
                    # Check if v is subset of u (since u has >= popcount)
                    if (masks[v] & masks[u]) == masks[v]:
                        valid_centers[v] = False
            
            # Filter neighbors
            for i in range(n):
                # Only keep valid centers
                # And sort by popcount
                valid_neighbors = [v for v in neighbors[i] if valid_centers[v]]
                neighbors[i] = sorted(valid_neighbors, key=lambda x: popcounts[x], reverse=True)

            max_pop = max(popcounts) if popcounts else 0
            
            solution = []
            
            def solve_recursive(uncovered, k_left):
                if uncovered == 0:
                    return True
                if k_left == 0:
                    return False
                
                try:
                    uc_count = uncovered.bit_count()
                except AttributeError:
                    uc_count = bin(uncovered).count('1')
                    
                if uc_count > k_left * max_pop:
                    return False
                
                lsb = (uncovered & -uncovered)
                u = int(lsb).bit_length() - 1
                
                for v in neighbors[u]:
                    solution.append(v)
                    new_uncovered = uncovered & ~masks[v]
                    
                    if solve_recursive(new_uncovered, k_left - 1):
                        return True
                    
                    solution.pop()
                
                return False

            if solve_recursive((1 << n) - 1, k):
                return [nodes[perm[i]] for i in solution]
            return None

        while low <= high:
            mid = (low + high) // 2
            limit = candidates[mid]
            
            res = get_dominating_set(limit)
            if res is not None:
                best_centers = res
                high = mid - 1
            else:
                low = mid + 1
        
        final_set = set(best_centers)
        if len(final_set) < k:
            for node in nodes:
                if node not in final_set:
                    final_set.add(node)
                    if len(final_set) == k: break
        return final_set