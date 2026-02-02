import sys
import collections

class Solver:
    def solve(self, problem, **kwargs):
        sys.setrecursionlimit(20000)
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]

        # Build Adjacency
        adj_g1 = [set() for _ in range(n)]
        for u, v in edges_g1:
            adj_g1[u].add(v)
            adj_g1[v].add(u)

        adj_g2 = [set() for _ in range(n)]
        for u, v in edges_g2:
            adj_g2[u].add(v)
            adj_g2[v].add(u)
            
        deg_g1 = [len(adj_g1[i]) for i in range(n)]
        deg_g2 = [len(adj_g2[i]) for i in range(n)]
        
        # Signatures: (degree, sorted_neighbor_degrees)
        def get_sigs(adj, deg):
            sigs = []
            for i in range(n):
                nb_degs = tuple(sorted([deg[nb] for nb in adj[i]]))
                sigs.append((deg[i], nb_degs))
            return sigs

        sig_g1 = get_sigs(adj_g1, deg_g1)
        sig_g2 = get_sigs(adj_g2, deg_g2)
        
        # Group G2 nodes by signature
        g2_by_sig = collections.defaultdict(list)
        for i in range(n):
            g2_by_sig[sig_g2[i]].append(i)
            
        # Determine domains for G1 nodes
        domains = []
        for i in range(n):
            s = sig_g1[i]
            cand = g2_by_sig.get(s)
            if cand is None:
                return {"mapping": [-1] * n}
            domains.append(cand)
            
        # Determine ordering: BFS starting from node with smallest domain
        start_node = min(range(n), key=lambda i: len(domains[i]))
        
        order = []
        visited = [False] * n
        queue = collections.deque([start_node])
        visited[start_node] = True
        
        while len(order) < n:
            if not queue:
                remaining = [i for i in range(n) if not visited[i]]
                if not remaining:
                    break
                rem_start = min(remaining, key=lambda i: len(domains[i]))
                queue.append(rem_start)
                visited[rem_start] = True
            
            u = queue.popleft()
            order.append(u)
            
            neighbors = sorted(list(adj_g1[u]), key=lambda x: len(domains[x]))
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        # Precompute backward neighbors
        pos_in_order = {node: i for i, node in enumerate(order)}
        backward_neighbors = []
        for k in range(n):
            u = order[k]
            bn = []
            for v in adj_g1[u]:
                if pos_in_order[v] < k:
                    bn.append(v)
            backward_neighbors.append(bn)

        # Bitmask optimization for small graphs
        use_bitmask = (n <= 64)
        if use_bitmask:
            adj_g2_masks = [0] * n
            for i in range(n):
                mask = 0
                for neighbor in adj_g2[i]:
                    mask |= (1 << neighbor)
                adj_g2_masks[i] = mask
        
        mapping = [-1] * n
        used_g2 = [False] * n

        # Bitmask optimization for small graphs
        use_bitmask = (n <= 64)
        if use_bitmask:
            adj_g2_masks = [0] * n
            for i in range(n):
                mask = 0
                for neighbor in adj_g2[i]:
                    mask |= (1 << neighbor)
                adj_g2_masks[i] = mask
            
            powers_of_2 = [1 << i for i in range(n)]
        
        mapping = [-1] * n
        used_g2 = [False] * n

        if use_bitmask:
            def solve_recursive_bitmask(k):
                if k == n:
                    return True
                
                u = order[k]
                bn = backward_neighbors[k]
                
                for v in domains[u]:
                    if used_g2[v]:
                        continue
                    
                    valid = True
                    mask_v = adj_g2_masks[v]
                    for nb in bn:
                        # Check if v is connected to mapped_nb
                        if not (mask_v & powers_of_2[mapping[nb]]):
                            valid = False
                            break
                    
                    if valid:
                        mapping[u] = v
                        used_g2[v] = True
                        if solve_recursive_bitmask(k + 1):
                            return True
                        used_g2[v] = False
                        mapping[u] = -1
                return False
            
            if solve_recursive_bitmask(0):
                return {"mapping": mapping}
            else:
                return {"mapping": [-1] * n}
                return {"mapping": [-1] * n}
        else:
            def solve_recursive(k):
                if k == n:
                    return True
                
                u = order[k]
                bn = backward_neighbors[k]
                
                for v in domains[u]:
                    if used_g2[v]:
                        continue
                    
                    valid = True
                    adj_v = adj_g2[v]
                    for nb in bn:
                        if mapping[nb] not in adj_v:
                            valid = False
                            break
                    
                    if valid:
                        mapping[u] = v
                        used_g2[v] = True
                        if solve_recursive(k + 1):
                            return True
                        used_g2[v] = False
                        mapping[u] = -1
                return False

            if solve_recursive(0):
                return {"mapping": mapping}
            else:
                return {"mapping": [-1] * n}