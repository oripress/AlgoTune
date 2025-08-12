import sys

class Solver:
    def __init__(self):
        # allow deeper recursion for backtracking
        sys.setrecursionlimit(10000)

    def solve(self, problem, **kwargs):
        n = problem["num_nodes"]
        edges1 = problem["edges_g1"]
        edges2 = problem["edges_g2"]

        # Build adjacency lists
        adj1 = [[] for _ in range(n)]
        adj2 = [set() for _ in range(n)]
        for u, v in edges1:
            adj1[u].append(v)
            adj1[v].append(u)
        for u, v in edges2:
            adj2[u].add(v)
            adj2[v].add(u)

        # 1-WL color refinement
        col1 = [len(adj1[i]) for i in range(n)]
        col2 = [len(adj2[i]) for i in range(n)]
        for _ in range(5):
            sigs = {}
            nxt = 0
            new1 = [0] * n
            new2 = [0] * n
            for u in range(n):
                nb = sorted(col1[v] for v in adj1[u])
                sig = (col1[u], tuple(nb))
                if sig not in sigs:
                    sigs[sig] = nxt
                    nxt += 1
                new1[u] = sigs[sig]
            for u in range(n):
                nb = sorted(col2[v] for v in adj2[u])
                sig = (col2[u], tuple(nb))
                if sig not in sigs:
                    sigs[sig] = nxt
                    nxt += 1
                new2[u] = sigs[sig]
            if new1 == col1 and new2 == col2:
                break
            col1, col2 = new1, new2

        # Group by color classes
        classes = {}
        for u, c in enumerate(col1):
            classes.setdefault(c, [[], []])[0].append(u)
        for u, c in enumerate(col2):
            classes.setdefault(c, [[], []])[1].append(u)

        # Check for mismatched class sizes
        for c, (l1, l2) in classes.items():
            if len(l1) != len(l2):
                import networkx as nx
                G1 = nx.Graph()
                G2 = nx.Graph()
                G1.add_nodes_from(range(n))
                G2.add_nodes_from(range(n))
                for u, v in edges1:
                    G1.add_edge(u, v)
                for u, v in edges2:
                    G2.add_edge(u, v)
                gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
                iso = next(gm.isomorphisms_iter())
                return {"mapping": [iso[u] for u in range(n)]}

        max_class = max(len(l1) for l1, _ in classes.values())
        # Direct mapping if fully refined
        if max_class == 1:
            mapping = [-1] * n
            for l1, l2 in classes.values():
                mapping[l1[0]] = l2[0]
            return {"mapping": mapping}

        THRESHOLD = 7
        if max_class > THRESHOLD:
            import networkx as nx
            G1 = nx.Graph()
            G2 = nx.Graph()
            G1.add_nodes_from(range(n))
            G2.add_nodes_from(range(n))
            for u, v in edges1:
                G1.add_edge(u, v)
            for u, v in edges2:
                G2.add_edge(u, v)
            gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
            iso = next(gm.isomorphisms_iter())
            return {"mapping": [iso[u] for u in range(n)]}

        # Build color mask for G2
        color_mask = {}
        for v in range(n):
            c = col2[v]
            color_mask[c] = color_mask.get(c, 0) | (1 << v)
        # Candidate masks for G1 vertices
        candidates = [color_mask[col1[u]] for u in range(n)]
        # Adjacency bit masks for G2
        bits2 = [0] * n
        for v in range(n):
            for w in adj2[v]:
                bits2[v] |= (1 << w)
        mapping = [-1] * n

        # Depth-first search with forward checking
        def dfs(depth, used):
            if depth == n:
                return True
            # select unassigned vertex with smallest domain
            best_u = -1
            best_dom = 0
            best_size = n + 1
            mask_unused = (~used)
            for u in range(n):
                if mapping[u] == -1:
                    dom = candidates[u] & mask_unused
                    size = dom.bit_count()
                    if size == 0:
                        return False
                    if size < best_size:
                        best_size = size
                        best_dom = dom
                        best_u = u
                        if size == 1:
                            break
            u = best_u
            dom = best_dom
            # try assignments
            while dom:
                v_mask = dom & -dom
                dom -= v_mask
                v = v_mask.bit_length() - 1
                mapping[u] = v
                changed = []
                ok = True
                for w in adj1[u]:
                    if mapping[w] == -1:
                        old = candidates[w]
                        new = old & bits2[v]
                        if new != old:
                            candidates[w] = new
                            changed.append((w, old))
                            if (new & ~(used | v_mask)).bit_count() == 0:
                                ok = False
                                break
                if ok and dfs(depth + 1, used | v_mask):
                    return True
                for w, old in changed:
                    candidates[w] = old
                mapping[u] = -1
            return False

        # run search
        if not dfs(0, 0):
            import networkx as nx
            G1 = nx.Graph()
            G2 = nx.Graph()
            G1.add_nodes_from(range(n))
            G2.add_nodes_from(range(n))
            for u, v in edges1:
                G1.add_edge(u, v)
            for u, v in edges2:
                G2.add_edge(u, v)
            gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
            iso = next(gm.isomorphisms_iter())
            mapping = [iso[u] for u in range(n)]
        return {"mapping": mapping}