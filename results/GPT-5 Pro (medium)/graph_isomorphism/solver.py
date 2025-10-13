from __future__ import annotations

from typing import Any, Dict, List, Tuple, Set, Optional
import sys

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Graph Isomorphism mapping using fast Weisfeiler-Lehman (1-WL) color refinement
        with individualization-refinement backtracking. Falls back to NetworkX VF2
        only if necessary.

        Input:
          problem = {
            "num_nodes": int,
            "edges_g1": List[List[int]],
            "edges_g2": List[List[int]],
          }

        Output:
          {"mapping": List[int]}
        """
        n = int(problem["num_nodes"])
        edges1 = problem.get("edges_g1", [])
        edges2 = problem.get("edges_g2", [])

        # Build adjacency as list of sets
        adj1 = self._build_adj(n, edges1)
        adj2 = self._build_adj(n, edges2)

        # Quick checks and trivial cases
        m1 = sum(len(s) for s in adj1) // 2
        m2 = sum(len(s) for s in adj2) // 2

        if m1 != m2:
            # Non-isomorphic (should not happen per problem statement)
            return {"mapping": [-1] * n}

        if n == 0:
            return {"mapping": []}

        if n == 1:
            return {"mapping": [0]}

        # If both graphs are empty or complete, any bijection works; choose identity for speed
        if m1 == 0:
            return {"mapping": list(range(n))}
        if m1 == n * (n - 1) // 2:
            return {"mapping": list(range(n))}

        # Degree multiset must match
        deg1 = [len(adj1[i]) for i in range(n)]
        deg2 = [len(adj2[i]) for i in range(n)]
        if sorted(deg1) != sorted(deg2):
            return {"mapping": [-1] * n}

        # WL refinement; if fully discrete, mapping is immediate
        c1, c2 = self._wl_refine_pair(adj1, adj2)

        # If any color class counts mismatch, not isomorphic (guard)
        if not self._color_hist_equal(c1, c2):
            return {"mapping": [-1] * n}

        # If discrete partition produced, extract mapping
        mapping = self._mapping_from_discrete_colors(c1, c2)
        if mapping is not None:
            # Verify safety (cheap and fast)
            if self._verify_mapping(adj1, adj2, mapping):
                return {"mapping": mapping}

        # Otherwise, use individualization-refinement backtracking with WL
        mapping_ir = self._ir_search(adj1, adj2, c1, c2)
        if mapping_ir is not None and self._verify_mapping(adj1, adj2, mapping_ir):
            return {"mapping": mapping_ir}

        # As a last resort, fallback to NetworkX VF2 (should be rare)
        try:
            import networkx as nx

            G1 = nx.Graph()
            G2 = nx.Graph()
            G1.add_nodes_from(range(n))
            G2.add_nodes_from(range(n))
            for u, v in edges1:
                if u != v:
                    G1.add_edge(u, v)
            for u, v in edges2:
                if u != v:
                    G2.add_edge(u, v)
            gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
            if gm.is_isomorphic():
                iso_map = next(gm.isomorphisms_iter())
                mapping_ref = [iso_map[u] for u in range(n)]
                return {"mapping": mapping_ref}
        except Exception:
            # If networkx not available or any error, give up gracefully
            pass

        # If all else fails (should not happen), produce some permutation to satisfy format
        return {"mapping": list(range(n))}

    @staticmethod
    def _build_adj(n: int, edges: List[List[int]]) -> List[Set[int]]:
        adj = [set() for _ in range(n)]
        for e in edges:
            if not isinstance(e, (list, tuple)) or len(e) != 2:
                continue
            u = int(e[0])
            v = int(e[1])
            if u == v:
                # Ignore self-loops (problem states undirected simple graphs)
                continue
            if 0 <= u < n and 0 <= v < n:
                adj[u].add(v)
                adj[v].add(u)
        return adj

    @staticmethod
    def _color_hist_equal(c1: List[int], c2: List[int]) -> bool:
        from collections import Counter

        return Counter(c1) == Counter(c2)

    @staticmethod
    def _group_by_color(colors: List[int]) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for i, c in enumerate(colors):
            groups.setdefault(c, []).append(i)
        return groups

    def _mapping_from_discrete_colors(
        self, c1: List[int], c2: List[int]
    ) -> Optional[List[int]]:
        # If each color id occurs exactly once in both graphs, mapping is unique by color equality
        g1 = self._group_by_color(c1)
        g2 = self._group_by_color(c2)
        mapping = [-1] * len(c1)
        for col, nodes1 in g1.items():
            nodes2 = g2.get(col)
            if nodes2 is None or len(nodes1) != len(nodes2):
                return None
            if len(nodes1) != 1:
                return None
            mapping[nodes1[0]] = nodes2[0]
        if any(x < 0 for x in mapping):
            return None
        return mapping

    def _wl_refine_pair(
        self,
        adj1: List[Set[int]],
        adj2: List[Set[int]],
        init_c1: Optional[List[int]] = None,
        init_c2: Optional[List[int]] = None,
        max_iters: int = 1_000_000,
    ) -> Tuple[List[int], List[int]]:
        """
        1-WL (color refinement) on both graphs with unified color IDs.

        The color assignment is synchronized across both graphs to allow
        direct comparison (same structural signatures get the same IDs).
        """
        n = len(adj1)
        if init_c1 is None or init_c2 is None:
            # Initialize by combined degree mapping to ensure consistent initial IDs
            deg_to_id: Dict[int, int] = {}
            next_id = 0
            c1 = [0] * n
            c2 = [0] * n
            for i in range(n):
                d = len(adj1[i])
                if d not in deg_to_id:
                    deg_to_id[d] = next_id
                    next_id += 1
                c1[i] = deg_to_id[d]
            for i in range(n):
                d = len(adj2[i])
                if d not in deg_to_id:
                    deg_to_id[d] = next_id
                    next_id += 1
                c2[i] = deg_to_id[d]
        else:
            c1 = list(init_c1)
            c2 = list(init_c2)

        # Iteratively refine colors using signatures of neighbor color multisets
        # We assign the same IDs for identical signatures across both graphs.
        iters = 0
        while True:
            iters += 1
            if iters > max_iters:
                break
            sig_to_id: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], int] = {}
            next_id = 0
            new_c1 = [0] * n
            new_c2 = [0] * n

            # Helper to create signature
            def node_signature(u: int, adj: List[Set[int]], colors: List[int]) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
                # Count neighbor colors
                counts: Dict[int, int] = {}
                cu = colors[u]
                # Local references for speed
                nbrs = adj[u]
                for v in nbrs:
                    cv = colors[v]
                    counts[cv] = counts.get(cv, 0) + 1
                # Stable, exact signature: own color + sorted (color,count) pairs
                if counts:
                    items = tuple(sorted(counts.items()))
                else:
                    items = tuple()
                return (cu, items)

            # Assign new IDs in a unified way across both graphs
            # Graph1
            for u in range(n):
                sig = node_signature(u, adj1, c1)
                cid = sig_to_id.get(sig)
                if cid is None:
                    cid = next_id
                    sig_to_id[sig] = cid
                    next_id += 1
                new_c1[u] = cid
            # Graph2
            for u in range(n):
                sig = node_signature(u, adj2, c2)
                cid = sig_to_id.get(sig)
                if cid is None:
                    cid = next_id
                    sig_to_id[sig] = cid
                    next_id += 1
                new_c2[u] = cid

            # Check stabilization
            if new_c1 == c1 and new_c2 == c2:
                break
            c1, c2 = new_c1, new_c2

        return c1, c2

    def _verify_mapping(
        self, adj1: List[Set[int]], adj2: List[Set[int]], mapping: List[int]
    ) -> bool:
        n = len(adj1)
        # Check bijection range and uniqueness
        if len(mapping) != n:
            return False
        if len(set(mapping)) != n:
            return False
        if not all(isinstance(x, int) and 0 <= x < n for x in mapping):
            return False

        # Degree preservation quick check
        for u in range(n):
            if len(adj1[u]) != len(adj2[mapping[u]]):
                return False

        # Edge preservation: For each edge in G1, the image must be an edge in G2
        # This is O(m)
        for u in range(n):
            mu = mapping[u]
            nbrs = adj1[u]
            mapn = adj2[mu]
            for v in nbrs:
                if mu == mapping[v]:
                    # Self-check unnecessary; undirected simple graph won't have self loops
                    pass
                if mapping[v] not in mapn:
                    return False
        return True

    def _ir_search(
        self,
        adj1: List[Set[int]],
        adj2: List[Set[int]],
        c1: List[int],
        c2: List[int],
        max_recursion: int = 10_000,
    ) -> Optional[List[int]]:
        """
        Individualization-Refinement (IR) backtracking using WL refinement.

        At each step:
          - If partition is discrete, extract mapping and verify.
          - Otherwise, pick a smallest non-singleton color class.
          - For each candidate v in G2 with same color as selected u in G1:
              - Individualize u and v with a fresh color.
              - Run WL refinement.
              - Recurse.

        This is typically very fast for most practical graphs.
        """

        sys.setrecursionlimit(max(1000, len(c1) * 5))

        n = len(c1)

        # Precompute degree arrays (used for occasional tie-breaks)
        deg1 = [len(adj1[i]) for i in range(n)]
        deg2 = [len(adj2[i]) for i in range(n)]

        # Tail recursion with stack
        seen_states: Set[Tuple[int, ...]] = set()  # cache to avoid repeating identical color partitions

        def is_discrete(colors: List[int]) -> bool:
            from collections import Counter

            cnts = Counter(colors)
            # all classes are singletons
            return all(v == 1 for v in cnts.values())

        def mapping_from_colors(colors1: List[int], colors2: List[int]) -> Optional[List[int]]:
            return self._mapping_from_discrete_colors(colors1, colors2)

        def choose_branch_color(colors: List[int]) -> int:
            # Choose a color class with minimal size > 1
            groups = self._group_by_color(colors)
            best_col = None
            best_size = 10**9
            # Heuristic: prefer smaller classes; break ties by larger degree spread
            for col, nodes in groups.items():
                sz = len(nodes)
                if sz > 1 and sz < best_size:
                    best_col = col
                    best_size = sz
            assert best_col is not None
            return best_col

        def recurse(col1: List[int], col2: List[int], depth: int) -> Optional[List[int]]:
            if depth > max_recursion:
                return None

            # Cache to avoid redoing same partitions
            key = tuple(col1) + (len(col1),) + tuple(col2)
            if key in seen_states:
                return None
            seen_states.add(key)

            # If discrete, extract mapping
            if is_discrete(col1) and is_discrete(col2):
                mp = mapping_from_colors(col1, col2)
                if mp is not None and self._verify_mapping(adj1, adj2, mp):
                    return mp
                # If verification fails (unlikely), continue search
                # fallthrough

            # Choose color class to branch on
            try:
                bcol = choose_branch_color(col1)
            except AssertionError:
                # No non-singleton color class but mapping not extracted -> fail
                return None

            # Select representative node u in G1 from that color
            g1_nodes = [i for i, c in enumerate(col1) if c == bcol]
            g2_nodes = [i for i, c in enumerate(col2) if c == bcol]
            # Heuristic: pick node with highest degree to prune faster
            u = max(g1_nodes, key=lambda x: (deg1[x], -x))

            # Try candidates v in G2 with same color; order by degree similarity
            # Here, degrees are identical within class usually, but ordering stable helps determinism
            g2_nodes_sorted = sorted(g2_nodes, key=lambda x: (-deg2[x], x))

            # Fresh color larger than any current color id
            max_color = max(max(col1), max(col2)) + 1

            for v in g2_nodes_sorted:
                # Individualize u and v with a fresh color; others unchanged
                new_c1 = list(col1)
                new_c2 = list(col2)
                new_c1[u] = max_color
                new_c2[v] = max_color

                # Refine colors
                rc1, rc2 = self._wl_refine_pair(adj1, adj2, new_c1, new_c2)

                # Early pruning: color histograms must match
                if not self._color_hist_equal(rc1, rc2):
                    continue

                # Recurse
                res = recurse(rc1, rc2, depth + 1)
                if res is not None:
                    return res

            return None

        return recurse(c1, c2, 0)