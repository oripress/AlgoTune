from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Compute a graph isomorphism mapping from G1 to G2.

        Strategy:
        - Build adjacency sets for O(1) edge queries.
        - Run 1-WL (Weisfeiler-Lehman) color refinement simultaneously on both graphs,
          ensuring consistent color IDs across graphs.
        - If refinement yields a discrete partition (all singleton color classes),
          return the mapping directly.
        - Otherwise, perform backtracking with individualization-refinement:
          pick a smallest non-singleton color class, try mapping one G1 node to each
          G2 node in that class, refine, and recurse. Use adjacency compatibility
          checks against already individualized pairs to prune.
        """
        n = int(problem["num_nodes"])
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]

        if n == 0:
            return {"mapping": []}
        if n == 1:
            return {"mapping": [0]}

        adj1 = self._build_adj(n, edges_g1)
        adj2 = self._build_adj(n, edges_g2)

        # Quick degree sanity check (should always pass for isomorphic graphs)
        deg1 = [len(adj1[i]) for i in range(n)]
        deg2 = [len(adj2[i]) for i in range(n)]
        if sorted(deg1) != sorted(deg2):
            # Not isomorphic, but per problem statement this shouldn't happen.
            return {"mapping": [-1] * n}

        # Initial colors: degree
        colors1 = deg1[:]
        colors2 = deg2[:]

        # Refinement
        refined = self._wl_refine_both(adj1, adj2, colors1, colors2)
        if refined is None:
            return {"mapping": [-1] * n}
        colors1, colors2 = refined

        # Try to solve directly or backtrack
        mapping = self._search_mapping(adj1, adj2, colors1, colors2)
        if mapping is None:
            # Fallback (should not happen): return a default invalid mapping
            return {"mapping": [-1] * n}
        return {"mapping": mapping}

    # -------------- Helpers --------------

    @staticmethod
    def _build_adj(n: int, edges: List[List[int]]) -> List[set]:
        adj: List[set] = [set() for _ in range(n)]
        for u, v in edges:
            if u == v:
                # Ignore self-loops for simple graph assumption
                continue
            # Ensure within range, assume input correctness per problem statement
            if 0 <= u < n and 0 <= v < n:
                # set.add is idempotent; no need to check membership
                adj[u].add(v)
                adj[v].add(u)
        return adj

    @staticmethod
    def _color_signature(
        node: int, colors: List[int], adj: List[set], tmp_count: Dict[int, int]
    ) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
        """
        Build the WL signature: (current_color, sorted multiset of neighbor colors as (color, count) pairs).
        tmp_count is a dict used as scratch space to avoid repeated allocations.

        Hybrid approach:
        - For small degrees (<= 32), use dict counting (reusing tmp_count) and sort its items.
        - For larger degrees, sort neighbor colors list and run-length encode for fewer dict ops.
        """
        deg = len(adj[node])
        if deg <= 32:
            tmp = tmp_count
            tmp.clear()
            for nb in adj[node]:
                c = colors[nb]
                tmp[c] = tmp.get(c, 0) + 1
            neigh_sig = tuple(sorted(tmp.items()))
        else:
            nb_cols = [colors[nb] for nb in adj[node]]
            nb_cols.sort()
            pairs: List[Tuple[int, int]] = []
            if nb_cols:
                cur = nb_cols[0]
                cnt = 1
                for c in nb_cols[1:]:
                    if c == cur:
                        cnt += 1
                    else:
                        pairs.append((cur, cnt))
                        cur, cnt = c, 1
                pairs.append((cur, cnt))
            neigh_sig = tuple(pairs)
        return colors[node], neigh_sig

    def _wl_refine_both(
        self,
        adj1: List[set],
        adj2: List[set],
        colors1: List[int],
        colors2: List[int],
        max_iters: Optional[int] = None,
    ) -> Optional[Tuple[List[int], List[int]]]:
        """
        Run WL refinement simultaneously on both graphs using a shared signature->color mapping.
        Returns refined colors if consistent (color class sizes match between graphs), else None.
        """
        n = len(colors1)
        # Normalize initial colors to [0..k-1] consistently across both graphs
        # using shared relabeling of values in colors1+colors2
        colors1, colors2 = self._normalize_colors(colors1, colors2)

        tmp_count: Dict[int, int] = {}
        iters = 0
        while True:
            iters += 1
            if max_iters is not None and iters > max_iters:
                break
            sign_to_id: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], int] = {}
            next_id = 0

            new1 = [0] * n
            new2 = [0] * n

            # Process graph 1 and accumulate color class counts on the fly
            count1: List[int] = []
            for u in range(n):
                sig = self._color_signature(u, colors1, adj1, tmp_count)
                cid = sign_to_id.get(sig)
                if cid is None:
                    cid = next_id
                    sign_to_id[sig] = cid
                    next_id += 1
                    count1.append(0)  # extend counts for new id
                new1[u] = cid
                count1[cid] += 1

            # Process graph 2 and check signature presence; accumulate counts
            count2: List[int] = [0] * next_id
            for v in range(n):
                sig = self._color_signature(v, colors2, adj2, tmp_count)
                cid = sign_to_id.get(sig)
                if cid is None:
                    # Signature not seen in graph1 -> inconsistent
                    return None
                new2[v] = cid
                count2[cid] += 1

            # Ensure that each color id appears same number of times in both graphs
            if count1 != count2:
                return None

            if new1 == colors1 and new2 == colors2:
                # Stable
                return new1, new2

            colors1, colors2 = new1, new2

    @staticmethod
    def _counts(colors: List[int]) -> Dict[int, int]:
        c: Dict[int, int] = {}
        for x in colors:
            c[x] = c.get(x, 0) + 1
        return c

    @staticmethod
    def _normalize_colors(c1: List[int], c2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Relabel colors so that equal values across c1 and c2 correspond to same canonical ids.
        """
        mapping: Dict[int, int] = {}
        next_id = 0
        out1 = [0] * len(c1)
        for i, x in enumerate(c1):
            if x not in mapping:
                mapping[x] = next_id
                next_id += 1
            out1[i] = mapping[x]
        # Ensure colors in c2 that are present in c1 keep same ids; any new colors get new ids
        out2 = [0] * len(c2)
        for i, x in enumerate(c2):
            if x not in mapping:
                mapping[x] = next_id
                next_id += 1
            out2[i] = mapping[x]
        return out1, out2

    def _classes(
        self, colors1: List[int], colors2: List[int]
    ) -> Optional[Dict[int, Tuple[List[int], List[int]]]]:
        """
        Compute color classes and ensure sizes match between graphs.
        Returns dict color -> (nodes_in_G1, nodes_in_G2) or None if mismatch.
        """
        n = len(colors1)
        groups1: Dict[int, List[int]] = {}
        groups2: Dict[int, List[int]] = {}
        for u in range(n):
            groups1.setdefault(colors1[u], []).append(u)
            groups2.setdefault(colors2[u], []).append(u)
        if set(groups1.keys()) != set(groups2.keys()):
            return None
        classes: Dict[int, Tuple[List[int], List[int]]] = {}
        for col in groups1:
            a = groups1[col]
            b = groups2[col]
            if len(a) != len(b):
                return None
            classes[col] = (a, b)
        return classes

    def _final_mapping_from_colors(
        self, colors1: List[int], colors2: List[int]
    ) -> Optional[List[int]]:
        """
        If each color class has size 1 in both graphs, produce the mapping accordingly.
        """
        classes = self._classes(colors1, colors2)
        if classes is None:
            return None
        mapping = [-1] * len(colors1)
        for col, (a, b) in classes.items():
            if len(a) != 1:
                return None
            u = a[0]
            v = b[0]
            mapping[u] = v
        return mapping

    def _search_mapping(
        self,
        adj1: List[set],
        adj2: List[set],
        colors1: List[int],
        colors2: List[int],
    ) -> Optional[List[int]]:
        """
        Backtracking with individualization-refinement. Returns mapping list or None.
        """
        # Quick attempt: if discrete, return directly
        direct = self._final_mapping_from_colors(colors1, colors2)
        if direct is not None:
            return direct

        # Build classes and choose a split cell
        classes = self._classes(colors1, colors2)
        if classes is None:
            return None

        # Select a color class with minimum size > 1
        candidate_col = None
        min_size = 10**9
        for col, (nodes1, nodes2) in classes.items():
            size = len(nodes1)
            if size > 1 and size < min_size:
                min_size = size
                candidate_col = col
        if candidate_col is None:
            # Should not happen as direct would have caught all-singleton case
            return None

        nodes1_class, nodes2_class = classes[candidate_col]
        # Choose pivot u in G1 from chosen color class.
        # Heuristic: pick the node with the most distinct neighbor colors to refine quickly.
        u = self._select_pivot(nodes1_class, colors1, adj1)

        # Build fixed pairs using already computed classes (avoid re-scanning arrays)
        fixed_pairs: Dict[int, int] = {}
        for col, (a, b) in classes.items():
            if len(a) == 1:
                fixed_pairs[a[0]] = b[0]

        # Try mapping u to each candidate v in the same class in G2
        # Order candidates using a heuristic: similar degree and neighbor color profile to u
        ordered_candidates = self._order_candidates(u, nodes2_class, adj1, adj2, colors1, colors2)
        for v in ordered_candidates:
            # Adjacency compatibility check with already fixed pairs
            if not self._adj_compatible(u, v, adj1, adj2, fixed_pairs):
                continue

            # Individualize: assign a new unique color to u and v, then refine
            # Use len(classes) as next unique id (colors are normalized to 0..k-1 after refinement)
            new_color_id = len(classes)
            new_c1 = colors1[:]  # copy
            new_c2 = colors2[:]
            new_c1[u] = new_color_id
            new_c2[v] = new_color_id

            refined = self._wl_refine_both(adj1, adj2, new_c1, new_c2)
            if refined is None:
                continue
            rc1, rc2 = refined

            # Recurse
            res = self._search_mapping(adj1, adj2, rc1, rc2)
            if res is not None:
                return res

        # No candidate worked
        return None

    @staticmethod
    def _select_pivot(nodes: List[int], colors: List[int], adj: List[set]) -> int:
        # Pick node with highest number of distinct neighbor colors; tie-break by degree, then id
        best_u = nodes[0]
        best_key = (-1, -1, -1)
        for u in nodes:
            neigh_colors = set(colors[v] for v in adj[u])
            key = (len(neigh_colors), len(adj[u]), -u)
            if key > best_key:
                best_key = key
                best_u = u
        return best_u

    @staticmethod
    def _fixed_pairs(colors1: List[int], colors2: List[int]) -> Dict[int, int]:
        """
        Return mapping for colors that are singleton classes already.
        """
        classes: Dict[int, List[int]] = {}
        for i, c in enumerate(colors1):
            classes.setdefault(c, []).append(i)
        classes2: Dict[int, List[int]] = {}
        for i, c in enumerate(colors2):
            classes2.setdefault(c, []).append(i)
        mapping: Dict[int, int] = {}
        for col, nodes1 in classes.items():
            nodes2 = classes2.get(col)
            if nodes2 is None:
                continue
            if len(nodes1) == 1 and len(nodes2) == 1:
                mapping[nodes1[0]] = nodes2[0]
        return mapping

    @staticmethod
    def _adj_compatible(
        u: int, v: int, adj1: List[set], adj2: List[set], fixed_pairs: Dict[int, int]
    ) -> bool:
        """
        Check adjacency consistency of mapping u->v with already fixed pairs.
        For every fixed a->b: edge(u,a) <=> edge(v,b).
        """
        Nu = adj1[u]
        Nv = adj2[v]
        for a, b in fixed_pairs.items():
            if (a in Nu) != (b in Nv):
                return False
        return True

    @staticmethod
    def _neighbor_color_profile(node: int, adj: List[set], colors: List[int]) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
        """
        Return (degree, sorted (color,count) multiset) for ordering candidates.
        """
        deg = len(adj[node])
        counts: Dict[int, int] = {}
        for nb in adj[node]:
            c = colors[nb]
            counts[c] = counts.get(c, 0) + 1
        prof = tuple(sorted(counts.items()))
        return deg, prof

    def _order_candidates(
        self,
        u: int,
        candidates: List[int],
        adj1: List[set],
        adj2: List[set],
        colors1: List[int],
        colors2: List[int],
    ) -> List[int]:
        """
        Order candidate v for mapping to u, preferring those with matching degree and similar neighbor color profile.
        """
        deg_u, prof_u = self._neighbor_color_profile(u, adj1, colors1)
        # Score: exact degree match prioritized, then similarity of profile length, etc.
        scored: List[Tuple[int, int, int, int]] = []
        for v in candidates:
            deg_v, prof_v = self._neighbor_color_profile(v, adj2, colors2)
            # Primary: degree equality (0 if equal, 1 otherwise -> smaller is better)
            deg_mismatch = 0 if deg_u == deg_v else 1
            # Secondary: profile length mismatch
            len_mismatch = abs(len(prof_u) - len(prof_v))
            # Tertiary: heuristic lexicographic difference between profiles
            prof_mismatch = 0 if prof_u == prof_v else 1
            scored.append((deg_mismatch, len_mismatch, prof_mismatch, v))
        scored.sort()
        return [v for _, _, _, v in scored]