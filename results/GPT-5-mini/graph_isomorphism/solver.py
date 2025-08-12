import sys
from typing import Any, Dict, List, Set, Tuple

# Increase recursion limit for backtracking on larger graphs
sys.setrecursionlimit(10000)

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[int]]:
        """
        Solve graph isomorphism by:
         1. Building adjacency sets for both graphs.
         2. Running Weisfeiler-Lehman color refinement jointly on both graphs
            to obtain a partition of nodes into color classes.
         3. If refinement yields unique colors, directly build mapping.
         4. Otherwise, build candidate sets per node (nodes in G2 with same color)
            and run a backtracking search with forward-checking to find a full
            bijection that preserves adjacency.

        Returns:
            {"mapping": mapping_list} where mapping_list[u] = v maps node u in G1 to v in G2.
        """
        n = int(problem.get("num_nodes", 0))
        if n == 0:
            return {"mapping": []}

        edges_g1 = problem.get("edges_g1", [])
        edges_g2 = problem.get("edges_g2", [])

        # Build adjacency sets
        adj1: List[Set[int]] = [set() for _ in range(n)]
        adj2: List[Set[int]] = [set() for _ in range(n)]

        for a, b in edges_g1:
            if a == b:
                continue
            adj1[a].add(b)
            adj1[b].add(a)
        for a, b in edges_g2:
            if a == b:
                continue
            adj2[a].add(b)
            adj2[b].add(a)

        # Quick check: number of edges must match (guaranteed by problem statement)
        # but we can use it to early-detect issues.
        # Compute degrees
        deg1 = [len(adj1[i]) for i in range(n)]
        deg2 = [len(adj2[i]) for i in range(n)]

        # Initial color: compress degrees into small integers (shared across both graphs)
        uniq_degs = sorted(set(deg1) | set(deg2))
        deg_to_color = {d: i for i, d in enumerate(uniq_degs)}
        colors1 = [deg_to_color[d] for d in deg1]
        colors2 = [deg_to_color[d] for d in deg2]

        # Weisfeiler-Lehman like refinement: refine colors jointly for both graphs
        # until stable.
        while True:
            signatures: List[Tuple[int, Tuple[Tuple[int, int], ...]]] = []
            # G1 signatures
            for i in range(n):
                cnt = {}
                for nb in adj1[i]:
                    c = colors1[nb]
                    cnt[c] = cnt.get(c, 0) + 1
                # signature: (current_color, sorted list of (neighbor_color, count))
                signatures.append((colors1[i], tuple(sorted(cnt.items()))))
            # G2 signatures
            for i in range(n):
                cnt = {}
                for nb in adj2[i]:
                    c = colors2[nb]
                    cnt[c] = cnt.get(c, 0) + 1
                signatures.append((colors2[i], tuple(sorted(cnt.items()))))

            # Map signatures to compact color ids (jointly)
            sig_to_id: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], int] = {}
            new_combined: List[int] = [0] * (2 * n)
            nxt = 0
            for idx, sig in enumerate(signatures):
                if sig in sig_to_id:
                    new_combined[idx] = sig_to_id[sig]
                else:
                    sig_to_id[sig] = nxt
                    new_combined[idx] = nxt
                    nxt += 1

            new_colors1 = new_combined[:n]
            new_colors2 = new_combined[n:]
            # Stop if stable
            if new_colors1 == colors1 and new_colors2 == colors2:
                colors1 = new_colors1
                colors2 = new_colors2
                break
            colors1 = new_colors1
            colors2 = new_colors2

        # Build color classes
        color_to_nodes1: Dict[int, List[int]] = {}
        color_to_nodes2: Dict[int, List[int]] = {}
        for i, c in enumerate(colors1):
            color_to_nodes1.setdefault(c, []).append(i)
        for i, c in enumerate(colors2):
            color_to_nodes2.setdefault(c, []).append(i)

        # If any color class sizes differ, something's inconsistent (shouldn't happen).
        for c, lst in color_to_nodes1.items():
            if len(lst) != len(color_to_nodes2.get(c, [])):
                # Fallback: attempt trivial mapping by sorted degree and index, but this is unexpected.
                # We'll construct mapping by stable pairing: sort nodes in each class by neighbor-color vector.
                mapping = self._fallback_map(n, adj1, adj2, colors1, colors2)
                return {"mapping": mapping}

        # If all color classes are singletons, mapping is immediate
        all_singleton = all(len(lst) == 1 for lst in color_to_nodes1.values())
        if all_singleton:
            mapping = [-1] * n
            for c, lst in color_to_nodes1.items():
                mapping[lst[0]] = color_to_nodes2[c][0]
            return {"mapping": mapping}

        # Build candidate sets: for each u in G1, candidates are nodes in G2 with same color
        candidates: List[Set[int]] = [set() for _ in range(n)]
        for u in range(n):
            candidates[u] = set(color_to_nodes2[colors1[u]])

        # Prepare state for search
        mapping: List[int] = [-1] * n
        used: List[bool] = [False] * n
        unmapped: Set[int] = set(range(n))

        # Deterministic propagation: assign all variables with singleton domains first
        queue: List[int] = [u for u in range(n) if len(candidates[u]) == 1]
        contradiction = False
        while queue:
            u = queue.pop()
            if mapping[u] != -1:
                continue
            v = next(iter(candidates[u]))
            if used[v]:
                contradiction = True
                break
            # assign u -> v
            mapping[u] = v
            used[v] = True
            if u in unmapped:
                unmapped.remove(u)
            # forward-check neighbors and non-neighbors
            nb_u = adj1[u]
            # iterate over a snapshot of unmapped to avoid mutation during iteration
            for w in list(unmapped):
                if mapping[w] != -1:
                    continue
                old = candidates[w]
                if w in nb_u:
                    new = old & adj2[v]
                else:
                    # w is non-neighbor of u: must map to non-neighbor of v
                    new = old - adj2[v]
                if not new:
                    contradiction = True
                    break
                if new is not old:
                    candidates[w] = new
                    if len(new) == 1:
                        queue.append(w)
            if contradiction:
                break

        if not contradiction and all(x != -1 for x in mapping):
            # full assignment found via propagation
            final_map = mapping
            return {"mapping": final_map}

        # Otherwise use backtracking search with forward-checking.
        # Prepare a small helper closure to do recursive search.
        # For speed, precompute degrees for tie-breaking.
        deg2_list = deg2

        # We'll pick the variable with minimum remaining values (MRV), tie-break by degree descending.
        def select_var() -> int:
            best_u = -1
            best_len = 10**9
            best_deg = -1
            for u in unmapped:
                l = len(candidates[u])
                if l == 0:
                    return u  # immediate dead-end
                if l < best_len or (l == best_len and deg1[u] > best_deg):
                    best_len = l
                    best_deg = deg1[u]
                    best_u = u
            return best_u

        # We use recursion; saved modifications are restored upon backtrack.
        def backtrack() -> bool:
            if not unmapped:
                return True
            u = select_var()
            # If domain empty, fail
            dom = candidates[u]
            if not dom:
                return False
            # Order candidates by degree (heuristic: try higher-degree nodes first)
            # convert to list to iterate deterministically
            cand_list = sorted(dom, key=lambda x: -deg2_list[x])
            # Snapshot unmapped for iteration safety (we will modify unmapped inside)
            for v in cand_list:
                if used[v]:
                    continue
                # Quick consistency check with already mapped neighbors
                ok = True
                for nb in adj1[u]:
                    mapped_nb = mapping[nb]
                    if mapped_nb != -1:
                        if mapped_nb not in adj2[v]:
                            ok = False
                            break
                if not ok:
                    continue

                # Assign u -> v
                mapping[u] = v
                used[v] = True
                unmapped.remove(u)

                # Forward-check: restrict domains and record changes
                changed: List[Tuple[int, Set[int]]] = []
                nb_u = adj1[u]
                fail = False
                for w in list(unmapped):
                    if mapping[w] != -1:
                        continue
                    old = candidates[w]
                    if w in nb_u:
                        new = old & adj2[v]
                    else:
                        new = old - adj2[v]
                    if not new:
                        fail = True
                        break
                    if new is not old:
                        changed.append((w, old))
                        candidates[w] = new
                if not fail:
                    if backtrack():
                        return True

                # Undo assignment and restore domains
                for (w, old) in changed:
                    candidates[w] = old
                mapping[u] = -1
                used[v] = False
                unmapped.add(u)
            return False

        # Start backtracking
        success = backtrack()
        if not success:
            # As a very last resort, attempt a deterministic fallback pairing by sorting
            final_map = self._fallback_map(n, adj1, adj2, colors1, colors2)
            return {"mapping": final_map}

        # Validate final mapping and return
        final_map = mapping
        # Ensure mapping is a permutation; if not, use fallback
        if len(set(final_map)) != n or any(not (0 <= x < n) for x in final_map):
            final_map = self._fallback_map(n, adj1, adj2, colors1, colors2)
        return {"mapping": final_map}

    def _fallback_map(
        self,
        n: int,
        adj1: List[Set[int]],
        adj2: List[Set[int]],
        colors1: List[int],
        colors2: List[int],
    ) -> List[int]:
        """
        Deterministic fallback mapping:
        For each color class, sort nodes in G1 and G2 by a stable signature
        (degree, sorted neighbor colors) and pair them in order.
        This is only used as a last resort.
        """
        # Build neighbor-color multisets for signatures
        sig1 = [None] * n
        sig2 = [None] * n
        for i in range(n):
            nb_colors = sorted([colors1[j] for j in adj1[i]])
            sig1[i] = (len(adj1[i]), tuple(nb_colors), i)
        for i in range(n):
            nb_colors = sorted([colors2[j] for j in adj2[i]])
            sig2[i] = (len(adj2[i]), tuple(nb_colors), i)

        # Group by color
        col_to_nodes1: Dict[int, List[int]] = {}
        col_to_nodes2: Dict[int, List[int]] = {}
        for i, c in enumerate(colors1):
            col_to_nodes1.setdefault(c, []).append(i)
        for i, c in enumerate(colors2):
            col_to_nodes2.setdefault(c, []).append(i)

        mapping = [-1] * n
        for c, nodes1 in col_to_nodes1.items():
            nodes2 = col_to_nodes2.get(c, [])
            # Stable sort both lists by signature
            nodes1_sorted = sorted(nodes1, key=lambda x: sig1[x])
            nodes2_sorted = sorted(nodes2, key=lambda x: sig2[x])
            # Pair in order
            for u, v in zip(nodes1_sorted, nodes2_sorted):
                mapping[u] = v
        # As a last step, if mapping invalid (not permutation), fall back to identity-like mapping
        if len(set(mapping)) != n or any(x is None or x < 0 for x in mapping):
            # Fill remaining with unused nodes in order
            used = set(x for x in mapping if x is not None and x >= 0)
            unused = [i for i in range(n) if i not in used]
            for i in range(n):
                if mapping[i] is None or mapping[i] < 0:
                    mapping[i] = unused.pop()
        return mapping