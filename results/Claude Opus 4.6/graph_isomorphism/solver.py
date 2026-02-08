import sys
import numpy as np
from collections import defaultdict
from typing import Any
import numba


@numba.njit(cache=True)
def _backtrack(order, prev_nbrs_flat, prev_nbrs_starts, prev_nbrs_lens,
               candidates_flat, cand_starts, cand_lens,
               adj2_mat, adj1_mat, adj2_list_flat, adj2_starts, adj2_lens,
               mapping, inv_mapping, n):
    """Numba-accelerated backtracking for graph isomorphism."""
    # Stack-based iterative backtracking
    # stack[idx] = current candidate index for position idx
    stack = np.zeros(n, dtype=np.int32)
    idx = 0

    while idx >= 0:
        if idx == n:
            return True

        u = order[idx]
        pn_start = prev_nbrs_starts[idx]
        pn_len = prev_nbrs_lens[idx]
        cand_start = cand_starts[u]
        cand_len = cand_lens[u]

        found = False
        ci = stack[idx]

        while ci < cand_len:
            v = candidates_flat[cand_start + ci]
            ci += 1

            if inv_mapping[v] != -1:
                continue

            # Forward check: mapped neighbors of u must be mapped to neighbors of v
            ok = True
            for pi in range(pn_len):
                w = prev_nbrs_flat[pn_start + pi]
                mw = mapping[w]
                if not adj2_mat[v, mw]:
                    ok = False
                    break
            if not ok:
                continue

            # Reverse check: mapped neighbors of v must map back to neighbors of u
            a2_start = adj2_starts[v]
            a2_len = adj2_lens[v]
            for ai in range(a2_len):
                nb_v = adj2_list_flat[a2_start + ai]
                w = inv_mapping[nb_v]
                if w != -1 and not adj1_mat[u, w]:
                    ok = False
                    break
            if not ok:
                continue

            # Assignment
            mapping[u] = v
            inv_mapping[v] = u
            stack[idx] = ci
            found = True
            break

        if found:
            idx += 1
            if idx < n:
                stack[idx] = 0
        else:
            # Undo assignment if we made one at this level
            if mapping[u] != -1:
                v_old = mapping[u]
                mapping[u] = -1
                inv_mapping[v_old] = -1
            stack[idx] = 0
            idx -= 1
            if idx >= 0:
                # Undo assignment at parent level
                u_parent = order[idx]
                v_parent = mapping[u_parent]
                if v_parent != -1:
                    mapping[u_parent] = -1
                    inv_mapping[v_parent] = -1

    return False


class Solver:
    def solve(self, problem, **kwargs) -> Any:
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]

        if n == 0:
            return {"mapping": []}
        if n == 1:
            return {"mapping": [0]}
        if not edges_g1 and not edges_g2:
            return {"mapping": list(range(n))}

        # Build adjacency lists
        adj1 = [[] for _ in range(n)]
        adj2 = [[] for _ in range(n)]

        for u, v in edges_g1:
            adj1[u].append(v)
            adj1[v].append(u)

        for u, v in edges_g2:
            adj2[u].append(v)
            adj2[v].append(u)

        # WL color refinement
        color1 = [len(adj1[i]) for i in range(n)]
        color2 = [len(adj2[i]) for i in range(n)]

        for _ in range(min(n + 1, 30)):
            num_before = len(set(color1) | set(color2))
            if num_before == 2 * n:
                break

            sig1 = [(color1[i], tuple(sorted(color1[j] for j in adj1[i]))) for i in range(n)]
            sig2 = [(color2[i], tuple(sorted(color2[j] for j in adj2[i]))) for i in range(n)]

            all_sigs = set(sig1) | set(sig2)
            if len(all_sigs) == num_before:
                break
            sig_map = {s: idx for idx, s in enumerate(all_sigs)}

            color1 = [sig_map[s] for s in sig1]
            color2 = [sig_map[s] for s in sig2]

        # Build candidate lists
        c2_by_color = defaultdict(list)
        for i in range(n):
            c2_by_color[color2[i]].append(i)

        candidates = [list(c2_by_color[color1[i]]) for i in range(n)]

        # Quick check: unique mapping
        if all(len(c) == 1 for c in candidates):
            return {"mapping": [c[0] for c in candidates]}

        # Processing order
        order = []
        in_order = [False] * n

        start = min(range(n), key=lambda x: len(candidates[x]))
        order.append(start)
        in_order[start] = True

        frontier = set()
        for nb in adj1[start]:
            if not in_order[nb]:
                frontier.add(nb)

        while len(order) < n:
            if frontier:
                best = None
                best_key = None
                for x in frontier:
                    conn = sum(1 for nb in adj1[x] if in_order[nb])
                    key = (-conn, len(candidates[x]))
                    if best_key is None or key < best_key:
                        best = x
                        best_key = key
                frontier.discard(best)
            else:
                best = None
                best_cand = float('inf')
                for i in range(n):
                    if not in_order[i] and len(candidates[i]) < best_cand:
                        best = i
                        best_cand = len(candidates[i])

            order.append(best)
            in_order[best] = True
            for nb in adj1[best]:
                if not in_order[nb]:
                    frontier.add(nb)

        # Precompute prev_neighbors
        order_pos = [0] * n
        for i, u in enumerate(order):
            order_pos[u] = i

        prev_nbrs_list = [[] for _ in range(n)]
        for i, u in enumerate(order):
            for w in adj1[u]:
                if order_pos[w] < i:
                    prev_nbrs_list[i].append(w)

        # Flatten arrays for numba
        order_arr = np.array(order, dtype=np.int32)

        # Flatten prev_nbrs
        pn_flat = []
        pn_starts = np.zeros(n, dtype=np.int32)
        pn_lens = np.zeros(n, dtype=np.int32)
        for i in range(n):
            pn_starts[i] = len(pn_flat)
            pn_lens[i] = len(prev_nbrs_list[i])
            pn_flat.extend(prev_nbrs_list[i])
        pn_flat_arr = np.array(pn_flat, dtype=np.int32) if pn_flat else np.zeros(0, dtype=np.int32)

        # Flatten candidates
        cand_flat = []
        cand_starts = np.zeros(n, dtype=np.int32)
        cand_lens = np.zeros(n, dtype=np.int32)
        for i in range(n):
            cand_starts[i] = len(cand_flat)
            cand_lens[i] = len(candidates[i])
            cand_flat.extend(candidates[i])
        cand_flat_arr = np.array(cand_flat, dtype=np.int32)

        # Adjacency matrices
        adj1_mat = np.zeros((n, n), dtype=np.bool_)
        adj2_mat = np.zeros((n, n), dtype=np.bool_)
        for u, v in edges_g1:
            adj1_mat[u, v] = True
            adj1_mat[v, u] = True
        for u, v in edges_g2:
            adj2_mat[u, v] = True
            adj2_mat[v, u] = True

        # Flatten adj2 lists
        a2_flat = []
        a2_starts = np.zeros(n, dtype=np.int32)
        a2_lens = np.zeros(n, dtype=np.int32)
        for i in range(n):
            a2_starts[i] = len(a2_flat)
            a2_lens[i] = len(adj2[i])
            a2_flat.extend(adj2[i])
        a2_flat_arr = np.array(a2_flat, dtype=np.int32) if a2_flat else np.zeros(0, dtype=np.int32)

        mapping = np.full(n, -1, dtype=np.int32)
        inv_mapping = np.full(n, -1, dtype=np.int32)

        _backtrack(order_arr, pn_flat_arr, pn_starts, pn_lens,
                   cand_flat_arr, cand_starts, cand_lens,
                   adj2_mat, adj1_mat, a2_flat_arr, a2_starts, a2_lens,
                   mapping, inv_mapping, n)

        return {"mapping": mapping.tolist()}