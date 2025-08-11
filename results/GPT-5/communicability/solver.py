from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]], **kwargs) -> Dict[str, Any]:
        """
        Compute communicability C(u, v) = (e^A)_{uv} for an undirected graph.

        Steps:
        - Parse adjacency list; union edges (u < v) to find connected components (DSU)
        - Group nodes by component
        - For each component, compute exp(Adjacency) via symmetric eigendecomposition
        - Use closed forms for special families (complete graphs, stars, complete bipartite, cycles)
        - Assemble result into a dense matrix and return as dict-of-dicts
        """
        adj_list = problem.get("adjacency_list", [])
        n = len(adj_list)

        # Handle empty graph
        if n == 0:
            return {"communicability": {}}

        # Union-Find (Disjoint Set Union) to find connected components
        parent = list(range(n))
        size = [1] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        # 1) Union pass (avoid set overhead; only consider u < v)
        for u, nbrs in enumerate(adj_list):
            for v in nbrs:
                if 0 <= v < n and u < v:
                    union(u, v)

        # Precompute roots after unions (avoid repeated find calls)
        roots = [find(i) for i in range(n)]

        # 2) Gather nodes per component root
        comp_nodes_map: Dict[int, List[int]] = {}
        for i, r in enumerate(roots):
            lst = comp_nodes_map.get(r)
            if lst is None:
                comp_nodes_map[r] = [i]
            else:
                lst.append(i)

        # Prepare output matrix; start with identity for isolated nodes
        C = np.eye(n, dtype=np.float64)

        # Fast constants for 2-node components
        ch1 = float(np.cosh(1.0))
        sh1 = float(np.sinh(1.0))

        # Preallocate reusable buffers
        max_m = 0
        for comp in comp_nodes_map.values():
            if len(comp) > max_m:
                max_m = len(comp)
        A_buf = np.empty((max_m, max_m), dtype=np.float64, order="F") if max_m > 0 else None
        idx_buf = np.empty(max_m, dtype=np.int64) if max_m > 0 else None

        # Reusable global-to-local index map (reset on each component)
        g2l = np.full(n, -1, dtype=np.int64)

        # Process each component
        for r, comp in comp_nodes_map.items():
            m = len(comp)
            if m == 1:
                # exp([[0]]) = [[1]], already set by identity
                continue

            if m == 2:
                # Two nodes connected by a single edge -> adjacency [[0,1],[1,0]]
                u, v = comp[0], comp[1]
                C[u, u] = ch1
                C[v, v] = ch1
                C[u, v] = sh1
                C[v, u] = sh1
                continue

            # Degrees (within component)
            degs = [len(adj_list[u]) for u in comp]

            # Closed form: Complete graph K_m
            if all(d == m - 1 for d in degs):
                idx = idx_buf[:m]
                idx[:] = comp
                # e^A = e^{-1} I + ((e^{m-1} - e^{-1})/m) J
                e_m1 = float(np.exp(m - 1.0))
                e_neg1 = float(np.exp(-1.0))
                alpha = (e_m1 - e_neg1) / m
                # Fill block quickly
                C[idx[:, None], idx[None, :]] = alpha
                C[idx, idx] += e_neg1
                continue

            # Closed form: Cycle graph C_m (connected 2-regular)
            if m >= 3 and all(d == 2 for d in degs):
                idx = idx_buf[:m]
                idx[:] = comp
                k = np.arange(m, dtype=np.float64)
                lambdas = 2.0 * np.cos(2.0 * np.pi * k / m)
                e_l = np.exp(lambdas)
                y = np.fft.ifft(e_l).real  # first row of exp(A) (circulant)
                a = np.arange(m, dtype=np.int64)
                D = (a[None, :] - a[:, None]) % m
                E_comp = y[D]
                C[idx[:, None], idx[None, :]] = E_comp
                continue

            # Closed form: Star graph S_m (one center degree m-1, others degree 1)
            max_deg = max(degs)
            if max_deg == m - 1:
                cnt_max = degs.count(max_deg)
                if cnt_max == 1 and degs.count(1) == m - 1:
                    # Identify center and leaves
                    center_local = int(np.argmax(degs))
                    center = comp[center_local]
                    leaves = [u for u in comp if u != center]
                    idx_leaves = np.array(leaves, dtype=np.int64)
                    # Parameters
                    L = m - 1
                    s = float(np.sqrt(L))
                    ch = float(np.cosh(s))
                    sh_over_s = float(np.sinh(s) / s)
                    diff = (ch - 1.0) / L
                    # Assign values
                    # Leaves x Leaves block
                    C[idx_leaves[:, None], idx_leaves[None, :]] = diff
                    C[idx_leaves, idx_leaves] += 1.0
                    # Center diagonal
                    C[center, center] = ch
                    # Center-leaves cross
                    C[center, idx_leaves] = sh_over_s
                    C[idx_leaves, center] = sh_over_s
                    continue

            # Closed form: Complete bipartite K_{p,q} (generalization of star)
            # Degrees take two values a and b with counts c_a and c_b such that c_a = b and c_b = a
            uniq = set(degs)
            if len(uniq) == 2:
                a, b = sorted(uniq)
                ca = degs.count(a)
                cb = degs.count(b)
                if ca == b and cb == a:
                    # Partition sets by degree
                    setA = [u for u, d in zip(comp, degs) if d == a]  # size ca
                    setB = [u for u, d in zip(comp, degs) if d == b]  # size cb
                    # Verify no edges within partitions (sufficient given degree counts)
                    # Check the smaller set to reduce work
                    S = setA if len(setA) <= len(setB) else setB
                    S_set = set(S)
                    is_complete_bip = True
                    for u in S:
                        # If any neighbor in same set, not complete bipartite
                        for v in adj_list[u]:
                            if v in S_set:
                                is_complete_bip = False
                                break
                        if not is_complete_bip:
                            break
                    if is_complete_bip:
                        p = len(setA)
                        q = len(setB)
                        s = float(np.sqrt(p * q))
                        ch = float(np.cosh(s))
                        sh_over_s = float(np.sinh(s) / s)
                        alphaA = (ch - 1.0) / p
                        alphaB = (ch - 1.0) / q
                        idxA = np.array(setA, dtype=np.int64)
                        idxB = np.array(setB, dtype=np.int64)
                        # Left-left block
                        C[idxA[:, None], idxA[None, :]] = alphaA
                        C[idxA, idxA] += 1.0
                        # Right-right block
                        C[idxB[:, None], idxB[None, :]] = alphaB
                        C[idxB, idxB] += 1.0
                        # Cross blocks
                        C[idxA[:, None], idxB[None, :]] = sh_over_s
                        C[idxB[:, None], idxA[None, :]] = sh_over_s
                        continue

            # Build local index map using reusable array
            for i, node in enumerate(comp):
                g2l[node] = i

            # Build dense adjacency matrix for the component (Fortran order for LAPACK)
            A = A_buf[:m, :m]
            A.fill(0.0)
            for u in comp:
                lu = g2l[u]
                for v in adj_list[u]:
                    if u < v:
                        lv = g2l[v]
                        if lv != -1:
                            A[lu, lv] = 1.0
                            A[lv, lu] = 1.0

            # Symmetric eigendecomposition: A = Q diag(w) Q^T  => exp(A) = Q diag(exp(w)) Q^T
            w, Q = np.linalg.eigh(A)
            # In-place exp of eigenvalues to reduce allocation
            np.exp(w, out=w)
            E_comp = (Q * w) @ Q.T  # scale columns by exp(w) without mutating Q

            # Place block into global matrix
            idx = idx_buf[:m]
            idx[:] = comp
            C[idx[:, None], idx[None, :]] = E_comp

            # Reset g2l for this component
            g2l[idx] = -1

        # Convert to required dict-of-dicts using single tolist() to reduce overhead
        rows = C.tolist()  # nested Python lists
        comm_dict: Dict[int, Dict[int, float]] = {u: dict(enumerate(rows[u])) for u in range(n)}

        return {"communicability": comm_dict}