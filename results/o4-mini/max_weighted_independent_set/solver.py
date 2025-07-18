import numpy as np
from ortools.sat.python import cp_model

# Try to use Cython optimized version, fallback to Numba JIT
try:
    from solver_cy import mwis_nb
except ImportError:
    from numba import njit

    @njit(cache=True)
    def mwis_nb(n, neighbor_masks, weights):
        # Iterative branch-and-bound for MWIS (bitmask up to 64 nodes)
        stack_cand = np.empty(n, dtype=np.uint64)
        stack_weight = np.empty(n, dtype=np.int64)
        stack_mask = np.empty(n, dtype=np.uint64)
        best_w = np.int64(0)
        best_mask = np.uint64(0)
        full = np.uint64((1 << n) - 1)
        # initial state
        stack_cand[0] = full
        stack_weight[0] = 0
        stack_mask[0] = np.uint64(0)
        pos = 1
        while pos > 0:
            pos -= 1
            cand = stack_cand[pos]
            cw = stack_weight[pos]
            cmask = stack_mask[pos]
            rem = cand
            rem_sum = np.int64(0)
            while rem:
                lsb = rem & -rem
                idx = int(lsb.bit_length() - 1)
                rem_sum += weights[idx]
                rem &= rem - np.uint64(1)
            if cw + rem_sum <= best_w:
                continue
            if cand == np.uint64(0):
                if cw > best_w:
                    best_w = cw
                    best_mask = cmask
                continue
            cand_bits = cand
            # pivot selection: single trailing‐zero bit (fast)
            lsb2 = cand & -cand
            idx2 = 0
            tmp2 = lsb2
            while (tmp2 & 1) == 0:
                tmp2 >>= 1
                idx2 += 1
            bitv = np.uint64(1) << idx2
            excl = neighbor_masks[idx2] | bitv
            # branch without this vertex
            stack_cand[pos] = cand & (~bitv)
            stack_weight[pos] = cw
            stack_mask[pos] = cmask
            pos += 1
            # branch with this vertex
            stack_cand[pos] = cand & (~excl)
            stack_weight[pos] = cw + weights[idx2]
            stack_mask[pos] = cmask | bitv
            pos += 1

class Solver:
    def __init__(self):
        # Pre-compile mwis_nb (not counted in solve time)
        dummy_masks = np.zeros(1, dtype=np.uint64)
        dummy_weights = np.zeros(1, dtype=np.int64)
        mwis_nb(1, dummy_masks, dummy_weights)

    def solve(self, problem, **kwargs):
        adj = problem["adj_matrix"]
        weights_py = problem["weights"]
        n = len(weights_py)
        # find connected components
        visited = [False] * n
        sol = []
        for u in range(n):
            if not visited[u]:
                stack = [u]
                visited[u] = True
                comp = []
                while stack:
                    v = stack.pop()
                    comp.append(v)
                    for w in range(n):
                        if adj[v][w] and not visited[w]:
                            visited[w] = True
                            stack.append(w)
                k = len(comp)
                if k == 0:
                    continue
                if k <= 64:
                    neighbor_masks = np.zeros(k, dtype=np.uint64)
                    weights_arr = np.zeros(k, dtype=np.int64)
                    for idx_local, vtx in enumerate(comp):
                        weights_arr[idx_local] = weights_py[vtx]
                        m = np.uint64(0)
                        row = adj[vtx]
                        for jdx_local, wtx in enumerate(comp):
                            if row[wtx]:
                                m |= np.uint64(1 << jdx_local)
                        neighbor_masks[idx_local] = m
                    _, mask = mwis_nb(k, neighbor_masks, weights_arr)
                    mask_int = int(mask)
                    for idx_local, vtx in enumerate(comp):
                        if (mask_int >> idx_local) & 1:
                            sol.append(vtx)
                else:
                    model = cp_model.CpModel()
                    x = [model.NewBoolVar(f"x_{vtx}") for vtx in comp]
                    for i_local, vtx_i in enumerate(comp):
                        for j_local, vtx_j in enumerate(comp[i_local+1:], start=i_local+1):
                            if adj[vtx_i][vtx_j]:
                                model.Add(x[i_local] + x[j_local] <= 1)
                    model.Maximize(sum(weights_py[vtx] * x[i_local] for i_local, vtx in enumerate(comp)))
                    solver = cp_model.CpSolver()
                    solver.parameters.num_search_workers = 1
                    status = solver.Solve(model)
                    if status == cp_model.OPTIMAL:
                        for i_local, vtx in enumerate(comp):
                            if solver.Value(x[i_local]):
                                sol.append(vtx)
        sol.sort()
        return sol
        return []