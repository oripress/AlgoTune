import numpy as np
from numba import njit
from typing import Any, Dict, List

@njit(cache=True)
def lsc_index(x):
    idx = 0
    while True:
        if x & 1:
            return idx
        x >>= 1
        idx += 1

@njit(cache=True)
def compute_tsp_dp(D, nodes, depot):
    m = nodes.shape[0]
    M = 1 << m
    DP = np.full((M, m), np.inf, dtype=np.float64)
    PREV = np.full((M, m), -1, dtype=np.int32)
    # single-node tours
    for i in range(m):
        DP[1 << i, i] = D[depot, nodes[i]]
    # build DP
    for mask in range(M):
        if mask == 0 or (mask & (mask - 1)) == 0:
            continue
        sub = mask
        while sub:
            lsb = sub & -sub
            i = lsc_index(lsb)
            sub ^= lsb
            prev_mask = mask ^ (1 << i)
            best = np.inf
            best_j = -1
            tmp = prev_mask
            while tmp:
                lsb2 = tmp & -tmp
                j = lsc_index(lsb2)
                tmp ^= lsb2
                cost = DP[prev_mask, j] + D[nodes[j], nodes[i]]
                if cost < best:
                    best = cost
                    best_j = j
            DP[mask, i] = best
            PREV[mask, i] = best_j
    return DP, PREV

@njit(cache=True)
def compute_cost_s_and_end(D, nodes, DP, depot):
    m = nodes.shape[0]
    M = DP.shape[0]
    cost_s = np.empty(M, dtype=np.float64)
    end_node = np.empty(M, dtype=np.int32)
    cost_s[0] = 0.0
    end_node[0] = -1
    for mask in range(1, M):
        tmp = mask
        best = np.inf
        best_i = -1
        while tmp:
            lsb = tmp & -tmp
            i = lsc_index(lsb)
            tmp ^= lsb
            c = DP[mask, i] + D[nodes[i], depot]
            if c < best:
                best = c
                best_i = i
        cost_s[mask] = best
        end_node[mask] = best_i
    return cost_s, end_node

@njit(cache=True)
def partition_dp(cost_s, M, K):
    DP2 = np.full((M, K+1), np.inf, dtype=np.float64)
    PRE2 = np.full((M, K+1), -1, dtype=np.int32)
    DP2[0, 0] = 0.0
    for mask in range(1, M):
        for k in range(1, K+1):
            sub = mask
            while sub:
                prev = mask ^ sub
                c = DP2[prev, k-1] + cost_s[sub]
                if c < DP2[mask, k]:
                    DP2[mask, k] = c
                    PRE2[mask, k] = sub
                sub = (sub - 1) & mask
    return DP2, PRE2

# Warm up Numba compilation
_dummy_D = np.zeros((1,1), dtype=np.float64)
_dummy_nodes = np.zeros(0, dtype=np.int32)
_dummy_DP, _dummy_PREV = compute_tsp_dp(_dummy_D, _dummy_nodes, 0)
_dummy_cost_s, _dummy_end = compute_cost_s_and_end(_dummy_D, _dummy_nodes, _dummy_DP, 0)
_dummy_DP2, _dummy_PRE2 = partition_dp(_dummy_cost_s, 1, 1)

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[List[int]]:
        D_list = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D_list)
        D = np.array(D_list, dtype=np.float64)
        # Nodes excluding depot
        nodes = np.array([i for i in range(n) if i != depot], dtype=np.int32)
        m = nodes.shape[0]
        if m == 0:
            return [[depot, depot] for _ in range(K)]
        # Compute exact TSP costs for all subsets
        DP, PREV = compute_tsp_dp(D, nodes, depot)
        cost_s, end_node = compute_cost_s_and_end(D, nodes, DP, depot)
        # Partition into K_eff tours
        K_eff = min(K, m)
        M = 1 << m
        DP2, PRE2 = partition_dp(cost_s, M, K_eff)
        routes_eff: List[List[int]] = []
        mask = M - 1
        # Backtrack partition
        for k in range(K_eff, 0, -1):
            sub = int(PRE2[mask, k])
            # Reconstruct tour for subset 'sub'
            path: List[int] = []
            cur = sub
            last = int(end_node[sub])
            while last >= 0:
                path.append(int(nodes[last]))
                pm = cur
                cur ^= (1 << last)
                last = int(PREV[pm, last])
            path.reverse()
            routes_eff.append([depot] + path + [depot])
            mask ^= sub
        # Add empty tours if K > K_eff
        routes = routes_eff + [[depot, depot] for _ in range(K - K_eff)]
        return routes