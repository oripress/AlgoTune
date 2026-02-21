import numpy as np
import networkx as nx
from numba import njit

@njit
def popcount(c):
    count = 0
    while c:
        c &= c - np.uint64(1)
        count += 1
    return count

@njit
def solve_numba(problem_matrix):
    n = problem_matrix.shape[0]
    adj = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        mask = np.uint64(0)
        for j in range(n):
            if problem_matrix[i, j]:
                mask |= (np.uint64(1) << np.uint64(j))
        adj[i] = mask
        
    max_set = np.uint64(0)
    max_len = 0
    
    stack_candidates = np.zeros(n + 1, dtype=np.uint64)
    stack_current_set = np.zeros(n + 1, dtype=np.uint64)
    stack_current_len = np.zeros(n + 1, dtype=np.int32)
    
    stack_ptr = 0
    stack_candidates[0] = (np.uint64(1) << np.uint64(n)) - np.uint64(1)
    stack_current_set[0] = np.uint64(0)
    stack_current_len[0] = 0
    
    while stack_ptr >= 0:
        candidates = stack_candidates[stack_ptr]
        current_set = stack_current_set[stack_ptr]
        current_len = stack_current_len[stack_ptr]
        stack_ptr -= 1
        
        if current_len + popcount(candidates) <= max_len:
            continue
            
        if candidates == 0:
            if current_len > max_len:
                max_len = current_len
                max_set = current_set
            continue
            
        lsb = candidates & (~candidates + np.uint64(1))
        v = 0
        temp = lsb
        while temp > np.uint64(1):
            temp >>= np.uint64(1)
            v += 1
            
        stack_ptr += 1
        stack_candidates[stack_ptr] = candidates & ~lsb
        stack_current_set[stack_ptr] = current_set
        stack_current_len[stack_ptr] = current_len
        
        stack_ptr += 1
        stack_candidates[stack_ptr] = candidates & ~adj[v] & ~lsb
        stack_current_set[stack_ptr] = current_set | lsb
        stack_current_len[stack_ptr] = current_len + 1

    return max_set

class Solver:
    def __init__(self):
        dummy = np.zeros((1, 1), dtype=np.int32)
        solve_numba(dummy)

    def solve(self, problem: list[list[int]], **kwargs) -> list[int]:
        n = len(problem)
        if n == 0:
            return []
        if n <= 64:
            problem_matrix = np.array(problem, dtype=np.int32)
            max_set = solve_numba(problem_matrix)
            result = []
            for i in range(n):
                if max_set & (1 << i):
                    result.append(i)
            return result
        else:
            G_comp = nx.Graph()
            G_comp.add_nodes_from(range(n))
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if problem[i][j] == 0:
                        edges.append((i, j))
            G_comp.add_edges_from(edges)
            clique, _ = nx.max_weight_clique(G_comp, weight=None)
            return sorted(clique)