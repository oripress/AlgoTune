from typing import Any, List
import numpy as np
from numba import njit

@njit(inline='always')
def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    x = x + (x >> np.uint64(8))
    x = x + (x >> np.uint64(16))
    x = x + (x >> np.uint64(32))
    return int(x & np.uint64(0x7f))

@njit(inline='always')
def ctz64(v):
    return popcount64((v ^ (v - np.uint64(1))) >> np.uint64(1))

@njit
def get_mis(adj):
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)
        
    num_words = (n + 63) // 64
    
    comp_adj_mask = np.zeros((n, num_words), dtype=np.uint64)
    for i in range(n):
        for j in range(n):
            if i != j and not adj[i, j]:
                comp_adj_mask[i, j // 64] |= (np.uint64(1) << np.uint64(j % 64))
                
    deg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for w in range(num_words):
            deg[i] += popcount64(comp_adj_mask[i, w])

    deg_order = np.zeros(n, dtype=np.int32)
    active = np.ones(n, dtype=np.int8)
    for i in range(n):
        min_deg = 1000000
        min_v = -1
        for j in range(n):
            if active[j] and deg[j] < min_deg:
                min_deg = deg[j]
                min_v = j
        deg_order[i] = min_v
        active[min_v] = 0
        for w in range(num_words):
            word = comp_adj_mask[min_v, w]
            while word != 0:
                idx = w * 64 + ctz64(word)
                if active[idx]:
                    deg[idx] -= 1
                word &= word - np.uint64(1)
                
    mapping = np.zeros(n, dtype=np.int32)
    reverse_mapping = np.zeros(n, dtype=np.int32)
    for i in range(n):
        v = deg_order[n - 1 - i]
        mapping[v] = i
        reverse_mapping[i] = v
        
    new_comp_adj_mask = np.zeros((n, num_words), dtype=np.uint64)
    for i in range(n):
        new_i = mapping[i]
        for w in range(num_words):
            word = comp_adj_mask[i, w]
            while word != 0:
                j = w * 64 + ctz64(word)
                new_j = mapping[j]
                new_comp_adj_mask[new_i, new_j // 64] |= (np.uint64(1) << np.uint64(new_j % 64))
                word &= word - np.uint64(1)
                
    max_clique_size = 0
    best_clique = np.zeros(n, dtype=np.int8)
    
    U_stack = np.zeros((n + 1, num_words), dtype=np.uint64)
    C_added = np.zeros(n + 1, dtype=np.int32)
    
    U_verts = np.zeros((n + 1, n), dtype=np.int32)
    U_colors = np.zeros((n + 1, n), dtype=np.int32)
    U_sizes = np.zeros(n + 1, dtype=np.int32)
    U_idx = np.zeros(n + 1, dtype=np.int32)
    
    all_color_classes = np.zeros((n + 1, n, num_words), dtype=np.uint64)
    all_colors = np.zeros((n + 1, n), dtype=np.int32)
    all_temp_U = np.zeros((n + 1, n), dtype=np.int32)
    all_bucket_counts = np.zeros((n + 1, n), dtype=np.int32)
    all_bucket_starts = np.zeros((n + 1, n), dtype=np.int32)
    
    num_colors = 0
    for i in range(n):
        u = i
        c_idx = 0
        while c_idx < num_colors:
            conflict = False
            for w in range(num_words):
                if (new_comp_adj_mask[u, w] & all_color_classes[0, c_idx, w]) != 0:
                    conflict = True
                    break
            if not conflict:
                break
            c_idx += 1
            
        if c_idx == num_colors:
            for w in range(num_words):
                all_color_classes[0, c_idx, w] = 0
            num_colors += 1
            
        all_color_classes[0, c_idx, u // 64] |= (np.uint64(1) << np.uint64(u % 64))
        all_colors[0, i] = c_idx
        
    for i in range(num_colors):
        all_bucket_counts[0, i] = 0
    for i in range(n):
        all_bucket_counts[0, all_colors[0, i]] += 1
        
    start = 0
    for c_idx in range(num_colors - 1, -1, -1):
        all_bucket_starts[0, c_idx] = start
        start += all_bucket_counts[0, c_idx]
        
    for i in range(n):
        c_idx = all_colors[0, i]
        pos = all_bucket_starts[0, c_idx]
        U_verts[0, pos] = i
        U_colors[0, pos] = c_idx
        U_stack[0, i // 64] |= (np.uint64(1) << np.uint64(i % 64))
        all_bucket_starts[0, c_idx] += 1
        
    U_sizes[0] = n
    U_idx[0] = 0
    
    top = 0
    
    while top >= 0:
        if U_idx[top] >= U_sizes[top]:
            top -= 1
            continue
            
        i = U_idx[top]
        v = U_verts[top, i]
        c = U_colors[top, i]
        U_idx[top] += 1
        
        if top + c + 1 <= max_clique_size:
            top -= 1
            continue
            
        U_stack[top, v // 64] &= ~(np.uint64(1) << np.uint64(v % 64))
        
        C_added[top] = v
        
        temp_U_size = 0
        for w in range(num_words):
            word = U_stack[top, w] & new_comp_adj_mask[v, w]
            U_stack[top + 1, w] = word
            while word != 0:
                idx = w * 64 + ctz64(word)
                all_temp_U[top + 1, temp_U_size] = idx
                temp_U_size += 1
                word &= word - np.uint64(1)
                
        if temp_U_size == 0:
            if top + 1 > max_clique_size:
                max_clique_size = top + 1
                best_clique[:] = 0
                for k in range(top + 1):
                    best_clique[reverse_mapping[C_added[k]]] = 1
            continue
            
        num_colors = 0
        for j in range(temp_U_size):
            u = all_temp_U[top + 1, j]
            c_idx = 0
            while c_idx < num_colors:
                conflict = False
                for w in range(num_words):
                    if (new_comp_adj_mask[u, w] & all_color_classes[top + 1, c_idx, w]) != 0:
                        conflict = True
                        break
                if not conflict:
                    break
                c_idx += 1
                
            if c_idx == num_colors:
                for w in range(num_words):
                    all_color_classes[top + 1, c_idx, w] = 0
                num_colors += 1
                
            all_color_classes[top + 1, c_idx, u // 64] |= (np.uint64(1) << np.uint64(u % 64))
            all_colors[top + 1, j] = c_idx
            
        if top + 1 + num_colors <= max_clique_size:
            continue
            
        for j in range(num_colors):
            all_bucket_counts[top + 1, j] = 0
        for j in range(temp_U_size):
            all_bucket_counts[top + 1, all_colors[top + 1, j]] += 1
            
        start = 0
        for c_idx in range(num_colors - 1, -1, -1):
            all_bucket_starts[top + 1, c_idx] = start
            start += all_bucket_counts[top + 1, c_idx]
            
        for j in range(temp_U_size):
            c_idx = all_colors[top + 1, j]
            pos = all_bucket_starts[top + 1, c_idx]
            U_verts[top + 1, pos] = all_temp_U[top + 1, j]
            U_colors[top + 1, pos] = c_idx
            all_bucket_starts[top + 1, c_idx] += 1
            
        U_sizes[top + 1] = temp_U_size
        U_idx[top + 1] = 0
        
        top += 1

    return best_clique

class Solver:
    def __init__(self):
        dummy_adj = np.zeros((2, 2), dtype=np.int8)
        get_mis(dummy_adj)
        
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []
            
        adj = np.array(problem, dtype=np.int8)
        mis = get_mis(adj)
        
        return [i for i in range(n) if mis[i] == 0]