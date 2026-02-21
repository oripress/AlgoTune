import numpy as np
from numba import njit
from typing import Any
@njit
def solve_mcs_iterative(A, B, n, m):
    mapped_G = np.full(n, -1, dtype=np.int32)
    mapped_H = np.full(m, -1, dtype=np.int32)
    
    best_mapping = np.full(n, -1, dtype=np.int32)
    best_size = 0
    
    choice = np.full(n, -1, dtype=np.int32)
    
    i = 0
    current_size = 0
    
    while i >= 0:
        max_possible = current_size + min(n - i, m - current_size)
        if max_possible <= best_size:
            i -= 1
            if i >= 0:
                p = choice[i]
                if p < m:
                    mapped_G[i] = -1
                    mapped_H[p] = -1
                    current_size -= 1
            continue
            
        if i == n:
            if current_size > best_size:
                best_size = current_size
                for k in range(n):
                    best_mapping[k] = mapped_G[k]
            i -= 1
            if i >= 0:
                p = choice[i]
                if p < m:
                    mapped_G[i] = -1
                    mapped_H[p] = -1
                    current_size -= 1
            continue
            
        start_p = choice[i] + 1
        found = False
        for p in range(start_p, m + 1):
            if p == m:
                choice[i] = m
                found = True
                break
            else:
                if mapped_H[p] != -1:
                    continue
                
                compatible = True
                for j in range(i):
                    q = mapped_G[j]
                    if q != -1:
                        if A[i, j] != B[p, q]:
                            compatible = False
                            break
                
                if compatible:
                    choice[i] = p
                    mapped_G[i] = p
                    mapped_H[p] = i
                    current_size += 1
                    found = True
                    break
                    
        if found:
            i += 1
            if i < n:
                choice[i] = -1
        else:
            i -= 1
            if i >= 0:
                p = choice[i]
                if p < m:
                    mapped_G[i] = -1
                    mapped_H[p] = -1
                    current_size -= 1

    return best_mapping

@njit
def solve_mcs_bitset_iterative(A, B, n, m):
    B_adj = np.zeros(m, dtype=np.uint64)
    B_nonadj = np.zeros(m, dtype=np.uint64)
    for p in range(m):
        for q in range(m):
            if p != q:
                if B[p, q]:
                    B_adj[p] |= (np.uint64(1) << np.uint64(q))
                else:
                    B_nonadj[p] |= (np.uint64(1) << np.uint64(q))

    best_mapping = np.full(n, -1, dtype=np.int32)
    best_size = 0
    
    domains = np.zeros((n + 1, n), dtype=np.uint64)
    initial_domain = (np.uint64(1) << np.uint64(m)) - np.uint64(1)
    for j in range(n):
        domains[0, j] = initial_domain
        
    mapped_G = np.full(n, -1, dtype=np.int32)
    
    state = np.zeros(n, dtype=np.int32)
    rem_domain = np.zeros(n, dtype=np.uint64)
    
    i = 0
    current_size = 0
    
    while i >= 0:
        if current_size > best_size:
            best_size = current_size
            for k in range(n):
                best_mapping[k] = mapped_G[k]
                
        if i == n:
            i -= 1
            if i >= 0 and state[i] == 1:
                current_size -= 1
                mapped_G[i] = -1
            continue
            
        if current_size + min(n - i, m - current_size) <= best_size:
            state[i] = 0
            i -= 1
            if i >= 0 and state[i] == 1:
                current_size -= 1
                mapped_G[i] = -1
            continue
            
        if state[i] == 0:
            rem_domain[i] = domains[i, i]
            state[i] = 1
            
        if state[i] == 1:
            dom = rem_domain[i]
            if dom > 0:
                p_uint = dom & (~dom + np.uint64(1))
                rem_domain[i] ^= p_uint
                
                p = 0
                temp = p_uint
                if (temp & np.uint64(0xFFFFFFFF)) == 0:
                    temp >>= np.uint64(32)
                    p += 32
                if (temp & np.uint64(0xFFFF)) == 0:
                    temp >>= np.uint64(16)
                    p += 16
                if (temp & np.uint64(0xFF)) == 0:
                    temp >>= np.uint64(8)
                    p += 8
                if (temp & np.uint64(0xF)) == 0:
                    temp >>= np.uint64(4)
                    p += 4
                if (temp & np.uint64(0x3)) == 0:
                    temp >>= np.uint64(2)
                    p += 2
                if (temp & np.uint64(0x1)) == 0:
                    p += 1
                    
                mapped_G[i] = p
                current_size += 1
                
                mask = ~(np.uint64(1) << np.uint64(p))
                for j in range(i + 1, n):
                    if A[i, j]:
                        domains[i + 1, j] = domains[i, j] & B_adj[p] & mask
                    else:
                        domains[i + 1, j] = domains[i, j] & B_nonadj[p] & mask
                    
                i += 1
                if i < n:
                    state[i] = 0
            else:
                state[i] = 2
                
        elif state[i] == 2:
            mapped_G[i] = -1
            for j in range(i + 1, n):
                domains[i + 1, j] = domains[i, j]
                
            state[i] = 3
            i += 1
            if i < n:
                state[i] = 0
                
        elif state[i] == 3:
            state[i] = 0
            i -= 1
            if i >= 0 and state[i] == 1:
                current_size -= 1
                mapped_G[i] = -1

    return best_mapping
class Solver:
    def solve(self, problem: dict[str, list[list[int]]], **kwargs) -> Any:
        A = np.array(problem["A"], dtype=np.int32)
        B = np.array(problem["B"], dtype=np.int32)
        n, m = len(A), len(B)
        
        if m <= 64:
            best_mapping = solve_mcs_bitset_iterative(A, B, n, m)
        else:
            best_mapping = solve_mcs_iterative(A, B, n, m)
        
        result = []
        for i in range(n):
            if best_mapping[i] != -1:
                result.append((int(i), int(best_mapping[i])))
                
        return result