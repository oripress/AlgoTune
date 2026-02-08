import sys
import numpy as np
from typing import Any


class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> Any:
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        sys.setrecursionlimit(max(10000, n * 10))
        
        # Build adjacency bitmasks
        adj = [0] * n
        for i in range(n):
            row = problem[i]
            for j in range(n):
                if row[j]:
                    adj[i] |= (1 << j)
        
        # Quick check: no edges
        if all(a == 0 for a in adj):
            return list(range(n))
        
        # Connected components via BFS
        visited = 0
        result_mask = 0
        
        for start in range(n):
            start_bit = 1 << start
            if visited & start_bit:
                continue
            comp = start_bit
            visited |= start_bit
            queue = [start]
            while queue:
                u = queue.pop()
                nbrs = adj[u] & ~visited
                while nbrs:
                    lsb = nbrs & -nbrs
                    v_idx = lsb.bit_length() - 1
                    visited |= lsb
                    comp |= lsb
                    queue.append(v_idx)
                    nbrs ^= lsb
            
            result_mask |= _solve_component(adj, comp, n)
        
        return sorted([i for i in range(n) if result_mask & (1 << i)])


def _solve_component(adj, comp, n):
    comp_size = bin(comp).count('1')
    if comp_size <= 1:
        return comp
    
    # Check if no edges in component
    temp = comp
    has_edge = False
    while temp:
        lsb = temp & -temp
        u = lsb.bit_length() - 1
        if adj[u] & comp:
            has_edge = True
            break
        temp ^= lsb
    if not has_edge:
        return comp
    
    best_mask = 0
    best_size = 0
    
    # Greedy initial solution (min-degree heuristic)
    P = comp
    I = 0
    sz = 0
    while P:
        temp = P
        v = -1
        min_deg = n + 1
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            deg = bin(adj[u] & P).count('1')
            if deg < min_deg:
                min_deg = deg
                v = u
            temp ^= lsb
        I |= (1 << v)
        sz += 1
        P &= ~adj[v] & ~(1 << v)
    best_mask = I
    best_size = sz
    
    # Stack-based branch and bound to avoid Python recursion overhead
    # Each stack frame: (P, I, sz, phase)
    # phase 0: first visit - do reductions, compute bounds, push include branch
    # phase 1: after include branch returns, push exclude branch
    
    stack = [(comp, 0, 0, 0, -1)]  # (P, I, sz, phase, branch_v)
    
    while stack:
        P, I, sz, phase, branch_v = stack.pop()
        
        if phase == 1:
            # Exclude branch_v
            v_mask = 1 << branch_v
            P2 = P & ~v_mask
            stack.append((P2, I, sz, 0, -1))
            continue
        
        # phase == 0: Process this node
        # Reductions
        while True:
            changed = False
            temp = P
            while temp:
                lsb = temp & (-temp)
                temp ^= lsb
                if not (P & lsb):
                    continue
                u = lsb.bit_length() - 1
                neighbors = adj[u] & P
                if neighbors == 0:
                    I |= lsb
                    sz += 1
                    P ^= lsb
                    changed = True
                elif (neighbors & (neighbors - 1)) == 0:
                    I |= lsb
                    sz += 1
                    P ^= lsb
                    P &= ~neighbors
                    changed = True
            if not changed:
                break
        
        if P == 0:
            if sz > best_size:
                best_size = sz
                best_mask = I
            continue
        
        # Matching upper bound
        num_matched = 0
        available = P
        temp = P
        while temp:
            lsb = temp & (-temp)
            temp ^= lsb
            if not (available & lsb):
                continue
            u = lsb.bit_length() - 1
            nb = adj[u] & available
            if nb:
                partner = nb & (-nb)
                available ^= lsb | partner
                num_matched += 1
        
        remaining = bin(P).count('1')
        upper_bound = remaining - num_matched
        if sz + upper_bound <= best_size:
            continue
        
        # Branch on max-degree vertex
        temp = P
        branch_v = -1
        max_deg = -1
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            deg = bin(adj[u] & P).count('1')
            if deg > max_deg:
                max_deg = deg
                branch_v = u
            temp ^= lsb
        
        v_mask = 1 << branch_v
        
        # Push exclude branch (will be executed after include returns)
        stack.append((P, I, sz, 1, branch_v))
        
        # Push include branch (will be executed next)
        P_inc = P & ~adj[branch_v] & ~v_mask
        stack.append((P_inc, I | v_mask, sz + 1, 0, -1))
    
    return best_mask