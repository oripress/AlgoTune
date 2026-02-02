import numpy as np
from numba import njit
from numba.typed import List

@njit(cache=True)
def greedy_coloring(adj, n):
    """Greedy coloring to get upper bound on clique size."""
    colors = np.zeros(n, dtype=np.int32)
    for v in range(n):
        used = set()
        for u in range(n):
            if adj[v, u] and colors[u] > 0:
                used.add(colors[u])
        c = 1
        while c in used:
            c += 1
        colors[v] = c
    return colors

@njit(cache=True)
def bron_kerbosch_pivot(adj, R, P, X, best, n):
    """Bron-Kerbosch with pivot selection."""
    if len(P) == 0 and len(X) == 0:
        if len(R) > len(best):
            best = R.copy()
        return best
    
    if len(R) + len(P) <= len(best):
        return best
    
    # Choose pivot with max connections to P
    pivot = -1
    max_conn = -1
    for u in P:
        conn = 0
        for v in P:
            if adj[u, v]:
                conn += 1
        if conn > max_conn:
            max_conn = conn
            pivot = u
    for u in X:
        conn = 0
        for v in P:
            if adj[u, v]:
                conn += 1
        if conn > max_conn:
            max_conn = conn
            pivot = u
    
    # Vertices in P not adjacent to pivot
    candidates = List()
    for v in P:
        if not adj[pivot, v]:
            candidates.append(v)
    
    for v in candidates:
        new_R = List()
        for x in R:
            new_R.append(x)
        new_R.append(v)
        
        new_P = List()
        for u in P:
            if adj[v, u]:
                new_P.append(u)
        
        new_X = List()
        for u in X:
            if adj[v, u]:
                new_X.append(u)
        
        best = bron_kerbosch_pivot(adj, new_R, new_P, new_X, best, n)
        
        # Remove v from P, add to X
        new_P2 = List()
        for u in P:
            if u != v:
                new_P2.append(u)
        P = new_P2
        X.append(v)
    
    return best

@njit(cache=True)
def find_max_clique(adj, n):
    """Find maximum clique using Bron-Kerbosch."""
    R = List()
    R.append(-1)
    R.pop()
    
    P = List()
    for i in range(n):
        P.append(i)
    
    X = List()
    X.append(-1)
    X.pop()
    
    best = List()
    best.append(-1)
    best.pop()
    
    return bron_kerbosch_pivot(adj, R, P, X, best, n)

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)
        
        if n == 0 or m == 0:
            return []
        
        # Convert to numpy arrays
        A_np = np.array(A, dtype=np.int8)
        B_np = np.array(B, dtype=np.int8)
        
        # Build modular product graph adjacency matrix
        nm = n * m
        adj = np.zeros((nm, nm), dtype=np.bool_)
        
        for i in range(n):
            for j in range(i + 1, n):
                aij = A_np[i, j]
                for p in range(m):
                    for q in range(m):
                        if p != q and aij == B_np[p, q]:
                            u = i * m + p
                            v = j * m + q
                            adj[u, v] = True
                            adj[v, u] = True
        
        # Find maximum clique
        clique = find_max_clique(adj, nm)
        
        # Convert back to (i, p) pairs
        result = []
        for node in clique:
            i = node // m
            p = node % m
            result.append((i, p))
        
        return result