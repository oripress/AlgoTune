import sys
sys.setrecursionlimit(100000)

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        # Build adjacency bitmasks (no self-loops)
        adj = [0] * n
        for i in range(n):
            row = problem[i]
            m = 0
            for j in range(n):
                if row[j] and i != j:
                    m |= 1 << j
            adj[i] = m
        
        # Greedy initial clique from multiple starting vertices
        best_size = 0
        best_clique = []
        
        degs = [bin(adj[i]).count('1') for i in range(n)]
        sorted_v = sorted(range(n), key=lambda x: -degs[x])
        
        for start in sorted_v[:min(n, 10)]:
            clq = [start]
            cand = adj[start]
            while cand:
                bv = -1
                bd = -1
                p = cand
                while p:
                    lsb = p & (-p)
                    v = lsb.bit_length() - 1
                    p ^= lsb
                    d = bin(adj[v] & cand).count('1')
                    if d > bd:
                        bd = d
                        bv = v
                clq.append(bv)
                cand &= adj[bv]
            if len(clq) > best_size:
                best_size = len(clq)
                best_clique = clq[:]
        
        # Reduce graph: remove vertices that can't be in a larger clique
        all_v = (1 << n) - 1
        changed = True
        while changed:
            changed = False
            p = all_v
            while p:
                lsb = p & (-p)
                v = lsb.bit_length() - 1
                p ^= lsb
                if bin(adj[v] & all_v).count('1') < best_size:
                    all_v &= ~lsb
                    changed = True
        
        # Check if B&B is needed
        if bin(all_v).count('1') <= best_size:
            return sorted(best_clique)
        
        # MCQ-style Branch and Bound with coloring bound
        def expand(cs, P, clq):
            nonlocal best_size, best_clique
            
            if not P:
                if cs > best_size:
                    best_size = cs
                    best_clique = clq[:]
                return
            
            # Quick popcount bound
            if cs + bin(P).count('1') <= best_size:
                return
            
            # Greedy coloring for tighter upper bound
            cm = []  # color class bitmasks
            vl = []  # (color_num, vertex)
            p = P
            while p:
                lsb = p & (-p)
                v = lsb.bit_length() - 1
                p ^= lsb
                av = adj[v]
                cn = 0
                for ci in range(len(cm)):
                    if not (av & cm[ci]):
                        cm[ci] |= lsb
                        cn = ci + 1
                        break
                if cn == 0:
                    cm.append(lsb)
                    cn = len(cm)
                vl.append((cn, v))
            
            if cs + len(cm) <= best_size:
                return
            
            vl.sort()
            
            for i in range(len(vl) - 1, -1, -1):
                cn, v = vl[i]
                if cs + cn <= best_size:
                    return
                nP = P & adj[v]
                clq.append(v)
                expand(cs + 1, nP, clq)
                clq.pop()
                P &= ~(1 << v)
        
        expand(0, all_v, [])
        
        return sorted(best_clique)