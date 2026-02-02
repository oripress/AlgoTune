import networkx as nx
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Hybrid solver:
        - If graph is dense (complement is sparse), use NetworkX Max Clique on complement.
        - If graph is sparse (complement is dense), use PySAT Max Independent Set on original.
        """
        n = len(problem)
        if n == 0:
            return []
            
        # Count edges to estimate density
        num_edges = 0
        for i in range(n):
            row = problem[i]
            for j in range(i + 1, n):
                if row[j] == 1:
                    num_edges += 1
                    
        density = 2 * num_edges / (n * (n - 1)) if n > 1 else 0
        
        # Threshold for switching.
        # If density is high (> 0.6), complement is sparse -> NetworkX is fast.
        # If density is low (< 0.6), original is sparse -> PySAT is fast.
        # I'll set it to 0.6 based on intuition.
        
        if density > 0.6:
            # Use NetworkX on complement
            edges = []
            for i in range(n):
                row = problem[i]
                for j in range(i + 1, n):
                    if row[j] == 0:
                        edges.append((i, j))
            
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)
            clique, _ = nx.max_weight_clique(G, weight=None)
            return sorted(clique)
        else:
            # Use PySAT on original
            wcnf = WCNF()
            for i in range(n):
                wcnf.append([i + 1], weight=1)
            
            for i in range(n):
                row = problem[i]
                for j in range(i + 1, n):
                    if row[j] == 1:
                        wcnf.append([-(i + 1), -(j + 1)])
                        
            with RC2(wcnf) as rc2:
                model = rc2.compute()
                
            if model:
                return [i - 1 for i in model if i > 0]
            return []