import networkx as nx

class Solver:
    def solve(self, problem: dict[str, list[list[int]]]) -> list[tuple[int, int]]:
        A = problem["A"]
        B = problem["B"]
        n, m = len(A), len(B)
        
        # Pre-compute edges by value for faster lookup
        A_edges_0 = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i][j] == 0]
        A_edges_1 = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i][j] == 1]
        B_edges_0 = [(p, q) for p in range(m) for q in range(p + 1, m) if B[p][q] == 0]
        B_edges_1 = [(p, q) for p in range(m) for q in range(p + 1, m) if B[p][q] == 1]
        
        # Build modular product graph
        G = nx.Graph()
        
        # Add edges for matching 0s
        for i, j in A_edges_0:
            for p, q in B_edges_0:
                G.add_edge((i, p), (j, q))
                G.add_edge((i, q), (j, p))
        
        # Add edges for matching 1s
        for i, j in A_edges_1:
            for p, q in B_edges_1:
                G.add_edge((i, p), (j, q))
                G.add_edge((i, q), (j, p))
        
        # Find maximum clique
        return max(nx.find_cliques(G), key=len, default=[])