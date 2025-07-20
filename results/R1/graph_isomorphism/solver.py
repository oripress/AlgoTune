import networkx as nx
from typing import Any
from collections import defaultdict

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]
        
        # Build adjacency lists
        adj1 = [[] for _ in range(n)]
        for u, v in edges_g1:
            adj1[u].append(v)
            adj1[v].append(u)
        
        adj2 = [[] for _ in range(n)]
        for u, v in edges_g2:
            adj2[u].append(v)
            adj2[v].append(u)
        
        # Compute initial degrees (0th iteration labels)
        deg1 = [len(neighbors) for neighbors in adj1]
        deg2 = [len(neighbors) for neighbors in adj2]
        
        # Compute 1st iteration Weisfeiler-Lehman labels
        wl1 = [tuple(sorted([deg1[j] for j in adj1[i]])) for i in range(n)]
        wl2 = [tuple(sorted([deg2[j] for j in adj2[i]])) for i in range(n)]
        
        # Create combined signatures (degree + WL1 label)
        sigs1 = [(deg1[i], wl1[i]) for i in range(n)]
        sigs2 = [(deg2[i], wl2[i]) for i in range(n)]
        
        # Check if signatures are unique and match
        if set(sigs1) == set(sigs2) and len(set(sigs2)) == n:
            sig_to_node = {sig: i for i, sig in enumerate(sigs2)}
            mapping = [sig_to_node[sig] for sig in sigs1]
            return {"mapping": mapping}
        
        # Build graphs for VF2++
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))
        G1.add_edges_from(edges_g1)
        G2.add_edges_from(edges_g2)
        
        # Use VF2++ isomorphism
        try:
            from networkx.algorithms.isomorphism import vf2pp_isomorphism
            result = vf2pp_isomorphism(G1, G2)
            if isinstance(result, dict):
                iso_map = result
            else:
                iso_map = next(result, None)
            
            if iso_map is None:
                # Fallback to GraphMatcher
                gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
                if not gm.is_isomorphic():
                    return {"mapping": [-1] * n}
                iso_map = next(gm.isomorphisms_iter())
            return {"mapping": [iso_map[i] for i in range(n)]}
        except ImportError:
            # Fallback to GraphMatcher if VF2++ not available
            gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
            if not gm.is_isomorphic():
                return {"mapping": [-1] * n}
            iso_map = next(gm.isomorphisms_iter())
            return {"mapping": [iso_map[i] for i in range(n)]}