import itertools
from typing import Any, Dict, List
import networkx as nx

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        n = problem["num_nodes"]
        edges1 = problem["edges_g1"]
        edges2 = problem["edges_g2"]
        # build adjacency
        neigh1: List[List[int]] = [[] for _ in range(n)]
        neigh2: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges1:
            neigh1[u].append(v)
            neigh1[v].append(u)
        for u, v in edges2:
            neigh2[u].append(v)
            neigh2[v].append(u)
        # degrees
        deg1 = [len(neigh1[i]) for i in range(n)]
        deg2 = [len(neigh2[i]) for i in range(n)]
        # signatures: (deg, sorted neighbor-deg multiset)
        sig1: Dict[Any, List[int]] = {}
        sig2: Dict[Any, List[int]] = {}
        for u in range(n):
            key = (deg1[u], tuple(sorted(deg1[v] for v in neigh1[u])))
            sig1.setdefault(key, []).append(u)
        for v in range(n):
            key = (deg2[v], tuple(sorted(deg2[w] for w in neigh2[v])))
            sig2.setdefault(key, []).append(v)
        # initial mapping from unique signatures
        mapping = [-1] * n
        ambiguous: List[tuple] = []
        for key, group1 in sig1.items():
            group2 = sig2.get(key, [])
            if len(group1) != len(group2):
                # inconsistent or missing, fallback immediately
                break
            if len(group1) == 1:
                u0 = group1[0]
                v0 = group2[0]
                mapping[u0] = v0
            else:
                ambiguous.append((group1, group2))
        else:
            # try to resolve small ambiguous groups by brute-force
            ok = True
            for group1, group2 in ambiguous:
                l = len(group1)
                if l <= 6:
                    found = False
                    for perm in itertools.permutations(group2):
                        valid = True
                        # check consistency with already mapped neighbors
                        for ui, vi in zip(group1, perm):
                            for w in neigh1[ui]:
                                mv = mapping[w]
                                if mv >= 0 and mv not in neigh2[vi]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        # check edges within the group
                        for i in range(l):
                            ui, vi = group1[i], perm[i]
                            for j in range(i+1, l):
                                uj, vj = group1[j], perm[j]
                                e1 = (uj in neigh1[ui])
                                e2 = (vj in neigh2[vi])
                                if e1 != e2:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        # accept
                        for ui, vi in zip(group1, perm):
                            mapping[ui] = vi
                        found = True
                        break
                    if not found:
                        ok = False
                        break
                else:
                    ok = False
                    break
            # if all assigned, return
            if ok and all(x >= 0 for x in mapping):
                return {"mapping": mapping}
        # fallback to NetworkX VF2
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))
        for u, v in edges1:
            G1.add_edge(u, v)
        for u, v in edges2:
            G2.add_edge(u, v)
        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        if not gm.is_isomorphic():
            return {"mapping": [-1]*n}
        iso = next(gm.isomorphisms_iter())
        return {"mapping": [iso[u] for u in range(n)]}