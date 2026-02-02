from __future__ import annotations

from typing import Any, Optional
MASK64 = (1 << 64) - 1

def _splitmix64(x: int) -> int:
    """Deterministic 64-bit mixer (splitmix64)."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & MASK64
    return z ^ (z >> 31)

class _Graph:
    __slots__ = ("n", "adj", "deg", "adj_set")

    def __init__(
        self,
        n: int,
        adj: list[list[int]],
        deg: list[int],
        adj_set: Optional[list[set[int]]] = None,
    ) -> None:
        self.n = n
        self.adj = adj
        self.deg = deg
        self.adj_set = adj_set
class Solver:
    """
    Fast graph isomorphism via:
      - degree initialization
      - 1-WL color refinement (hashed neighbor multisets)
      - individualization/backtracking with strong local adjacency consistency

    Falls back to NetworkX VF2 only if needed.
    """

    def __init__(self) -> None:
        # Precomputed color->randomish 64-bit values for refinement hashing
        self._mix1: list[int] = []
        self._mix2: list[int] = []
        self._mix3: list[int] = []

    def _ensure_mix(self, k: int) -> None:
        cur = len(self._mix1)
        if cur >= k:
            return
        m1, m2, m3 = self._mix1, self._mix2, self._mix3
        for c in range(cur, k):
            m1.append(_splitmix64(c + 0x1234))
            m2.append(_splitmix64(c + 0x9ABCDEF0))
            m3.append(_splitmix64(c + 0x31415926))

    @staticmethod
    def _build_graph(n: int, edges: list[list[int]], need_set: bool) -> _Graph:
        adj = [[] for _ in range(n)]
        deg = [0] * n
        adj_set = [set() for _ in range(n)] if need_set else None
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
            if need_set:
                adj_set[u].add(v)  # type: ignore[index]
                adj_set[v].add(u)  # type: ignore[index]
        return _Graph(n=n, adj=adj, deg=deg, adj_set=adj_set)

    @staticmethod
    def _initial_colors(g1: _Graph, g2: _Graph) -> tuple[list[int], list[int], int]:
        # Degree-based partition with shared color ids.
        mp: dict[int, int] = {}
        nxt = 0
        c1 = [0] * g1.n
        c2 = [0] * g2.n
        for i, d in enumerate(g1.deg):
            cid = mp.get(d)
            if cid is None:
                cid = nxt
                mp[d] = nxt
                nxt += 1
            c1[i] = cid
        for i, d in enumerate(g2.deg):
            cid = mp.get(d)
            if cid is None:
                cid = nxt
                mp[d] = nxt
                nxt += 1
            c2[i] = cid
        return c1, c2, nxt

    def _refine_pair(
        self, g1: _Graph, g2: _Graph, c1: list[int], c2: list[int]
    ) -> tuple[list[int], list[int], int]:
        """
        Simultaneous 1-WL refinement on both graphs with aligned color ids.
        Signature: (current_color, degree, sum(mix1), sum(mix2), xor(mix3)).
        """
        n = g1.n
        adj1, adj2 = g1.adj, g2.adj
        deg1, deg2 = g1.deg, g2.deg

        while True:
            # Determine how many mix values are needed.
            maxc = 0
            for x in c1:
                if x > maxc:
                    maxc = x
            for x in c2:
                if x > maxc:
                    maxc = x
            k = maxc + 1
            self._ensure_mix(k)
            mix1, mix2, mix3 = self._mix1, self._mix2, self._mix3

            sig_to_id: dict[tuple[int, int, int, int, int], int] = {}
            nxt = 0
            new1 = [0] * n
            new2 = [0] * n

            for u in range(n):
                cu = c1[u]
                s1 = 0
                s2 = 0
                x3 = 0
                for w in adj1[u]:
                    cw = c1[w]
                    s1 = (s1 + mix1[cw]) & MASK64
                    s2 = (s2 + mix2[cw]) & MASK64
                    x3 ^= mix3[cw]
                sig = (cu, deg1[u], s1, s2, x3 & MASK64)
                cid = sig_to_id.get(sig)
                if cid is None:
                    cid = nxt
                    sig_to_id[sig] = nxt
                    nxt += 1
                new1[u] = cid

            for u in range(n):
                cu = c2[u]
                s1 = 0
                s2 = 0
                x3 = 0
                for w in adj2[u]:
                    cw = c2[w]
                    s1 = (s1 + mix1[cw]) & MASK64
                    s2 = (s2 + mix2[cw]) & MASK64
                    x3 ^= mix3[cw]
                sig = (cu, deg2[u], s1, s2, x3 & MASK64)
                cid = sig_to_id.get(sig)
                if cid is None:
                    cid = nxt
                    sig_to_id[sig] = nxt
                    nxt += 1
                new2[u] = cid

            if new1 == c1 and new2 == c2:
                return new1, new2, nxt
            c1, c2 = new1, new2

    @staticmethod
    def _classes(colors: list[int], mapping: list[int]) -> dict[int, list[int]]:
        d: dict[int, list[int]] = {}
        for i, c in enumerate(colors):
            if mapping[i] == -1:
                d.setdefault(c, []).append(i)
        return d

    @staticmethod
    def _check_new_pair(g1: _Graph, g2: _Graph, map1: list[int], map2: list[int], u: int, v: int) -> bool:
        # Degree constraint
        if g1.deg[u] != g2.deg[v]:
            return False

        adj2v = g2.adj_set[v]  # type: ignore[index]
        adj1u = g1.adj_set[u]  # type: ignore[index]

        # For edges in G1 from u to already-mapped nodes => must exist in G2.
        for nb in g1.adj[u]:
            vb = map1[nb]
            if vb != -1 and vb not in adj2v:
                return False

        # For edges in G2 from v to already-mapped nodes => must exist in G1.
        for nb2 in g2.adj[v]:
            pre = map2[nb2]
            if pre != -1 and pre not in adj1u:
                return False

        return True

    def _search(
        self,
        g1: _Graph,
        g2: _Graph,
        c1: list[int],
        c2: list[int],
        map1: list[int],
        map2: list[int],
    ) -> Optional[list[int]]:
        c1, c2, nxt_color = self._refine_pair(g1, g2, c1, c2)

        # Propagate singleton classes.
        while True:
            d1 = self._classes(c1, map1)
            d2 = self._classes(c2, map2)

            singles: list[tuple[int, int]] = []
            for col, nodes1 in d1.items():
                nodes2 = d2.get(col)
                if nodes2 is None or len(nodes2) != len(nodes1):
                    return None
                if len(nodes1) == 1:
                    singles.append((nodes1[0], nodes2[0]))

            if not singles:
                break

            changed = False
            for u, v in singles:
                if map1[u] != -1:
                    continue
                if map2[v] != -1:
                    return None
                if not self._check_new_pair(g1, g2, map1, map2, u, v):
                    return None
                map1[u] = v
                map2[v] = u
                c1[u] = nxt_color
                c2[v] = nxt_color
                nxt_color += 1
                changed = True

            if not changed:
                break
            c1, c2, nxt_color = self._refine_pair(g1, g2, c1, c2)

        if -1 not in map1:
            return map1

        # Pick smallest ambiguous color class.
        d1 = self._classes(c1, map1)
        d2 = self._classes(c2, map2)
        best_col = -1
        best_size = 1 << 30
        for col, nodes in d1.items():
            sz = len(nodes)
            if sz <= 1:
                continue
            if sz < best_size:
                best_size = sz
                best_col = col
                if sz == 2:
                    break
        if best_col == -1:
            return None

        nodes1 = d1[best_col]
        nodes2 = d2.get(best_col)
        if nodes2 is None or len(nodes2) != len(nodes1):
            return None

        u = nodes1[0]
        # Try candidates in corresponding class in G2.
        for v in nodes2:
            if map2[v] != -1:
                continue
            if not self._check_new_pair(g1, g2, map1, map2, u, v):
                continue

            map1_n = map1.copy()
            map2_n = map2.copy()
            c1_n = c1.copy()
            c2_n = c2.copy()

            map1_n[u] = v
            map2_n[v] = u

            new_col = max(max(c1_n), max(c2_n)) + 1
            c1_n[u] = new_col
            c2_n[v] = new_col

            res = self._search(g1, g2, c1_n, c2_n, map1_n, map2_n)
            if res is not None:
                return res

        return None

    @staticmethod
    def _verify_mapping(problem: dict[str, Any], mapping: list[int]) -> bool:
        n = problem["num_nodes"]
        if len(mapping) != n:
            return False
        if len(set(mapping)) != n:
            return False
        adj2 = [set() for _ in range(n)]
        for x, y in problem["edges_g2"]:
            adj2[x].add(y)
            adj2[y].add(x)
        for u, v in problem["edges_g1"]:
            mu = mapping[u]
            mv = mapping[v]
            if mv not in adj2[mu]:
                return False
        return True

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[int]]:
        n = problem["num_nodes"]
        if n <= 1:
            return {"mapping": list(range(n))}
        if not problem["edges_g1"] and not problem["edges_g2"]:
            return {"mapping": list(range(n))}

        # Build sets for both: used for strong consistency in backtracking.
        g1 = self._build_graph(n, problem["edges_g1"], need_set=True)
        g2 = self._build_graph(n, problem["edges_g2"], need_set=True)

        c1, c2, _ = self._initial_colors(g1, g2)
        map1 = [-1] * n
        map2 = [-1] * n

        res = self._search(g1, g2, c1, c2, map1, map2)
        if res is not None and self._verify_mapping(problem, res):
            return {"mapping": res}

        # Fallback to always-correct VF2.
        import networkx as nx

        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))
        G1.add_edges_from(problem["edges_g1"])
        G2.add_edges_from(problem["edges_g2"])
        gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
        iso_map = next(gm.isomorphisms_iter())
        return {"mapping": [iso_map[u] for u in range(n)]}