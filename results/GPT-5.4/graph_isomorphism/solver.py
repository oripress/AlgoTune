from __future__ import annotations

from typing import Any

class Solver:
    def __init__(self) -> None:
        self._adj1: list[list[int]] = []
        self._adj2: list[list[int]] = []
        self._n = 0
        self._steps = 0
        self._step_cap = 0

    def _refine(
        self, colors1: list[int], colors2: list[int]
    ) -> tuple[
        list[int] | None,
        list[int] | None,
        list[list[int]] | None,
        list[list[int]] | None,
    ]:
        n = self._n
        adj1 = self._adj1
        adj2 = self._adj2

        while True:
            max_color = -1
            for c in colors1:
                if c > max_color:
                    max_color = c
            for c in colors2:
                if c > max_color:
                    max_color = c
            cnts = [0] * (max_color + 1 if max_color >= 0 else 0)

            sig_to_color: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
            next_color = 0
            new1 = [0] * n
            new2 = [0] * n

            for u in range(n):
                seen: list[int] = []
                for v in adj1[u]:
                    c = colors1[v]
                    if cnts[c] == 0:
                        seen.append(c)
                    cnts[c] += 1
                if len(seen) > 1:
                    seen.sort()
                sig = (colors1[u], tuple((c, cnts[c]) for c in seen))
                col = sig_to_color.get(sig)
                if col is None:
                    col = next_color
                    sig_to_color[sig] = col
                    next_color += 1
                new1[u] = col
                for c in seen:
                    cnts[c] = 0

            for u in range(n):
                seen = []
                for v in adj2[u]:
                    c = colors2[v]
                    if cnts[c] == 0:
                        seen.append(c)
                    cnts[c] += 1
                if len(seen) > 1:
                    seen.sort()
                sig = (colors2[u], tuple((c, cnts[c]) for c in seen))
                col = sig_to_color.get(sig)
                if col is None:
                    col = next_color
                    sig_to_color[sig] = col
                    next_color += 1
                new2[u] = col
                for c in seen:
                    cnts[c] = 0

            classes1 = [[] for _ in range(next_color)]
            classes2 = [[] for _ in range(next_color)]
            for u, c in enumerate(new1):
                classes1[c].append(u)
            for u, c in enumerate(new2):
                classes2[c].append(u)

            for c in range(next_color):
                if len(classes1[c]) != len(classes2[c]):
                    return None, None, None, None

            if new1 == colors1 and new2 == colors2:
                return new1, new2, classes1, classes2
            colors1 = new1
            colors2 = new2

    def _mapping_from_colors(self, colors1: list[int], colors2: list[int]) -> list[int]:
        n = self._n
        mapping = [0] * n
        color_to_node2 = [0] * n
        for v, c in enumerate(colors2):
            color_to_node2[c] = v
        for u, c in enumerate(colors1):
            mapping[u] = color_to_node2[c]
        return mapping

    def _validate_mapping(self, mapping: list[int]) -> bool:
        adj2_sets = [set(nei) for nei in self._adj2]
        for u in range(self._n):
            mu = mapping[u]
            for v in self._adj1[u]:
                if mapping[v] not in adj2_sets[mu]:
                    return False
        return True

    def _fast_unique_mapping(
        self, colors1: list[int], colors2: list[int]
    ) -> list[int] | None:
        n = self._n
        adj1 = self._adj1
        adj2 = self._adj2
        mask = (1 << 64) - 1

        for _ in range(4):
            sig_to_color: dict[tuple[int, int, int, int, int], int] = {}
            next_color = 0
            new1 = [0] * n
            new2 = [0] * n
            counts1: dict[int, int] = {}
            counts2: dict[int, int] = {}

            for u in range(n):
                s1 = 0
                s2 = 0
                s3 = 0
                for v in adj1[u]:
                    x = colors1[v] + 1
                    y = (x * 11400714819323198485) & mask
                    s1 = (s1 + y) & mask
                    s2 = (s2 + ((y * x) & mask)) & mask
                    s3 = (s3 + ((y * ((x * x) & mask)) & mask)) & mask
                sig = (colors1[u], len(adj1[u]), s1, s2, s3)
                col = sig_to_color.get(sig)
                if col is None:
                    col = next_color
                    sig_to_color[sig] = col
                    next_color += 1
                new1[u] = col
                counts1[col] = counts1.get(col, 0) + 1

            for u in range(n):
                s1 = 0
                s2 = 0
                s3 = 0
                for v in adj2[u]:
                    x = colors2[v] + 1
                    y = (x * 11400714819323198485) & mask
                    s1 = (s1 + y) & mask
                    s2 = (s2 + ((y * x) & mask)) & mask
                    s3 = (s3 + ((y * ((x * x) & mask)) & mask)) & mask
                sig = (colors2[u], len(adj2[u]), s1, s2, s3)
                col = sig_to_color.get(sig)
                if col is None:
                    col = next_color
                    sig_to_color[sig] = col
                    next_color += 1
                new2[u] = col
                counts2[col] = counts2.get(col, 0) + 1

            if counts1 != counts2:
                return None

            if next_color == n:
                mapping = self._mapping_from_colors(new1, new2)
                if self._validate_mapping(mapping):
                    return mapping
                return None

            if new1 == colors1 and new2 == colors2:
                break
            colors1 = new1
            colors2 = new2

        return None

    def _search_refined(
        self,
        refined1: list[int],
        refined2: list[int],
        classes1: list[list[int]],
        classes2: list[list[int]],
    ) -> list[int] | None:
        self._steps += 1
        if self._steps > self._step_cap:
            return None

        if len(classes1) == self._n:
            return self._mapping_from_colors(refined1, refined2)

        chosen_color = -1
        chosen_bucket1: list[int] | None = None
        best_size = self._n + 1
        for c, bucket in enumerate(classes1):
            size = len(bucket)
            if 1 < size < best_size:
                best_size = size
                chosen_color = c
                chosen_bucket1 = bucket

        if chosen_bucket1 is None:
            return None

        u = chosen_bucket1[0]
        bucket2 = classes2[chosen_color]
        individualized_color = len(classes1)

        branches: list[
            tuple[
                int,
                list[int],
                list[int],
                list[list[int]],
                list[list[int]],
            ]
        ] = []
        for v in bucket2:
            next1 = refined1.copy()
            next2 = refined2.copy()
            next1[u] = individualized_color
            next2[v] = individualized_color
            out1, out2, next_classes1, next_classes2 = self._refine(next1, next2)
            if (
                out1 is None
                or out2 is None
                or next_classes1 is None
                or next_classes2 is None
            ):
                continue
            if len(next_classes1) == self._n:
                return self._mapping_from_colors(out1, out2)
            branches.append((len(next_classes1), out1, out2, next_classes1, next_classes2))

        branches.sort(key=lambda item: item[0], reverse=True)
        for _, out1, out2, next_classes1, next_classes2 in branches:
            result = self._search_refined(out1, out2, next_classes1, next_classes2)
            if result is not None:
                return result
        return None

    def _search(
        self,
        colors1: list[int],
        colors2: list[int],
    ) -> list[int] | None:
        refined1, refined2, classes1, classes2 = self._refine(colors1, colors2)
        if refined1 is None or refined2 is None or classes1 is None or classes2 is None:
            return None
        return self._search_refined(refined1, refined2, classes1, classes2)

    def _fallback(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        import networkx as nx

        n = problem["num_nodes"]
        g1 = nx.Graph()
        g2 = nx.Graph()
        g1.add_nodes_from(range(n))
        g2.add_nodes_from(range(n))
        g1.add_edges_from(problem["edges_g1"])
        g2.add_edges_from(problem["edges_g2"])

        gm = nx.algorithms.isomorphism.GraphMatcher(g1, g2)
        try:
            iso = next(gm.isomorphisms_iter())
            return {"mapping": [iso[i] for i in range(n)]}
        except StopIteration:
            return {"mapping": list(range(n))}

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        n = problem["num_nodes"]
        self._n = n

        if n <= 1:
            return {"mapping": list(range(n))}

        total_pairs = n * (n - 1) // 2
        m = len(problem["edges_g1"])
        if m == 0 or m == total_pairs:
            return {"mapping": list(range(n))}

        adj1 = [[] for _ in range(n)]
        adj2 = [[] for _ in range(n)]
        for u, v in problem["edges_g1"]:
            adj1[u].append(v)
            adj1[v].append(u)
        for u, v in problem["edges_g2"]:
            adj2[u].append(v)
            adj2[v].append(u)

        self._adj1 = adj1
        self._adj2 = adj2

        deg1 = [len(x) for x in adj1]
        deg2 = [len(x) for x in adj2]

        count1: dict[int, int] = {}
        count2: dict[int, int] = {}
        for d in deg1:
            count1[d] = count1.get(d, 0) + 1
        for d in deg2:
            count2[d] = count2.get(d, 0) + 1
        if count1 != count2:
            return self._fallback(problem)

        unique_deg = sorted(count1)
        deg_to_color = {d: i for i, d in enumerate(unique_deg)}
        colors1 = [deg_to_color[d] for d in deg1]
        colors2 = [deg_to_color[d] for d in deg2]

        self._steps = 0
        self._step_cap = 200 + 20 * n
        mapping = self._search(colors1, colors2)
        if mapping is not None:
            return {"mapping": mapping}
        return self._fallback(problem)