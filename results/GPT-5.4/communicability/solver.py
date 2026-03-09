from __future__ import annotations

import math
from typing import Any

import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        adj_list = problem["adjacency_list"]
        n = len(adj_list)

        if n == 0:
            return {"communicability": {}}

        exp_neg1 = math.exp(-1.0)

        # Match the reference graph reconstruction:
        # add undirected edge (u, v) only when encountered with u < v.
        # Ignore duplicate neighbor entries.
        graph_adj = [[] for _ in range(n)]
        edge_count = 0
        for u, neighbors in enumerate(adj_list):
            last_v = -1
            for v in neighbors:
                if v == last_v:
                    continue
                last_v = v
                if u < v:
                    graph_adj[u].append(v)
                    graph_adj[v].append(u)
                    edge_count += 1

        if edge_count == 0:
            zero_row = {v: 0.0 for v in range(n)}
            result = {}
            for u in range(n):
                row = zero_row.copy()
                row[u] = 1.0
                result[u] = row
            return {"communicability": result}

        if edge_count == n * (n - 1) // 2:
            off_diag = (float(np.exp(n - 1.0)) - exp_neg1) / n
            diag = exp_neg1 + off_diag
            base_row = {v: off_diag for v in range(n)}
            result = {}
            for u in range(n):
                row = base_row.copy()
                row[u] = diag
                result[u] = row
            return {"communicability": result}

        degrees = [len(nei) for nei in graph_adj]

        visited = [False] * n
        components: list[list[int]] = []
        for start in range(n):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in graph_adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            components.append(comp)

        max_component_size = max(len(comp) for comp in components)
        direct_output = len(components) > 1 and max_component_size <= 16

        if direct_output:
            zero_row = {v: 0.0 for v in range(n)}
            result = {u: zero_row.copy() for u in range(n)}
            comm = None
        else:
            result = None
            comm = np.zeros((n, n), dtype=np.float64)

        def write_block(nodes: list[int], sub_comm: np.ndarray) -> None:
            if result is not None:
                for i, u in enumerate(nodes):
                    row_u = result[u]
                    row_vals = sub_comm[i]
                    for j, v in enumerate(nodes):
                        row_u[v] = row_vals[j]
            else:
                comm[np.ix_(nodes, nodes)] = sub_comm

        cosh1 = math.cosh(1.0)
        sinh1 = math.sinh(1.0)

        for nodes in components:
            c = len(nodes)

            if c == 1:
                u = nodes[0]
                if result is not None:
                    result[u][u] = 1.0
                else:
                    comm[u, u] = 1.0
                continue

            if c == 2:
                u, v = nodes
                if result is not None:
                    row_u = result[u]
                    row_v = result[v]
                    row_u[u] = cosh1
                    row_u[v] = sinh1
                    row_v[u] = sinh1
                    row_v[v] = cosh1
                else:
                    comm[u, u] = cosh1
                    comm[v, v] = cosh1
                    comm[u, v] = sinh1
                    comm[v, u] = sinh1
                continue

            if all(degrees[u] == c - 1 for u in nodes):
                off_diag = (float(np.exp(c - 1.0)) - exp_neg1) / c
                diag = exp_neg1 + off_diag
                sub_comm = np.full((c, c), off_diag, dtype=np.float64)
                np.fill_diagonal(sub_comm, diag)
                write_block(nodes, sub_comm)
                continue

            center = -1
            is_star = True
            for u in nodes:
                deg = degrees[u]
                if deg == c - 1:
                    if center != -1:
                        is_star = False
                        break
                    center = u
                elif deg != 1:
                    is_star = False
                    break

            if is_star and center != -1:
                m = c - 1
                s = math.sqrt(m)
                center_center = math.cosh(s)
                center_leaf = math.sinh(s) / s
                leaf_off = (math.cosh(s) - 1.0) / m
                leaf_diag = 1.0 + leaf_off

                sub_comm = np.full((c, c), leaf_off, dtype=np.float64)
                np.fill_diagonal(sub_comm, leaf_diag)

                center_idx = nodes.index(center)
                sub_comm[center_idx, :] = center_leaf
                sub_comm[:, center_idx] = center_leaf
                sub_comm[center_idx, center_idx] = center_center
                write_block(nodes, sub_comm)
                continue

            pos = {node: i for i, node in enumerate(nodes)}
            sub_a = np.zeros((c, c), dtype=np.float64)
            for u in nodes:
                iu = pos[u]
                for v in graph_adj[u]:
                    if u < v:
                        iv = pos[v]
                        sub_a[iu, iv] = 1.0
                        sub_a[iv, iu] = 1.0

            vals, vecs = np.linalg.eigh(sub_a)
            sub_comm = (vecs * np.exp(vals)) @ vecs.T
            write_block(nodes, sub_comm)

        if result is not None:
            return {"communicability": result}

        final_result = {}
        for u in range(n):
            final_result[u] = dict(enumerate(comm[u].tolist()))
        return {"communicability": final_result}