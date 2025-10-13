from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[List[int]]], **kwargs) -> Any:
        """
        Compute the global efficiency of an undirected graph given as an adjacency list.

        The reference implementation constructs a NetworkX Graph by adding an undirected edge
        for each pair (u, v) encountered where u < v while iterating over the input adjacency list.
        To match this precisely (including handling of asymmetric adjacency lists), we replicate
        that edge construction rule, then compute global efficiency via repeated BFS.

        Args:
            problem: Dict with key "adjacency_list": list of sorted neighbor lists.

        Returns:
            Dict with key "global_efficiency" and a float value.
        """
        adj_in = problem.get("adjacency_list", [])
        n = len(adj_in)

        # Edge cases
        if n <= 1:
            return {"global_efficiency": 0.0}

        # Build undirected adjacency as per the reference:
        # add edge (u, v) iff u < v and v is in adj_in[u].
        # Deduplicate consecutive duplicates in adj_in[u] (since it's sorted).
        adj: List[List[int]] = [[] for _ in range(n)]
        for u in range(n):
            prev = -1
            for v in adj_in[u]:
                if v == prev:
                    continue  # skip duplicates within the same list
                prev = v
                if u < v:
                    adj[u].append(v)
                    adj[v].append(u)

        # Prepare BFS structures
        seen = [0] * n
        dist = [0] * n
        stamp = 0

        inv_cache = [0.0, 1.0]  # inv_cache[d] = 1.0/d for d >= 1

        sum_inv_once = 0.0  # sum over unordered pairs (u < v) of 1/d(u, v)

        # BFS from each node, only accumulate contributions to nodes with index > source
        for s in range(n):
            stamp += 1
            seen[s] = stamp
            dist[s] = 0
            q = [s]
            head = 0

            while head < len(q):
                u = q[head]
                head += 1
                du_next = dist[u] + 1
                for v in adj[u]:
                    if seen[v] != stamp:
                        seen[v] = stamp
                        dist[v] = du_next
                        q.append(v)
                        # Only count each unordered pair once (s < v)
                        if v > s:
                            while du_next >= len(inv_cache):
                                inv_cache.append(1.0 / len(inv_cache))
                            sum_inv_once += inv_cache[du_next]

        denom = n * (n - 1)
        efficiency = (2.0 * sum_inv_once) / denom if denom > 0 else 0.0
        return {"global_efficiency": float(efficiency)}