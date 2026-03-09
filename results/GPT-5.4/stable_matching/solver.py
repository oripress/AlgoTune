from typing import Any

class Solver:
    __slots__ = ()

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list[int]]:
        prop_raw = problem["proposer_prefs"]
        recv_raw = problem["receiver_prefs"]

        if type(prop_raw) is list and type(recv_raw) is list:
            proposer_prefs = prop_raw
            receiver_prefs = recv_raw
            n = len(proposer_prefs)

            recv_rank = [[0] * n for _ in range(n)]
            for r in range(n):
                rank_row = recv_rank[r]
                prefs = receiver_prefs[r]
                for rank, p in enumerate(prefs):
                    rank_row[p] = rank

            next_prop = [0] * n
            recv_match = [-1] * n
            free = list(range(n))

            free_pop = free.pop
            free_append = free.append
            recv_rank_local = recv_rank
            recv_match_local = recv_match
            next_prop_local = next_prop
            proposer_prefs_local = proposer_prefs

            while free:
                p = free_pop()
                i = next_prop_local[p]
                r = proposer_prefs_local[p][i]
                next_prop_local[p] = i + 1

                cur = recv_match_local[r]
                if cur == -1:
                    recv_match_local[r] = p
                elif recv_rank_local[r][p] < recv_rank_local[r][cur]:
                    recv_match_local[r] = p
                    free_append(cur)
                else:
                    free_append(p)

            matching = [0] * n
            for r, p in enumerate(recv_match_local):
                matching[p] = r
            return {"matching": matching}

        if isinstance(prop_raw, dict):
            n = len(prop_raw)
            proposer_prefs = [prop_raw[i] for i in range(n)]
        else:
            proposer_prefs = list(prop_raw)
            n = len(proposer_prefs)

        if isinstance(recv_raw, dict):
            receiver_prefs = [recv_raw[i] for i in range(n)]
        else:
            receiver_prefs = list(recv_raw)

        recv_rank = [[0] * n for _ in range(n)]
        for r in range(n):
            rank_row = recv_rank[r]
            prefs = receiver_prefs[r]
            for rank, p in enumerate(prefs):
                rank_row[p] = rank

        next_prop = [0] * n
        recv_match = [-1] * n
        free = list(range(n))

        free_pop = free.pop
        free_append = free.append
        recv_rank_local = recv_rank
        recv_match_local = recv_match
        next_prop_local = next_prop
        proposer_prefs_local = proposer_prefs

        while free:
            p = free_pop()
            i = next_prop_local[p]
            r = proposer_prefs_local[p][i]
            next_prop_local[p] = i + 1

            cur = recv_match_local[r]
            if cur == -1:
                recv_match_local[r] = p
            elif recv_rank_local[r][p] < recv_rank_local[r][cur]:
                recv_match_local[r] = p
                free_append(cur)
            else:
                free_append(p)

        matching = [0] * n
        for r, p in enumerate(recv_match_local):
            matching[p] = r

        return {"matching": matching}