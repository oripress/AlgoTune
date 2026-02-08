from ortools.sat.python import cp_model
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        customers = [i for i in range(n) if i != depot]
        nc = len(customers)

        if nc == 0:
            return [[depot, depot] for _ in range(K)]

        # If K > nc, we can't give each vehicle a customer
        # Use effective_K = min(K, nc) for the circuit, pad with empty routes
        effective_K = min(K, nc)

        model = cp_model.CpModel()
        arcs = []
        obj_vars = []
        obj_coeffs = []

        # Customer -> Customer arcs
        for i in range(nc):
            ci = customers[i]
            for j in range(nc):
                if i != j:
                    cj = customers[j]
                    lit = model.NewBoolVar("")
                    arcs.append((i, j, lit))
                    d = int(round(D[ci][cj]))
                    if d:
                        obj_vars.append(lit)
                        obj_coeffs.append(d)

        # Depot copy k -> Customer i
        for k in range(effective_K):
            dk = nc + k
            for i in range(nc):
                ci = customers[i]
                lit = model.NewBoolVar("")
                arcs.append((dk, i, lit))
                d = int(round(D[depot][ci]))
                if d:
                    obj_vars.append(lit)
                    obj_coeffs.append(d)

        # Customer i -> Depot copy k
        for i in range(nc):
            ci = customers[i]
            for k in range(effective_K):
                dk = nc + k
                lit = model.NewBoolVar("")
                arcs.append((i, dk, lit))
                d = int(round(D[ci][depot]))
                if d:
                    obj_vars.append(lit)
                    obj_coeffs.append(d)

        # No depot-to-depot arcs: every vehicle must visit at least one customer
        # Chain depot copies in the circuit
        if effective_K > 1:
            # We need to connect depot copies in a chain within the circuit
            # But WITHOUT allowing empty skips
            # Actually for the circuit to work, each depot copy must have
            # exactly one incoming and one outgoing arc.
            # If we don't add depot-to-depot, each must connect to/from customers.
            pass

        model.AddCircuit(arcs)

        if obj_vars:
            model.Minimize(cp_model.LinearExpr.WeightedSum(obj_vars, obj_coeffs))

        solver = cp_model.CpSolver()
        solver.parameters.num_workers = 1
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            nxt = {}
            for h, t, lit in arcs:
                if solver.Value(lit):
                    nxt[h] = t

            routes = []
            for k in range(effective_K):
                dk = nc + k
                cur = nxt.get(dk)
                if cur is not None and cur < nc:
                    route = [depot]
                    while cur < nc:
                        route.append(customers[cur])
                        cur = nxt[cur]
                    route.append(depot)
                    routes.append(route)

            # Pad with empty routes if K > effective_K
            while len(routes) < K:
                routes.append([depot, depot])

            return routes

        return []