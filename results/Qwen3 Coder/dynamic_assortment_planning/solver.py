from typing import Any, List, Dict
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[int]:
        """
        Solve the DAP exactly with a binary integer program (CP‑SAT).
        
        Returns
        -------
        List[int]
            offer[t] ∈ {‑1,0,…,N−1}.  ‑1 ⇒ offer nothing in period *t*.
        """
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        # Preprocessing: Remove products with zero probability or zero price
        valid_products = []
        for i in range(N):
            if prices[i] > 0 and any(probs[t][i] > 0 for t in range(T)):
                valid_products.append(i)
        
        model = cp_model.CpModel()

        # Decision vars: x[t,i] = 1 ⇔ offer product i in period t
        x = {}
        for t in range(T):
            for i in valid_products:
                x[(t, i)] = model.NewBoolVar(f"x_{t}_{i}")

        # Each period at most one product (only for valid products)
        for t in range(T):
            if valid_products:
                model.Add(sum(x[(t, i)] for i in valid_products) <= 1)
            else:
                # No valid products, so no offers
                pass

        # Capacity limits
        for i in valid_products:
            model.Add(sum(x[(t, i)] for t in range(T)) <= capacities[i])

        # Objective: expected revenue (only for valid products)
        obj_vars = []
        obj_coeffs = []
        for t in range(T):
            for i in valid_products:
                obj_vars.append(x[(t, i)])
                obj_coeffs.append(prices[i] * probs[t][i])
        if obj_vars:
            model.Maximize(cp_model.LinearExpr.WeightedSum(obj_vars, obj_coeffs))
        else:
            # No valid products, maximize a dummy variable
            dummy = model.NewIntVar(0, 0, "dummy")
            model.Maximize(dummy)

        # Set solver parameters for faster solving
        solver = cp_model.CpSolver()
        # Set solver parameters for faster solving
        solver = cp_model.CpSolver()
        # Reduce time limit and disable most optimizations for faster solving of small instances
        solver.parameters.max_time_in_seconds = 0.1
        solver.parameters.num_workers = 1
        solver.parameters.log_search_progress = False
        solver.parameters.cp_model_probing_level = 0
        solver.parameters.cp_model_presolve = False

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T

        offer = []
        for t in range(T):
            chosen = -1
            for i in valid_products:
                if solver.Value(x[(t, i)]) == 1:
                    chosen = i
                    break
            offer.append(chosen)
        return offer