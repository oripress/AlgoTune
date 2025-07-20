from typing import Any
import pulp

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        """
        Solve the Dynamic Assortment Planning problem using linear programming.
        
        Returns a list of length T where element t is -1 (offer nothing) 
        or a product index in 0...N-1.
        """
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]
        
        # Create the LP model
        model = pulp.LpProblem("DAP", pulp.LpMaximize)
        
        # Decision variables: x[t][i] = 1 if we offer product i in period t
        x = {}
        for t in range(T):
            for i in range(N):
                x[t, i] = pulp.LpVariable(f"x_{t}_{i}", cat='Binary')
        
        # Objective: maximize expected revenue
        objective = []
        for t in range(T):
            for i in range(N):
                objective.append(prices[i] * probs[t][i] * x[t, i])
        model += pulp.lpSum(objective)
        
        # Constraint 1: At most one product per period
        for t in range(T):
            model += pulp.lpSum(x[t, i] for i in range(N)) <= 1
        
        # Constraint 2: Capacity constraints
        for i in range(N):
            model += pulp.lpSum(x[t, i] for t in range(T)) <= capacities[i]
        
        # Solve the model
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract the solution
        offer = []
        for t in range(T):
            chosen = -1
            for i in range(N):
                if x[t, i].value() and x[t, i].value() > 0.5:  # Check if binary var is 1
                    chosen = i
                    break
            offer.append(chosen)
        
        return offer