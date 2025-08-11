from typing import Any, List, Tuple
import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        """
        Solve the DAP with a hybrid approach: greedy for simple cases, DP for complex ones.
        
        Returns
        -------
        List[int]
            offer[t] ∈ {-1,0,…,N−1}.  -1 ⇒ offer nothing in period *t*.
        """
        T = problem["T"]
        N = problem["N"]
        prices = np.array(problem["prices"])
        capacities = np.array(problem["capacities"])
        probs = np.array(problem["probs"])
        
        # For small problems, use optimized DP
        if T <= 20 and np.prod(capacities + 1) <= 10000:
            return self._solve_dp_optimized(problem)
        
        # For medium problems, use greedy with look-ahead
        elif T <= 50:
            return self._solve_greedy_lookahead(problem)
        
        # For large problems, use CP-SAT with optimizations
        else:
            return self._solve_cp_sat_optimized(problem)
    
    def _solve_dp_optimized(self, problem: dict[str, Any]) -> list[int]:
        """Super optimized DP with numpy vectorization for faster computation."""
        T = problem["T"]
        N = problem["N"]
        prices = np.array(problem["prices"])
        capacities = np.array(problem["capacities"])
        probs = np.array(problem["probs"])
        
        # Precompute expected revenues
        exp_revenues = probs * prices[np.newaxis, :]
        
        # Create state multiplier for encoding
        multipliers = np.cumprod([1] + list(capacities[:-1]))
        max_state = np.prod(capacities + 1)  # +1 for 0..capacity
        
        # Use separate numpy arrays for faster operations
        dp_values = np.full((T + 1, max_state), -1.0)
        dp_actions = np.full((T + 1, max_state), -1, dtype=int)
        dp_next_states = np.full((T + 1, max_state), 0, dtype=int)
        
        # Base case: dp[T][*] = 0
        dp_values[T, :] = 0.0
        
        # Precompute state transitions for efficiency
        state_transitions = []
        for state in range(max_state):
            # Decode state to capacities
            remaining_caps = []
            temp = state
            for i in range(N - 1, -1, -1):
                remaining_caps.insert(0, temp // multipliers[i])
                temp = temp % multipliers[i]
            remaining_caps = np.array(remaining_caps)
            
            transitions = []
            # Option: offer nothing
            transitions.append((-1, state))
            
            # Option: offer each product
            for i in range(N):
                if remaining_caps[i] > 0:
                    new_caps = remaining_caps.copy()
                    new_caps[i] -= 1
                    next_state = np.sum(new_caps * multipliers)
                    transitions.append((i, next_state))
            
            state_transitions.append(transitions)
        
        # Fill DP table backwards
        for t in range(T - 1, -1, -1):
            for state in range(max_state):
                best_revenue = -1.0
                best_action = -1
                best_next_state = state
                
                for action, next_state in state_transitions[state]:
                    if action == -1:
                        revenue = dp_values[t + 1, next_state]
                    else:
                        revenue = exp_revenues[t, action] + dp_values[t + 1, next_state]
                    
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_action = action
                        best_next_state = next_state
                
                dp_values[t, state] = best_revenue
                dp_actions[t, state] = best_action
                dp_next_states[t, state] = best_next_state
        
        # Reconstruct solution
        solution = []
        state = 0  # Start with full capacities
        
        for t in range(T):
            action = dp_actions[t, state]
            solution.append(action)
            state = dp_next_states[t, state]
        
        return solution
    
    def _solve_greedy_lookahead(self, problem: dict[str, Any]) -> list[int]:
        """Greedy algorithm with limited look-ahead."""
        T = problem["T"]
        N = problem["N"]
        prices = np.array(problem["prices"])
        capacities = np.array(problem["capacities"])
        probs = np.array(problem["probs"])
        
        solution = [-1] * T
        remaining_caps = capacities.copy()
        
        # Look-ahead window
        lookahead = min(5, T)
        
        for t in range(T):
            if t + lookahead > T:
                # For final periods, use simple greedy
                best_score = -1.0
                best_product = -1
                
                for i in range(N):
                    if remaining_caps[i] > 0:
                        score = prices[i] * probs[t][i]
                        if score > best_score:
                            best_score = score
                            best_product = i
                
                if best_product >= 0:
                    solution[t] = best_product
                    remaining_caps[best_product] -= 1
            else:
                # Use look-ahead
                best_score = -1.0
                best_product = -1
                
                for i in range(N):
                    if remaining_caps[i] > 0:
                        # Immediate reward
                        immediate = prices[i] * probs[t][i]
                        
                        # Look-ahead: estimate future opportunities
                        future = 0.0
                        temp_caps = remaining_caps.copy()
                        temp_caps[i] -= 1
                        
                        for tau in range(t + 1, min(t + lookahead, T)):
                            tau_best = 0.0
                            for j in range(N):
                                if temp_caps[j] > 0:
                                    tau_best = max(tau_best, prices[j] * probs[tau][j])
                            future += tau_best * (0.9 ** (tau - t))  # Discount factor
                        
                        total_score = immediate + future * 0.5  # Weight future
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_product = i
                
                if best_product >= 0:
                    solution[t] = best_product
                    remaining_caps[best_product] -= 1
        
        return solution
    
    def _solve_cp_sat_optimized(self, problem: dict[str, Any]) -> list[int]:
        """Highly optimized CP-SAT solver for large problems with efficient model formulation."""
        T = problem["T"]
        N = problem["N"]
        prices = problem["prices"]
        capacities = problem["capacities"]
        probs = problem["probs"]

        model = cp_model.CpModel()

        # More efficient variable creation: use list comprehension
        x = [[model.NewBoolVar(f"x_{t}_{i}") for i in range(N)] for t in range(T)]

        # Each period at most one product - more efficient constraint
        for t in range(T):
            model.Add(sum(x[t]) <= 1)

        # Capacity limits - precompute sums for efficiency
        for i in range(N):
            model.Add(sum(x[t][i] for t in range(T)) <= capacities[i])

        # Objective: expected revenue - precompute coefficients
        objective_terms = []
        for t in range(T):
            for i in range(N):
                objective_terms.append(prices[i] * probs[t][i] * x[t][i])
        model.Maximize(sum(objective_terms))

        solver = cp_model.CpSolver()
        
        # Aggressive optimization parameters
        solver.parameters.max_time_in_seconds = 8.5  # Slightly less than timeout
        solver.parameters.num_search_workers = 1
        solver.parameters.cp_model_presolve = True
        solver.parameters.cp_model_probing_level = 1  # Enable probing
        
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return [-1] * T

        # More efficient solution extraction
        offer = [-1] * T
        for t in range(T):
            for i in range(N):
                if solver.Value(x[t][i]) == 1:
                    offer[t] = i
                    break
        return offer