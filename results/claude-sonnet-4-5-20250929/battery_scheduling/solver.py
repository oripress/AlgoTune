import numpy as np
import highspy

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the battery scheduling problem using HighsPy directly.
        """
        # Extract problem parameters
        T = int(problem["T"])
        p = np.array(problem["p"], dtype=np.float64)
        u = np.array(problem["u"], dtype=np.float64)
        battery = problem["batteries"][0]
        
        # Extract battery parameters
        Q = float(battery["Q"])
        C = float(battery["C"])
        D = float(battery["D"])
        efficiency = float(battery["efficiency"])
        
        # Create HiGHS model
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        
        # Variables: [q_0, ..., q_{T-1}, c_in_0, ..., c_in_{T-1}, c_out_0, ..., c_out_{T-1}]
        num_vars = 3 * T
        
        # Variable bounds
        lower = np.concatenate([np.zeros(T), np.zeros(T), np.zeros(T)])
        upper = np.concatenate([np.full(T, Q), np.full(T, C), np.full(T, D)])
        
        # Objective coefficients
        obj_coeff = np.concatenate([np.zeros(T), p, -p])
        
        # Add variables
        h.addVars(num_vars, lower, upper)
        h.changeColsCost(num_vars, np.arange(num_vars), obj_coeff)
        
        # Build constraint matrix as lists
        starts = []
        indices = []
        values = []
        
        current_start = 0
        
        # Battery dynamics constraints
        for t in range(T-1):
            starts.append(current_start)
            # q[t+1] - q[t] - efficiency * c_in[t] + (1/efficiency) * c_out[t] = 0
            indices.extend([t+1, t, T+t, 2*T+t])
            values.extend([1.0, -1.0, -efficiency, 1.0/efficiency])
            current_start += 4
        
        # Cyclic constraint
        starts.append(current_start)
        indices.extend([0, T-1, T+T-1, 2*T+T-1])
        values.extend([1.0, -1.0, -efficiency, 1.0/efficiency])
        current_start += 4
        
        # Grid constraint: -c_in + c_out <= u
        for t in range(T):
            starts.append(current_start)
            indices.extend([T+t, 2*T+t])
            values.extend([-1.0, 1.0])
            current_start += 2
        
        starts.append(current_start)
        
        # Add constraints
        num_eq = T  # Battery dynamics + cyclic
        num_ineq = T  # Grid constraints
        
        lower_bounds = np.concatenate([np.zeros(num_eq), np.full(num_ineq, -np.inf)])
        upper_bounds = np.concatenate([np.zeros(num_eq), u])
        
        h.addRows(num_eq + num_ineq, lower_bounds, upper_bounds, 
                 len(indices), starts, indices, values)
        
        # Solve
        h.run()
        
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            return {"status": "failed", "optimal": False}
        
        # Get solution
        solution = h.getSolution()
        x = np.array(solution.col_value)
        
        q = x[:T]
        c_in = x[T:2*T]
        c_out = x[2*T:3*T]
        c_net = c_in - c_out
        
        # Calculate costs
        cost_without_battery = float(p @ u)
        cost_with_battery = float(p @ (u + c_net))
        savings = cost_without_battery - cost_with_battery
        
        # Return solution
        return {
            "status": "optimal",
            "optimal": True,
            "battery_results": [
                {
                    "q": q.tolist(),
                    "c": c_net.tolist(),
                    "c_in": c_in.tolist(),
                    "c_out": c_out.tolist(),
                    "cost": cost_with_battery,
                }
            ],
            "total_charging": c_net.tolist(),
            "cost_without_battery": cost_without_battery,
            "cost_with_battery": cost_with_battery,
            "savings": savings,
            "savings_percent": float(100 * savings / cost_without_battery),
        }