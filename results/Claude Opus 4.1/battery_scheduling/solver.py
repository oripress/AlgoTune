import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Optimized battery scheduling solver using CVXPY with ECOS solver.
        """
        # Extract problem parameters
        T = int(problem["T"])
        p = np.array(problem["p"], dtype=np.float64)
        u = np.array(problem["u"], dtype=np.float64)
        battery = problem["batteries"][0]  # Single battery
        
        # Extract battery parameters
        Q = float(battery["Q"])  # Battery capacity
        C = float(battery["C"])  # Max charging rate
        D = float(battery["D"])  # Max discharging rate
        efficiency = float(battery["efficiency"])  # Battery efficiency
        
        # Define variables
        q = cp.Variable(T)  # Energy stored
        c_in = cp.Variable(T)  # Charging rate (positive only)
        c_out = cp.Variable(T)  # Discharging rate (positive only)
        
        # Net charging rate
        c = c_in - c_out
        
        # Constraints list
        constraints = []
        
        # Battery capacity constraints
        constraints.append(q >= 0)
        constraints.append(q <= Q)
        
        # Non-negative charging/discharging
        constraints.append(c_in >= 0)
        constraints.append(c_out >= 0)
        
        # Charge/discharge rate constraints
        constraints.append(c_in <= C)
        constraints.append(c_out <= D)
        
        # Battery dynamics with efficiency losses (vectorized)
        # q[t+1] = q[t] + efficiency * c_in[t] - (1/efficiency) * c_out[t]
        A_dyn = np.zeros((T-1, T))
        for i in range(T-1):
            A_dyn[i, i] = -1
            A_dyn[i, i+1] = 1
        
        b_dyn = np.zeros(T-1)
        constraints.append(A_dyn @ q == efficiency * c_in[:-1] - (1/efficiency) * c_out[:-1])
        
        # Cyclic constraint: q[0] = q[T-1] + efficiency * c_in[T-1] - (1/efficiency) * c_out[T-1]
        constraints.append(q[0] == q[T-1] + efficiency * c_in[T-1] - (1/efficiency) * c_out[T-1])
        
        # No power back to grid constraint
        constraints.append(u + c >= 0)
        
        # Objective: minimize electricity cost
        objective = cp.Minimize(p @ c)
        
        # Define and solve the problem with ECOS solver
        prob = cp.Problem(objective, constraints)
        
        try:
            # Use ECOS solver which is typically faster for these problems
            prob.solve(solver=cp.ECOS, verbose=False)
            
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return {"status": prob.status, "optimal": False}
            
            # Extract values
            q_val = q.value
            c_in_val = c_in.value
            c_out_val = c_out.value
            c_net = c_in_val - c_out_val
            
            # Calculate costs
            cost_without_battery = float(p @ u)
            cost_with_battery = float(p @ (u + c_net))
            savings = cost_without_battery - cost_with_battery
            
            # Return solution
            return {
                "status": prob.status,
                "optimal": True,
                "battery_results": [
                    {
                        "q": q_val.tolist(),
                        "c": c_net.tolist(),
                        "c_in": c_in_val.tolist(),
                        "c_out": c_out_val.tolist(),
                        "cost": cost_with_battery,
                    }
                ],
                "total_charging": c_net.tolist(),
                "cost_without_battery": cost_without_battery,
                "cost_with_battery": cost_with_battery,
                "savings": savings,
                "savings_percent": float(100 * savings / cost_without_battery),
            }
            
        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}