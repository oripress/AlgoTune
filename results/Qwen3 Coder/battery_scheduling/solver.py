import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the battery scheduling problem using CVXPY with optimized formulation.
        
        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal schedules and costs
        """
        # Extract problem parameters
        T = int(problem["T"])
        p = np.array(problem["p"])
        u = np.array(problem["u"])
        battery = problem["batteries"][0]  # Single battery
        
        # Extract battery parameters
        Q = float(battery["Q"])  # Battery capacity
        C = float(battery["C"])  # Max charging rate
        D = float(battery["D"])  # Max discharging rate
        efficiency = float(battery["efficiency"])  # Battery efficiency
        
        # Define variables with better initial values for faster convergence
        q = cp.Variable(T)  # Energy stored
        c_in = cp.Variable(T)  # Charging rate (positive only)
        c_out = cp.Variable(T)  # Discharging rate (positive only)
        # Net charging rate (for objective and grid constraints)
        c = c_in - c_out
        # Battery dynamics constraints
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
        
        # Battery dynamics with efficiency losses
        # Using vectorized constraints for better performance
        constraints.append(q[1:] == q[:-1] + efficiency * c_in[:-1] - (1/efficiency) * c_out[:-1])
        
        # Cyclic constraint: end with same charge as start
        effective_charge_last = efficiency * c_in[-1] - (1/efficiency) * c_out[-1]
        constraints.append(q[0] == q[-1] + effective_charge_last)
        
        
        # No power back to grid constraint
        constraints.append(u + c >= 0)
        
        # Objective: minimize electricity cost
        objective = cp.Minimize(p @ c)
        
        # Define and solve the problem
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)  # Use CLARABEL solver with no verbose output
            
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return {"status": prob.status, "optimal": False}
            
            # Calculate net charging
            c_net = c_in.value - c_out.value
            
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
                        "q": q.value.tolist(),
                        "c": c_net.tolist(),
                        "c_in": c_in.value.tolist(),
                        "c_out": c_out.value.tolist(),
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
            # Fallback to default solver if CLARABEL fails
            try:
                prob.solve()
                if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                    return {"status": prob.status, "optimal": False}
                
                # Calculate net charging
                c_net = c_in.value - c_out.value
                
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
                            "q": q.value.tolist(),
                            "c": c_net.tolist(),
                            "c_in": c_in.value.tolist(),
                            "c_out": c_out.value.tolist(),
                            "cost": cost_with_battery,
                        }
                    ],
                    "total_charging": c_net.tolist(),
                    "cost_without_battery": cost_without_battery,
                    "cost_with_battery": cost_with_battery,
                    "savings": savings,
                    "savings_percent": float(100 * savings / cost_without_battery),
                }
            except Exception:
                return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}