import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
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

        # Define variables
        q = cp.Variable(T)  # Energy stored
        c_in = cp.Variable(T)  # Charging rate (positive only)
        c_out = cp.Variable(T)  # Discharging rate (positive only)

        # Net charging rate (for objective and grid constraints)
        c = c_in - c_out

        # Vectorized constraints for efficiency
        constraints = [
            q >= 0,
            q <= Q,
            c_in >= 0,
            c_in <= C,
            c_out >= 0,
            c_out <= D,
            # Battery dynamics (vectorized)
            q[1:] == q[:-1] + efficiency * c_in[:-1] - (1/efficiency) * c_out[:-1],
            # Cyclic constraint
            q[0] == q[T-1] + efficiency * c_in[T-1] - (1/efficiency) * c_out[T-1],
            # No power back to grid
            u + c >= 0
        ]

        # Objective: minimize electricity cost
        objective = cp.Minimize(p @ c)

        # Define and solve the problem
        prob = cp.Problem(objective, constraints)

        try:
            # Use ECOS solver with lower tolerance for faster convergence
            prob.solve(solver=cp.ECOS, max_iters=10000, abstol=1e-3, reltol=1e-3)
            
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
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}