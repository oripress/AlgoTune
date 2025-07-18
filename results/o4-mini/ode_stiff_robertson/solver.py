from scipy.integrate import solve_ivp

class Solver:
    def solve(self, problem, **kwargs):
        # Unpack and cast inputs to pure Python floats/lists
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        y0 = [float(problem["y0"][0]), float(problem["y0"][1]), float(problem["y0"][2])]
        k1 = float(problem["k"][0]); k2 = float(problem["k"][1]); k3 = float(problem["k"][2])

        # ODE right-hand side (stiff Robertson kinetics)
        def f(t, y):
            y1, y2, y3 = y
            return (
                -k1 * y1 + k3 * y2 * y3,
                k1 * y1 - k2 * y2 * y2 - k3 * y2 * y3,
                k2 * y2 * y2,
            )

        # Analytical Jacobian for BDF
        def jac(t, y):
            y1, y2, y3 = y
            return (
                [-k1,               k3 * y3,       k3 * y2],
                [ k1,  -2.0 * k2 * y2 - k3 * y3,      -k3 * y2],
                [ 0.0,            2.0 * k2 * y2,          0.0],
            )

        # Solve with BDF method, analytic Jacobian, only final value needed
        sol = solve_ivp(
            f, (t0, t1), y0,
            method="BDF",
            jac=jac,
            rtol=1e-7,
            atol=1e-9,
            dense_output=False,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        yf = sol.y[:, -1]
        return [yf[0], yf[1], yf[2]]