from typing import Any, Dict, List, Tuple

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

class Solver:
    def __init__(self) -> None:
        # Choose fast conic solvers if available
        self._preferred_solvers: List[str] = []
        try:
            installed = set(cp.installed_solvers())
        except Exception:
            installed = set()
        for s in ("ECOS", "SCS"):
            if s in installed:
                self._preferred_solvers.append(s)

    # -------------------- Fast smooth optimizer (L-BFGS-B over w) --------------------

    @staticmethod
    def _objective_and_grad_w(
        w_vec: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray,
        tau_half: float,
        M: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute objective and gradient wrt w (flattened) for:
            f(w) = sum_t ||w_t||^2 + (tau/2) * sum_t phi(||v_t||)
        where phi(r) = r^2 if r<=M else 2*M*r - M^2,
        v_t = y_t - C x_t, and x_{t+1} = A x_t + B w_t.

        Note: tau * huber(||v||, M) == (tau/2) * phi(||v||).
        Returns (f, grad_w_vec).
        """
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        if N == 0 or p == 0:
            return 0.0, np.zeros((N * p,), dtype=float)

        # Unflatten w
        w = w_vec.reshape(N, p)

        AT = A.T
        BT = B.T
        CT = C.T

        # Forward pass: x and v
        x = np.zeros((N + 1, n), dtype=float)
        x[0] = x0
        for t in range(N):
            x[t + 1] = A @ x[t] + B @ w[t]
        v = y - (C @ x[:N].T).T  # shape (N, m)

        # Objective value
        f_proc = float(np.sum(w * w))
        v_norms = np.linalg.norm(v, axis=1)
        small = v_norms <= M
        large = ~small
        phi = np.empty(N, dtype=float)
        phi[small] = v_norms[small] ** 2
        phi[large] = 2.0 * M * v_norms[large] - (M * M)
        f = f_proc + tau_half * float(np.sum(phi))

        # Gradient wrt v
        g_v = np.zeros_like(v)
        if np.any(small):
            g_v[small] = 2.0 * v[small]
        if np.any(large):
            scale = (2.0 * M) / v_norms[large]  # safe: large => v_norms > M > 0
            g_v[large] = (v[large].T * scale).T

        # Backward pass for gradients wrt w
        # q_t = -C^T g_v[t]
        q = -(CT @ g_v.T).T  # shape (N, n)

        grad_w = np.empty_like(w)
        r_next = np.zeros(n, dtype=float)  # r_{N} = 0
        for k in range(N - 1, -1, -1):
            # Gradient from measurement term wrt w_k: B^T r_{k+1}
            grad_meas_k = BT @ r_next
            grad_w[k] = 2.0 * w[k] + tau_half * grad_meas_k
            # Update r_k = q_k + A^T r_{k+1}
            r_next = q[k] + AT @ r_next

        return f, grad_w.reshape(-1)

    def _solve_lbfgs_w_only(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray,
        tau: float,
        M: float,
    ) -> Dict[str, Any]:
        """
        Solve by optimizing only over w with a fast L-BFGS-B method.
        Objective minimized:
            sum ||w_t||^2 + tau * huber(||v_t||, M)
        implemented as
            sum ||w_t||^2 + (tau/2) * phi(||v_t||),
        which is equivalent since phi = 2 * huber.
        """
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Trivial cases
        if N == 0:
            x_hat = np.asarray(x0, dtype=float).reshape(1, n)
            return {
                "x_hat": x_hat.tolist(),
                "w_hat": np.zeros((0, p), dtype=float).tolist(),
                "v_hat": np.zeros((0, m), dtype=float).tolist(),
            }

        if p == 0:
            # No control variables; dynamics fixed by A
            x_hat = np.zeros((N + 1, n), dtype=float)
            x_hat[0] = np.asarray(x0, dtype=float).reshape(n)
            for t in range(N):
                x_hat[t + 1] = (A @ x_hat[t]).reshape(n)
            w_hat = np.zeros((N, p), dtype=float)
            v_hat = y - (C @ x_hat[:-1].T).T
            return {
                "x_hat": x_hat.tolist(),
                "w_hat": w_hat.tolist(),
                "v_hat": v_hat.tolist(),
            }

        # Initial guess
        w0 = np.zeros((N * p,), dtype=float)

        # Optimize
        tau_half = 0.5 * tau
        fun = lambda wvec: self._objective_and_grad_w(wvec, A, B, C, y, x0, tau_half, M)
        try:
            res = minimize(
                fun=lambda wvec: fun(wvec)[0],
                x0=w0,
                jac=lambda wvec: fun(wvec)[1],
                method="L-BFGS-B",
                options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7, "iprint": -1},
            )
            w_vec = res.x
            if not np.isfinite(w_vec).all():
                raise ValueError("Non-finite result from optimizer.")
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        w_hat = w_vec.reshape(N, p)

        # Reconstruct x_hat and v_hat
        x_hat = np.zeros((N + 1, n), dtype=float)
        x_hat[0] = np.asarray(x0, dtype=float).reshape(n)
        for t in range(N):
            x_hat[t + 1] = (A @ x_hat[t] + B @ w_hat[t]).reshape(n)
        v_hat = y - (C @ x_hat[:-1].T).T

        return {
            "x_hat": x_hat.tolist(),
            "w_hat": w_hat.tolist(),
            "v_hat": v_hat.tolist(),
        }

    # -------------------- CVXPY-based formulations (fallbacks) --------------------

    def _solve_w_only_cvxpy(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray,
        tau: float,
        M: float,
    ) -> Dict[str, Any]:
        """
        CVXPY: Solve by eliminating x and v; optimize only over w.
        """
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Trivial case
        if N == 0:
            x_hat = np.asarray(x0, dtype=float).reshape(1, n)
            return {
                "x_hat": x_hat.tolist(),
                "w_hat": np.zeros((0, p), dtype=float).tolist(),
                "v_hat": np.zeros((0, m), dtype=float).tolist(),
            }

        # Decision variable: w in R^{N x p} (unless p==0)
        if p > 0:
            w = cp.Variable((N, p), name="w")
        else:
            w = None

        # Build v_t = y_t - C x_t, with x_t = A^t x0 + sum_{k=0}^{t-1} A^{t-1-k} B w_k
        xt_expr = x0  # constant ndarray
        v_rows = []
        for t in range(N):
            v_rows.append(y[t] - C @ xt_expr)
            if p > 0:
                xt_expr = A @ xt_expr + B @ w[t, :]
            else:
                xt_expr = A @ xt_expr

        V = cp.vstack(v_rows)  # (N, m)

        # Objective: sum ||w_t||^2 + tau * sum huber(||v_t||, M)
        if p > 0:
            process_noise_term = cp.sum_squares(w)
        else:
            process_noise_term = 0.0

        v_norms = cp.norm(V, axis=1)  # length-N vector
        measurement_noise_term = tau * cp.sum(cp.huber(v_norms, M))

        if p == 0:
            # No decision variables; return closed-form solution directly
            x_hat = np.zeros((N + 1, n), dtype=float)
            x_hat[0] = np.asarray(x0, dtype=float).reshape(n)
            for t in range(N):
                x_hat[t + 1] = (A @ x_hat[t]).reshape(n)
            w_hat = np.zeros((N, p), dtype=float)
            v_hat = y - (C @ x_hat[:-1].T).T
            return {
                "x_hat": x_hat.tolist(),
                "w_hat": w_hat.tolist(),
                "v_hat": v_hat.tolist(),
            }

        obj = cp.Minimize(process_noise_term + measurement_noise_term)
        prob = cp.Problem(obj)

        # Try preferred solvers and default
        solved = False
        trial_solvers: List[Any] = []
        for s in self._preferred_solvers:
            if s == "ECOS":
                trial_solvers.append(cp.ECOS)
        for s in self._preferred_solvers:
            if s == "SCS":
                trial_solvers.append(cp.SCS)
        trial_solvers.append(None)

        for solver in trial_solvers:
            try:
                if solver is None:
                    prob.solve(warm_start=True, verbose=False)
                elif solver is cp.ECOS:
                    prob.solve(solver=cp.ECOS, warm_start=True, verbose=False, max_iters=200000)
                elif solver is cp.SCS:
                    prob.solve(solver=cp.SCS, warm_start=True, verbose=False, max_iters=100000)
                else:
                    prob.solve(solver=solver, warm_start=True, verbose=False)
            except cp.SolverError:
                continue
            except Exception:
                continue
            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                solved = True
                break

        if not solved:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Reconstruct x_hat and v_hat
        w_hat = np.asarray(w.value, dtype=float).reshape(N, p)
        x_hat = np.zeros((N + 1, n), dtype=float)
        x_hat[0] = np.asarray(x0, dtype=float).reshape(n)
        for t in range(N):
            x_hat[t + 1] = (A @ x_hat[t] + B @ w_hat[t]).reshape(n)
        v_hat = y - (C @ x_hat[:-1].T).T  # y_t - C x_t

        return {
            "x_hat": x_hat.tolist(),
            "w_hat": w_hat.tolist(),
            "v_hat": v_hat.tolist(),
        }

    def _solve_full_fallback(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray,
        tau: float,
        M: float,
    ) -> Dict[str, Any]:
        """
        Fallback: Full formulation with variables x, w, v.
        """
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # Objective
        process_noise_term = cp.sum_squares(w)
        measurement_noise_term = tau * cp.sum(cp.huber(cp.norm(v, axis=1), M))
        obj = cp.Minimize(process_noise_term + measurement_noise_term)

        # Constraints
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])

        prob = cp.Problem(obj, constraints)

        solved = False
        trial_solvers: List[Any] = []
        for s in self._preferred_solvers:
            if s == "ECOS":
                trial_solvers.append(cp.ECOS)
        for s in self._preferred_solvers:
            if s == "SCS":
                trial_solvers.append(cp.SCS)
        trial_solvers.append(None)

        for solver in trial_solvers:
            try:
                if solver is None:
                    prob.solve(warm_start=True, verbose=False)
                elif solver is cp.ECOS:
                    prob.solve(solver=cp.ECOS, warm_start=True, verbose=False, max_iters=200000)
                elif solver is cp.SCS:
                    prob.solve(solver=cp.SCS, warm_start=True, verbose=False, max_iters=100000)
                else:
                    prob.solve(solver=solver, warm_start=True, verbose=False)
            except cp.SolverError:
                continue
            except Exception:
                continue
            if prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and x.value is not None:
                solved = True
                break

        if not solved or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }

    # -------------------- Public API --------------------

    def solve(self, problem: dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve the robust Kalman filtering/smoothing problem efficiently.

        Returns a dictionary with keys "x_hat", "w_hat", "v_hat".
        """
        # Parse inputs
        try:
            A = np.asarray(problem["A"], dtype=float)
            B = np.asarray(problem["B"], dtype=float)
            C = np.asarray(problem["C"], dtype=float)
            y = np.asarray(problem["y"], dtype=float)
            x0 = np.asarray(problem["x_initial"], dtype=float)
            tau = float(problem["tau"])
            M = float(problem["M"])
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Basic shape checks
        if any(arr.ndim != 2 for arr in (A, B, C)) or y.ndim != 2:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        if A.shape[0] != n or C.shape[1] != n or B.shape[0] != n or C.shape[0] != m:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        x0 = x0.reshape(n)

        # Fast path: CVXPY w-only formulation (eliminate x and v)
        result = self._solve_w_only_cvxpy(A, B, C, y, x0, tau, M)

        # Fallbacks if needed
        if not result.get("x_hat"):
            # Try full CVXPY model
            result = self._solve_full_fallback(A, B, C, y, x0, tau, M)
        if not result.get("x_hat"):
            # As last resort, try smooth L-BFGS-B on w
            result = self._solve_lbfgs_w_only(A, B, C, y, x0, tau, M)

        # Validate finiteness and format
        try:
            x_hat = np.asarray(result.get("x_hat", []), dtype=float)
            w_hat = np.asarray(result.get("w_hat", []), dtype=float)
            v_hat = np.asarray(result.get("v_hat", []), dtype=float)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if x_hat.size == 0 or not (
            np.isfinite(x_hat).all() and np.isfinite(w_hat).all() and np.isfinite(v_hat).all()
        ):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Ensure shapes
        if x_hat.shape != (N + 1, n) or w_hat.shape != (N, p) or v_hat.shape != (N, m):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x_hat.tolist(),
            "w_hat": w_hat.tolist(),
            "v_hat": v_hat.tolist(),
        }