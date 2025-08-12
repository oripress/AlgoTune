from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Fast IRLS-based solver for the robust Kalman smoothing problem.

        Returns a dictionary with keys "x_hat", "w_hat", "v_hat".
        """
        try:
            A = np.asarray(problem["A"], dtype=float)
            B = np.asarray(problem.get("B", []), dtype=float)
            C = np.asarray(problem["C"], dtype=float)
            y = np.asarray(problem["y"], dtype=float)
            x0 = np.asarray(problem["x_initial"], dtype=float).reshape(-1)
            tau = float(problem["tau"])
            M = float(problem["M"])
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Basic dims
        if y.size == 0:
            # No measurements: only initial state is known
            n = A.shape[1]
            return {"x_hat": [x0.reshape(-1).tolist()], "w_hat": [], "v_hat": []}

        if y.ndim == 1:
            # single measurement vector interpreted as N=1, m = len(y)
            y = y.reshape(1, -1)
        N, m = y.shape
        n = A.shape[1]
        # Ensure B has correct shape; if empty, p=0
        if B.size == 0:
            B = np.zeros((n, 0), dtype=float)
            p = 0
        else:
            if B.ndim == 1:
                B = B.reshape(n, -1)
            p = B.shape[1]

        # Sanity checks
        if A.shape[0] != n or C.shape[1] != n:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Variables ordering: z = [x_1, ..., x_N, w_0, ..., w_{N-1}]
        Nx_vars = N * n
        Nw_vars = N * p
        nv = Nx_vars + Nw_vars
        nc = N * n  # dynamics equality constraints count

        # Precompute measurement-related matrices
        Ct = C.T
        CtC = Ct @ C
        Cty = np.array([Ct @ y[t] for t in range(N)])  # N x n

        # H0 baseline (from process noise cost sum ||w||^2)
        H0 = np.zeros((nv, nv), dtype=float) if nv > 0 else np.zeros((0, 0), dtype=float)
        if p > 0:
            for t in range(N):
                iw = Nx_vars + t * p
                H0[iw:iw + p, iw:iw + p] = 2.0 * np.eye(p)

        # Build dynamics constraint E z = h
        if nc > 0 and nv > 0:
            E = np.zeros((nc, nv), dtype=float)
        elif nc > 0:
            E = np.zeros((nc, 0), dtype=float)
        else:
            E = np.zeros((0, nv), dtype=float)
        h = np.zeros(nc, dtype=float)

        def idx_x(t):
            # x_t for t in 1..N maps to index (t-1)*n .. +n
            return (t - 1) * n

        def idx_w(t):
            # w_t for t in 0..N-1 maps to Nx_vars + t*p
            return Nx_vars + t * p

        for t in range(N):
            r0 = t * n
            r1 = r0 + n
            # x_{t+1}
            ix_next = idx_x(t + 1)
            E[r0:r1, ix_next:ix_next + n] += np.eye(n)
            if t >= 1:
                ix_t = idx_x(t)
                E[r0:r1, ix_t:ix_t + n] += -A
            else:
                # move A x0 to rhs
                h[r0:r1] = A @ x0
            if p > 0:
                iw = idx_w(t)
                E[r0:r1, iw:iw + p] += -B

        # IRLS initialization
        s = np.ones(N, dtype=float)
        x_vars = np.zeros((N, n), dtype=float)
        w_vars = np.zeros((N, p), dtype=float) if p > 0 else np.zeros((0, 0), dtype=float)

        max_iter = int(kwargs.get("max_iter", 80))
        tol = float(kwargs.get("tol", 1e-8))
        J_prev = np.inf

        # IRLS loop
        for iteration in range(max_iter):
            if nv > 0:
                H = H0.copy()
                f = np.zeros(nv, dtype=float)
            else:
                H = np.zeros((0, 0), dtype=float)
                f = np.zeros(0, dtype=float)

            # Add measurement-weighted terms for t = 1..N-1 (x_1..x_{N-1} are unknown)
            for t in range(1, N):
                wgt = 2.0 * tau * s[t]  # factor for H (since 0.5 z^T H z form)
                ix = idx_x(t)
                H[ix:ix + n, ix:ix + n] += wgt * CtC
                f[ix:ix + n] += -2.0 * tau * s[t] * Cty[t]

            # Regularize H slightly for numeric stability
            if nv > 0:
                traceH = np.trace(H)
                base = max(abs(traceH) / max(1, nv), 1.0)
                reg = 1e-9 * base
                H_reg = H.copy()
                H_reg.flat[::nv + 1] += reg
            else:
                H_reg = H

            # Solve KKT using Schur complement
            try:
                if nc == 0 and nv > 0:
                    z = np.linalg.solve(H_reg, -f)
                elif nv == 0 and nc == 0:
                    z = np.zeros(0, dtype=float)
                else:
                    if nv > 0:
                        X = np.linalg.solve(H_reg, E.T)  # nv x nc
                        v = np.linalg.solve(H_reg, f)    # nv
                        S = E @ X
                    else:
                        X = np.zeros((0, nc), dtype=float)
                        v = np.zeros(0, dtype=float)
                        S = np.zeros((nc, nc), dtype=float)

                    if nc > 0:
                        traceS = np.trace(S)
                        sbase = max(abs(traceS) / max(1, nc), 1.0)
                        sreg = 1e-12 * sbase
                        if sreg > 0:
                            S.flat[::nc + 1] += sreg
                        rhs = -(h + E @ v)
                        lam = np.linalg.solve(S, rhs)
                    else:
                        lam = np.zeros(0, dtype=float)

                    z = -v - (X @ lam) if nv > 0 else np.zeros(0, dtype=float)
            except np.linalg.LinAlgError:
                # fallback full KKT
                KKT = np.zeros((nv + nc, nv + nc), dtype=float)
                if nv > 0:
                    KKT[:nv, :nv] = H_reg
                    KKT[:nv, nv:] = E.T
                    KKT[nv:, :nv] = E
                else:
                    KKT[:nv, nv:] = E.T
                    KKT[nv:, :nv] = E
                rhs = np.concatenate([-f, h])
                try:
                    sol = np.linalg.solve(KKT, rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
                z = sol[:nv] if nv > 0 else np.zeros(0, dtype=float)

            # Extract variables
            if Nx_vars > 0:
                x_vars = z[:Nx_vars].reshape((N, n)).copy()
            else:
                x_vars = np.zeros((N, n), dtype=float)

            # Full state sequence
            x_hat_seq = np.vstack((x0.reshape(1, n), x_vars))

            # Extract w from solution vector if present (ensures dynamics are satisfied)
            if p > 0:
                if Nw_vars > 0:
                    w_part = z[Nx_vars:Nx_vars + Nw_vars]
                    # reshape into (N, p) even if empty (p may be 0)
                    w_vars = np.asarray(w_part).reshape((N, p)).copy()
                else:
                    w_vars = np.zeros((N, p), dtype=float)
            else:
                # p == 0 -> empty per-time w vectors
                w_vars = np.zeros((N, p), dtype=float)

            # Measurement residuals and v estimate
            v_vars = np.zeros((N, m), dtype=float)
            for t in range(N):
                xt = x_hat_seq[t]
                v_vars[t, :] = y[t, :] - (C @ xt)

            # Compute residual norms and update IRLS weights
            r = np.linalg.norm(v_vars, axis=1)
            tiny = 1e-12
            new_s = np.where(r <= M + tiny, 1.0, M / np.maximum(r, tiny))
            # Compute objective J (Huber)
            J_w = float(np.sum(w_vars ** 2))
            J_v = 0.0
            for t in range(N):
                rt = r[t]
                if rt <= M:
                    J_v += rt * rt
                else:
                    J_v += 2.0 * M * rt - M * M
            J = J_w + tau * J_v

            if np.isfinite(J_prev):
                if abs(J_prev - J) <= tol * max(1.0, abs(J_prev)):
                    break
            J_prev = J
            s = new_s

        # Format outputs
        x_hat_list = [x_hat_seq[t, :].tolist() for t in range(N + 1)]
        if p > 0:
            w_hat_list = [w_vars[t, :].tolist() for t in range(N)]
        else:
            w_hat_list = [[] for _ in range(N)]
        v_hat_list = [v_vars[t, :].tolist() for t in range(N)]

        return {"x_hat": x_hat_list, "w_hat": w_hat_list, "v_hat": v_hat_list}