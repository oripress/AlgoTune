import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        try:
            # Parse inputs
            A = np.asarray(problem["A"], dtype=float)
            B = np.asarray(problem["B"], dtype=float)
            C = np.asarray(problem["C"], dtype=float)
            y = np.asarray(problem["y"], dtype=float)
            x0 = np.asarray(problem["x_initial"], dtype=float)
            tau = float(problem["tau"])

            # Dimensions
            N = y.shape[0]
            m = C.shape[0]
            n = A.shape[1]
            p = B.shape[1]

            # Covariances
            Q = B.dot(B.T)
            R = np.eye(m) / tau if m > 0 else np.zeros((0, 0))
            I_n = np.eye(n)

            # Allocate storage
            x_pred = np.zeros((N + 1, n))
            P_pred = np.zeros((N + 1, n, n))
            x_filt = np.zeros((N + 1, n))
            P_filt = np.zeros((N + 1, n, n))

            # Initial state
            x_pred[0] = x0
            P_pred[0] = np.zeros((n, n))

            # Forward Kalman filter
            for t in range(N):
                Pt = P_pred[t]
                Pt_CT = Pt.dot(C.T)
                S = C.dot(Pt_CT) + R
                if m > 0:
                    # Kalman gain
                    K = Pt_CT.dot(np.linalg.inv(S))
                    innov = y[t] - C.dot(x_pred[t])
                    x_filt[t] = x_pred[t] + K.dot(innov)
                    P_filt[t] = (I_n - K.dot(C)).dot(Pt)
                else:
                    x_filt[t] = x_pred[t]
                    P_filt[t] = Pt
                # Predict next
                x_pred[t + 1] = A.dot(x_filt[t])
                P_pred[t + 1] = A.dot(P_filt[t]).dot(A.T) + Q

            # At final step
            x_filt[N] = x_pred[N]
            P_filt[N] = P_pred[N]

            # Backward Rauch–Tung–Striebel smoother
            x_smooth = np.zeros_like(x_pred)
            x_smooth[N] = x_filt[N]
            for t in range(N - 1, -1, -1):
                Pnext = P_pred[t + 1]
                # Smoother gain
                J = P_filt[t].dot(A.T).dot(np.linalg.inv(Pnext))
                x_smooth[t] = x_filt[t] + J.dot(x_smooth[t + 1] - x_pred[t + 1])

            # Estimate noises
            dx = x_smooth[1:] - (A.dot(x_smooth[:-1].T)).T
            if p > 0:
                w = np.linalg.pinv(B).dot(dx.T).T
            else:
                w = np.zeros((N, 0))
            if m > 0:
                v = y - (C.dot(x_smooth[:-1].T)).T
            else:
                v = np.zeros((N, 0))

            return {
                "x_hat": x_smooth.tolist(),
                "w_hat": w.tolist(),
                "v_hat": v.tolist(),
            }
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

# Module‐level solve for harness compatibility
def solve(problem: dict, **kwargs) -> dict:
    try:
        return Solver().solve(problem, **kwargs)
    except Exception:
        return {"x_hat": [], "w_hat": [], "v_hat": []}