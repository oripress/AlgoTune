import numpy as np
from scipy import signal, linalg
from numba import njit

@njit
def simulate_foh(Phi, Gamma1, Gamma2, C, D, u, x0):
    n_steps = len(u)
    n_states = len(x0)
    x = x0.copy()
    y = np.empty(n_steps, dtype=np.float64)
    
    # Pre-allocate x_next to avoid allocation in loop
    x_next = np.empty(n_states, dtype=np.float64)
    
    for k in range(n_steps - 1):
        # y[k] = C @ x + D * u[k]
        val = 0.0
        for i in range(n_states):
            val += C[i] * x[i]
        y[k] = val + D * u[k]
        
        # Update state
        uk = u[k]
        uk1 = u[k+1]
        
        # x_next = Phi @ x + Gamma1 * uk + Gamma2 * uk1
        for i in range(n_states):
            acc = 0.0
            for j in range(n_states):
                acc += Phi[i, j] * x[j]
            acc += Gamma1[i] * uk + Gamma2[i] * uk1
            x_next[i] = acc
            
        for i in range(n_states):
            x[i] = x_next[i]
            
    # Last step
    k = n_steps - 1
    val = 0.0
    for i in range(n_states):
        val += C[i] * x[i]
    y[k] = val + D * u[k]
    
    return y

class Solver:
    def __init__(self):
        # Dummy call to compile JIT function
        Phi = np.eye(1, dtype=np.float64)
        Gamma1 = np.ones(1, dtype=np.float64)
        Gamma2 = np.ones(1, dtype=np.float64)
        C = np.ones(1, dtype=np.float64)
        D = 0.0
        u = np.zeros(2, dtype=np.float64)
        x0 = np.zeros(1, dtype=np.float64)
        simulate_foh(Phi, Gamma1, Gamma2, C, D, u, x0)

    def solve(self, problem: dict[str, np.ndarray], **kwargs) -> dict[str, list[float]]:
        num = problem["num"]
        den = problem["den"]
        u_list = problem["u"]
        t_list = problem["t"]
        
        # Convert to numpy arrays
        u = np.array(u_list, dtype=np.float64)
        t = np.array(t_list, dtype=np.float64)
        
        if len(t) < 2:
             return {"yout": [0.0] * len(t)}

        dt = t[1] - t[0]
        
        # Check uniformity
        # Using a tolerance relative to dt
        tol = 1e-5 * dt
        is_uniform = np.max(np.abs(np.diff(t) - dt)) < tol
        
        if is_uniform:
            try:
                # Optimization for 1st order system
                if len(den) == 2 and len(num) <= 2:
                    a0 = float(den[1])
                    a1 = float(den[0])
                    
                    if len(num) == 1:
                        b0 = float(num[0])
                        b1 = 0.0
                    else:
                        b0 = float(num[1])
                        b1 = float(num[0])
                    
                    if abs(a1) > 1e-12:
                        a1_inv = 1.0 / a1
                        A_val = -a0 * a1_inv
                        B_val = 1.0
                        D_val = b1 * a1_inv
                        C_val = (b0 - D_val * a0) * a1_inv
                        
                        if abs(A_val) < 1e-9:
                            Phi_val = 1.0
                            Psi1_val = B_val * dt
                            Psi2_val = B_val * dt**2 / 2.0
                        else:
                            Phi_val = np.exp(A_val * dt)
                            inv_A = 1.0 / A_val
                            Psi1_val = inv_A * (Phi_val - 1.0) * B_val
                            Psi2_val = (inv_A**2) * (Phi_val - 1.0 - A_val * dt) * B_val
                        
                        Gamma1_val = Psi1_val - Psi2_val / dt
                        Gamma2_val = Psi2_val / dt
                        
                        Phi = np.array([[Phi_val]], dtype=np.float64)
                        Gamma1 = np.array([Gamma1_val], dtype=np.float64)
                        Gamma2 = np.array([Gamma2_val], dtype=np.float64)
                        C_arr = np.array([C_val], dtype=np.float64)
                        x0 = np.zeros(1, dtype=np.float64)
                        
                        yout = simulate_foh(Phi, Gamma1, Gamma2, C_arr, D_val, u, x0)
                        return {"yout": yout.tolist()}

                # Convert to State Space
                A, B, C, D = signal.tf2ss(num, den)
                
                n = A.shape[0]
                
                # Construct M for FOH
                M = np.zeros((n + 2, n + 2))
                M[:n, :n] = A
                M[:n, n] = B.flatten()
                M[n, n+1] = 1.0
                
                # Matrix exponential
                E = linalg.expm(M * dt)
                
                Phi = E[:n, :n]
                Psi1 = E[:n, n]
                Psi2 = E[:n, n+1]
                
                Gamma1 = Psi1 - Psi2 / dt
                Gamma2 = Psi2 / dt
                
                C_arr = C.flatten()
                D_val = D.flatten()[0] if D.size > 0 else 0.0
                
                x0 = np.zeros(n, dtype=np.float64)
                
                # Ensure contiguous arrays for Numba
                Phi = np.ascontiguousarray(Phi)
                Gamma1 = np.ascontiguousarray(Gamma1)
                Gamma2 = np.ascontiguousarray(Gamma2)
                C_arr = np.ascontiguousarray(C_arr)
                
                yout = simulate_foh(Phi, Gamma1, Gamma2, C_arr, D_val, u, x0)
                return {"yout": yout.tolist()}
                
            except Exception:
                pass

        # Fallback
        system = signal.lti(num, den)
        tout, yout, xout = signal.lsim(system, u, t)
        return {"yout": yout.tolist()}