import numpy as np
from typing import Any

try:
    import cupy as xp
    GPU_ENABLED = True

    # --- Custom CUDA Kernels for Memory-Efficient Sinkhorn ---
    # These kernels compute operations involving K = exp(-M/reg) without
    # explicitly materializing the large kernel matrix K, saving memory and bandwidth.

    # Computes: out = K @ v = exp(-M/reg) @ v
    _matvec_kernel = xp.RawKernel(r'''
    extern "C" __global__
    void matvec(const double* M, const double* v, double reg, int n, int m, double* out) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;

        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += exp(-M[i * m + j] / reg) * v[j];
        }
        out[i] = sum;
    }
    ''', 'matvec')

    # Computes: out = K.T @ u = exp(-M.T/reg) @ u
    _matvec_T_kernel = xp.RawKernel(r'''
    extern "C" __global__
    void matvec_T(const double* M, const double* u, double reg, int n, int m, double* out) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= m) return;

        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += exp(-M[i * m + j] / reg) * u[i];
        }
        out[j] = sum;
    }
    ''', 'matvec_T')

    # Computes: G_ij = u_i * exp(-M_ij/reg) * v_j
    _reconstruct_kernel = xp.RawKernel(r'''
    extern "C" __global__
    void reconstruct(const double* u, const double* v, const double* M, double reg, int n, int m, double* G) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        if (i < n && j < m) {
            int idx = i * m + j;
            G[idx] = u[i] * exp(-M[idx] / reg) * v[j];
        }
    }
    ''', 'reconstruct')

except ImportError:
    xp = np
    GPU_ENABLED = False

def _sinkhorn_gpu_kernel(a, b, M, reg, max_iter, tol):
    """Runs the Sinkhorn loop using custom CUDA kernels."""
    n, m = M.shape
    u = xp.ones(n, dtype=xp.float64)
    
    # Pre-allocate output arrays for matrix-vector products
    KTu = xp.empty(m, dtype=xp.float64)
    Kv = xp.empty(n, dtype=xp.float64)

    # CUDA kernel launch parameters
    block_size_n = 256
    grid_size_n = (n + block_size_n - 1) // block_size_n
    block_size_m = 256
    grid_size_m = (m + block_size_m - 1) // block_size_m

    for i in range(max_iter):
        # Update v = b / (K.T @ u)
        _matvec_T_kernel((grid_size_m,), (block_size_m,), (M, u, reg, n, m, KTu))
        v = b / KTu
        
        # Update u = a / (K @ v)
        _matvec_kernel((grid_size_n,), (block_size_n,), (M, v, reg, n, m, Kv))
        u = a / Kv

        if i % 10 == 0:
            err = xp.linalg.norm(u * Kv - a)
            if err < tol:
                break
            if not xp.all(xp.isfinite(u)) or not xp.all(xp.isfinite(v)):
                break
    
    # Final update for v to ensure consistency
    _matvec_T_kernel((grid_size_m,), (block_size_m,), (M, u, reg, n, m, KTu))
    v = b / KTu
    return u, v

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        a_np = np.array(problem["source_weights"], dtype=np.float64)
        b_np = np.array(problem["target_weights"], dtype=np.float64)
        M_np = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])

        max_iter = 1000
        tol = 1e-9

        try:
            if GPU_ENABLED:
                # --- Main Path: Custom CUDA Kernels ---
                a, b, M = xp.asarray(a_np), xp.asarray(b_np), xp.asarray(M_np)
                
                u, v = _sinkhorn_gpu_kernel(a, b, M, reg, max_iter, tol)

                # Reconstruct G using the custom kernel
                n, m = M.shape
                G_gpu = xp.empty_like(M)
                block_dim = (16, 16)
                grid_dim = ((m + block_dim[0] - 1) // block_dim[0], (n + block_dim[1] - 1) // block_dim[1])
                _reconstruct_kernel(grid_dim, block_dim, (u, v, M, reg, n, m, G_gpu))
                G = xp.asnumpy(G_gpu)
            else:
                # --- Fallback Path: NumPy on CPU ---
                K = np.exp(-M_np / reg)
                u = np.ones(a_np.shape[0], dtype=np.float64)
                for i in range(max_iter):
                    v = b_np / (K.T @ u)
                    Kv = K @ v
                    u = a_np / Kv
                    if i % 10 == 0 and np.linalg.norm(u * Kv - a_np) < tol:
                        break
                v = b_np / (K.T @ u)
                G = u[:, None] * K * v[None, :]

            if not np.isfinite(G).all():
                raise ValueError("Non-finite values in transport plan. Try increasing 'reg'.")

            return {"transport_plan": G}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}