import numpy as np
import ctypes
import ctypes.util

# Try to find and load LAPACK
_lapack = None
for name in ['lapack', 'openblas', 'mkl_rt', 'blas']:
    path = ctypes.util.find_library(name)
    if path:
        try:
            _lapack = ctypes.CDLL(path)
            # Check if dgeqrf_ exists
            _lapack.dgeqrf_
            break
        except (OSError, AttributeError):
            _lapack = None

if _lapack is None:
    try:
        import scipy.linalg
        # Find the LAPACK library from scipy
        import os
        scipy_dir = os.path.dirname(scipy.linalg.__file__)
        for f in os.listdir(scipy_dir):
            if 'lapack' in f.lower() and f.endswith('.so'):
                try:
                    _lapack = ctypes.CDLL(os.path.join(scipy_dir, f))
                    _lapack.dgeqrf_
                    break
                except (OSError, AttributeError):
                    _lapack = None
    except Exception:
        pass

# Fallback imports
try:
    from qr_cy import qr_decomp_fast, qr_workspace_query
    _CY = True
except ImportError:
    _CY = False

from scipy.linalg.lapack import dgeqrf as _dgeqrf, dorgqr as _dorgqr

class Solver:
    def __init__(self):
        self._c = {}
    
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.float64)
        
        n = A.shape[0]
        m = A.shape[1]
        
        c = self._c.get((n, m))
        if c is None:
            if _CY:
                lq, lqr = qr_workspace_query(n, m)
                mx = max(lq, lqr)
                c = (lq, lqr, np.empty(mx, dtype=np.float64), np.empty(n, dtype=np.float64))
            else:
                tmp = np.empty((n, m), dtype=np.float64, order='F')
                _, _, wqr, _ = _dgeqrf(tmp, lwork=-1, overwrite_a=True)
                lq = int(wqr[0])
                tmp2 = np.empty((n, n), dtype=np.float64, order='F')
                tau_tmp = np.empty(n, dtype=np.float64)
                _, wq, _ = _dorgqr(tmp2, tau_tmp, lwork=-1, overwrite_a=True)
                lqr = int(wq[0])
                c = (lq, lqr)
            self._c[(n, m)] = c
        
        if _CY:
            Q, R = qr_decomp_fast(A, c[0], c[1], c[2], c[3])
        else:
            Af = np.asfortranarray(A, dtype=np.float64)
            qr, tau, _, _ = _dgeqrf(Af, lwork=c[0], overwrite_a=True)
            R = np.triu(qr[:n, :m])
            q = qr[:n, :n].copy(order='F')
            Q, _, _ = _dorgqr(q, tau, lwork=c[1], overwrite_a=True)
        
        return {"QR": {"Q": Q, "R": R}}