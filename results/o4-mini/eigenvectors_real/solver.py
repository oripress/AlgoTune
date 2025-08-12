import os
import multiprocessing
# Set BLAS/MKL/OpenBLAS threads to all available CPUs for parallel eigen decomposition
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(multiprocessing.cpu_count())

import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs):
        # Ensure Fortran-contiguous array to avoid internal copies in LAPACK
        A = np.array(problem, dtype=np.float64, order='F', copy=False)
        # Use SciPy’s divide-and-conquer driver for full symmetric eigensolution, no finiteness checks
        w, v = eigh(A, check_finite=False, driver='evd')
        # Return eigenvalues and eigenvectors in descending order
        return w[::-1].tolist(), v[:, ::-1].T.tolist()