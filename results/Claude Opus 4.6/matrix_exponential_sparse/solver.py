import numpy as np
from scipy import sparse
from scipy.linalg import expm as dense_expm
from scipy.linalg.lapack import dgesv

# Theta values from Higham (2005), Table 10.2
_THETA3 = 1.495585217958292e-2
_THETA5 = 2.539398330063230e-1
_THETA7 = 9.504178996162932e-1
_THETA9 = 2.097847961257068
_THETA13 = 5.371920351148152

# Pade-13 coefficients
_b13 = (64764752532480000., 32382376266240000., 7771770303897600.,
        1187353796428800., 129060195264000., 10559470521600.,
        670442572800., 33522128640., 1323241920., 40840800.,
        960960., 16380., 182., 1.)


def _fast_expm(A):
    """Fast matrix exponential using Pade approximation with scaling and squaring."""
    n = A.shape[0]
    
    # Shift by trace/n to reduce norm
    mu = np.trace(A) / n
    if mu != 0:
        np.fill_diagonal(A, A.diagonal() - mu)
    
    nA = np.linalg.norm(A, 1)
    
    if nA == 0:
        result = np.eye(n)
        if mu != 0:
            result *= np.exp(mu)
        return result
    
    I = np.eye(n)
    
    if nA <= _THETA3:
        A2 = A @ A
        U = A @ (A2 + 60.*I)
        V = 12.*A2 + 120.*I
    elif nA <= _THETA5:
        A2 = A @ A
        A4 = A2 @ A2
        U = A @ (A4 + 420.*A2 + 15120.*I)
        V = 30.*A4 + 3360.*A2 + 30240.*I
    elif nA <= _THETA7:
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A2 @ A4
        U = A @ (A6 + 1512.*A4 + 277200.*A2 + 8648640.*I)
        V = 56.*A6 + 25200.*A4 + 1995840.*A2 + 17297280.*I
    elif nA <= _THETA9:
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A2 @ A4
        A8 = A4 @ A4
        U = A @ (A8 + 3960.*A6 + 2162160.*A4 + 302702400.*A2 + 8821612800.*I)
        V = 90.*A8 + 110880.*A6 + 30270240.*A4 + 2075673600.*A2 + 17643225600.*I
    else:
        # Order 13 with scaling
        s = max(0, int(np.ceil(np.log2(nA / _THETA13))))
        if s > 0:
            A = A / (2. ** s)
        
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A2 @ A4
        
        b = _b13
        
        inner_u = b[13]*A6 + b[11]*A4 + b[9]*A2
        W = A6 @ inner_u
        W += b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I
        U = A @ W
        
        inner_v = b[12]*A6 + b[10]*A4 + b[8]*A2
        V = A6 @ inner_v
        V += b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
        
        VmU = V - U
        VpU = V + U
        _, _, X, _ = dgesv(VmU, VpU, overwrite_a=True, overwrite_b=True)
        
        for _ in range(s):
            X = X @ X
        
        if mu != 0:
            X *= np.exp(mu)
        return X
    
    # For lower orders (no scaling needed)
    VmU = V - U
    VpU = V + U
    _, _, X, _ = dgesv(VmU, VpU, overwrite_a=True, overwrite_b=True)
    
    if mu != 0:
        X *= np.exp(mu)
    return X


class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        n = A.shape[0]
        nnz = A.nnz
        
        if nnz == 0:
            return sparse.eye(n, format='csc')
        
        if n == 1:
            val = np.exp(A[0, 0])
            if val == 0:
                return sparse.csc_matrix((1, 1))
            return sparse.csc_matrix(np.array([[val]]))
        
        # Check if diagonal - fast path
        if nnz <= n:
            diag_vals = A.diagonal()
            off_diag_nnz = nnz - np.count_nonzero(diag_vals)
            if off_diag_nnz == 0:
                exp_diag = np.exp(diag_vals)
                return sparse.diags(exp_diag, offsets=0, shape=(n, n), format='csc')
        
        # Check for block-diagonal structure
        density = nnz / (n * n) if n > 0 else 1
        if density < 0.1 and n > 20:
            A_pattern = A.copy()
            A_pattern.data = np.ones_like(A_pattern.data)
            A_sym = A_pattern + A_pattern.T
            n_components, labels = sparse.csgraph.connected_components(
                A_sym, directed=False)
            
            if n_components > 1:
                return self._block_expm(A, n, n_components, labels)
        
        # Fast symmetry check
        is_symmetric = (A - A.T).nnz == 0
        
        A_dense = A.toarray()
        
        if is_symmetric:
            eigenvalues, Q = np.linalg.eigh(A_dense)
            exp_eigenvalues = np.exp(eigenvalues)
            result_dense = (Q * exp_eigenvalues) @ Q.T
        else:
            result_dense = _fast_expm(A_dense)
        
        return sparse.csc_matrix(result_dense)
    
    def _block_expm(self, A, n, n_components, labels):
        """Compute matrix exponential for block-diagonal matrices."""
        result_dense = np.zeros((n, n), dtype=np.float64)
        
        for comp in range(n_components):
            idx = np.where(labels == comp)[0]
            block_size = len(idx)
            
            if block_size == 1:
                result_dense[idx[0], idx[0]] = np.exp(A[idx[0], idx[0]])
            else:
                block = A[np.ix_(idx, idx)].toarray()
                
                if np.allclose(block, block.T, rtol=0, atol=1e-14):
                    eigenvalues, Q = np.linalg.eigh(block)
                    exp_eig = np.exp(eigenvalues)
                    exp_block = (Q * exp_eig) @ Q.T
                else:
                    exp_block = _fast_expm(block)
                
                result_dense[np.ix_(idx, idx)] = exp_block
        
        return sparse.csc_matrix(result_dense)