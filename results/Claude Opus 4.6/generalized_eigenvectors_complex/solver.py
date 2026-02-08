import numpy as np
from scipy.linalg.lapack import dggev

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        n = A.shape[0]
        
        if n == 0:
            return ([], [])
        
        if n == 1:
            b_val = B[0, 0]
            if abs(b_val) > 1e-15:
                lam = complex(A[0, 0] / b_val)
            else:
                lam = complex(float('inf'), 0)
            return ([lam], [[complex(1.0, 0.0)]])
        
        if n == 2:
            return self._solve_2x2(A, B)
        
        # Scale matrices for better numerical stability
        norm_B = np.linalg.norm(B)
        if norm_B > 0:
            inv_scale = 1.0 / np.sqrt(norm_B)
        else:
            inv_scale = 1.0
        
        A_work = np.asfortranarray(A * inv_scale, dtype=np.float64)
        B_work = np.asfortranarray(B * inv_scale, dtype=np.float64)
        
        # Call LAPACK dggev directly
        alphar, alphai, beta, vl, vr, info = dggev(
            A_work, B_work, compute_vl=0, compute_vr=1,
            overwrite_a=1, overwrite_b=1
        )
        
        if info != 0:
            from scipy.linalg import eig
            eigenvalues, eigvecs = eig(A_work, B_work)
            norms = np.linalg.norm(eigvecs, axis=0)
            norms[norms < 1e-15] = 1.0
            eigvecs /= norms
            idx = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
            return (eigenvalues[idx].tolist(), eigvecs[:, idx].T.tolist())
        
        # Compute eigenvalues
        abs_beta = np.abs(beta)
        zero_mask = abs_beta <= 1e-15
        safe_beta = np.where(zero_mask, 1.0, beta)
        inv_beta = 1.0 / safe_beta
        
        re_eig = alphar * inv_beta
        im_eig = alphai * inv_beta
        
        if np.any(zero_mask):
            re_eig[zero_mask] = float('inf')
            im_eig[zero_mask] = 0.0
        
        # Sort indices
        idx = np.lexsort((-im_eig, -re_eig))
        
        # Construct complex eigenvectors
        has_complex = np.any(alphai != 0)
        
        if not has_complex:
            sorted_vr = vr[:, idx]
            norms = np.linalg.norm(sorted_vr, axis=0)
            norms[norms < 1e-15] = 1.0
            sorted_vr /= norms
            eigenvectors_out = (sorted_vr + 0j).T.tolist()
        else:
            eigenvectors = np.empty((n, n), dtype=complex)
            i = 0
            while i < n:
                if alphai[i] == 0:
                    eigenvectors[:, i] = vr[:, i]
                    i += 1
                elif i + 1 < n and alphai[i] > 0:
                    re_part = vr[:, i]
                    im_part = vr[:, i + 1]
                    eigenvectors[:, i] = re_part + 1j * im_part
                    eigenvectors[:, i + 1] = re_part - 1j * im_part
                    i += 2
                else:
                    eigenvectors[:, i] = vr[:, i]
                    i += 1
            
            norms = np.linalg.norm(eigenvectors, axis=0)
            norms[norms < 1e-15] = 1.0
            eigenvectors /= norms
            eigenvectors_out = eigenvectors[:, idx].T.tolist()
        
        eigenvalues_out = (re_eig[idx] + 1j * im_eig[idx]).tolist()
        return (eigenvalues_out, eigenvectors_out)
    
    def _solve_2x2(self, A, B):
        """Solve 2x2 generalized eigenvalue problem analytically."""
        a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        e, f, g, h = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
        
        # det(A - λB) = 0
        # (a-λe)(d-λh) - (b-λf)(c-λg) = 0
        # λ²(eh - fg) - λ(ah + de - bg - cf) + (ad - bc) = 0
        
        A_coeff = e * h - f * g  # coefficient of λ²
        B_coeff = -(a * h + d * e - b * g - c * f)  # coefficient of λ
        C_coeff = a * d - b * c  # constant term
        
        import cmath
        
        if abs(A_coeff) < 1e-15:
            # Linear: B_coeff * λ + C_coeff = 0
            if abs(B_coeff) > 1e-15:
                lam1 = complex(-C_coeff / B_coeff)
                # Only one finite eigenvalue, second is inf
                lam2 = complex(float('inf'), 0)
                eigenvalues = [lam1, lam2]
            else:
                eigenvalues = [complex(float('inf'), 0), complex(float('inf'), 0)]
            
            # Compute eigenvectors
            vecs = []
            for lam in eigenvalues:
                if not (cmath.isfinite(lam)):
                    vecs.append([complex(1.0, 0.0), complex(0.0, 0.0)])
                    continue
                # (A - λB)v = 0
                m00 = a - lam * e
                m01 = b - lam * f
                m10 = c - lam * g
                m11 = d - lam * h
                
                # Find null space
                if abs(m00) > abs(m10):
                    if abs(m00) > 1e-15:
                        v = [-m01 / m00, complex(1.0)]
                    else:
                        v = [complex(1.0), complex(0.0)]
                else:
                    if abs(m10) > 1e-15:
                        v = [-m11 / m10, complex(1.0)]
                    else:
                        v = [complex(1.0), complex(0.0)]
                
                norm = abs(cmath.sqrt(v[0] * v[0].conjugate() + v[1] * v[1].conjugate()))
                if norm > 1e-15:
                    v = [v[0] / norm, v[1] / norm]
                vecs.append(v)
            
            # Sort
            pairs = list(zip(eigenvalues, vecs))
            pairs.sort(key=lambda p: (-p[0].real, -p[0].imag))
            eigenvalues_sorted, vecs_sorted = zip(*pairs)
            return (list(eigenvalues_sorted), list(vecs_sorted))
        
        discriminant = B_coeff * B_coeff - 4 * A_coeff * C_coeff
        sqrt_disc = cmath.sqrt(discriminant)
        
        lam1 = (-B_coeff + sqrt_disc) / (2 * A_coeff)
        lam2 = (-B_coeff - sqrt_disc) / (2 * A_coeff)
        
        eigenvalues = [complex(lam1), complex(lam2)]
        
        # Compute eigenvectors
        vecs = []
        for lam in eigenvalues:
            m00 = a - lam * e
            m01 = b - lam * f
            m10 = c - lam * g
            m11 = d - lam * h
            
            if abs(m00) > abs(m10):
                if abs(m00) > 1e-15:
                    v = [-m01 / m00, complex(1.0)]
                else:
                    v = [complex(1.0), complex(0.0)]
            else:
                if abs(m10) > 1e-15:
                    v = [-m11 / m10, complex(1.0)]
                else:
                    v = [complex(1.0), complex(0.0)]
            
            norm = abs(cmath.sqrt(v[0] * v[0].conjugate() + v[1] * v[1].conjugate()))
            if norm > 1e-15:
                v = [v[0] / norm, v[1] / norm]
            vecs.append(v)
        
        # Sort by descending real part, then descending imaginary part
        pairs = list(zip(eigenvalues, vecs))
        pairs.sort(key=lambda p: (-p[0].real, -p[0].imag))
        eigenvalues_sorted, vecs_sorted = zip(*pairs)
        return (list(eigenvalues_sorted), list(vecs_sorted))