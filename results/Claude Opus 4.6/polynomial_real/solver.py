import numpy as np
from scipy.linalg import lapack
import math
import numba as nb

_dgebal = lapack.dgebal
_dhseqr = lapack.dhseqr

@nb.njit(cache=True)
def _solve_deg2(p0, p1, p2):
    disc = p1*p1 - 4*p0*p2
    if disc < 0:
        disc = 0.0
    sd = math.sqrt(disc)
    inv_2a = 0.5 / p0
    r1 = (-p1 + sd) * inv_2a
    r2 = (-p1 - sd) * inv_2a
    if r1 >= r2:
        return r1, r2
    return r2, r1

@nb.njit(cache=True)
def _solve_deg3(p0, p1, p2, p3):
    inv_a = 1.0 / p0
    a = p1 * inv_a
    b = p2 * inv_a
    c = p3 * inv_a
    Q = (a*a - 3.0*b) / 9.0
    R = (2.0*a*a*a - 9.0*a*b + 27.0*c) / 54.0
    if Q < 0:
        Q = 0.0
    sqQ = math.sqrt(Q)
    shift = -a / 3.0
    if sqQ > 1e-30:
        ratio = R / (sqQ * sqQ * sqQ)
        if ratio > 1.0:
            ratio = 1.0
        elif ratio < -1.0:
            ratio = -1.0
        theta = math.acos(ratio)
        m = -2.0 * sqQ
        r1 = m * math.cos(theta / 3.0) + shift
        r2 = m * math.cos((theta + 2.0943951023931953) / 3.0) + shift
        r3 = m * math.cos((theta - 2.0943951023931953) / 3.0) + shift
    else:
        r1 = r2 = r3 = shift
    if r1 < r2:
        r1, r2 = r2, r1
    if r2 < r3:
        r2, r3 = r3, r2
    if r1 < r2:
        r1, r2 = r2, r1
    return r1, r2, r3

@nb.njit(cache=True)
def _solve_deg4(p0, p1, p2, p3, p4):
    inv_a = 1.0 / p0
    b = p1 * inv_a
    c = p2 * inv_a
    d = p3 * inv_a
    e = p4 * inv_a
    b4 = b * 0.25
    p_val = c - 6.0*b4*b4
    q_val = b*b*b*0.125 - b*c*0.5 + d
    r_val = e + b4*(-d + b4*(c - 3.0*b4*b4))
    
    roots = np.empty(4)
    if abs(q_val) < 1e-30:
        disc = p_val*p_val - 4.0*r_val
        if disc < 0:
            disc = 0.0
        sd = math.sqrt(disc)
        u1 = (-p_val + sd) * 0.5
        u2 = (-p_val - sd) * 0.5
        if u1 < 0:
            u1 = 0.0
        if u2 < 0:
            u2 = 0.0
        su1 = math.sqrt(u1)
        su2 = math.sqrt(u2)
        roots[0] = su1 - b4
        roots[1] = -su1 - b4
        roots[2] = su2 - b4
        roots[3] = -su2 - b4
    else:
        rc_b = -p_val * 0.5
        rc_c = -r_val
        rc_d = (4.0*p_val*r_val - q_val*q_val) * 0.125
        Q = (rc_b*rc_b - 3.0*rc_c) / 9.0
        R = (2.0*rc_b*rc_b*rc_b - 9.0*rc_b*rc_c + 27.0*rc_d) / 54.0
        if Q < 0:
            Q = 0.0
        sqQ = math.sqrt(Q)
        if sqQ > 1e-30:
            ratio = R / (sqQ * sqQ * sqQ)
            if ratio > 1.0:
                ratio = 1.0
            elif ratio < -1.0:
                ratio = -1.0
            theta = math.acos(ratio)
            m = -2.0 * sqQ * math.cos(theta / 3.0) - rc_b / 3.0
        else:
            m = -rc_b / 3.0
        disc1 = 2.0*m + p_val
        if disc1 < 0:
            disc1 = 0.0
        s = math.sqrt(disc1)
        if s > 1e-30:
            q2s = q_val / (2.0 * s)
            d1 = s*s - 4.0*(m - q2s)
            if d1 < 0:
                d1 = 0.0
            sd1 = math.sqrt(d1)
            roots[0] = (-s + sd1) * 0.5 - b4
            roots[1] = (-s - sd1) * 0.5 - b4
            d2 = s*s - 4.0*(m + q2s)
            if d2 < 0:
                d2 = 0.0
            sd2 = math.sqrt(d2)
            roots[2] = (s + sd2) * 0.5 - b4
            roots[3] = (s - sd2) * 0.5 - b4
        else:
            d1 = -4.0*m
            if d1 < 0:
                d1 = 0.0
            sd1 = math.sqrt(d1)
            roots[0] = sd1*0.5 - b4
            roots[1] = -sd1*0.5 - b4
            roots[2] = sd1*0.5 - b4
            roots[3] = -sd1*0.5 - b4
    for i in range(4):
        for j in range(i+1, 4):
            if roots[j] > roots[i]:
                roots[i], roots[j] = roots[j], roots[i]
    return roots[0], roots[1], roots[2], roots[3]

@nb.njit(cache=True)
def _build_companion_and_solve_small(coeffs, N):
    """Build companion matrix and compute eigenvalues for small N using QR iteration."""
    # This is for N=5..~15 where LAPACK overhead is large relative to computation
    inv_a = 1.0 / coeffs[0]
    
    # Build companion matrix (upper Hessenberg)
    A = np.zeros((N, N))
    for j in range(N):
        A[0, j] = -coeffs[j+1] * inv_a
    for i in range(1, N):
        A[i, i-1] = 1.0
    
    # Simple QR iteration with shifts for Hessenberg matrix
    # This is a basic implementation - may need refinement
    n = N
    roots = np.empty(N)
    root_idx = 0
    
    for iteration in range(1000):
        if n <= 0:
            break
        if n == 1:
            roots[root_idx] = A[0, 0]
            root_idx += 1
            break
        if n == 2:
            a11 = A[0, 0]
            a12 = A[0, 1]
            a21 = A[1, 0]
            a22 = A[1, 1]
            trace = a11 + a22
            det = a11*a22 - a12*a21
            disc = trace*trace - 4*det
            if disc < 0:
                disc = 0.0
            sd = math.sqrt(disc)
            roots[root_idx] = (trace + sd) * 0.5
            roots[root_idx+1] = (trace - sd) * 0.5
            root_idx += 2
            break
        
        # Check for convergence (bottom element)
        if abs(A[n-1, n-2]) < 1e-15 * (abs(A[n-1, n-1]) + abs(A[n-2, n-2])):
            roots[root_idx] = A[n-1, n-1]
            root_idx += 1
            n -= 1
            continue
        
        # Wilkinson shift
        a11 = A[n-2, n-2]
        a12 = A[n-2, n-1]
        a21 = A[n-1, n-2]
        a22 = A[n-1, n-1]
        trace = a11 + a22
        det = a11*a22 - a12*a21
        disc = trace*trace - 4*det
        if disc < 0:
            disc = 0.0
        sd = math.sqrt(disc)
        s1 = (trace + sd) * 0.5
        s2 = (trace - sd) * 0.5
        if abs(s1 - a22) < abs(s2 - a22):
            shift = s1
        else:
            shift = s2
        
        # QR step with shift on Hessenberg matrix using Givens rotations
        # A - shift*I
        for i in range(n):
            A[i, i] -= shift
        
        # QR factorization using Givens rotations
        cs = np.empty(n-1)
        sn = np.empty(n-1)
        for i in range(n-1):
            a = A[i, i]
            b = A[i+1, i]
            if abs(b) < 1e-30:
                cs[i] = 1.0
                sn[i] = 0.0
            else:
                r = math.sqrt(a*a + b*b)
                cs[i] = a / r
                sn[i] = b / r
            # Apply rotation to rows i and i+1
            for j in range(i, n):
                t1 = A[i, j]
                t2 = A[i+1, j]
                A[i, j] = cs[i]*t1 + sn[i]*t2
                A[i+1, j] = -sn[i]*t1 + cs[i]*t2
        
        # R * Q
        for i in range(n-1):
            for j in range(min(i+2, n)):
                t1 = A[j, i]
                t2 = A[j, i+1]
                A[j, i] = cs[i]*t1 + sn[i]*t2
                A[j, i+1] = -sn[i]*t1 + cs[i]*t2
        
        # Add shift back
        for i in range(n):
            A[i, i] += shift
    
    return roots


# Warm up
_solve_deg2(1.0, -3.0, 2.0)
_solve_deg3(1.0, -6.0, 11.0, -6.0)
_solve_deg4(1.0, -10.0, 35.0, -50.0, 24.0)
_p = np.array([1.0, -15.0, 85.0, -225.0, 274.0, -120.0])
_build_companion_and_solve_small(_p, 5)


class Solver:
    def solve(self, problem, **kwargs):
        coefficients = problem
        n = len(coefficients)
        
        start = 0
        while start < n and coefficients[start] == 0:
            start += 1
        if start == n:
            return []
        
        end = n - 1
        while end > start and coefficients[end] == 0:
            end -= 1
        
        trailing_zeros = n - 1 - end
        N = end - start
        
        if N <= 0:
            return [0.0] * trailing_zeros if trailing_zeros > 0 else []
        
        cs = coefficients
        
        if N == 1:
            r = -cs[start+1] / cs[start]
            if trailing_zeros == 0:
                return [r]
            result = [r] + [0.0] * trailing_zeros
            result.sort(reverse=True)
            return result
        
        if N == 2:
            r1, r2 = _solve_deg2(cs[start], cs[start+1], cs[start+2])
            if trailing_zeros == 0:
                return [r1, r2]
            result = [r1, r2] + [0.0] * trailing_zeros
            result.sort(reverse=True)
            return result
        
        if N == 3:
            r1, r2, r3 = _solve_deg3(cs[start], cs[start+1], cs[start+2], cs[start+3])
            if trailing_zeros == 0:
                return [r1, r2, r3]
            result = [r1, r2, r3] + [0.0] * trailing_zeros
            result.sort(reverse=True)
            return result
        
        if N == 4:
            r1, r2, r3, r4 = _solve_deg4(cs[start], cs[start+1], cs[start+2], cs[start+3], cs[start+4])
            if trailing_zeros == 0:
                return [r1, r2, r3, r4]
            result = [r1, r2, r3, r4] + [0.0] * trailing_zeros
            result.sort(reverse=True)
            return result
        
        # For small N (5-15), use numba QR iteration
        if N <= 15:
            p = np.array(coefficients[start:end+1], dtype=np.float64)
            roots = _build_companion_and_solve_small(p, N)
            if trailing_zeros > 0:
                roots = np.concatenate([roots, np.zeros(trailing_zeros)])
            roots.sort()
            return roots[::-1].tolist()
        
        # For larger N, use LAPACK
        p = np.array(coefficients[start:end+1], dtype=np.float64)
        inv_p0 = 1.0 / p[0]
        
        A_flat = np.zeros(N * N, dtype=np.float64)
        A_flat[::N] = -p[1:] * inv_p0
        A_flat[1::N+1] = 1.0
        A = A_flat.reshape((N, N), order='F')
        
        A, lo, hi, scale, info = _dgebal('B', A, overwrite_a=1)
        result_lapack = _dhseqr('E', 'N', N, lo, hi, A, overwrite_a=1)
        roots = result_lapack[0]
        
        if trailing_zeros > 0:
            roots = np.concatenate([roots, np.zeros(trailing_zeros)])
        
        roots.sort()
        return roots[::-1].tolist()