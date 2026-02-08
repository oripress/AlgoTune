import numpy as np
import numba as nb
from scipy.linalg import solve_toeplitz

@nb.njit(cache=True)
def levinson_nb(a, y):
    """
    Direct port of scipy's levinson to numba.
    a: Toeplitz vector of length 2*n-1, T[i,j] = a[(n-1) + i - j]
    y: RHS vector of length n
    """
    n = len(y)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    
    M = n - 1  # index of the main diagonal in a
    
    if n == 1:
        return np.array([y[0] / a[M]])
    
    # Based on scipy's levinson implementation:
    # https://github.com/scipy/scipy/blob/main/scipy/linalg/_solve_toeplitz.py
    
    # Using modified Levinson-Trench-Zohar algorithm
    # Following Golub & Van Loan notation
    
    An = np.zeros(n)  # forward polynomial coefficients
    Cn = np.zeros(n)  # backward polynomial coefficients
    An_new = np.zeros(n)
    Cn_new = np.zeros(n)
    
    x = np.zeros(n)
    x_new = np.zeros(n)
    
    # Normalize by a[M]
    t0 = a[M]
    
    # Initialize
    An[0] = 1.0
    Cn[0] = 1.0
    
    # Solve step 0: x[0] = y[0] / t0
    x[0] = y[0] / t0
    
    # For the forward prediction: sum of An[j] * a[M+1-j]
    # Actually, let me follow the scipy code precisely
    
    # In scipy source:
    # An, Cn are coefficient vectors of length M+1
    # The recurrence is:
    #   alpha_n = sum_j (An[j] * a[M-n+j]) for j=0..n-1
    #   beta_n  = sum_j (Cn[j] * a[M+n-j]) for j=0..n-1
    #   kn = -(alpha_n) / ...
    
    # Let me just implement it step by step matching scipy exactly
    
    # Step 0 completed above
    # Now iterate
    
    alpha = a[M + 1]  # first subdiagonal
    beta = a[M - 1]   # first superdiagonal
    
    # gamma is the error/denominator
    gamma = t0
    
    # Update An for step 1
    An_new[0] = 1.0
    An_new[1] = -alpha / gamma
    
    Cn_new[0] = -beta / gamma
    Cn_new[1] = 1.0
    
    # Update gamma
    gamma_new = gamma - alpha * beta / gamma
    # equivalently gamma_new = gamma * (1 - (alpha/gamma) * (beta/gamma))
    
    # Update x for step 1
    # lambda_n = y[1] - sum_j (x[j] * a[M+1-j]) for j=0..0
    lambda_1 = y[1] - x[0] * a[M + 1]
    
    x_new[0] = x[0] + (lambda_1 / gamma_new) * Cn_new[0]
    x_new[1] = (lambda_1 / gamma_new) * Cn_new[1]
    
    # Copy back
    for j in range(2):
        An[j] = An_new[j]
        Cn[j] = Cn_new[j]
        x[j] = x_new[j]
    gamma = gamma_new
    
    for i in range(2, n):
        # Compute alpha_n and beta_n
        alpha_n = 0.0
        beta_n = 0.0
        for j in range(i):
            alpha_n += An[j] * a[M + i - j]
            beta_n += Cn[j] * a[M - i + j]
        
        # Reflection coefficients
        kn = -alpha_n / gamma
        mu = -beta_n / gamma
        
        # Update An
        An_new[0] = 1.0  # An always starts with 1
        for j in range(1, i):
            An_new[j] = An[j] + kn * Cn[j - 1]
        An_new[i] = kn * Cn[i - 1]  # = kn since Cn[i-1] = 1 (last element)
        
        # Update Cn
        for j in range(i - 1):
            Cn_new[j] = Cn[j] + mu * An[j + 1]
        Cn_new[i - 1] = mu * An[i - 1]  # Hmm, need to check
        Cn_new[i] = 1.0  # Cn always ends with 1? No...
        
        # Wait, I think I'm getting confused. Let me be more careful.
        # Actually, Cn_new should end with 1? Or maybe not.
        
        # Let me reconsider. Looking at scipy source more carefully...
        
        # Actually, from the scipy source I inspected:
        # An[0] = 1 always
        # Cn[n-1] = 1 always (Cn[-1] in 0-indexed for current order)
        
        # So Cn is a backwards polynomial with Cn[current_order - 1] = 1
        # For order i:
        # Cn has nonzero entries from 0 to i-1
        # Cn[i-1] = 1
        
        # Let me re-do this. For the Cn update:
        # Cn_new[j] = mu * An[j+1] + Cn[j] for j=0..i-2
        # Cn_new[i-1] = mu * An[i] + Cn[i-1]  -- but An[i] is 0 and Cn[i-1] = 1
        # Hmm, An has entries 0..i-1 (for current order i)
        # After update, An has entries 0..i
        
        # I think the issue is that An has i entries (0..i-1) before update
        # and i+1 entries (0..i) after update
        
        # Let me just trust the formulas and fix end points:
        
        # For An update (length goes from i to i+1):
        An_new[0] = 1.0
        for j in range(1, i):
            An_new[j] = An[j] + kn * Cn[i - 1 - j]
        An_new[i] = kn  # kn * Cn[i-1-(i-1)] = kn * Cn[0]? No...
        
        # Hmm this is wrong. Let me think again.
        # Actually in the standard Levinson recursion:
        # An_new = An + kn * Cn_reversed
        # where Cn_reversed[j] = Cn[i-1-j]
        
        # So An_new[j] = An[j] + kn * Cn[i-1-j] for j=0..i-1
        # An_new[i] = 0 + kn * ... need to extend
        
        # I think the arrays need careful handling of lengths
        # This is getting very confusing. Let me take a different approach.
        break
    
    return x


class Solver:
    def __init__(self):
        # Warm up numba JIT
        a_test = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        y_test = np.array([1.0, 1.0], dtype=np.float64)
        levinson_nb(a_test, y_test)
    
    def solve(self, problem, **kwargs):
        c = np.array(problem["c"], dtype=np.float64)
        r = np.array(problem["r"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        return solve_toeplitz((c, r), b, check_finite=False).tolist()