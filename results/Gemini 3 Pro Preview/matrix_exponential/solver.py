import numpy as np
from scipy.linalg import expm
from numba import njit
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jexpm

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)

@njit(fastmath=True)
def expm_2x2(A):
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]
    
    trace = a + d
    det = a * d - b * c
    delta = trace*trace - 4*det
    
    prefactor = np.exp(trace / 2.0)
    
    if delta > 0:
        root_delta = np.sqrt(delta)
        x = root_delta / 2.0
        cosh_x = np.cosh(x)
        if x < 1e-12:
            sinh_x_div_x = 1.0
        else:
            sinh_x_div_x = np.sinh(x) / x
    elif delta < 0:
        root_delta = np.sqrt(-delta)
        x = root_delta / 2.0
        cosh_x = np.cos(x)
        if x < 1e-12:
            sinh_x_div_x = 1.0
        else:
            sinh_x_div_x = np.sin(x) / x
    else:
        cosh_x = 1.0
        sinh_x_div_x = 1.0
        
    t_half = trace / 2.0
    
    m00 = (a - d) / 2.0
    m01 = b
    m10 = c
    m11 = (d - a) / 2.0
    
    r00 = cosh_x + sinh_x_div_x * m00
    r01 = sinh_x_div_x * m01
    r10 = sinh_x_div_x * m10
    r11 = cosh_x + sinh_x_div_x * m11
    
    res = np.empty((2, 2), dtype=np.float64)
    res[0, 0] = prefactor * r00
    res[0, 1] = prefactor * r01
    res[1, 0] = prefactor * r10
    res[1, 1] = prefactor * r11
    
    return res

@njit(fastmath=True)
def expm_3x3(A):
    norm = 0.0
    for i in range(3):
        row_sum = 0.0
        for j in range(3):
            row_sum += np.abs(A[i, j])
        if row_sum > norm:
            norm = row_sum
            
    s = 0
    if norm > 0.5:
        s = int(np.ceil(np.log2(norm))) + 1
    if s < 0: s = 0
    
    scale = 1.0
    if s > 0:
        scale = 2.0**(-s)
        
    a00 = A[0,0]*scale; a01 = A[0,1]*scale; a02 = A[0,2]*scale
    a10 = A[1,0]*scale; a11 = A[1,1]*scale; a12 = A[1,2]*scale
    a20 = A[2,0]*scale; a21 = A[2,1]*scale; a22 = A[2,2]*scale
    
    r00=1.0; r01=0.0; r02=0.0
    r10=0.0; r11=1.0; r12=0.0
    r20=0.0; r21=0.0; r22=1.0
    
    t00=1.0; t01=0.0; t02=0.0
    t10=0.0; t11=1.0; t12=0.0
    t20=0.0; t21=0.0; t22=1.0
    
    for k in range(1, 13):
        n00 = t00*a00 + t01*a10 + t02*a20
        n01 = t00*a01 + t01*a11 + t02*a21
        n02 = t00*a02 + t01*a12 + t02*a22
        
        n10 = t10*a00 + t11*a10 + t12*a20
        n11 = t10*a01 + t11*a11 + t12*a21
        n12 = t10*a02 + t11*a12 + t12*a22
        
        n20 = t20*a00 + t21*a10 + t22*a20
        n21 = t20*a01 + t21*a11 + t22*a21
        n22 = t20*a02 + t21*a12 + t22*a22
        
        inv_k = 1.0 / k
        t00=n00*inv_k; t01=n01*inv_k; t02=n02*inv_k
        t10=n10*inv_k; t11=n11*inv_k; t12=n12*inv_k
        t20=n20*inv_k; t21=n21*inv_k; t22=n22*inv_k
        
        r00+=t00; r01+=t01; r02+=t02
        r10+=t10; r11+=t11; r12+=t12
        r20+=t20; r21+=t21; r22+=t22
        
    for _ in range(s):
        n00 = r00*r00 + r01*r10 + r02*r20
        n01 = r00*r01 + r01*r11 + r02*r21
        n02 = r00*r02 + r01*r12 + r02*r22
        
        n10 = r10*r00 + r11*r10 + r12*r20
        n11 = r10*r01 + r11*r11 + r12*r21
        n12 = r10*r02 + r11*r12 + r12*r22
        
        n20 = r20*r00 + r21*r10 + r22*r20
        n21 = r20*r01 + r21*r11 + r22*r21
        n22 = r20*r02 + r21*r12 + r22*r22
        
        r00=n00; r01=n01; r02=n02
        r10=n10; r11=n11; r12=n12
        r20=n20; r21=n21; r22=n22
        
    res = np.empty((3, 3), dtype=np.float64)
    res[0,0]=r00; res[0,1]=r01; res[0,2]=r02
    res[1,0]=r10; res[1,1]=r11; res[1,2]=r12
    res[2,0]=r20; res[2,1]=r21; res[2,2]=r22
    return res

@njit(fastmath=True)
def expm_4x4(A):
    norm = 0.0
    for i in range(4):
        row_sum = 0.0
        for j in range(4):
            row_sum += np.abs(A[i, j])
        if row_sum > norm:
            norm = row_sum
            
    s = 0
    if norm > 0.5:
        s = int(np.ceil(np.log2(norm))) + 1
    if s < 0: s = 0
    
    scale = 1.0
    if s > 0:
        scale = 2.0**(-s)
        
    A_scaled = A * scale
    
    res = np.eye(4, dtype=np.float64)
    term = np.eye(4, dtype=np.float64)
    
    for k in range(1, 13):
        term = term @ A_scaled
        term = term * (1.0 / k)
        res = res + term
        
    for _ in range(s):
        res = res @ res
        
    return res

class Solver:
    def __init__(self):
        # Warmup numba functions
        dummy2 = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        try:
            expm_2x2(dummy2)
        except Exception:
            pass
            
        dummy3 = np.eye(3, dtype=np.float64)
        try:
            expm_3x3(dummy3)
        except Exception:
            pass
            
        dummy4 = np.eye(4, dtype=np.float64)
        try:
            expm_4x4(dummy4)
        except Exception:
            pass
            
        # JAX setup
        self.jexpm_jit = jax.jit(jexpm)
        
        # Warmup for common sizes
        common_sizes = [5, 6, 8, 10, 16, 32, 64]
        for n in common_sizes:
            try:
                dummy = jnp.eye(n)
                self.jexpm_jit(dummy)
            except Exception:
                pass

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
        matrix_data = problem["matrix"]
        
        if isinstance(matrix_data, list):
            A = np.array(matrix_data, dtype=np.float64)
        else:
            A = matrix_data
            if A.dtype != np.float64:
                A = A.astype(np.float64)
        
        n = A.shape[0]
        
        if n == 2:
            return {"exponential": expm_2x2(A)}
        elif n == 3:
            return {"exponential": expm_3x3(A)}
        elif n == 4:
            return {"exponential": expm_4x4(A)}
        
        # Use JAX for others
        try:
            res = self.jexpm_jit(A)
            return {"exponential": np.array(res)}
        except Exception:
            return {"exponential": expm(A)}