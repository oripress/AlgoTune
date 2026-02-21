import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def do_fft(x):
    N = x.shape[0]
    R = jnp.fft.rfftn(x)
    c = N // 2 + 1
    
    row_idx = (N - jnp.arange(N)) % N
    col_idx = N - jnp.arange(c, N)
    
    right_part = jnp.conj(R[row_idx][:, col_idx])
    
    return jnp.concatenate([R, right_part], axis=1)

class Solver:
    def solve(self, problem, **kwargs):
        return np.asarray(do_fft(problem))