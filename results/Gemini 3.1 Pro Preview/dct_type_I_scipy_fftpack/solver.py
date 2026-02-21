import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def dct2_type1(x):
    # Apply 1D DCT Type I along axis 0
    pad_width_0 = ((0, x.shape[0] - 2), (0, 0))
    x_mirrored_0 = jnp.pad(x, pad_width_0, mode='reflect')
    X0 = jnp.real(jnp.fft.rfft(x_mirrored_0, axis=0))
    
    # Apply 1D DCT Type I along axis 1
    pad_width_1 = ((0, 0), (0, X0.shape[1] - 2))
    x_mirrored_1 = jnp.pad(X0, pad_width_1, mode='reflect')
    X1 = jnp.real(jnp.fft.rfft(x_mirrored_1, axis=1))
    
    return X1

class Solver:
    def __init__(self):
        # Precompile for common shapes if possible
        pass

    def solve(self, problem, **kwargs):
        x = jax.device_put(problem)
        return np.asarray(dct2_type1(x))