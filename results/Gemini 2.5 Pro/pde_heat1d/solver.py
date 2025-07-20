from typing import Any
import numpy as np
# Use the older, more established fftpack module, which has a well-defined
# behavior for the DST-I/IDST-I pair used in this spectral method.
from scipy.fftpack import dst, idst

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Solves the 1D heat equation using a spectral method with Discrete Sine Transform.
        This method is very efficient for this specific problem with Dirichlet boundary conditions.
        """
        y0 = np.array(problem["y0"], dtype=np.float64)
        t0, t1 = problem["t0"], problem["t1"]
        params = problem["params"]
        
        alpha = params["alpha"]
        dx = params["dx"]
        num_points = params["num_points"]
        
        delta_t = t1 - t0

        # The spectral method involves transforming to Fourier space, evolving, and transforming back.
        # 1. Analysis: Find coefficients b_k for the sine series of y0.
        #    y0[n] = sum_k b_k * sin(pi*(k+1)*(n+1)/(N+1))
        #    b_k = (2/(N+1)) * sum_n y0[n] * sin(...)
        # 2. Evolution: b_k(t) = b_k(0) * exp(lambda_k * t)
        # 3. Synthesis: y(t)[n] = sum_k b_k(t) * sin(...)
        #
        # The scipy.fftpack functions are defined as:
        #   dst(x,1) = 2 * sum_n x[n] * sin(...)
        #   idst(b,1) = sum_k b_k * sin(...)
        #
        # Therefore, we compute b_k = dst(y0,1)/(N+1) and y_final = idst(b_k(t),1).

        # Step 1: Compute the true Fourier-Sine coefficients of the initial condition.
        bk0 = dst(y0, type=1) / (num_points + 1.0)

        # Step 2: Define the wave numbers (modes).
        k = np.arange(1, num_points + 1)

        # Step 3: Use the eigenvalues of the discrete Laplacian to evolve the modes.
        discrete_eigenvalues = -4 * (alpha / dx**2) * (np.sin(k * np.pi / (2 * (num_points + 1))))**2
        decay_factor = np.exp(discrete_eigenvalues * delta_t)

        # Step 4: Apply the decay to the Fourier coefficients.
        bkt1 = bk0 * decay_factor

        # Step 5: Synthesize the solution from the evolved coefficients.
        y_final = idst(bkt1, type=1)

        return y_final.tolist()