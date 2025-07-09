import logging
import random

import numpy as np
import scipy.optimize

from AlgoTuneTasks.base import register_task, Task


# Define the vectorized function and derivative used in the benchmark
def _task_f_vec(x, *a):
    # Ensure x is treated as numpy array for vectorization
    x = np.asarray(x)
    # a contains (a0, a1, a2, a3, a4, a5) where a0, a1 can be arrays
    a0, a1, a2, a3, a4, a5 = a
    b = a0 + x * a3  # b will be array if a0 or x is array
    term1 = a2 * (np.exp(b / a5) - 1.0)
    term2 = b / a4
    return a1 - term1 - term2 - x


def _task_f_vec_prime(x, *a):
    x = np.asarray(x)
    a0, a1, a2, a3, a4, a5 = a
    b_deriv_term = a3 / a5
    exp_term = np.exp((a0 + x * a3) / a5)
    return -a2 * exp_term * b_deriv_term - a3 / a4 - 1.0


@register_task("vectorized_newton")
class VectorizedNewton(Task):
    def __init__(self, **kwargs):
        """
        Initialize the VectorizedNewton task.

        Finds roots for `n` instances of a parameterized function simultaneously
        using a vectorized call to `scipy.optimize.newton` (Newton-Raphson).
        The function `f(x, *params)` and its derivative `fprime(x, *params)`
        are evaluated element-wise over arrays of initial guesses `x0` and
        corresponding parameter arrays `a0`, `a1`.
        """
        super().__init__(**kwargs)
        self.func = _task_f_vec
        self.fprime = _task_f_vec_prime
        # Fixed parameters from benchmark, except a0, a1 which scale with n
        self.a2 = 1e-09
        self.a3 = 0.004
        self.a4 = 10.0
        self.a5 = 0.27456

    def generate_problem(self, n: int, random_seed: int = 1) -> dict[str, list[float]]:
        """
        Generates `n` initial guesses `x0` and corresponding parameter arrays `a0`, `a1`.

        Uses the specific function structure from the `NewtonArray` benchmark.

        :param n: The number of simultaneous root-finding problems (size of arrays).
        :param random_seed: Seed for reproducibility.
        :return: A dictionary with keys:
                 "x0": List of `n` initial guesses.
                 "a0": List of `n` corresponding 'a0' parameter values.
                 "a1": List of `n` corresponding 'a1' parameter values.
        """
        logging.debug(f"Generating Vectorized Newton problem with n={n}, random_seed={random_seed}")
        random.seed(random_seed)
        np.random.seed(random_seed)
        rng = np.random.default_rng(random_seed)

        # Generate arrays based on benchmark example structure
        x0 = rng.uniform(6.0, 8.0, size=n).tolist()  # Around the expected root area
        # Generate a0 similar to example
        a0 = rng.uniform(1.0, 6.0, size=n)
        a0[::10] = rng.uniform(5.0, 5.5, size=len(a0[::10]))  # Add some variation
        a0 = a0.tolist()
        # Generate a1 similar to example
        a1 = (np.sin(rng.uniform(0, 2 * np.pi, size=n)) + 1.0) * 7.0
        a1 = a1.tolist()

        problem = {"x0": x0, "a0": a0, "a1": a1}
        logging.debug(f"Generated {n} vectorized problem instances.")
        return problem

    def solve(self, problem: dict[str, list[float]]) -> dict[str, list[float]]:
        """
        Finds roots using a single vectorized call to scipy.optimize.newton.

        :param problem: Dict with lists "x0", "a0", "a1".
        :return: Dictionary with key "roots": List of `n` found roots. Uses NaN on failure.
        """
        try:
            x0_arr = np.array(problem["x0"])
            a0_arr = np.array(problem["a0"])
            a1_arr = np.array(problem["a1"])
            n = len(x0_arr)
            if len(a0_arr) != n or len(a1_arr) != n:
                raise ValueError("Input arrays have mismatched lengths")
        except Exception as e:
            logging.error(f"Failed to reconstruct input arrays: {e}")
            # Cannot determine n reliably, return empty list?
            return {"roots": []}

        # Assemble args tuple for vectorized function
        args = (a0_arr, a1_arr, self.a2, self.a3, self.a4, self.a5)

        roots_list = []
        try:
            # Perform vectorized call
            roots_arr = scipy.optimize.newton(self.func, x0_arr, fprime=self.fprime, args=args)
            roots_list = roots_arr
            # Check if newton returned a scalar unexpectedly (e.g., if n=1)
            if np.isscalar(roots_list):
                roots_list = np.array([roots_list])

            # Pad with NaN if output length doesn't match input (shouldn't happen with vectorization)
            if len(roots_list) != n:
                logging.warning(
                    f"Vectorized Newton output length {len(roots_list)} != input {n}. Padding with NaN."
                )
                roots_list.extend([float("nan")] * (n - len(roots_list)))

        except RuntimeError as e:
            # Vectorized call might fail entirely or partially? SciPy docs are unclear.
            # Assume it raises error if *any* fail to converge. Return all NaNs.
            logging.warning(
                f"Vectorized Newton failed to converge (may affect all elements): {e}. Returning NaNs."
            )
            roots_list = [float("nan")] * n
        except Exception as e:
            logging.error(f"Unexpected error in vectorized Newton: {e}. Returning NaNs.")
            roots_list = [float("nan")] * n

        solution = {"roots": roots_list}
        return solution

    def is_solution(
        self, problem: dict[str, list[float]], solution: dict[str, list[float]]
    ) -> bool:
        """
        Check if the provided roots are correct for the vectorized problem.

        Checks length and compares element-wise against the reference vectorized output.

        :param problem: The problem definition dictionary.
        :param solution: The proposed solution dictionary.
        :return: True if the solution is valid and correct, False otherwise.
        """
        required_keys = ["x0", "a0", "a1"]
        if not all(k in problem for k in required_keys):
            logging.error(f"Problem dictionary missing required keys: {required_keys}")
            return False
        try:
            # Determine n from a key expected to be present
            n = len(problem["x0"])
        except Exception:
            logging.error("Cannot determine problem size n from input.")
            return False

        if not isinstance(solution, dict) or "roots" not in solution:
            logging.error("Solution format invalid: missing 'roots' key.")
            return False

        proposed_roots = solution["roots"]
        if not isinstance(proposed_roots, list):
            # Handle case where solve might return non-list if n=1 or scalar result
            if isinstance(proposed_roots, float | int) and n == 1:
                proposed_roots = [proposed_roots]
            else:
                logging.error(f"'roots' is not a list (type: {type(proposed_roots)}).")
                return False

        if len(proposed_roots) != n:
            logging.error(f"'roots' list has incorrect length ({len(proposed_roots)} != {n}).")
            return False

        # Recompute reference solution
        ref_solution = self.solve(problem)
        ref_roots = ref_solution["roots"]

        # Compare proposed roots with reference roots (handling NaNs)
        rtol = 1e-7
        atol = 1e-9
        try:
            is_close = np.allclose(proposed_roots, ref_roots, rtol=rtol, atol=atol, equal_nan=True)
        except TypeError as e:
            logging.error(
                f"Comparison failed, possibly due to non-numeric data in roots lists: {e}"
            )
            # Log lists for inspection
            logging.error(f"Proposed roots: {proposed_roots}")
            logging.error(f"Reference roots: {ref_roots}")
            return False

        if not is_close:
            logging.error("Proposed roots do not match reference roots within tolerance.")
            num_mismatch = np.sum(
                ~np.isclose(proposed_roots, ref_roots, rtol=rtol, atol=atol, equal_nan=True)
            )
            logging.error(f"Number of mismatches: {num_mismatch} / {n}")
            # Log first few mismatches
            count = 0
            for i in range(n):
                if not np.allclose(
                    proposed_roots[i], ref_roots[i], rtol=rtol, atol=atol, equal_nan=True
                ):
                    logging.error(
                        f"Mismatch at index {i}: Proposed={proposed_roots[i]}, Ref={ref_roots[i]}"
                    )
                    count += 1
                    if count >= 5:
                        break
            return False

        # Optional: Basic validity check f(root) ~= 0
        func_tol = 1e-8
        try:
            prop_roots_arr = np.array(proposed_roots)
            finite_mask = np.isfinite(prop_roots_arr)
            if np.any(finite_mask):  # Only check if there are finite roots
                a0_arr = np.array(problem["a0"])[finite_mask]
                a1_arr = np.array(problem["a1"])[finite_mask]
                args_check = (a0_arr, a1_arr, self.a2, self.a3, self.a4, self.a5)
                f_vals = self.func(prop_roots_arr[finite_mask], *args_check)
                if np.any(np.abs(f_vals) > func_tol * 10):
                    bad_indices = np.where(np.abs(f_vals) > func_tol * 10)[0]
                    logging.warning(
                        f"Some roots ({len(bad_indices)}) do not satisfy |f(root)| <= {func_tol * 10}. Max |f(root)| = {np.max(np.abs(f_vals))}"
                    )
                    # Example bad index:
                    idx = bad_indices[0]
                    logging.warning(
                        f"Example: Index {idx}, Root={prop_roots_arr[finite_mask][idx]}, f(root)={f_vals[idx]}"
                    )
                    # Don't fail here, primary check is np.allclose with reference.
                    # return False
        except Exception as e:
            logging.warning(f"Could not perform basic validity check |f(root)|~0 due to error: {e}")

        logging.debug("Solution verification successful.")
        return True
