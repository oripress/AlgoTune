import logging
import random
from typing import Any

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KernelDensity

from AlgoTuneTasks.base import register_task, Task


@register_task("kernel_density_estimation")
class KernelDensityEstimation(Task):
    def __init__(self, **kwargs):
        """
        Initialize the Kernel Density Estimation (KDE) task.

        In this task, given a set of data points X, a set of query points X_q,
        a kernel type, and a bandwidth, the goal is to estimate the probability
        density function from X and evaluate the log-density at the points in X_q.
        """
        super().__init__(**kwargs)
        self.available_kernels = [
            "gaussian",
            "tophat",
            "epanechnikov",
            "exponential",
            "linear",
            "cosine",
        ]

    def _generate_data(
        self, distribution_type: str, num_points: int, dims: int, dist_params: dict | None = None
    ) -> np.ndarray:
        """Internal helper to generate data based on distribution type."""

        if dist_params is None:
            dist_params = {}

        if distribution_type == "normal":
            mean = dist_params.get("mean", np.zeros(dims))
            cov_base = dist_params.get("cov", np.eye(dims))
            # Ensure mean and cov have correct dimensions
            if len(mean) != dims:
                mean = np.zeros(dims)
            cov_base = np.array(cov_base)
            if cov_base.shape != (dims, dims):
                cov_base = np.eye(dims)
            # Add small jitter for numerical stability if cov is identity
            cov = (
                cov_base + np.eye(dims) * 1e-6 if np.allclose(cov_base, np.eye(dims)) else cov_base
            )
            try:
                return np.random.multivariate_normal(mean, cov, size=num_points)
            except np.linalg.LinAlgError:
                logging.warning(
                    "Covariance matrix not positive definite for normal generation, using identity."
                )
                return np.random.multivariate_normal(mean, np.eye(dims), size=num_points)

        elif distribution_type == "uniform":
            low = dist_params.get("low", 0.0)
            high = dist_params.get("high", 1.0)
            return np.random.uniform(low, high, size=(num_points, dims))

        elif distribution_type == "mixture":
            # Expect 'components': List[Tuple[weight, type, params]]
            components = dist_params.get(
                "components",
                [
                    (0.5, "normal", {"mean": np.zeros(dims) - 2, "cov": np.eye(dims) * 0.5}),
                    (0.5, "normal", {"mean": np.zeros(dims) + 2, "cov": np.eye(dims) * 0.5}),
                ],
            )
            logging.debug(f"Generating mixture with components: {components}")
            data = np.zeros((num_points, dims))
            counts = np.random.multinomial(num_points, [comp[0] for comp in components])
            start_idx = 0
            for i, (weight, comp_type, comp_params) in enumerate(components):
                n_comp_points = counts[i]
                if n_comp_points > 0:
                    end_idx = start_idx + n_comp_points
                    # Ensure component parameters match overall dimensions
                    if "mean" in comp_params and len(comp_params["mean"]) != dims:
                        comp_params["mean"] = np.resize(comp_params["mean"], dims)
                        logging.warning(f"Resized mean for component {i} to {dims} dims")
                    if "cov" in comp_params and np.array(comp_params["cov"]).shape != (dims, dims):
                        cov = np.array(comp_params["cov"])
                        comp_params["cov"] = np.resize(cov, (dims, dims)) * np.eye(
                            dims
                        )  # Simplistic resize
                        logging.warning(f"Resized cov for component {i} to {dims}x{dims}")

                    data[start_idx:end_idx, :] = self._generate_data(
                        comp_type, n_comp_points, dims, comp_params
                    )
                    start_idx = end_idx
            return data

        elif distribution_type == "analytical_normal":
            # Same generation as normal, just indicates true density is known
            mean = dist_params.get("mean", np.zeros(dims))
            cov = np.array(dist_params.get("cov", np.eye(dims)))
            if len(mean) != dims:
                mean = np.zeros(dims)
            if cov.shape != (dims, dims):
                cov = np.eye(dims)
            try:
                return np.random.multivariate_normal(mean, cov, size=num_points)
            except np.linalg.LinAlgError:
                logging.warning(
                    "Covariance matrix not positive definite for analytical_normal, using identity."
                )
                return np.random.multivariate_normal(mean, np.eye(dims), size=num_points)

        elif distribution_type == "analytical_uniform":
            # Same generation as uniform
            low = dist_params.get("low", 0.0)
            high = dist_params.get("high", 1.0)
            return np.random.uniform(low, high, size=(num_points, dims))

        else:
            logging.warning(
                f"Unknown distribution type '{distribution_type}'. Falling back to standard normal."
            )
            return np.random.randn(num_points, dims)

    def generate_problem(self, n: int, random_seed: int = 42) -> dict[str, Any]:
        """
        Generate a KDE problem instance.

        The complexity and characteristics (dimensions, sample sizes, distributions,
        kernel, bandwidth) are determined based on the complexity parameter 'n' and
        the random seed.

        :param n: Base scaling parameter controlling complexity.
        :param random_seed: Seed for reproducibility.
        :return: A dictionary representing the KDE problem.
        """
        # Determine parameters based on n and random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Determine parameters: Use n for scaling
        max_dims = 64  # Cap dimensions to prevent excessive runtime
        dims = random.randint(1, min(max(2, n // 2), max_dims))
        num_points = random.randint(n * 5, n * 15)
        num_query_points = random.randint(n, n * 3)
        kernel = random.choice(self.available_kernels)
        # Adjust bandwidth based on dims and n? Maybe just dims.
        bandwidth = (
            random.uniform(0.1, 1.5) * np.sqrt(dims) * (1 + n / 100.0) ** 0.5
        )  # Slightly scale with n

        # Determine distributions and their parameters
        possible_distributions = [
            "normal",
            "uniform",
            "mixture",
            "analytical_normal",
            "analytical_uniform",
        ]
        if n < 10:
            data_distribution = random.choices(
                possible_distributions, weights=[0.4, 0.3, 0.1, 0.1, 0.1], k=1
            )[0]
        else:
            data_distribution = random.choices(
                possible_distributions, weights=[0.3, 0.2, 0.2, 0.15, 0.15], k=1
            )[0]

        dist_params = None
        if data_distribution == "normal" or data_distribution == "analytical_normal":
            mean = np.random.randn(dims) * (n / 10.0)  # Mean scales with n?
            # Generate a random positive semi-definite covariance matrix
            A = np.random.rand(dims, dims)
            cov = np.dot(A, A.transpose()) + np.eye(dims) * 1e-3  # Ensure positive definite
            cov *= 1 + n / 50.0  # Scale covariance with n
            dist_params = {"mean": mean.tolist(), "cov": cov.tolist()}
        elif data_distribution == "uniform" or data_distribution == "analytical_uniform":
            width = 1 + n / 5.0
            center = random.uniform(-n / 10.0, n / 10.0)
            dist_params = {"low": center - width / 2, "high": center + width / 2}
        elif data_distribution == "mixture":
            num_components = random.randint(2, max(3, n // 5))
            components = []
            weights = np.random.dirichlet(np.ones(num_components))  # Random weights summing to 1
            for i in range(num_components):
                comp_type = random.choice(["normal", "uniform"])
                comp_params = {}
                if comp_type == "normal":
                    mean = np.random.randn(dims) * (n / 8.0)
                    A = np.random.rand(dims, dims)
                    cov = np.dot(A, A.transpose()) + np.eye(dims) * 1e-3
                    cov *= (1 + n / 40.0) * random.uniform(0.5, 1.5)  # Vary covariance size
                    comp_params = {"mean": mean.tolist(), "cov": cov.tolist()}
                elif comp_type == "uniform":
                    width = (1 + n / 4.0) * random.uniform(0.5, 1.5)
                    center = random.uniform(-n / 8.0, n / 8.0)
                    comp_params = {"low": center - width / 2, "high": center + width / 2}
                components.append((weights[i], comp_type, comp_params))
            dist_params = {"components": components}

        # Query points generation (can be simpler, e.g., mostly normal/uniform)
        query_distribution = random.choice(["normal", "uniform"])
        query_dist_params = None
        if query_distribution == "normal":
            mean = np.random.randn(dims) * (n / 10.0)
            A = np.random.rand(dims, dims)
            cov = np.dot(A, A.transpose()) + np.eye(dims) * 1e-3
            cov *= 1 + n / 50.0
            query_dist_params = {"mean": mean.tolist(), "cov": cov.tolist()}
        elif query_distribution == "uniform":
            width = 1 + n / 5.0
            center = random.uniform(-n / 10.0, n / 10.0)
            query_dist_params = {"low": center - width / 2, "high": center + width / 2}

        logging.debug(
            f"Generating KDE problem with derived params: n={n}, seed={random_seed}, "
            f"dims={dims}, num_points={num_points}, num_query={num_query_points}, "
            f"kernel={kernel}, bw={bandwidth:.3f}, data_dist={data_distribution}"
        )

        # --- The rest of the generation logic uses the derived parameters ---
        X = self._generate_data(data_distribution, num_points, dims, dist_params)
        X_q = self._generate_data(query_distribution, num_query_points, dims, query_dist_params)

        problem = {
            "data_points": X.tolist(),
            "query_points": X_q.tolist(),
            "kernel": kernel,
            "bandwidth": bandwidth,
        }

        # Optionally add analytical info (existing logic remains the same)
        if data_distribution == "analytical_normal":
            # ... (existing analytical normal logic) ...
            mean = dist_params.get("mean", np.zeros(dims))
            cov = dist_params.get("cov", np.eye(dims))
            cov = np.array(cov)
            mean = np.array(mean)
            if len(mean) != dims:
                mean = np.zeros(dims)
            if cov.shape != (dims, dims):
                cov = np.eye(dims)
            try:
                multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            except Exception as e:
                logging.warning(f"Could not create analytical normal PDF object: {e}")
                problem["data_distribution_type"] = "normal"  # Downgrade if analytical fails

        elif data_distribution == "analytical_uniform":
            # ... (existing analytical uniform logic) ...
            low = dist_params.get("low", 0.0)
            high = dist_params.get("high", 1.0)
            volume = (high - low) ** dims
            if volume <= 0:
                log_density_inside = -np.inf
            else:
                log_density_inside = -np.log(volume)

            def uniform_logpdf(x):
                x = np.asarray(x)
                in_bounds = np.all((x >= low) & (x <= high), axis=-1)
                return np.where(in_bounds, log_density_inside, -np.inf)

        logging.debug(
            f"Generated KDE problem: {num_points} pts, {num_query_points} query, {dims}D, kernel='{kernel}', bw={bandwidth:.3f}, data_dist='{data_distribution}'"
        )
        return problem

    def solve(
        self, problem: dict[str, Any]
    ) -> dict[str, Any]:  # Return type includes error possibility
        try:
            X = np.array(problem["data_points"])
            X_q = np.array(problem["query_points"])
            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]
            # Infer dimensions from data robustly
            if X.ndim != 2 or X_q.ndim != 2:
                raise ValueError("Data points or query points are not 2D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X_q.shape[0] == 0:
                # Return empty list if no query points
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")

            # Basic validation of inputs needed for solving
            if not isinstance(bandwidth, float | int) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            if kernel not in self.available_kernels:
                raise ValueError(f"Unknown kernel: {kernel}")

            # Initialize and fit the KDE model
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
            kde.fit(X)

            # Evaluate the log-density at query points
            log_density = kde.score_samples(X_q)

            solution = {"log_density": log_density.tolist()}
            return solution

        except KeyError as e:
            logging.error(f"Missing key in problem dictionary: {e}")
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError, NotFittedError, np.linalg.LinAlgError) as e:
            logging.error(f"Error during KDE computation: {e}")
            return {"error": f"Computation error: {e}"}
        except Exception as e:
            logging.error(f"An unexpected error occurred during solve: {e}", exc_info=True)
            return {"error": f"Unexpected error: {e}"}

    def is_solution(self, problem: dict[str, Any], solution: dict[str, list[float]]) -> bool:
        """
        Validate the KDE solution.

        Compares the provided solution against a reference solution generated
        by the internal `solve` method. Also performs basic structural checks.

        :param problem: A dictionary representing the KDE problem.
        :param solution: A dictionary containing the KDE solution.
        :return: True if solution is valid, else False.
        """
        # Check if solution indicates an error occurred during student's solve
        if "error" in solution:
            logging.error(f"Solution indicates an error state: {solution['error']}")
            return False

        # Basic structural checks on the problem dict
        required_problem_keys = [
            "data_points",
            "query_points",
            "kernel",
            "bandwidth",
        ]
        for key in required_problem_keys:
            if key not in problem:
                logging.error(f"Problem dictionary is missing the key: '{key}'.")
                return False

        # Check solution structure
        if "log_density" not in solution:
            logging.error("Solution does not contain 'log_density' key.")
            return False

        try:
            # Get query points to determine expected size
            query_points = np.array(problem["query_points"])
            num_query_points = len(query_points)

            # Handle case of zero query points
            if num_query_points == 0:
                if isinstance(solution["log_density"], list) and len(solution["log_density"]) == 0:
                    logging.debug("Validation successful for zero query points.")
                    return True  # Correct empty list for zero queries
                else:
                    logging.error(
                        "Expected empty list for 'log_density' when num_query_points is 0."
                    )
                    return False

            # Proceed for non-zero query points
            log_density_sol = np.array(solution["log_density"])

            # Check shape
            if log_density_sol.ndim != 1 or len(log_density_sol) != num_query_points:
                logging.error(
                    f"Solution 'log_density' has incorrect shape. Expected ({num_query_points},), got {log_density_sol.shape}."
                )
                return False

            # Re-compute the reference solution for comparison
            reference_solution = self.solve(problem)
            if "error" in reference_solution:
                logging.error(
                    f"Failed to compute reference solution for validation: {reference_solution['error']}"
                )
                # Cannot validate if reference calculation itself fails
                return False

            log_density_ref = np.array(reference_solution["log_density"])

            # Compare solutions
            # Increased tolerance slightly, as minor implementation details can shift results
            if not np.allclose(log_density_sol, log_density_ref, rtol=1e-5, atol=1e-7):
                max_abs_diff = np.max(np.abs(log_density_sol - log_density_ref))
                max_rel_diff = np.max(
                    np.abs(log_density_sol - log_density_ref) / (np.abs(log_density_ref) + 1e-8)
                )  # Avoid division by zero
                logging.error(
                    f"Solution 'log_density' values do not match reference within tolerance. Max abs diff: {max_abs_diff:.4e}, Max rel diff: {max_rel_diff:.4e}"
                )
                return False

        except (TypeError, ValueError) as e:
            logging.error(
                f"Error during validation (e.g., converting to numpy array or comparison): {e}"
            )
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during validation: {e}", exc_info=True)
            return False

        # All checks passed
        return True
