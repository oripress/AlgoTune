# Copyright (c) 2025 Ori Press and the AlgoTune contributors
# https://github.com/oripress/AlgoTune
import logging
import random
from typing import Any

import faiss
import numpy as np

from AlgoTuneTasks.base import register_task, Task


@register_task("kd_tree")
class KDTree(Task):
    def __init__(self, **kwargs):
        """
        Initialize the KD-tree construction and search task.

        In this task, you are given a set of points in d-dimensional space and query points.
        Your goal is to construct a KD-tree data structure for efficient nearest neighbor search
        and use it to find the k nearest neighbors for each query point.
        """
        super().__init__(**kwargs)

    def generate_problem(
        self,
        n: int,
        random_seed: int,
    ) -> dict[str, Any]:
        """
        Generate a random set of points and query points with flexible parameters.

        Defaults for n_points, n_queries, dim, k, distribution, point_config are set internally based on n.

        :param n: Base scaling parameter for the dataset size
        :param random_seed: Seed for reproducibility
        :return: A dictionary representing the KD-tree problem
        """
        logging.debug(f"Generating KD-tree problem with n={n}, seed={random_seed}")
        random.seed(random_seed)
        np.random.seed(random_seed)

        # --- Set parameters based on n (defaults for the removed signature params) ---
        n_points = n * 100
        n_queries = max(10, n * 10)
        dim = max(2, n // 2)
        k = min(10, n_points // 10)
        distribution = "uniform"  # Default distribution
        point_config = {}  # Default config
        # --- End setting parameters ---

        # Generate points based on the specified distribution
        if distribution == "uniform":
            points = np.random.uniform(0, 1, (n_points, dim))
            queries = np.random.uniform(0, 1, (n_queries, dim))

        elif distribution == "normal":
            mean = point_config.get("mean", np.zeros(dim))
            std = point_config.get("std", 1.0)
            points = np.random.normal(mean, std, (n_points, dim))
            queries = np.random.normal(mean, std, (n_queries, dim))

        elif distribution == "cluster":
            n_centers = point_config.get("centers", min(5, dim))
            cluster_std = point_config.get("cluster_std", 0.1)
            centers = np.random.uniform(0, 1, (n_centers, dim))
            points = np.zeros((n_points, dim))
            cluster_indices = np.random.choice(n_centers, n_points)
            for i in range(n_points):
                points[i] = centers[cluster_indices[i]] + np.random.normal(0, cluster_std, dim)
            queries = np.zeros((n_queries, dim))
            query_cluster_indices = np.random.choice(n_centers, n_queries)
            for i in range(n_queries):
                queries[i] = centers[query_cluster_indices[i]] + np.random.normal(
                    0, cluster_std, dim
                )

        elif distribution == "linear":
            slope = point_config.get("slope", np.ones(dim) / np.sqrt(dim))
            noise = point_config.get("noise", 0.1)
            t = np.random.uniform(0, 1, n_points)
            points = np.outer(t, slope) + np.random.normal(0, noise, (n_points, dim))
            t_queries = np.random.uniform(0, 1, n_queries)
            queries = np.outer(t_queries, slope) + np.random.normal(0, noise, (n_queries, dim))

        elif distribution == "grid":
            spacing = point_config.get("spacing", 0.1)
            jitter = point_config.get("jitter", 0.01)
            points_per_dim = max(2, int(np.floor(n_points ** (1 / dim))))
            actual_n_points = points_per_dim**dim
            n_points = min(n_points, actual_n_points)
            grid_coords = np.linspace(0, 1 - spacing, points_per_dim)
            grid_points = np.array(np.meshgrid(*[grid_coords for _ in range(dim)])).T.reshape(
                -1, dim
            )
            if len(grid_points) > n_points:
                indices = np.random.choice(len(grid_points), n_points, replace=False)
                grid_points = grid_points[indices]
            points = grid_points + np.random.uniform(-jitter, jitter, (len(grid_points), dim))
            queries = np.random.uniform(0, 1, (n_queries, dim))

        elif distribution == "hypercube_shell":
            points = np.zeros((n_points, dim))
            for i in range(n_points):
                face_dim = random.randint(0, dim - 1)
                face_value = random.choice([0, 1])
                point = np.random.uniform(0, 1, dim)
                point[face_dim] = face_value
                noise = np.random.normal(0, 0.01, dim)
                noise[face_dim] = 0
                points[i] = point + noise
            points = np.clip(points, 0, 1)
            queries = np.random.uniform(0, 1, (n_queries, dim))

        elif distribution == "skewed":
            skew_factor = point_config.get("skew_factor", 2.0)
            points = np.random.beta(1, skew_factor, (n_points, dim))
            queries = np.random.uniform(0, 1, (n_queries, dim))

        elif distribution == "sparse":
            sparsity = point_config.get("sparsity", 0.9)
            points = np.zeros((n_points, dim))
            for i in range(n_points):
                non_zero_dims = np.random.choice(dim, int(dim * (1 - sparsity)), replace=False)
                points[i, non_zero_dims] = np.random.uniform(0, 1, len(non_zero_dims))
            queries = np.zeros((n_queries, dim))
            for i in range(n_queries):
                non_zero_dims = np.random.choice(dim, int(dim * (1 - sparsity)), replace=False)
                queries[i, non_zero_dims] = np.random.uniform(0, 1, len(non_zero_dims))

        elif distribution == "outliers":
            outlier_percentage = point_config.get("outlier_percentage", 0.05)
            outlier_scale = point_config.get("outlier_scale", 5.0)
            main_count = int(n_points * (1 - outlier_percentage))
            points = np.random.uniform(0, 1, (n_points, dim))
            outlier_count = n_points - main_count
            outlier_indices = np.random.choice(n_points, outlier_count, replace=False)
            points[outlier_indices] = np.random.uniform(1, outlier_scale, (outlier_count, dim))
            queries = np.random.uniform(0, 1, (n_queries, dim))
            query_outlier_count = int(n_queries * outlier_percentage)
            outlier_query_indices = np.random.choice(n_queries, query_outlier_count, replace=False)
            queries[outlier_query_indices] = np.random.uniform(
                1, outlier_scale, (query_outlier_count, dim)
            )

        elif distribution == "duplicates":
            duplicate_percentage = point_config.get("duplicate_percentage", 0.2)
            unique_count = int(n_points * (1 - duplicate_percentage))
            unique_points = np.random.uniform(0, 1, (unique_count, dim))
            duplicate_count = n_points - unique_count
            duplicate_indices = np.random.choice(unique_count, duplicate_count, replace=True)
            duplicate_points = unique_points[duplicate_indices] + np.random.normal(
                0, 0.001, (duplicate_count, dim)
            )
            points = np.vstack([unique_points, duplicate_points])
            np.random.shuffle(points)
            queries = np.random.uniform(0, 1, (n_queries, dim))

        else:
            raise ValueError(f"Unknown distribution type: {distribution}")

        k = min(k, n_points)
        problem = {
            "k": k,
            "points": points,
            "queries": queries,
            "distribution": distribution,
            "point_config": point_config,
        }
        logging.debug(
            f"Generated KD-tree problem with {n_points} points in {dim} dimensions, "
            f"{n_queries} queries, distribution={distribution}"
        )
        return problem

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = np.array(problem["points"])
        queries = np.array(problem["queries"])
        k = problem["k"]
        dim = points.shape[1]

        index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(points.astype(np.float32), np.arange(len(points)))

        k = min(k, len(points))
        distances, indices = index.search(queries.astype(np.float32), k)

        solution = {"indices": indices.tolist(), "distances": distances.tolist()}

        # — Inject true boundary queries for “hypercube_shell” tests —
        if problem.get("distribution") == "hypercube_shell":
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float32)
                q0[d] = 0.0
                q1 = np.ones(dim, dtype=np.float32)
                q1[d] = 1.0
                bqs.extend([q0, q1])
            bqs = np.stack(bqs, axis=0)
            bq_dist, bq_idx = index.search(bqs, k)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()

        return solution

    def is_solution(self, problem: dict[str, Any], solution: dict[str, Any]) -> bool:
        for key in ["points", "queries", "k"]:
            if key not in problem:
                logging.error(f"Problem does not contain '{key}' key.")
                return False
        for key in ["indices", "distances"]:
            if key not in solution:
                logging.error(f"Solution does not contain '{key}' key.")
                return False

        try:
            points = np.array(problem["points"])
            queries = np.array(problem["queries"])
            indices = np.array(solution["indices"])
            distances = np.array(solution["distances"])
        except Exception as e:
            logging.error(f"Error converting lists to numpy arrays: {e}")
            return False

        n_queries = len(queries)
        n_points = len(points)
        k = min(problem["k"], n_points)
        dim = points.shape[1]

        if indices.shape != (n_queries, k):
            logging.error(
                f"Indices array incorrect. Expected ({n_queries}, {k}), got {indices.shape}."
            )
            return False
        if distances.shape != (n_queries, k):
            logging.error(
                f"Distances array incorrect. Expected ({n_queries}, {k}), got {distances.shape}."
            )
            return False
        if np.any((indices < 0) | (indices >= n_points)):
            logging.error("Indices out of valid range.")
            return False
        if np.any(distances < 0):
            logging.error("Negative distances found.")
            return False

        exact_index = faiss.IndexFlatL2(dim)
        exact_index.add(points.astype(np.float32))

        def _validate_neighbor_block(
            query_points: np.ndarray,
            candidate_indices: np.ndarray,
            candidate_distances: np.ndarray,
            block_name: str,
        ) -> bool:
            num_query_points = len(query_points)

            if candidate_indices.shape != (num_query_points, k):
                logging.error(
                    f"{block_name} indices shape incorrect. Expected {(num_query_points, k)}, "
                    f"got {candidate_indices.shape}."
                )
                return False
            if candidate_distances.shape != (num_query_points, k):
                logging.error(
                    f"{block_name} distances shape incorrect. Expected {(num_query_points, k)}, "
                    f"got {candidate_distances.shape}."
                )
                return False
            try:
                if np.any((candidate_indices < 0) | (candidate_indices >= n_points)):
                    logging.error(f"{block_name} indices out of valid range.")
                    return False
            except Exception as e:
                logging.error(f"Failed to validate {block_name} index range: {e}")
                return False
            if np.any(candidate_distances < 0):
                logging.error(f"Negative {block_name} distances found.")
                return False
            if num_query_points == 0 or k == 0:
                return True

            if k > 1:
                sorted_idx = np.sort(candidate_indices, axis=1)
                dup_rows = np.where(np.any(np.diff(sorted_idx, axis=1) == 0, axis=1))[0]
                if dup_rows.size > 0:
                    logging.error(
                        f"Duplicate {block_name} neighbor indices detected for query "
                        f"{int(dup_rows[0])}."
                    )
                    return False

                unsorted_rows = np.where(
                    np.any(np.diff(candidate_distances, axis=1) < -1e-5, axis=1)
                )[0]
                if unsorted_rows.size > 0:
                    logging.error(
                        f"{block_name} distances not sorted ascending for query "
                        f"{int(unsorted_rows[0])}."
                    )
                    return False

            try:
                computed_distances = np.sum(
                    (points[candidate_indices] - query_points[:, None, :]) ** 2,
                    axis=2,
                )
            except Exception as e:
                logging.error(f"Failed to compute {block_name} distances: {e}")
                return False

            if not np.allclose(computed_distances, candidate_distances, rtol=1e-4, atol=1e-4):
                row_errors = np.max(np.abs(computed_distances - candidate_distances), axis=1)
                bad_row = int(np.argmax(row_errors))
                logging.error(
                    f"{block_name} distance mismatch for query {bad_row}. "
                    f"Max absolute error: {float(row_errors[bad_row]):.6g}"
                )
                return False

            ref_distances, _ = exact_index.search(query_points.astype(np.float32), k)
            # Match the exact top-k distance profile for every query row; ties remain valid
            # because any neighbor ids at the same squared distance are accepted.
            if not np.allclose(ref_distances, candidate_distances, rtol=1e-4, atol=1e-4):
                row_errors = np.max(np.abs(ref_distances - candidate_distances), axis=1)
                bad_row = int(np.argmax(row_errors))
                logging.error(
                    f"{block_name} does not match exact top-k distances for query {bad_row}. "
                    f"Max absolute error: {float(row_errors[bad_row]):.6g}"
                )
                return False

            return True

        if not _validate_neighbor_block(queries, indices, distances, "Main"):
            return False

        # ======== Boundary case handling ========
        if problem.get("distribution") == "hypercube_shell":
            bq_idx = np.array(solution.get("boundary_indices", []))
            bqs = []
            for d in range(dim):
                q0 = np.zeros(dim, dtype=np.float64)
                q0[d] = 0.0
                q1 = np.ones(dim, dtype=np.float64)
                q1[d] = 1.0
                bqs.extend([q0, q1])
            bqs = np.stack(bqs, axis=0)
            if "boundary_distances" in solution:
                try:
                    bq_dist = np.array(solution["boundary_distances"])
                except Exception as e:
                    logging.error(f"Could not convert boundary distances to numpy array: {e}")
                    return False
            else:
                try:
                    bq_dist = np.sum((points[bq_idx] - bqs[:, None, :]) ** 2, axis=2)
                except Exception as e:
                    logging.error(f"Could not derive boundary distances from boundary indices: {e}")
                    return False

            if not _validate_neighbor_block(bqs, bq_idx, bq_dist, "Boundary"):
                return False
        else:
            logging.debug(
                "Skipping boundary checks for distribution=%s", problem.get("distribution")
            )

        return True
