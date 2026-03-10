import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path


def _load_polynomial_mixed_module():
    module_path = Path(__file__).resolve().parents[2] / "AlgoTuneTasks" / "polynomial_mixed" / "polynomial_mixed.py"
    fake_package = types.ModuleType("AlgoTuneTasks")
    fake_base = types.ModuleType("AlgoTuneTasks.base")

    class _Task:
        def __init__(self, **kwargs):
            pass

    def _register_task(name):
        def _decorator(cls):
            return cls

        return _decorator

    fake_base.Task = _Task
    fake_base.register_task = _register_task
    fake_package.base = fake_base

    original_package = sys.modules.get("AlgoTuneTasks")
    original_base = sys.modules.get("AlgoTuneTasks.base")
    try:
        sys.modules["AlgoTuneTasks"] = fake_package
        sys.modules["AlgoTuneTasks.base"] = fake_base
        spec = importlib.util.spec_from_file_location("test_polynomial_mixed_task", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_package is not None:
            sys.modules["AlgoTuneTasks"] = original_package
        else:
            sys.modules.pop("AlgoTuneTasks", None)
        if original_base is not None:
            sys.modules["AlgoTuneTasks.base"] = original_base
        else:
            sys.modules.pop("AlgoTuneTasks.base", None)


PolynomialMixed = _load_polynomial_mixed_module().PolynomialMixed


def _solve_in_subprocess(coefficients, n_threads: int) -> list[complex]:
    child = subprocess.run(
        [sys.executable, "-c", _CHILD_SOLVER_SCRIPT],
        input=json.dumps(coefficients),
        text=True,
        stdout=subprocess.PIPE,
        check=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": str(n_threads),
            "OPENBLAS_NUM_THREADS": str(n_threads),
            "MKL_NUM_THREADS": str(n_threads),
            "NUMEXPR_NUM_THREADS": str(n_threads),
            "VECLIB_MAXIMUM_THREADS": str(n_threads),
        },
    )
    return [complex(real, imag) for real, imag in json.loads(child.stdout)]


_CHILD_SOLVER_SCRIPT = """
import json
import numpy as np
import sys

coefficients = json.loads(sys.stdin.read())
roots = np.roots(coefficients)
roots = sorted(roots, key=lambda z: (z.real, z.imag), reverse=True)
json.dump([[float(z.real), float(z.imag)] for z in roots], sys.stdout)
"""


class PolynomialMixedValidatorTests(unittest.TestCase):
    def test_reference_solver_passes_validation(self):
        task = PolynomialMixed()
        problem = task.generate_problem(n=5, random_seed=0)

        self.assertTrue(task.is_solution(problem, task.solve(problem)))

    def test_validator_rejects_constant_nan_shortcut(self):
        task = PolynomialMixed()
        problem = task.generate_problem(n=5, random_seed=0)

        self.assertFalse(task.is_solution(problem, complex(float("nan"), float("nan"))))

    def test_validator_accepts_single_thread_solver_output_for_large_problem(self):
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
        except Exception as exc:
            self.skipTest(f"threadpoolctl unavailable: {exc}")

        max_threads = max((pool.get("num_threads", 0) for pool in threadpool_info()), default=0)
        if max_threads < 2:
            self.skipTest("BLAS backend exposes fewer than 2 threads")

        task = PolynomialMixed()
        problem = task.generate_problem(n=415, random_seed=142)
        child_solution = _solve_in_subprocess(problem, n_threads=1)

        with threadpool_limits(limits=min(4, max_threads)):
            self.assertTrue(task.is_solution(problem, child_solution))


if __name__ == "__main__":
    unittest.main()
