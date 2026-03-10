import importlib.util
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load_firls_module():
    module_path = Path(__file__).resolve().parents[2] / "AlgoTuneTasks" / "firls" / "firls.py"
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
        spec = importlib.util.spec_from_file_location("test_firls_task", module_path)
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


FIRLS = _load_firls_module().FIRLS


def _solve_in_subprocess(problem, n_threads: int) -> list[float]:
    child = subprocess.run(
        [sys.executable, "-c", _CHILD_SOLVER_SCRIPT],
        input=json.dumps(problem),
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
    return [float(value) for value in json.loads(child.stdout)]


_CHILD_SOLVER_SCRIPT = """
import json
import sys
from scipy import signal

n, edges = json.loads(sys.stdin.read())
coeffs = signal.firls(2 * int(n) + 1, (0.0, float(edges[0]), float(edges[1]), 1.0), [1, 1, 0, 0])
json.dump([float(x) for x in coeffs], sys.stdout)
"""


class FIRLSValidatorTests(unittest.TestCase):
    def test_reference_solver_passes_validation(self):
        task = FIRLS()
        problem = task.generate_problem(n=32, random_seed=0)

        self.assertTrue(task.is_solution(problem, task.solve(problem)))

    def test_validator_accepts_single_thread_solver_output_for_large_problem(self):
        try:
            from threadpoolctl import threadpool_info, threadpool_limits
        except Exception as exc:
            self.skipTest(f"threadpoolctl unavailable: {exc}")

        max_threads = max((pool.get("num_threads", 0) for pool in threadpool_info()), default=0)
        if max_threads < 2:
            self.skipTest("BLAS backend exposes fewer than 2 threads")

        task = FIRLS()
        problem = task.generate_problem(n=1113, random_seed=42)
        child_solution = np.asarray(_solve_in_subprocess(problem, n_threads=1))

        with threadpool_limits(limits=min(4, max_threads)):
            self.assertTrue(task.is_solution(problem, child_solution))


if __name__ == "__main__":
    unittest.main()
