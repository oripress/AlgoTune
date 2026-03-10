import sys
import types
import unittest
import importlib.util
from pathlib import Path

from AlgoTuner.security.code_validator import check_code_for_tampering


def _load_solution_checks_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "utils" / "evaluator" / "solution_checks.py"
    )
    spec = importlib.util.spec_from_file_location("test_solution_checks", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_solution_checks = _load_solution_checks_module()
capture_validation_dependency_snapshot = _solution_checks.capture_validation_dependency_snapshot
detect_validation_dependency_tampering = _solution_checks.detect_validation_dependency_tampering
find_nonconcrete_solution = _solution_checks.find_nonconcrete_solution
prepare_isolated_solver_result_for_validation = (
    _solution_checks.prepare_isolated_solver_result_for_validation
)


class _LazyListProxy(list):
    def __init__(self, values):
        self._values = list(values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)


class _LazyDictProxy(dict):
    def __init__(self, payload):
        self._payload = dict(payload)

    def items(self):
        return self._payload.items()

    def __getitem__(self, key):
        return self._payload[key]

    def get(self, key, default=None):
        return self._payload.get(key, default)

    def __iter__(self):
        return iter(self._payload)

    def __len__(self):
        return len(self._payload)


_fake_scipy = types.ModuleType("scipy")
_fake_ndimage = types.ModuleType("scipy.ndimage")


def _baseline_affine_transform(problem, solution):
    return (problem, solution)


_fake_ndimage.affine_transform = _baseline_affine_transform
_fake_scipy.ndimage = _fake_ndimage


def _guard_helper():
    return True


class _ValidatorDependencyTask:
    def is_solution(self, problem, solution):
        return (
            _guard_helper()
            and _fake_scipy.ndimage.affine_transform(problem, solution) is not None
        )


class SolverValidationGuardTests(unittest.TestCase):
    def test_find_nonconcrete_solution_rejects_list_proxy_subclass(self):
        reason = find_nonconcrete_solution(_LazyListProxy([1.0, 2.0, 3.0]))

        self.assertIsNotNone(reason)
        self.assertIn("list-like proxy", reason)

    def test_find_nonconcrete_solution_rejects_dict_proxy_subclass(self):
        reason = find_nonconcrete_solution(_LazyDictProxy({"transport_plan": [[1.0]]}))

        self.assertIsNotNone(reason)
        self.assertIn("dict-like proxy", reason)

    def test_prepare_isolated_solver_result_preserves_materialized_child_result(self):
        materialized_result = {"rotated_image": [[1.0, 2.0], [3.0, 4.0]]}
        benchmark_result = {"success": True, "result": materialized_result}

        prepared = prepare_isolated_solver_result_for_validation(benchmark_result)

        self.assertIs(prepared["result"], materialized_result)
        self.assertNotIn("validation_result", prepared)

    def test_prepare_isolated_solver_result_fails_closed_when_result_missing(self):
        benchmark_result = {"success": True, "result": None}

        prepared = prepare_isolated_solver_result_for_validation(benchmark_result)

        self.assertFalse(prepared["validation_result"]["success"])
        self.assertEqual(prepared["validation_result"]["error_type"], "invalid_solution")
        self.assertTrue(prepared["result"]["stripped_after_validation"])
        self.assertEqual(prepared["result"]["reason"], "timed_result_unavailable")

    def test_validation_dependency_snapshot_detects_global_helper_mutation(self):
        global _guard_helper

        task = _ValidatorDependencyTask()
        snapshot = capture_validation_dependency_snapshot(task, task.is_solution)
        original_helper = _guard_helper

        def _tampered_guard_helper():
            return False

        _guard_helper = _tampered_guard_helper
        try:
            reason = detect_validation_dependency_tampering(task, snapshot)
        finally:
            _guard_helper = original_helper

        self.assertIsNotNone(reason)
        self.assertIn("_guard_helper", reason)

    def test_validation_dependency_snapshot_detects_module_attribute_mutation(self):
        task = _ValidatorDependencyTask()
        snapshot = capture_validation_dependency_snapshot(task, task.is_solution)
        original_affine_transform = _fake_ndimage.affine_transform
        _fake_ndimage.affine_transform = lambda *args, **kwargs: None
        try:
            reason = detect_validation_dependency_tampering(task, snapshot)
        finally:
            _fake_ndimage.affine_transform = original_affine_transform

        self.assertIsNotNone(reason)
        self.assertIn("_fake_scipy.ndimage.affine_transform", reason)

    def test_code_validator_rejects_alias_based_protected_module_monkeypatch(self):
        code = (
            "import scipy.ndimage as _ndi\n"
            "_ndi.affine_transform = lambda *args, **kwargs: None\n"
        )

        reason = check_code_for_tampering(code)

        self.assertIsNotNone(reason)
        self.assertIn("affine_transform", reason)


if __name__ == "__main__":
    unittest.main()
