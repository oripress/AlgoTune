"""Runtime patches that make PySAT behave correctly in forked or spawned
processes.

Call ``apply_pysat_fixes()`` **once per interpreter** (e.g., at the start of a
multiprocessing worker). The function is idempotent – calling it multiple times
is safe and cheap.

Why these patches are needed
===========================
1. ``MainThread.check``
   PySAT's C extensions call ``MainThread.check()``. After a ``fork`` this
   reference leaks from the parent and leads to crashes or the infamous
   ``TypeError: integer expected``. Overriding the check to always return
   ``True`` avoids the issue inside child processes.

2. ``Solver.solve`` default parameter
   ``Solver.solve(self, assumptions=[])`` uses a *mutable* default list. The
   list is shared **across calls** and can accumulate foreign objects, which
   later break the underlying SAT solver. Wrapping the method ensures each call
   receives a fresh list.
"""
from __future__ import annotations

import logging
from types import MethodType

_logger = logging.getLogger(__name__)

# IMMEDIATE PATCH: Apply MainThread fix as soon as this module is imported
def _immediate_mainthread_patch():
    """Apply MainThread patch immediately when module is imported."""
    try:
        from pysat._utils import MainThread  # type: ignore
        
        _logger.debug("PySAT IMMEDIATE: Attempting to patch MainThread.check on module import")
        
        # Check original state
        original_check = getattr(MainThread, 'check', None)
        _logger.debug(f"PySAT IMMEDIATE: Original MainThread.check = {original_check}")
        
        def patched_check():
            return 1
        
        MainThread.check = staticmethod(patched_check)
        _logger.debug("PySAT IMMEDIATE: Patched MainThread.check directly")
        
        # Also patch at module level
        import pysat._utils
        pysat._utils.MainThread.check = staticmethod(patched_check)
        _logger.debug("PySAT IMMEDIATE: Patched pysat._utils.MainThread.check")
        
        # Verify patch worked
        test_result = MainThread.check()
        _logger.debug(f"PySAT IMMEDIATE: Test call MainThread.check() = {test_result} (type: {type(test_result)})")
        
        _logger.debug("PySAT IMMEDIATE: MainThread.check patched successfully on import")
        return True
    except Exception as exc:
        _logger.warning(f"PySAT IMMEDIATE: MainThread patch failed on import: {exc}")
        return False

# Apply the patch immediately
_immediate_mainthread_patch()


def _patch_mainthread_check() -> None:
    """Replace problematic MainThread.check to return integer 1."""
    try:
        from pysat._utils import MainThread  # type: ignore

        _logger.debug("PySAT PATCH: Starting MainThread.check patch")

        if getattr(MainThread, "_algo_tuner_patched", False):
            _logger.debug("PySAT PATCH: MainThread patch already applied, skipping")
            return

        # Store original before patching
        _original_check = getattr(MainThread, 'check', None)
        _logger.debug(f"PySAT PATCH: Original MainThread.check = {_original_check}")
        
        # Simple but effective fix: just make check always return 1
        def patched_check():
            return 1
        
        MainThread.check = staticmethod(patched_check)
        MainThread._algo_tuner_patched = True  # type: ignore[attr-defined]
        _logger.debug(f"PySAT PATCH: Applied MainThread.check patch (was: {_original_check})")
        
        # CRITICAL: Also patch at module level to handle C extension imports
        try:
            import pysat._utils
            pysat._utils.MainThread.check = staticmethod(patched_check)
            _logger.debug("PySAT PATCH: Also patched module-level MainThread.check")
            
            # Verify both patches work
            direct_result = MainThread.check()
            module_result = pysat._utils.MainThread.check()
            _logger.debug(f"PySAT PATCH: Verification - direct: {direct_result}, module: {module_result}")
            
        except Exception as module_patch_exc:
            _logger.error(f"PySAT PATCH: Module-level MainThread patch failed: {module_patch_exc}")
            
    except Exception as exc:  # noqa: BLE001
        # PySAT might not be installed – that's okay.
        _logger.error(f"PySAT PATCH: MainThread patch failed completely: {exc}")


def _patch_solver_solve() -> None:
    """Wrap ``Solver.solve`` to avoid the shared default `assumptions=[]`."""
    try:
        from pysat.solvers import Solver as _Solver  # type: ignore

        _logger.debug("PySAT PATCH: Starting Solver.solve patch")

        if getattr(_Solver.solve, "_algo_tuner_patched", False):
            _logger.debug("PySAT PATCH: Solver.solve patch already applied, skipping")
            return

        _orig_solve = _Solver.solve  # type: ignore[misc]
        _logger.debug(f"PySAT PATCH: Original Solver.solve = {_orig_solve}")

        def _sanitize(assumptions):  # type: ignore
            """Return a *fresh* list of plain ints suitable for the C extension."""
            if assumptions is None:
                return []
            
            # Handle different input types
            if not hasattr(assumptions, '__iter__'):
                return []
            
            result = []
            for a in assumptions:
                try:
                    # Convert to int, handling bools, numpy ints, etc.
                    if isinstance(a, (bool, int, float)):
                        result.append(int(a))
                    elif hasattr(a, 'item'):  # numpy scalars
                        result.append(int(a.item()))
                    else:
                        # Skip non-numeric items that can't be converted
                        _logger.debug(f"PySAT _sanitize: skipping non-numeric assumption: {a} (type: {type(a)})")
                        continue
                except (ValueError, TypeError, AttributeError):
                    # Skip items that can't be converted to int
                    _logger.debug(f"PySAT _sanitize: failed to convert assumption to int: {a} (type: {type(a)})")
                    continue
            
            return result

        def _solve_fixed(self, assumptions=None, *args, **kwargs):  # type: ignore[override]
            clean_assumptions = _sanitize(assumptions)
            _logger.debug(f"PySAT SOLVE: Called with assumptions: {assumptions} -> sanitized: {clean_assumptions}")
            
            # FALLBACK ERROR HANDLER: If solve still fails with TypeError, try with empty assumptions
            try:
                result = _orig_solve(self, clean_assumptions, *args, **kwargs)
                _logger.debug(f"PySAT SOLVE: Success with sanitized assumptions, result: {result}")
                return result
            except TypeError as te:
                if "integer expected" in str(te):
                    _logger.warning(f"PySAT SOLVE: Failed with 'integer expected', retrying with empty assumptions. Original error: {te}")
                    try:
                        result = _orig_solve(self, [], *args, **kwargs)
                        _logger.info(f"PySAT SOLVE: Retry with empty assumptions succeeded, result: {result}")
                        return result
                    except Exception as retry_exc:
                        _logger.error(f"PySAT SOLVE: Retry with empty assumptions also failed: {retry_exc}")
                        raise te  # Re-raise original error
                else:
                    _logger.error(f"PySAT SOLVE: Different TypeError occurred: {te}")
                    raise  # Re-raise if it's a different TypeError
            except Exception as other_exc:
                _logger.error(f"PySAT SOLVE: Non-TypeError exception: {other_exc}")
                raise

        _solve_fixed._algo_tuner_patched = True  # type: ignore[attr-defined]
        _Solver.solve = _solve_fixed  # type: ignore[assignment]
        _logger.debug("PySAT PATCH: Applied Solver.solve patch")

        # Also patch solve_limited which has the same signature / bug
        if hasattr(_Solver, "solve_limited"):
            _orig_solve_limited = _Solver.solve_limited  # type: ignore
            _logger.debug("PySAT PATCH: Also patching Solver.solve_limited")

            def _solve_limited_fixed(self, assumptions=None, *args, **kwargs):  # type: ignore[override]
                clean_assumptions = _sanitize(assumptions)
                _logger.debug(f"PySAT SOLVE_LIMITED: Called with assumptions: {assumptions} -> sanitized: {clean_assumptions}")
                try:
                    result = _orig_solve_limited(self, clean_assumptions, *args, **kwargs)
                    _logger.debug(f"PySAT SOLVE_LIMITED: Success, result: {result}")
                    return result
                except TypeError as te:
                    if "integer expected" in str(te):
                        _logger.warning(f"PySAT SOLVE_LIMITED: Failed with 'integer expected', retrying with empty assumptions. Original error: {te}")
                        try:
                            result = _orig_solve_limited(self, [], *args, **kwargs)
                            _logger.info(f"PySAT SOLVE_LIMITED: Retry succeeded, result: {result}")
                            return result
                        except Exception as retry_exc:
                            _logger.error(f"PySAT SOLVE_LIMITED: Retry also failed: {retry_exc}")
                            raise te
                    else:
                        _logger.error(f"PySAT SOLVE_LIMITED: Different TypeError: {te}")
                        raise

            _solve_limited_fixed._algo_tuner_patched = True  # type: ignore[attr-defined]
            _Solver.solve_limited = _solve_limited_fixed  # type: ignore[assignment]
            _logger.debug("PySAT PATCH: Applied Solver.solve_limited patch")

        _logger.debug("PySAT PATCH: Solver patches applied successfully")
    except Exception as exc:  # noqa: BLE001
        _logger.error(f"PySAT PATCH: Solver patch failed completely: {exc}")


def apply_pysat_fixes() -> None:
    """Apply all PySAT runtime patches (safe to call multiple times)."""
    _logger.debug("PySAT APPLY: Starting to apply all PySAT fixes")
    _patch_mainthread_check()
    _patch_solver_solve()
    _logger.debug("PySAT APPLY: Finished applying all PySAT fixes") 