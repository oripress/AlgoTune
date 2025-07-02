from __future__ import annotations

from typing import Any


def validation_already_done(res: Any) -> bool:
    """Return True if *res* looks like a placeholder produced *after* the
    worker has already validated the solution.

    Several different phases can strip or replace the original solver result
    with a small metadata dict to avoid expensive inter-process transfers.
    We treat the presence of any of the known flags as proof that validation
    already happened inside the worker process and can safely be skipped in
    the parent process.
    """
    # If it's a ResultMetadata instance we also know validation already happened
    try:
        from AlgoTuner.utils.smart_result_handler import ResultMetadata  # local import to avoid cycle
        if isinstance(res, ResultMetadata):
            return True
    except ImportError:
        pass

    if not isinstance(res, dict):
        return False

    known_flags = (
        "validation_completed",     # set explicitly when worker validated
        "stripped_after_validation",  # worker replaced big array after validation
        "stripped_in_timing",       # timing phase replacement
    )
    return any(res.get(flag) for flag in known_flags) 