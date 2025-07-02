from __future__ import annotations

from importlib import import_module
import sys as _sys
from typing import Any

__all__ = ["CommandHandlers"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute hook
    if name == "CommandHandlers":
        # Real import done only at first access – avoids circular deps.
        module = import_module("interfaces.commands.handlers")
        obj = module.CommandHandlers
        # Cache on the package module so future look‑ups are fast.
        setattr(_sys.modules[__name__], "CommandHandlers", obj)
        return obj
    raise AttributeError(name)