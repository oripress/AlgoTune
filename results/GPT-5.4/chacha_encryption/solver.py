from __future__ import annotations

import ast
from typing import Any

import fastchacha

class Solver:
    def __init__(self) -> None:
        self._encrypt = fastchacha.encrypt

    def solve(self, problem, **kwargs) -> Any:
        if isinstance(problem, str):
            problem = ast.literal_eval(problem)
        return self._encrypt(
            problem["key"],
            problem["nonce"],
            problem["plaintext"],
            problem["associated_data"],
        )