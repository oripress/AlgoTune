from __future__ import annotations

import ast
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

class Solver:
    def __init__(self) -> None:
        # Cache full 8D closures by constants tuple (used in fallback).
        self._cache8: Dict[Tuple[float, ...], Tuple[Callable, Callable]] = {}

    @staticmethod
    def _make_fun_and_jac_8(constants: Union[list[float], np.ndarray]) -> tuple[Callable, Callable]:
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = map(float, constants)

        out = np.empty(8, dtype=float)
        j = np.zeros((8, 8), dtype=float)

        def fun(_t: float, y: np.ndarray) -> np.ndarray:
            y1, y2, y3, y4, y5, y6, y7, y8 = y
            y6y8 = y6 * y8
            out[0] = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            out[1] = c1 * y1 - c5 * y2
            out[2] = -c6 * y3 + c2 * y4 + c7 * y5
            out[3] = c3 * y2 + c1 * y3 - c8 * y4
            out[4] = -c9 * y5 + c2 * y6 + c2 * y7
            out[5] = -c10 * y6y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7
            out[6] = c10 * y6y8 - c12 * y7
            out[7] = -c10 * y6y8 + c12 * y7
            return out

        def jac(_t: float, y: np.ndarray) -> np.ndarray:
            y6 = float(y[5])
            y8 = float(y[7])

            j.fill(0.0)

            j[0, 0] = -c1
            j[0, 1] = c2
            j[0, 2] = c3

            j[1, 0] = c1
            j[1, 1] = -c5

            j[2, 2] = -c6
            j[2, 3] = c2
            j[2, 4] = c7

            j[3, 1] = c3
            j[3, 2] = c1
            j[3, 3] = -c8

            j[4, 4] = -c9
            j[4, 5] = c2
            j[4, 6] = c2

            j[5, 3] = c11
            j[5, 4] = c1
            j[5, 5] = -(c10 * y8 + c2)
            j[5, 6] = c11
            j[5, 7] = -(c10 * y6)

            j[6, 5] = c10 * y8
            j[6, 6] = -c12
            j[6, 7] = c10 * y6

            j[7, 5] = -(c10 * y8)
            j[7, 6] = c12
            j[7, 7] = -(c10 * y6)

            return j

        return fun, jac

    @staticmethod
    def _make_fun_and_jac_7(
        constants: Union[list[float], np.ndarray], s0: float
    ) -> tuple[Callable, Callable]:
        # Reduce system using invariant y7+y8 = const = s0, so y8 = s0 - y7.
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = map(float, constants)

        out = np.empty(7, dtype=float)
        j = np.zeros((7, 7), dtype=float)

        def fun(_t: float, y: np.ndarray) -> np.ndarray:
            # y = [y1..y6,y7], y8 derived
            y1, y2, y3, y4, y5, y6, y7v = y
            y8v = s0 - y7v
            y6y8 = y6 * y8v

            out[0] = -c1 * y1 + c2 * y2 + c3 * y3 + c4
            out[1] = c1 * y1 - c5 * y2
            out[2] = -c6 * y3 + c2 * y4 + c7 * y5
            out[3] = c3 * y2 + c1 * y3 - c8 * y4
            out[4] = -c9 * y5 + c2 * y6 + c2 * y7v
            out[5] = -c10 * y6y8 + c11 * y4 + c1 * y5 - c2 * y6 + c11 * y7v
            out[6] = c10 * y6y8 - c12 * y7v
            return out

        def jac(_t: float, y: np.ndarray) -> np.ndarray:
            y6 = float(y[5])
            y7v = float(y[6])
            y8v = s0 - y7v

            j.fill(0.0)

            j[0, 0] = -c1
            j[0, 1] = c2
            j[0, 2] = c3

            j[1, 0] = c1
            j[1, 1] = -c5

            j[2, 2] = -c6
            j[2, 3] = c2
            j[2, 4] = c7

            j[3, 1] = c3
            j[3, 2] = c1
            j[3, 3] = -c8

            j[4, 4] = -c9
            j[4, 5] = c2
            j[4, 6] = c2

            # f6 = -c10*y6*(s0-y7) + c11*y4 + c1*y5 - c2*y6 + c11*y7
            j[5, 3] = c11
            j[5, 4] = c1
            j[5, 5] = -(c10 * y8v + c2)
            j[5, 6] = c10 * y6 + c11

            # f7 = c10*y6*(s0-y7) - c12*y7
            j[6, 5] = c10 * y8v
            j[6, 6] = -(c10 * y6 + c12)

            return j

        return fun, jac

    def solve(self, problem: Any, **kwargs: Any) -> Any:
        if isinstance(problem, str):
            problem = ast.literal_eval(problem)

        y0 = np.asarray(problem["y0"], dtype=float)
        t0 = float(problem["t0"])
        t1 = float(problem["t1"])
        constants = problem["constants"]

        rtol = float(kwargs.get("rtol", 1e-8))
        atol = float(kwargs.get("atol", 1e-9))

        # Reduced 7D solve using invariant y7+y8 const.
        s0 = float(y0[6] + y0[7])
        y0_7 = np.array((y0[0], y0[1], y0[2], y0[3], y0[4], y0[5], y0[6]), dtype=float)
        fun7, jac7 = self._make_fun_and_jac_7(constants, s0)

        sol7 = solve_ivp(
            fun7,
            (t0, t1),
            y0_7,
            method="LSODA",
            jac=jac7,
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False,
        )

        if sol7.success:
            y7 = sol7.y[:, -1]
            y8_final = s0 - float(y7[6])
            return [
                float(y7[0]),
                float(y7[1]),
                float(y7[2]),
                float(y7[3]),
                float(y7[4]),
                float(y7[5]),
                float(y7[6]),
                y8_final,
            ]

        # Fallback to full 8D (cached) for robustness.
        key = tuple(map(float, constants))
        fj8 = self._cache8.get(key)
        if fj8 is None:
            fj8 = self._make_fun_and_jac_8(constants)
            self._cache8[key] = fj8
        fun8, jac8 = fj8

        sol = solve_ivp(
            fun8,
            (t0, t1),
            y0,
            method="LSODA",
            jac=jac8,
            rtol=rtol,
            atol=atol,
            t_eval=None,
            dense_output=False,
        )

        if not sol.success:
            sol = solve_ivp(
                fun8,
                (t0, t1),
                y0,
                method="BDF",
                jac=jac8,
                rtol=rtol,
                atol=atol,
                t_eval=None,
                dense_output=False,
            )
            if not sol.success:
                sol = solve_ivp(
                    fun8,
                    (t0, t1),
                    y0,
                    method="Radau",
                    jac=jac8,
                    rtol=min(rtol, 1e-9),
                    atol=min(atol, 1e-10),
                    t_eval=None,
                    dense_output=False,
                )
                if not sol.success:
                    raise RuntimeError(f"Solver failed: {sol.message}")

        return sol.y[:, -1].tolist()