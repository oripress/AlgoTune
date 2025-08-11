import ot
_emd = ot.lp.emd

class Solver:
    solve = staticmethod(
        lambda p, **k: {
            "transport_plan": _emd(
                p["source_weights"],
                p["target_weights"],
                p["cost_matrix"],
                False
            )
        }
    )