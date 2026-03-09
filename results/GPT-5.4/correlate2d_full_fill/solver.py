from scipy._lib import array_api_compat
from scipy.signal._signaltools import _freq_domain_conv

class Solver:
    mode = "full"
    boundary = "fill"

    __slots__ = ()

    def __init__(self):
        pass

    def solve(self, problem, **kwargs):
        a, b = problem
        return _freq_domain_conv(array_api_compat.numpy, a, b[::-1, ::-1], axes=(0, 1), shape=(a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1), calc_fast_len=True)