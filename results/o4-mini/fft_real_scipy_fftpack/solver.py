try:
    from scipy.fft import fftn
except ImportError:
    from numpy.fft import fftn

class Solver:
    solve = staticmethod(fftn)