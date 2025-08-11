import numpy as np
from scipy import signal

class Solver:
    def __init__(self):
        self.mode = 'full'
        self.boundary = 'fill'
    
    def solve(self, problem):
        """
        Compute the 2D correlation of arrays a and b using "full" mode and "fill" boundary.
        
        :param problem: A tuple (a, b) of 2D arrays.
        :return: A 2D array containing the correlation result.
        """
        a, b = problem
        
        # For "fill" boundary, we need to pad the first array with zeros
        # The padding size should be the size of b minus 1
        pad_height = b.shape[0] - 1
        pad_width = b.shape[1] - 1
        
        # Pad array a with zeros (fill boundary)
        a_padded = np.pad(a, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
        
        # Correlation is convolution with flipped kernel
        # So we flip b and use fftconvolve which is faster for large arrays
        b_flipped = np.flip(b)
        
        # Use FFT-based convolution which is O(n log n) instead of O(n^2)
        result = signal.fftconvolve(a_padded, b_flipped, mode='valid')
        
        return result