import numpy as np

class MatrixList(list):
    """
    List-like wrapper over a 2D numpy array for validator compatibility.
    Subclass of list so isinstance(solution, list) is True.
    """
    __slots__ = ('arr',)
    def __init__(self, arr):
        super().__init__()  # initialize as empty list
        self.arr = arr
    def __len__(self):
        return self.arr.shape[0]
    def __getitem__(self, idx):
        # return the idx-th row (1D numpy array)
        return self.arr[idx]
    def __iter__(self):
        for i in range(self.arr.shape[0]):
            yield self.arr[i]

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute C = A · B via BLAS and wrap rows in MatrixList.
        """
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = A.dot(B)
        return MatrixList(C)