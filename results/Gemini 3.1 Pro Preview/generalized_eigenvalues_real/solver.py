import scipy.linalg

class Solver:
    def solve(self, problem, **kwargs):
        A, B = problem
        eigenvalues = scipy.linalg.eigh(A, B, eigvals_only=True, check_finite=False, driver='gvx', overwrite_a=True, overwrite_b=True, subset_by_index=[0, A.shape[0]-1])
        return eigenvalues[::-1].tolist()