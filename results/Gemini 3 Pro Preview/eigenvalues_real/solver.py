import numpy as np
import torch

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The solution returned is a list of eigenvalues in descending order.
        """
        # Convert to torch tensor. Ensure float64 for precision.
        t_problem = torch.tensor(problem, dtype=torch.float64)
        
        # Compute eigenvalues using torch.linalg.eigvalsh
        # If linter complains, we can try to ignore it or use getattr
        try:
            eigvalsh = torch.linalg.eigvalsh
        except AttributeError:
            # Fallback for older torch versions
            eigvalsh = torch.symeig
            # symeig returns (eigenvalues, eigenvectors)
            e, _ = eigvalsh(t_problem, eigenvectors=False)
            return e.flip(0).tolist()

        eigenvalues = eigvalsh(t_problem)
        
        # Reverse and convert to list
        return eigenvalues.flip(0).tolist()