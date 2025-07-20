from typing import Any, Dict
import numpy as np
import torch

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Load matrix as float32 C-contiguous
        A = np.array(problem["matrix"], dtype=np.float32, order="C")
        n, m = A.shape
        k = problem["n_components"]
        matrix_type = problem.get("matrix_type", "")

        # trivial case
        if k == 0:
            return {
                "U": np.zeros((n, 0), dtype=A.dtype),
                "S": np.zeros((0,), dtype=A.dtype),
                "V": np.zeros((m, 0), dtype=A.dtype),
            }

        # choose power iterations for ill-conditioned problems
        niter = 2 if matrix_type == "ill_conditioned" else 0

        # perform randomized PCA/SVD using PyTorch's optimized backend
        A_t = torch.from_numpy(A)  # float32
        U_t, S_t, V_t = torch.pca_lowrank(A_t, q=k, center=False, niter=niter)
        U = U_t.numpy()
        S = S_t.numpy()
        V = V_t.numpy()

        return {"U": U, "S": S, "V": V}