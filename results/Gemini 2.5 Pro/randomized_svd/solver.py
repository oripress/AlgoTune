from typing import Any
import numpy as np
import torch
from scipy.linalg import svd

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Computes randomized SVD using a manual, hybrid GPU/CPU implementation for
        maximum speed, leveraging a GPU if available.
        - PyTorch on a CUDA device is used for all large matrix operations.
        - SciPy on the CPU is used for the fast SVD of the small intermediate matrix.
        """
        # Use float32 for faster matrix operations.
        A_np = np.asarray(problem["matrix"], dtype=np.float32)
        n_components = problem["n_components"]
        m, n = A_np.shape

        # Use proven hyperparameter tuning.
        n_oversamples = 5
        k = n_components + n_oversamples
        n_iter = 2 if problem.get("matrix_type") == "ill_conditioned" else 0

        # Detect and set the device (GPU if available, otherwise CPU).
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the main matrix to the selected device.
        A = torch.from_numpy(A_np).to(device)
        torch.manual_seed(42)

        # Step 1: Create a random projection matrix on the device.
        Omega = torch.randn(n, k, dtype=torch.float32, device=device)

        # Step 2: Form an initial sample matrix Y and orthonormalize it into Q.
        # All these operations happen on the GPU if available.
        Y = A @ Omega
        Q, _ = torch.qr(Y)

        # Step 3: Perform power iterations (subspace iteration) for accuracy.
        for _ in range(n_iter):
            Y_star = A.T @ Q
            Q_star, _ = torch.qr(Y_star)
            Y = A @ Q_star
            Q, _ = torch.qr(Y)

        # Step 4: Project A onto the basis Q. B is a small k x n matrix on the device.
        B = Q.T @ A

        # Step 5: Compute SVD on the small matrix B.
        # Move B to CPU to use SciPy's fast SVD.
        B_cpu = B.cpu().numpy()
        U_hat_cpu, s_cpu, Vt_cpu = svd(B_cpu, full_matrices=False)

        # Step 6: Project U_hat back to the original space.
        # Move U_hat back to the device for the final matrix multiplication.
        U_hat = torch.from_numpy(U_hat_cpu).to(device)
        U = Q @ U_hat

        # Step 7: Truncate to the desired number of components.
        U_final = U[:, :n_components]
        s_final = s_cpu[:n_components]
        V_final = Vt_cpu.T[:, :n_components]

        # Return results as NumPy arrays, ensuring U is moved back to CPU.
        return {
            "U": U_final.cpu().numpy(),
            "S": s_final,
            "V": V_final
        }