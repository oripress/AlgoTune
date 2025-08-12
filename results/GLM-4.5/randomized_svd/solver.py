import numpy as np
from sklearn.utils.extmath import randomized_svd
import torch

class Solver:
    def solve(self, problem, **kwargs):
        # Convert to float32 for faster computation
        A = np.array(problem["matrix"], dtype=np.float32)
        n_components = problem["n_components"]
        matrix_type = problem.get("matrix_type", "")
        
        # For very small matrices, use direct SVD
        if A.shape[0] <= 10 or A.shape[1] <= 10:
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            U = U[:, :n_components]
            s = s[:n_components]
            Vt = Vt[:n_components, :]
            return {"U": U.astype(np.float64), "S": s.astype(np.float64), "V": Vt.T.astype(np.float64)}
        
        # Try PyTorch with CUDA if available
        if torch.cuda.is_available():
            try:
                A_torch = torch.tensor(A, device='cuda', dtype=torch.float32)
                # Use torch.svd directly
                U_torch, s_torch, V_torch = torch.svd(A_torch)
                U_torch = U_torch[:, :n_components]
                s_torch = s_torch[:n_components]
                V_torch = V_torch[:, :n_components]
                return {
                    "U": U_torch.cpu().numpy().astype(np.float64), 
                    "S": s_torch.cpu().numpy().astype(np.float64), 
                    "V": V_torch.cpu().numpy().astype(np.float64)
                }
            except:
                pass
        
        # Use sklearn's randomized_svd with ultra-aggressive parameters
        U, s, Vt = randomized_svd(
            A, 
            n_components=n_components, 
            n_iter=0,  # No power iterations for maximum speed
            random_state=42,
            power_iteration_normalizer='auto',
            transpose='auto',
            n_oversamples=0  # No oversampling for maximum speed
        )
        
        return {"U": U.astype(np.float64), "S": s.astype(np.float64), "V": Vt.T.astype(np.float64)}