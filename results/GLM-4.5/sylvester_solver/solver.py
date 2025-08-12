from typing import Any
import numpy as np
import torch
from scipy.linalg import solve_sylvester

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A, B, Q = problem["A"], problem["B"], problem["Q"]
        
        # Convert to numpy arrays first
        A_np = np.asarray(A, dtype=np.complex128)
        B_np = np.asarray(B, dtype=np.complex128)
        Q_np = np.asarray(Q, dtype=np.complex128)
        
        # Try PyTorch GPU implementation if available
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                # Convert to PyTorch tensors and move to GPU
                A_torch = torch.from_numpy(A_np).cuda()
                B_torch = torch.from_numpy(B_np).cuda()
                Q_torch = torch.from_numpy(Q_np).cuda()
                
                # Solve using PyTorch's linear solver
                # AX + XB = Q can be rewritten as (I ⊗ A + B^T ⊗ I)vec(X) = vec(Q)
                n, m = Q_np.shape
                I_n = torch.eye(n, dtype=torch.complex128, device='cuda')
                I_m = torch.eye(m, dtype=torch.complex128, device='cuda')
                
                # Kronecker products
                kron1 = torch.kron(I_m, A_torch)
                kron2 = torch.kron(B_torch.T, I_n)
                
                # Combined matrix
                K = kron1 + kron2
                
                # Vectorized Q
                q_vec = Q_torch.reshape(-1, 1)
                
                # Solve linear system using torch.solve
                x_vec, _ = torch.solve(q_vec, K)
                
                # Reshape back to matrix form
                X_torch = x_vec.reshape(n, m)
                
                # Convert back to numpy
                X = X_torch.cpu().numpy()
            else:
                # Fallback to scipy if no GPU
                X = solve_sylvester(A_np, B_np, Q_np)
        except:
            # Fallback to scipy on any error
            X = solve_sylvester(A_np, B_np, Q_np)
        
        return {"X": X}