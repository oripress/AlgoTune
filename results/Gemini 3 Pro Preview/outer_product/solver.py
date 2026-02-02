import numpy as np

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        vec1, vec2 = problem
        # Broadcasting avoids np.outer overhead. 
        # We assume inputs are numpy arrays as per type hint.
        return vec1[:, None] * vec2
        # torch.from_numpy shares memory, so it's fast.
        # We need to ensure inputs are contiguous for torch, or torch handles it.
        # np.outer handles flattening, torch.outer expects 1D input.
        
        # We must flatten inputs as np.outer does.
        v1 = vec1.ravel()
        v2 = vec2.ravel()
        
        # Convert to torch
        t1 = torch.from_numpy(v1)
        t2 = torch.from_numpy(v2)
        
        # Compute outer product
        res = torch.outer(t1, t2)
        
        # Convert back to numpy
        return res.numpy()