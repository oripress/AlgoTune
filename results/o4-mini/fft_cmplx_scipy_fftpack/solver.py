import numpy as np
from scipy.fft import fftn as scipy_fftn

class Solver:
    """
    Hybrid FFT solver: large transforms use PyTorch FFT via MKL;
    fall back to SciPy pocketfft for other sizes.
    """
    TORCH_CPU_THRESHOLD = 1 << 17  # 131072 elements
    TORCH_GPU_THRESHOLD = 1 << 22  # 4194304 elements

    def solve(self, problem, **kwargs) -> np.ndarray:
        # Create a NumPy view of the input
        arr0 = np.asarray(problem)
        n = arr0.size

        # Use PyTorch FFT for sufficiently large problems
        if n >= self.TORCH_CPU_THRESHOLD:
            try:
                torch = __import__('torch')
                torch_fft = __import__('torch.fft', fromlist=['fftn'])
                # Cast to single‐precision complex for speed
                arr64 = arr0.astype(np.complex64, copy=False, order='C')
                tensor = torch.from_numpy(arr64)
                # Offload to GPU if very large and available
                if n >= self.TORCH_GPU_THRESHOLD and torch.cuda.is_available():
                    tensor = tensor.cuda()
                    out = torch_fft.fftn(tensor)
                    res = out.cpu().numpy()
                else:
                    out = torch_fft.fftn(tensor)
                    res = out.numpy()
                # Back to double precision
                return res.astype(np.complex128)
            except Exception:
                pass

        # Fallback to SciPy's pocketfft with multithreading
        arr128 = arr0.astype(np.complex128, copy=False, order='C')
        return scipy_fftn(arr128, overwrite_x=True, workers=-1)