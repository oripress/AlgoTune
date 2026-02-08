import numpy as np
from scipy.optimize import leastsq

def _safe_exp(z):
    return np.exp(np.clip(z, -50.0, 50.0))

class Solver:
    def solve(self, problem, **kwargs):
        x_data = np.asarray(problem["x_data"], dtype=np.float64)
        y_data = np.asarray(problem["y_data"], dtype=np.float64)
        model_type = problem["model_type"]
        
        if model_type == "polynomial":
            deg = problem["degree"]
            params = np.polyfit(x_data, y_data, deg)
            return {"params": params.tolist()}
        
        elif model_type == "exponential":
            def residual(p):
                a, b, c = p
                return y_data - (a * _safe_exp(b * x_data) + c)
            guess = np.array([1.0, 0.05, 0.0])
            
        elif model_type == "logarithmic":
            def residual(p):
                a, b, c, d = p
                return y_data - (a * np.log(b * x_data + c) + d)
            guess = np.array([1.0, 1.0, 1.0, 0.0])
            
        elif model_type == "sigmoid":
            def residual(p):
                a, b, c, d = p
                return y_data - (a / (1 + _safe_exp(-b * (x_data - c))) + d)
            y_min, y_max = np.min(y_data), np.max(y_data)
            a_guess = y_max - y_min
            d_guess = y_min
            c_guess = np.median(x_data)
            b_guess = 0.5
            guess = np.array([a_guess, b_guess, c_guess, d_guess])
            
        elif model_type == "sinusoidal":
            def residual(p):
                a, b, c, d = p
                return y_data - (a * np.sin(b * x_data + c) + d)
            d_guess = np.mean(y_data)
            y_centered = y_data - d_guess
            a_guess = np.max(np.abs(y_centered))
            n = len(x_data)
            b_guess = 1.0
            if n > 4:
                dx = np.median(np.diff(np.sort(x_data)))
                if dx > 0:
                    fft_vals = np.fft.rfft(y_centered)
                    freqs = np.fft.rfftfreq(n, d=dx)
                    magnitudes = np.abs(fft_vals[1:])
                    if len(magnitudes) > 0:
                        peak_idx = np.argmax(magnitudes) + 1
                        b_guess = 2 * np.pi * freqs[peak_idx]
            guess = np.array([a_guess, b_guess, 0.0, d_guess])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        params_opt, _ = leastsq(residual, guess, maxfev=10000)
        return {"params": params_opt.tolist()}