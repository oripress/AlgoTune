import numpy as np

class Solver:
    def __init__(self):
        # Fixed parameters for the task
        self.a2 = 0.005
        self.a3 = 0.02
        self.a4 = 50.0
        self.a5 = 0.05

    def func(self, x, a0, a1, a2, a3, a4, a5):
        """The function f(x, a0..a5) = a1 - a2*(exp((a0+x*a3)/a5) - 1) - (a0+x*a3)/a4 - x"""
        return a1 - a2 * (np.exp((a0 + x * a3) / a5) - 1) - (a0 + x * a3) / a4 - x

    def fprime(self, x, a0, a1, a2, a3, a4, a5):
        """Derivative of the function with respect to x"""
        return -a2 * np.exp((a0 + x * a3) / a5) * (a3 / a5) - (a3 / a4) - 1

    def solve(self, problem):
        """
        Finds roots using a vectorized Newton-Raphson method.
        
        :param problem: Dict with lists "x0", "a0", "a1".
        :return: Dictionary with key "roots": list of roots.
        """
        # Convert inputs to numpy arrays
        x0 = np.array(problem["x0"])
        a0 = np.array(problem["a0"])
        a1 = np.array(problem["a1"])
        
        # Initialize x with the initial guess
        x = x0.copy()
        
        # Vectorized Newton-Raphson iterations
        for i in range(50):  # Limit iterations
            # Calculate function value and derivative at current x
            f_val = self.func(x, a0, a1, self.a2, self.a3, self.a4, self.a5)
            f_prime = self.fprime(x, a0, a1, self.a2, self.a3, self.a4, self.a5)
            
            # Newton-Raphson update: x_new = x - f(x)/f'(x)
            delta = f_val / f_prime
            x_new = x - delta
            
            # Check for convergence
            if np.all(np.abs(f_val) < 1e-10):
                break
                
            x = x_new
        
        # Return as a list
        # Simple conversion to list
        roots_list = x.tolist() if hasattr(x, 'tolist') else list(x)
        # Ensure it's definitely a list
        if not isinstance(roots_list, list):
            roots_list = [roots_list]
        return {"roots": roots_list}