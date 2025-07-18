import numpy as np
from scipy.integrate import cumulative_simpson

# Test with a simple array
y = np.ones((2, 2, 5))
dx = 0.1

result = cumulative_simpson(y, dx=dx)
print(f"Input shape: {y.shape}")
print(f"Output shape: {result.shape}")
print(f"Result: {result}")

# Test with odd number of points
y2 = np.ones((2, 2, 7))
result2 = cumulative_simpson(y2, dx=dx)
print(f"\nInput shape (odd): {y2.shape}")
print(f"Output shape (odd): {result2.shape}")