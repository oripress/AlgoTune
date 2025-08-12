import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.order = 3
        self.mode = 'constant'
    
    def solve(self, problem, **kwargs):
        image = np.array(problem['image'], dtype=np.float64)
        shift_row, shift_col = problem['shift']
        
        # Separate into integer and fractional parts
        int_shift_row = int(round(shift_row))
        frac_shift_row = shift_row - int_shift_row
        
        int_shift_col = int(round(shift_col))
        frac_shift_col = shift_col - int_shift_col
        
        # Handle integer shift with optimized slicing
        if abs(frac_shift_row) < 1e-10 and abs(frac_shift_col) < 1e-10:
            h, w = image.shape
            shifted = np.zeros_like(image)
            
            # Calculate source and target regions
            src_row_start = max(0, -int_shift_row)
            src_row_end = min(h, h - int_shift_row)
            src_col_start = max(0, -int_shift_col)
            src_col_end = min(w, w - int_shift_col)
            
            tgt_row_start = max(0, int_shift_row)
            tgt_row_end = min(h, h + int_shift_row)
            tgt_col_start = max(0, int_shift_col)
            tgt_col_end = min(w, w + int_shift_col)
            
            # Perform the shift with numpy slicing
            shifted[tgt_row_start:tgt_row_end, tgt_col_start:tgt_col_end] = \
                image[src_row_start:src_row_end, src_col_start:src_col_end]
            
            return {"shifted_image": shifted.tolist()}
        
        # Handle fractional shift with scipy
        shifted = scipy.ndimage.shift(
            image, 
            (shift_row, shift_col),
            order=self.order,
            mode=self.mode
        )
        
        return {"shifted_image": shifted.tolist()}