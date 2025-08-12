import itertools
from typing import List, Tuple, Any

class Solver:
    def solve(self, problem: Tuple[int, int, List[Tuple[int, int, bool]]], **kwargs) -> Any:
        # Parse the problem
        W, H, rectangles = problem
        
        # Sort rectangles by area in descending order to place larger rectangles first
        indexed_rectangles = [(i, w, h, r) for i, (w, h, r) in enumerate(rectangles)]
        indexed_rectangles.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        # Keep track of placed rectangles
        placed = []  # (index, x, y, rotated)
        occupied_areas = []  # List of (x1, y1, x2, y2) for placed rectangles
        
        def overlaps(x1, y1, x2, y2, occupied):
            for ox1, oy1, ox2, oy2 in occupied:
                if not (x2 <= ox1 or ox2 <= x1 or y2 <= oy1 or oy2 <= y1):
                    return True
            return False
            
        def try_place_rectangle(idx, w, h, can_rotate):
            orientations = [(w, h, False)]
            if can_rotate:
                orientations.append((h, w, True))
                
            # Try to place in the first available position
            # Use a better placement strategy: try to place as low as possible, then as left as possible
            best_x, best_y = None, None
            best_rw, best_rh = 0, 0
            best_rotated = False
            
            for rw, rh, rotated in orientations:
                if rw > W or rh > H:
                    continue
                    
                # Try different placement strategies
                # Strategy 1: Place at (0,0) if possible
                if not overlaps(0, 0, rw, rh, occupied_areas):
                    best_x, best_y = 0, 0
                    best_rw, best_rh = rw, rh
                    best_rotated = rotated
                    break
                    
                # Strategy 2: Try to place as low as possible, then as left as possible
                for y in range(H - rh + 1):
                    for x in range(W - rw + 1):
                        if not overlaps(x, y, x + rw, y + rh, occupied_areas):
                            if best_x is None or y < best_y or (y == best_y and x < best_x):
                                best_x, best_y = x, y
                                best_rw, best_rh = rw, rh
                                best_rotated = rotated
                                
                if best_x is not None and best_x == 0 and best_y == 0:
                    break
            
            if best_x is not None:
                placed.append((idx, best_x, best_y, best_rotated))
                occupied_areas.append((best_x, best_y, best_x + best_rw, best_y + best_rh))
                return True
            return False
        
        # Try to place each rectangle
        for idx, w, h, can_rotate in indexed_rectangles:
            try_place_rectangle(idx, w, h, can_rotate)
            
        return placed