from typing import NamedTuple, Any, List, Tuple, Set
import itertools

class Rectangle(NamedTuple):
    width: int
    height: int
    rotatable: bool

class Instance(NamedTuple):
    container_width: int
    container_height: int
    rectangles: list[Rectangle]

class RectanglePlacement(NamedTuple):
    index: int
    x: int
    y: int
    rotated: bool

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """Fast rectangle packing solver using branch and bound."""
        # Convert to typed instance
        if isinstance(problem, Instance):
            instance = problem
        else:
            instance = Instance(problem[0], problem[1], [Rectangle(*r) for r in problem[2]])
        
        n = len(instance.rectangles)
        
        # For very small instances, try exact solution
        if n <= 7:
            return self._solve_exact(instance)
        else:
            # For larger instances, use fast heuristic with local search
            return self._solve_heuristic(instance)
    
    def _solve_exact(self, instance: Instance) -> List[RectanglePlacement]:
        """Solve small instances using branch and bound."""
        n = len(instance.rectangles)
        best_solution = []
        best_count = 0
        
        # Pre-compute all valid positions for each rectangle
        valid_positions = []
        for i, rect in enumerate(instance.rectangles):
            positions = []
            orientations = [(rect.width, rect.height, False)]
            if rect.rotatable:
                orientations.append((rect.height, rect.width, True))
            
            for w, h, rot in orientations:
                if w <= instance.container_width and h <= instance.container_height:
                    # Only consider a subset of positions to reduce search space
                    for y in range(0, instance.container_height - h + 1, max(1, h // 4)):
                        for x in range(0, instance.container_width - w + 1, max(1, w // 4)):
                            positions.append((i, x, y, w, h, rot))
            valid_positions.append(positions)
        
        # Branch and bound search
        def search(level: int, placed: List[RectanglePlacement], occupied: Set[Tuple[int, int]]):
            nonlocal best_solution, best_count
            
            if len(placed) > best_count:
                best_count = len(placed)
                best_solution = placed[:]
            
            if level >= n or len(placed) + (n - level) <= best_count:
                return
            
            # Try placing current rectangle
            for i, x, y, w, h, rot in valid_positions[level]:
                # Check if position is valid
                valid = True
                for dx in range(w):
                    for dy in range(h):
                        if (x + dx, y + dy) in occupied:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Place rectangle
                    new_occupied = set()
                    for dx in range(w):
                        for dy in range(h):
                            new_occupied.add((x + dx, y + dy))
                    
                    placed.append(RectanglePlacement(i, x, y, rot))
                    occupied.update(new_occupied)
                    
                    search(level + 1, placed, occupied)
                    
                    placed.pop()
                    occupied.difference_update(new_occupied)
            
            # Try not placing current rectangle
            search(level + 1, placed, occupied)
        
        search(0, [], set())
        return best_solution
    
    def _solve_heuristic(self, instance: Instance) -> List[RectanglePlacement]:
        """Fast heuristic for larger instances."""
        # Sort rectangles by different criteria and pick best
        indexed_rects = [(i, rect) for i, rect in enumerate(instance.rectangles)]
        
        # Try different sorting strategies
        strategies = [
            # Area descending
            lambda x: -x[1].width * x[1].height,
            # Max dimension descending  
            lambda x: -max(x[1].width, x[1].height),
            # Width descending
            lambda x: -x[1].width,
            # Perimeter descending
            lambda x: -(x[1].width + x[1].height),
        ]
        
        best_solution = []
        
        for key_func in strategies:
            sorted_rects = sorted(indexed_rects, key=key_func)
            solution = self._fast_pack(instance, sorted_rects)
            if len(solution) > len(best_solution):
                best_solution = solution
        
        # Try to improve solution with local search
        improved = self._local_search(instance, best_solution)
        if len(improved) > len(best_solution):
            best_solution = improved
        
        return best_solution
    
    def _fast_pack(self, instance: Instance, sorted_rects: List[Tuple[int, Rectangle]]) -> List[RectanglePlacement]:
        """Fast greedy packing using skyline algorithm."""
        # Skyline representation: list of (x, y) points representing the skyline
        skyline = [(0, 0), (instance.container_width, 0)]
        solution = []
        
        for idx, rect in sorted_rects:
            best_fit = None
            best_waste = float('inf')
            
            # Try both orientations
            orientations = [(rect.width, rect.height, False)]
            if rect.rotatable:
                orientations.append((rect.height, rect.width, True))
            
            for width, height, rotated in orientations:
                # Try to fit at each skyline position
                for i in range(len(skyline) - 1):
                    x = skyline[i][0]
                    y = skyline[i][1]
                    
                    # Check if rectangle fits
                    if x + width > instance.container_width:
                        continue
                    if y + height > instance.container_height:
                        continue
                    
                    # Find the highest point in the range [x, x+width]
                    max_y = y
                    j = i
                    while j < len(skyline) - 1 and skyline[j][0] < x + width:
                        max_y = max(max_y, skyline[j][1])
                        j += 1
                    
                    if max_y + height > instance.container_height:
                        continue
                    
                    # Calculate wasted area
                    waste = 0
                    j = i
                    while j < len(skyline) - 1 and skyline[j][0] < x + width:
                        w = min(skyline[j+1][0], x + width) - max(skyline[j][0], x)
                        waste += w * (max_y - skyline[j][1])
                        j += 1
                    
                    if waste < best_waste:
                        best_waste = waste
                        best_fit = (x, max_y, width, height, rotated, i)
            
            if best_fit:
                x, y, width, height, rotated, start_idx = best_fit
                solution.append(RectanglePlacement(idx, x, y, rotated))
                
                # Update skyline
                new_skyline = []
                i = 0
                
                # Add points before the rectangle
                while i < len(skyline) and skyline[i][0] < x:
                    new_skyline.append(skyline[i])
                    i += 1
                
                # Add the rectangle's skyline
                if len(new_skyline) == 0 or new_skyline[-1][0] < x:
                    new_skyline.append((x, y + height))
                else:
                    new_skyline[-1] = (x, y + height)
                
                # Skip points covered by the rectangle
                while i < len(skyline) and skyline[i][0] < x + width:
                    i += 1
                
                # Add the right edge of the rectangle
                if x + width < instance.container_width:
                    new_skyline.append((x + width, y + height))
                
                # Add remaining points
                while i < len(skyline):
                    if skyline[i][0] >= x + width:
                        new_skyline.append(skyline[i])
                    i += 1
                
                # Merge consecutive points with same height
                merged = []
                for point in new_skyline:
                    if len(merged) == 0 or merged[-1][1] != point[1]:
                        merged.append(point)
                
                skyline = merged
        
        return solution
    
    def _local_search(self, instance: Instance, initial_solution: List[RectanglePlacement]) -> List[RectanglePlacement]:
        """Try to improve solution by adding more rectangles."""
        placed_indices = {p.index for p in initial_solution}
        unplaced = [i for i in range(len(instance.rectangles)) if i not in placed_indices]
        
        if not unplaced:
            return initial_solution
        
        # Build occupied grid
        grid = [[False] * instance.container_width for _ in range(instance.container_height)]
        for p in initial_solution:
            rect = instance.rectangles[p.index]
            w, h = (rect.height, rect.width) if p.rotated else (rect.width, rect.height)
            for dy in range(h):
                for dx in range(w):
                    if p.y + dy < instance.container_height and p.x + dx < instance.container_width:
                        grid[p.y + dy][p.x + dx] = True
        
        solution = initial_solution[:]
        
        # Try to place remaining rectangles
        for idx in unplaced:
            rect = instance.rectangles[idx]
            placed = False
            
            orientations = [(rect.width, rect.height, False)]
            if rect.rotatable:
                orientations.append((rect.height, rect.width, True))
            
            for w, h, rot in orientations:
                if w > instance.container_width or h > instance.container_height:
                    continue
                
                for y in range(instance.container_height - h + 1):
                    for x in range(instance.container_width - w + 1):
                        # Check if position is free
                        valid = True
                        for dy in range(h):
                            for dx in range(w):
                                if grid[y + dy][x + dx]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            # Place rectangle
                            solution.append(RectanglePlacement(idx, x, y, rot))
                            for dy in range(h):
                                for dx in range(w):
                                    grid[y + dy][x + dx] = True
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
        
        return solution