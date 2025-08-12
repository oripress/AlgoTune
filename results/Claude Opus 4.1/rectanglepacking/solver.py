from typing import Any, NamedTuple, List
from ortools.sat.python import cp_model
import itertools

class Rectangle(NamedTuple):
    width: int
    height: int
    rotatable: bool

class Instance(NamedTuple):
    container_width: int
    container_height: int
    rectangles: List[Rectangle]

class RectanglePlacement(NamedTuple):
    index: int
    x: int
    y: int
    rotated: bool

class Solver:
    def __init__(self):
        pass
    
    def _typesafe_instance(self, instance) -> Instance:
        if isinstance(instance, Instance):
            return instance
        return Instance(instance[0], instance[1], [Rectangle(*r) for r in instance[2]])
    
    def solve(self, problem, **kwargs) -> Any:
        """Optimized OR-Tools solver for rectangle packing"""
        problem = self._typesafe_instance(problem)
        
        model = cp_model.CpModel()
        
        n = len(problem.rectangles)
        W = problem.container_width
        H = problem.container_height
        
        # Decision variables
        x1_vars = []  # bottom-left x
        y1_vars = []  # bottom-left y
        x2_vars = []  # top-right x
        y2_vars = []  # top-right y
        placed = []
        rotated = []
        
        for i, rect in enumerate(problem.rectangles):
            x1_vars.append(model.NewIntVar(0, W, f'x1_{i}'))
            y1_vars.append(model.NewIntVar(0, H, f'y1_{i}'))
            x2_vars.append(model.NewIntVar(0, W, f'x2_{i}'))
            y2_vars.append(model.NewIntVar(0, H, f'y2_{i}'))
            placed.append(model.NewBoolVar(f'placed_{i}'))
            rotated.append(model.NewBoolVar(f'rotated_{i}'))
        
        # Constraints for dimensions based on rotation
        for i, rect in enumerate(problem.rectangles):
            if rect.rotatable:
                # Not rotated
                model.Add(x2_vars[i] == x1_vars[i] + rect.width).OnlyEnforceIf([placed[i], rotated[i].Not()])
                model.Add(y2_vars[i] == y1_vars[i] + rect.height).OnlyEnforceIf([placed[i], rotated[i].Not()])
                # Rotated
                model.Add(x2_vars[i] == x1_vars[i] + rect.height).OnlyEnforceIf([placed[i], rotated[i]])
                model.Add(y2_vars[i] == y1_vars[i] + rect.width).OnlyEnforceIf([placed[i], rotated[i]])
            else:
                # Not rotatable
                model.Add(x2_vars[i] == x1_vars[i] + rect.width).OnlyEnforceIf(placed[i])
                model.Add(y2_vars[i] == y1_vars[i] + rect.height).OnlyEnforceIf(placed[i])
                model.Add(rotated[i] == 0)
            
            # If not placed, set all coordinates to 0
            model.Add(x1_vars[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(y1_vars[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(x2_vars[i] == 0).OnlyEnforceIf(placed[i].Not())
            model.Add(y2_vars[i] == 0).OnlyEnforceIf(placed[i].Not())
        
        # Non-overlapping constraints for placed rectangles
        for i, j in itertools.combinations(range(n), 2):
            # Create boolean variables for relative positions
            left = model.NewBoolVar(f'{i}_left_of_{j}')
            right = model.NewBoolVar(f'{i}_right_of_{j}')
            below = model.NewBoolVar(f'{i}_below_{j}')
            above = model.NewBoolVar(f'{i}_above_{j}')
            
            # If both are placed, at least one separation must be true
            model.Add(left + right + below + above >= 1).OnlyEnforceIf([placed[i], placed[j]])
            
            # Define the separations
            model.Add(x2_vars[i] <= x1_vars[j]).OnlyEnforceIf([placed[i], placed[j], left])
            model.Add(x1_vars[i] >= x2_vars[j]).OnlyEnforceIf([placed[i], placed[j], right])
            model.Add(y2_vars[i] <= y1_vars[j]).OnlyEnforceIf([placed[i], placed[j], below])
            model.Add(y1_vars[i] >= y2_vars[j]).OnlyEnforceIf([placed[i], placed[j], above])
        
        # Objective: maximize number of placed rectangles
        model.Maximize(sum(placed))
        
        # Solve with optimized parameters for speed
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0  # Reduced time limit
        solver.parameters.num_search_workers = 1
        solver.parameters.linearization_level = 2
        solver.parameters.cp_model_presolve = True
        
        status = solver.Solve(model)
        
        result = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(n):
                if solver.Value(placed[i]):
                    result.append(RectanglePlacement(
                        i,
                        solver.Value(x1_vars[i]),
                        solver.Value(y1_vars[i]),
                        bool(solver.Value(rotated[i]))
                    ))
        
        return result