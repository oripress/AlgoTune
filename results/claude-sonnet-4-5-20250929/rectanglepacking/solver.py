from typing import NamedTuple, Any

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
    def _typesafe_instance(self, instance) -> Instance:
        if isinstance(instance, Instance):
            return instance
        return Instance(instance[0], instance[1], [Rectangle(*r) for r in instance[2]])

    def solve(self, problem, **kwargs) -> Any:
        from ortools.sat.python import cp_model
        
        problem = self._typesafe_instance(problem)
        
        model = cp_model.CpModel()
        
        # Create variables for position and rotation
        x_vars = [
            model.new_int_var(0, problem.container_width, f"x_{i}")
            for i in range(len(problem.rectangles))
        ]
        y_vars = [
            model.new_int_var(0, problem.container_height, f"y_{i}")
            for i in range(len(problem.rectangles))
        ]
        rotated_vars = [
            model.new_bool_var(f"rotated_{i}") for i in range(len(problem.rectangles))
        ]
        placed_vars = [
            model.new_bool_var(f"placed_{i}") for i in range(len(problem.rectangles))
        ]
        
        # Create interval variables for each rectangle
        x_intervals = []
        y_intervals = []
        
        for i, rect in enumerate(problem.rectangles):
            if rect.rotatable:
                # Create size variables
                x_size = model.new_int_var(0, max(rect.width, rect.height), f"x_size_{i}")
                y_size = model.new_int_var(0, max(rect.width, rect.height), f"y_size_{i}")
                
                # Constrain sizes based on rotation
                model.add(x_size == rect.width).only_enforce_if(rotated_vars[i].Not())
                model.add(y_size == rect.height).only_enforce_if(rotated_vars[i].Not())
                model.add(x_size == rect.height).only_enforce_if(rotated_vars[i])
                model.add(y_size == rect.width).only_enforce_if(rotated_vars[i])
                
                # Create end variables
                x_end = model.new_int_var(0, problem.container_width, f"x_end_{i}")
                y_end = model.new_int_var(0, problem.container_height, f"y_end_{i}")
                model.add(x_end == x_vars[i] + x_size)
                model.add(y_end == y_vars[i] + y_size)
                
                # Ensure rectangles stay within container when placed
                model.add(x_end <= problem.container_width).only_enforce_if(placed_vars[i])
                model.add(y_end <= problem.container_height).only_enforce_if(placed_vars[i])
                
                # Create intervals
                x_interval = model.new_optional_interval_var(
                    x_vars[i], x_size, x_end, placed_vars[i], f"x_interval_{i}"
                )
                y_interval = model.new_optional_interval_var(
                    y_vars[i], y_size, y_end, placed_vars[i], f"y_interval_{i}"
                )
            else:
                # Not rotatable - use constant sizes
                model.add(rotated_vars[i] == 0)
                
                # Ensure rectangles stay within container when placed
                model.add(x_vars[i] + rect.width <= problem.container_width).only_enforce_if(placed_vars[i])
                model.add(y_vars[i] + rect.height <= problem.container_height).only_enforce_if(placed_vars[i])
                
                x_interval = model.new_optional_interval_var(
                    x_vars[i], rect.width, x_vars[i] + rect.width,
                    placed_vars[i], f"x_interval_{i}"
                )
                y_interval = model.new_optional_interval_var(
                    y_vars[i], rect.height, y_vars[i] + rect.height,
                    placed_vars[i], f"y_interval_{i}"
                )
            
            x_intervals.append(x_interval)
            y_intervals.append(y_interval)
        
        # Add NoOverlap2D constraint
        model.add_no_overlap_2d(x_intervals, y_intervals)
        
        # Maximize placed rectangles
        model.maximize(sum(placed_vars))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 900.0
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)
        
        solution = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(len(problem.rectangles)):
                if solver.Value(placed_vars[i]):
                    x = solver.Value(x_vars[i])
                    y = solver.Value(y_vars[i])
                    rotated = solver.Value(rotated_vars[i]) == 1
                    solution.append(RectanglePlacement(i, x, y, rotated))
        
        return solution