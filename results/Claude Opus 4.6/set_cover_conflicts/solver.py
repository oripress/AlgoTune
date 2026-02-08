from typing import Any
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        if isinstance(problem, tuple):
            n, sets_list, conflicts = problem
        else:
            n, sets_list, conflicts = problem.n, problem.sets, problem.conflicts
        
        num_sets = len(sets_list)
        
        # Precompute which sets cover each object
        obj_to_sets = [[] for _ in range(n)]
        for i, s in enumerate(sets_list):
            for obj in s:
                obj_to_sets[obj].append(i)
        
        # Build conflict lookup
        conflict_groups = {}
        for conflict in conflicts:
            for idx in conflict:
                if idx not in conflict_groups:
                    conflict_groups[idx] = []
                conflict_groups[idx].append(conflict)
        
        # Preprocessing: propagate forced selections
        fixed_selected = set()
        fixed_forbidden = set()
        covered = set()
        
        changed = True
        while changed:
            changed = False
            for obj in range(n):
                if obj in covered:
                    continue
                available = [i for i in obj_to_sets[obj] if i not in fixed_forbidden]
                if len(available) == 1:
                    s_idx = available[0]
                    if s_idx not in fixed_selected:
                        fixed_selected.add(s_idx)
                        covered.update(sets_list[s_idx])
                        if s_idx in conflict_groups:
                            for conflict in conflict_groups[s_idx]:
                                for other in conflict:
                                    if other != s_idx and other not in fixed_selected:
                                        if other not in fixed_forbidden:
                                            fixed_forbidden.add(other)
                                            changed = True
                        changed = True
        
        remaining_objs = [obj for obj in range(n) if obj not in covered]
        
        if not remaining_objs:
            return sorted(fixed_selected)
        
        remaining_set = set(remaining_objs)
        active_sets = []
        for i in range(num_sets):
            if i in fixed_selected or i in fixed_forbidden:
                continue
            if remaining_set & set(sets_list[i]):
                active_sets.append(i)
        
        set_id_map = {s: idx for idx, s in enumerate(active_sets)}
        n_vars = len(active_sets)
        
        if n_vars == 0:
            return sorted(fixed_selected)
        
        reduced_conflicts = []
        for conflict in conflicts:
            reduced = [set_id_map[i] for i in conflict if i in set_id_map]
            if len(reduced) >= 2:
                reduced_conflicts.append(reduced)
        
        obj_covering = []
        for obj in remaining_objs:
            covering = [set_id_map[i] for i in obj_to_sets[obj] if i in set_id_map]
            obj_covering.append(covering)
        
        n_cover = len(remaining_objs)
        
        # Try SAT-based approach for small instances
        if n_vars <= 200 and n_cover <= 200:
            result_vars = self._solve_sat(n_vars, obj_covering, reduced_conflicts)
            if result_vars is not None:
                solution = list(fixed_selected)
                for i in result_vars:
                    solution.append(active_sets[i])
                return sorted(solution)
        
        # Fall back to HiGHS MIP
        n_conflict = len(reduced_conflicts)
        total_constraints = n_cover + n_conflict
        
        result_vars = self._solve_highs(n_vars, n_cover, total_constraints, obj_covering, reduced_conflicts)
        
        if result_vars is not None:
            solution = list(fixed_selected)
            for i in result_vars:
                solution.append(active_sets[i])
            return sorted(solution)
        else:
            return list(range(n))
    
    def _solve_sat(self, n_vars, obj_covering, reduced_conflicts):
        """Solve using iterative SAT (binary search on objective)."""
        from pysat.solvers import Glucose4
        from pysat.card import CardEnc, EncType
        
        # Variable mapping: var i -> SAT variable (i+1)
        # SAT variables are 1-indexed
        
        # Get greedy upper bound first
        ub = self._greedy_ub(n_vars, obj_covering, reduced_conflicts)
        if ub is None:
            return None
        
        lb_val = 1
        ub_val = len(ub)
        best_solution = ub
        
        # Binary search on the number of sets
        while lb_val <= ub_val:
            mid = (lb_val + ub_val) // 2
            result = self._sat_check(n_vars, obj_covering, reduced_conflicts, mid)
            if result is not None:
                best_solution = result
                ub_val = mid - 1
            else:
                lb_val = mid + 1
        
        return best_solution
    
    def _sat_check(self, n_vars, obj_covering, reduced_conflicts, max_sets):
        from pysat.solvers import Glucose4
        from pysat.card import CardEnc, EncType
        
        clauses = []
        
        # Coverage constraints: at least one set covers each object
        for covering in obj_covering:
            clauses.append([v + 1 for v in covering])
        
        # Conflict constraints: at most one from each conflict group
        top_var = n_vars
        for conflict in reduced_conflicts:
            # At most one: pairwise negation
            for i in range(len(conflict)):
                for j in range(i + 1, len(conflict)):
                    clauses.append([-(conflict[i] + 1), -(conflict[j] + 1)])
        
        # Cardinality constraint: at most max_sets selected
        card_clauses = CardEnc.atmost(
            lits=[i + 1 for i in range(n_vars)],
            bound=max_sets,
            top_id=top_var,
            encoding=EncType.totalizer
        )
        clauses.extend(card_clauses)
        
        solver = Glucose4()
        for clause in clauses:
            solver.add_clause(clause)
        
        if solver.solve():
            model = solver.get_model()
            solver.delete()
            return [i for i in range(n_vars) if model[i] > 0]
        else:
            solver.delete()
            return None
    
    def _greedy_ub(self, n_vars, obj_covering, reduced_conflicts):
        """Fast greedy solution."""
        var_covers = [set() for _ in range(n_vars)]
        all_objs = set()
        for obj_idx, covering in enumerate(obj_covering):
            all_objs.add(obj_idx)
            for v in covering:
                var_covers[v].add(obj_idx)
        
        uncovered = set(all_objs)
        selected = set()
        forbidden = set()
        
        cg = {}
        for conflict in reduced_conflicts:
            for idx in conflict:
                if idx not in cg:
                    cg[idx] = []
                cg[idx].append(conflict)
        
        while uncovered:
            best_var = -1
            best_cover = 0
            
            for i in range(n_vars):
                if i in selected or i in forbidden:
                    continue
                cover = len(uncovered & var_covers[i])
                if cover > best_cover:
                    best_cover = cover
                    best_var = i
            
            if best_var == -1 or best_cover == 0:
                return None
            
            selected.add(best_var)
            uncovered -= var_covers[best_var]
            
            if best_var in cg:
                for conflict in cg[best_var]:
                    for s in conflict:
                        if s != best_var:
                            forbidden.add(s)
        
        return selected
    
    def _solve_highs(self, n_vars, n_cover, total_constraints, obj_covering, reduced_conflicts):
        import highspy
        
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("mip_abs_gap", 0.999)
        
        col_lower = np.zeros(n_vars, dtype=np.float64)
        col_upper = np.ones(n_vars, dtype=np.float64)
        col_cost = np.ones(n_vars, dtype=np.float64)
        integrality = np.full(n_vars, highspy.HighsVarType.kInteger, dtype=np.int32)
        
        h.addVars(n_vars, col_lower, col_upper)
        h.changeColsIntegralityByRange(0, n_vars - 1, integrality)
        h.changeColsCostByRange(0, n_vars - 1, col_cost)
        h.changeObjectiveSense(highspy.ObjSense.kMinimize)
        
        row_lower = np.empty(total_constraints, dtype=np.float64)
        row_upper = np.empty(total_constraints, dtype=np.float64)
        row_lower[:n_cover] = 1.0
        row_upper[:n_cover] = highspy.kHighsInf
        row_lower[n_cover:] = -highspy.kHighsInf
        row_upper[n_cover:] = 1.0
        
        starts = []
        indices = []
        idx = 0
        for covering in obj_covering:
            starts.append(idx)
            indices.extend(covering)
            idx += len(covering)
        for conflict in reduced_conflicts:
            starts.append(idx)
            indices.extend(conflict)
            idx += len(conflict)
        
        nnz = idx
        h.addRows(total_constraints, row_lower, row_upper, nnz,
                   np.array(starts, dtype=np.int32),
                   np.array(indices, dtype=np.int32),
                   np.ones(nnz, dtype=np.float64))
        
        h.run()
        
        info = h.getInfoValue("primal_solution_status")
        if info[1] >= 1:
            vals = h.getSolution().col_value
            return [i for i in range(n_vars) if vals[i] > 0.5]
        return None