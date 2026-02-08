from ortools.sat.python import cp_model


class Solver:
    def solve(self, problem, **kwargs):
        W = problem[0]
        H = problem[1]
        rectangles = problem[2]
        n = len(rectangles)

        if n == 0:
            return []

        model = cp_model.CpModel()

        placed = {}
        rect_info = {}
        all_x_intervals = []
        all_y_intervals = []

        for i in range(n):
            w = rectangles[i][0]
            h = rectangles[i][1]
            r = rectangles[i][2]

            fits_normal = w <= W and h <= H
            fits_rotated = r and (h <= W) and (w <= H) and (w != h)

            if not fits_normal and not fits_rotated:
                continue

            placed[i] = model.NewBoolVar(f"p_{i}")

            if fits_normal and fits_rotated:
                pn = model.NewBoolVar(f"pn_{i}")
                pr = model.NewBoolVar(f"pr_{i}")
                model.Add(pn + pr == placed[i])

                # Normal orientation
                xn = model.NewIntVar(0, W - w, f"xn_{i}")
                yn = model.NewIntVar(0, H - h, f"yn_{i}")
                xn_end = model.NewIntVar(w, W, f"xne_{i}")
                yn_end = model.NewIntVar(h, H, f"yne_{i}")
                xi_n = model.NewOptionalIntervalVar(
                    xn, w, xn_end, pn, f"xin_{i}"
                )
                yi_n = model.NewOptionalIntervalVar(
                    yn, h, yn_end, pn, f"yin_{i}"
                )

                # Rotated orientation
                xr = model.NewIntVar(0, W - h, f"xr_{i}")
                yr = model.NewIntVar(0, H - w, f"yr_{i}")
                xr_end = model.NewIntVar(h, W, f"xre_{i}")
                yr_end = model.NewIntVar(w, H, f"yre_{i}")
                xi_r = model.NewOptionalIntervalVar(
                    xr, h, xr_end, pr, f"xir_{i}"
                )
                yi_r = model.NewOptionalIntervalVar(
                    yr, w, yr_end, pr, f"yir_{i}"
                )

                all_x_intervals.extend([xi_n, xi_r])
                all_y_intervals.extend([yi_n, yi_r])
                rect_info[i] = ("both", pn, pr, xn, yn, xr, yr)

            elif fits_normal:
                x = model.NewIntVar(0, W - w, f"x_{i}")
                y = model.NewIntVar(0, H - h, f"y_{i}")
                x_end = model.NewIntVar(w, W, f"xe_{i}")
                y_end = model.NewIntVar(h, H, f"ye_{i}")
                xi = model.NewOptionalIntervalVar(
                    x, w, x_end, placed[i], f"xi_{i}"
                )
                yi = model.NewOptionalIntervalVar(
                    y, h, y_end, placed[i], f"yi_{i}"
                )
                all_x_intervals.append(xi)
                all_y_intervals.append(yi)
                rect_info[i] = ("normal", x, y)

            else:
                x = model.NewIntVar(0, W - h, f"x_{i}")
                y = model.NewIntVar(0, H - w, f"y_{i}")
                x_end = model.NewIntVar(h, W, f"xe_{i}")
                y_end = model.NewIntVar(w, H, f"ye_{i}")
                xi = model.NewOptionalIntervalVar(
                    x, h, x_end, placed[i], f"xi_{i}"
                )
                yi = model.NewOptionalIntervalVar(
                    y, w, y_end, placed[i], f"yi_{i}"
                )
                all_x_intervals.append(xi)
                all_y_intervals.append(yi)
                rect_info[i] = ("rotated", x, y)

        feasible_indices = list(placed.keys())

        if not feasible_indices:
            return []

        model.AddNoOverlap2D(all_x_intervals, all_y_intervals)

        model.Maximize(sum(placed[i] for i in feasible_indices))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300.0

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        solution = []
        for i in feasible_indices:
            if solver.Value(placed[i]):
                info = rect_info[i]
                if info[0] == "both":
                    _, pn, pr, xn, yn, xr, yr = info
                    if solver.Value(pn):
                        solution.append(
                            (i, solver.Value(xn), solver.Value(yn), False)
                        )
                    else:
                        solution.append(
                            (i, solver.Value(xr), solver.Value(yr), True)
                        )
                elif info[0] == "normal":
                    _, x, y = info
                    solution.append(
                        (i, solver.Value(x), solver.Value(y), False)
                    )
                else:
                    _, x, y = info
                    solution.append(
                        (i, solver.Value(x), solver.Value(y), True)
                    )

        return solution