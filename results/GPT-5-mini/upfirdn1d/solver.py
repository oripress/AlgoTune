import numpy as np
from typing import Any, Sequence

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute upfirdn for each item in `problem`.

        Accepted item formats:
          - (h, x, up, down)
          - (h, x)  (uses up/down from kwargs or defaults to 1)

        `problem` may be:
          - a list/tuple/ndarray of items as above
          - a single item (h,x,up,down) or (h,x)

        Returns a list with one numpy 1D array per item (or None for items that failed to parse).
        """
        items = self._ensure_items(problem)
        results = []

        for item in items:
            try:
                if isinstance(item, (list, tuple, np.ndarray)):
                    ilen = len(item)
                    if ilen == 4:
                        h, x, up, down = item
                    elif ilen == 2:
                        h, x = item
                        up = kwargs.get("up", 1)
                        down = kwargs.get("down", 1)
                    elif ilen >= 2:
                        # Fallback: accept first two entries as (h,x)
                        h = item[0]
                        x = item[1]
                        up = kwargs.get("up", 1)
                        down = kwargs.get("down", 1)
                    else:
                        # Cannot interpret this item
                        results.append(None)
                        continue
                else:
                    results.append(None)
                    continue

                up = int(up)
                down = int(down)
                if up <= 0 or down <= 0:
                    results.append(None)
                    continue
            except Exception:
                results.append(None)
                continue

            try:
                out = self._upfirdn_1d(h, x, up, down)
            except Exception:
                out = None
            results.append(out)

        return results

    def _ensure_items(self, problem):
        """
        Normalize `problem` into a list of items.

        Heuristics:
          - If `problem` is a sequence whose elements are sequences of length 2 or 4,
            treat it as a list of items.
          - If `problem` itself is a sequence of length 2 or 4, treat it as a single item.
          - Otherwise wrap `problem` in a list.
        """
        if problem is None:
            return []

        if isinstance(problem, (list, tuple, np.ndarray)):
            try:
                plen = len(problem)
            except Exception:
                return [problem]

            if plen == 0:
                return []

            # Check if each element looks like a problem item
            is_list_of_items = True
            for el in problem:
                if not isinstance(el, (list, tuple, np.ndarray)):
                    is_list_of_items = False
                    break
                try:
                    if len(el) not in (2, 4):
                        is_list_of_items = False
                        break
                except Exception:
                    is_list_of_items = False
                    break

            if is_list_of_items:
                return list(problem)

            # If the top-level object itself looks like a single problem, return it wrapped
            if plen in (2, 4):
                return [problem]

            # Fallback: return a shallow copy to preserve iteration semantics
            try:
                return list(problem)
            except Exception:
                return [problem]

        # Non-sequence -> single item
        return [problem]

    def _upfirdn_1d(self, h: Sequence, x: Sequence, up: int, down: int) -> np.ndarray:
        """
        Efficient polyphase upfirdn (1D).

        Computes downsample(conv(upsample(x, up), h), down) using polyphase decomposition of h.
        """
        h_arr = np.asarray(h)
        x_arr = np.asarray(x)

        # Empty inputs -> empty output (use a safe dtype)
        if h_arr.size == 0 or x_arr.size == 0:
            return np.zeros(0, dtype=np.result_type(h_arr, x_arr, np.float64))

        up = int(up)
        down = int(down)

        # Computation dtype consistent with numpy/scipy conventions
        dtype = np.result_type(h_arr, x_arr, np.float64)
        h = np.asarray(h_arr, dtype=dtype).ravel()
        x = np.asarray(x_arr, dtype=dtype).ravel()

        Lh = h.size
        Nx = x.size

        # Fast path: simple convolution when no resampling
        if up == 1 and down == 1:
            return np.convolve(x, h)

        # Length after upsampling and convolution (before downsampling)
        N_up = Nx * up + Lh - 1
        if N_up <= 0:
            return np.zeros(0, dtype=dtype)

        # Number of output samples after downsampling (ceil division)
        Ny = (N_up + down - 1) // down
        if Ny <= 0:
            return np.zeros(0, dtype=dtype)

        y = np.zeros(Ny, dtype=dtype)

        # target sample positions in the upsampled domain
        k = np.arange(Ny, dtype=np.int64)
        t = k * down
        r_idx = (t % up).astype(np.int64)
        # q index into the convolution result for each target sample
        q_idx = ((t - r_idx) // up).astype(np.int64)

        # Process only phases that actually occur
        phases = np.unique(r_idx)
        for r in phases:
            hr = h[r::up]
            if hr.size == 0:
                continue

            # Convolve x with this polyphase component
            conv_r = np.convolve(x, hr)  # length Nx + len(hr) - 1

            sel = np.nonzero(r_idx == r)[0]
            if sel.size == 0:
                continue

            q_sel = q_idx[sel]
            # select valid indices inside the convolution result
            valid = (q_sel >= 0) & (q_sel < conv_r.size)
            if not np.any(valid):
                continue

            sel_valid = sel[valid]
            q_valid = q_sel[valid]
            y[sel_valid] = conv_r[q_valid]

        return y