import numpy as np
import threading
import multiprocessing
from scipy.ndimage import affine_transform, spline_filter

class Solver:
    def solve(self, problem, **kwargs):
        image = np.asarray(problem["image"], dtype=np.float64)
        matrix = np.asarray(problem["matrix"], dtype=np.float64)
        H, W = image.shape
        # For small images, use direct SciPy call
        if H * W <= 4096:
            out = affine_transform(
                image,
                matrix,
                output_shape=(H, W),
                order=3,
                mode='constant',
                cval=0.0,
                prefilter=True
            )
            return {"transformed_image": out.tolist()}
        # Prefilter once for cubic spline interpolation (mirror mode as in SciPy)
        coeffs = spline_filter(image, order=3)
        linear = matrix[:, :2]
        offset0 = matrix[:, 2]
        out = np.zeros((H, W), dtype=np.float64)
        # Determine stripe-based threading
        nthreads = min(multiprocessing.cpu_count(), H)
        splits = [int(i * H / nthreads) for i in range(nthreads + 1)]
        threads = []
        for i in range(nthreads):
            start, end = splits[i], splits[i+1]
            if end <= start:
                continue
            # Adjust offset for stripe start
            off = (
                offset0[0] + matrix[0, 0] * start,
                offset0[1] + matrix[1, 0] * start
            )
            def worker(s=start, e=end, offp=off):
                sub = affine_transform(
                    coeffs,
                    linear,
                    offset=offp,
                    output_shape=(e - s, W),
                    order=3,
                    mode='constant',
                    cval=0.0,
                    prefilter=False
                )
                out[s:e, :] = sub
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return {"transformed_image": out.tolist()}