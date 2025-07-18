import cython

cpdef tuple integrate(double t0, double t1, double x0, double y0, double A, double B):
    cdef double nt = t1 - t0
    cdef int nsteps = <int>(50.0 * nt)
    if nsteps < 1:
        nsteps = 1
    cdef double dt = nt / nsteps
    cdef double x = x0
    cdef double y = y0
    cdef int i
    cdef double k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y
    cdef double xt, yt, xt2, yt2, xt3, yt3
    cdef double B1 = B + 1.0
    for i in range(nsteps):
        k1x = A + x*x*y - B1 * x
        k1y = B * x - x*x*y
        xt = x + 0.5 * dt * k1x
        yt = y + 0.5 * dt * k1y
        k2x = A + xt*xt*yt - B1 * xt
        k2y = B * xt - xt*xt*yt
        xt2 = x + 0.5 * dt * k2x
        yt2 = y + 0.5 * dt * k2y
        k3x = A + xt2*xt2*yt2 - B1 * xt2
        k3y = B * xt2 - xt2*xt2*yt2
        xt3 = x + dt * k3x
        yt3 = y + dt * k3y
        k4x = A + xt3*xt3*yt3 - B1 * xt3
        k4y = B * xt3 - xt3*xt3*yt3
        x += dt * (k1x + 2.0*k2x + 2.0*k3x + k4x) / 6.0
        y += dt * (k1y + 2.0*k2y + 2.0*k3y + k4y) / 6.0
    return x, y