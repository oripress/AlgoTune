# cython: boundscheck=False, wraparound=False, cdivision=True
def rk4_lv(double x0, double y0, double alpha, double beta,
           double delta, double gamma, double t_span, int N):
    cdef double dt = t_span / N
    cdef double dt2 = dt * 0.5
    cdef double dt6 = dt / 6.0
    cdef double x = x0, y = y0
    cdef double k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y
    cdef double x2, y2, x3, y3, x4, y4
    cdef int i
    for i in range(N):
        # k1
        k1x = alpha * x - beta * x * y
        k1y = delta * x * y - gamma * y
        # k2
        x2 = x + k1x * dt2
        y2 = y + k1y * dt2
        k2x = alpha * x2 - beta * x2 * y2
        k2y = delta * x2 * y2 - gamma * y2
        # k3
        x3 = x + k2x * dt2
        y3 = y + k2y * dt2
        k3x = alpha * x3 - beta * x3 * y3
        k3y = delta * x3 * y3 - gamma * y3
        # k4
        x4 = x + k3x * dt
        y4 = y + k3y * dt
        k4x = alpha * x4 - beta * x4 * y4
        k4y = delta * x4 * y4 - gamma * y4
        # update
        x += dt6 * (k1x + 2*k2x + 2*k3x + k4x)
        y += dt6 * (k1y + 2*k2y + 2*k3y + k4y)
    return x, y