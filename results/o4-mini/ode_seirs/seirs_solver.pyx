# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np

def integrate(np.ndarray[np.double_t, ndim=1] y0, double t0, double t1,
              double beta, double sigma, double gamma, double omega):
    cdef double span = t1 - t0
    cdef int N = <int>(span * 50.0)
    if N < 500:
        N = 500
    cdef double dt = span / N
    cdef double S = y0[0]
    cdef double E = y0[1]
    cdef double I = y0[2]
    cdef double R = y0[3]
    cdef double dS1, dE1, dI1, dR1
    cdef double S2, E2, I2, R2
    cdef double dS2, dE2, dI2, dR2
    cdef double S3, E3, I3, R3
    cdef double dS3, dE3, dI3, dR3
    cdef double S4, E4, I4, R4
    cdef double dS4, dE4, dI4, dR4
    for i in range(N):
        dS1 = -beta * S * I + omega * R
        dE1 = beta * S * I - sigma * E
        dI1 = sigma * E - gamma * I
        dR1 = gamma * I - omega * R

        S2 = S + 0.5 * dt * dS1
        E2 = E + 0.5 * dt * dE1
        I2 = I + 0.5 * dt * dI1
        R2 = R + 0.5 * dt * dR1

        dS2 = -beta * S2 * I2 + omega * R2
        dE2 = beta * S2 * I2 - sigma * E2
        dI2 = sigma * E2 - gamma * I2
        dR2 = gamma * I2 - omega * R2

        S3 = S + 0.5 * dt * dS2
        E3 = E + 0.5 * dt * dE2
        I3 = I + 0.5 * dt * dI2
        R3 = R + 0.5 * dt * dR2

        dS3 = -beta * S3 * I3 + omega * R3
        dE3 = beta * S3 * I3 - sigma * E3
        dI3 = sigma * E3 - gamma * I3
        dR3 = gamma * I3 - omega * R3

        S4 = S + dt * dS3
        E4 = E + dt * dE3
        I4 = I + dt * dI3
        R4 = R + dt * dR3

        dS4 = -beta * S4 * I4 + omega * R4
        dE4 = beta * S4 * I4 - sigma * E4
        dI4 = sigma * E4 - gamma * I4
        dR4 = gamma * I4 - omega * R4

        S  += dt * (dS1 + 2*dS2 + 2*dS3 + dS4) / 6.0
        E  += dt * (dE1 + 2*dE2 + 2*dE3 + dE4) / 6.0
        I  += dt * (dI1 + 2*dI2 + 2*dI3 + dI4) / 6.0
        R  += dt * (dR1 + 2*dR2 + 2*dR3 + dR4) / 6.0

    cdef np.ndarray[np.double_t, ndim=1] out = np.empty(4, dtype=np.float64)
    out[0] = S
    out[1] = E
    out[2] = I
    out[3] = R
    return out