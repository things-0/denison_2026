import numpy as np

# Trapezoidal rule with error propagation
def integrate_flux_with_err(lam, flux, sigma):
    # Trapezoidal: integral = sum of (f_i + f_{i+1})/2 * dx_i
    dx = np.diff(lam)
    
    # Flux integral (trapezoidal)
    flux_integral = np.sum((flux[:-1] + flux[1:]) / 2 * dx)
    
    # Error propagation: each flux point contributes to two trapezoids
    # (except endpoints which contribute to one)
    weights = np.zeros_like(lam)
    weights[0] = dx[0] / 2
    weights[-1] = dx[-1] / 2
    weights[1:-1] = (dx[:-1] + dx[1:]) / 2
    
    flux_err = np.sqrt(np.sum((weights * sigma)**2))
    
    return flux_integral, flux_err