import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from lmfit.models import GaussianModel
from scipy.integrate import simpson
import matplotlib.pyplot as plt

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

def integrate_flux_with_err(
    lam: np.ndarray,
    spec_flux_density: np.ndarray,
    spec_flux_density_err: np.ndarray
) -> tuple[float, float]:
    flux_integral = np.trapz(spec_flux_density, x=lam)

    dx = np.diff(lam)
    # Weights for interior points are 1.0 (0.5 from left trap + 0.5 from right trap)
    # Weights for endpoints are 0.5
    
    # General case for non-uniform spacing:
    # Each sigma_i is multiplied by (dx_{i-1} + dx_i) / 2
    err_weights = np.zeros_like(lam)
    err_weights[1:] += dx / 2
    err_weights[:-1] += dx / 2
    
    flux_integral_err = np.sqrt(np.sum((err_weights * spec_flux_density_err)**2))

    return flux_integral, flux_integral_err



def integrate_flux(
    lam: np.ndarray,
    spec_flux_density: np.ndarray,
    spec_flux_density_err: np.ndarray,
    lam_bounds: tuple[float, float],
    use_mult_gaussians: bool = False,
    n_gaussians: int = 1,
    n_trials: int = 1000  # Monte Carlo samples
) -> tuple[float, float]:
    
    valid_mask = np.where((lam > lam_bounds[0]) & (lam < lam_bounds[1]))
    x = lam[valid_mask]
    y = spec_flux_density[valid_mask]
    sig = spec_flux_density_err[valid_mask]
    sig[sig == 0] = np.min(sig[sig > 0])  # avoid division by zero
    weights = 1.0 / sig
    
    if use_mult_gaussians:
        # 1. Detect peaks and estimate initial params
        peaks, properties = find_peaks(y, height=np.std(y)*3, distance=3)
        peaks = peaks[:n_gaussians]
        mu_guesses = x[peaks]
        amp_guesses = y[peaks]  # Use y[peaks] instead of properties["peak_heights"] 
                                 # since peaks was shortened
        results_half = peak_widths(y, peaks, rel_height=0.5)
        sigma_guesses = results_half[0] * (x[1] - x[0]) / (2*np.sqrt(2*np.log(2)))
        
        # 2. Build multi-Gaussian model
        model, params = None, None
        for i, (amp, mu, sigma) in enumerate(zip(amp_guesses, mu_guesses, sigma_guesses)):
            prefix = f"Gaussian_model_{i+1}_"
            g = GaussianModel(prefix=prefix)
            if model is None:
                model, params = g, g.make_params()
            else:
                model += g
                params.update(g.make_params())
            params[prefix+'amplitude'].set(value=amp*np.sqrt(2*np.pi)*sigma, min=0)
            params[prefix+'center'].set(value=mu)
            params[prefix+'sigma'].set(value=sigma, min=0.1)
        
        # 3. Monte Carlo integration (Tyler's approach)
        flux_list = []
        for _ in range(n_trials):
            y_perturbed = y + np.random.normal(0, sig)
            result_mc = model.fit(y_perturbed, params.copy(), x=x, weights=weights)
            flux_list.append(np.trapz(result_mc.best_fit, x))
        
        flux = np.mean(flux_list)
        flux_err = np.std(flux_list)

    else:
        flux = np.trapz(spec_flux_density, x=lam)

        dx = np.diff(lam)
        # Weights for interior points are 1.0 (0.5 from left trap + 0.5 from right trap)
        # Weights for endpoints are 0.5
        
        # General case for non-uniform spacing:
        # Each sigma_i is multiplied by (dx_{i-1} + dx_i) / 2
        err_weights = np.zeros_like(lam)
        err_weights[1:] += dx / 2
        err_weights[:-1] += dx / 2
        
        flux_err = np.sqrt(np.sum((err_weights * spec_flux_density_err)**2))
    
    return flux, flux_err

def integrate_flux(
    lam: np.ndarray,
    flux_diff: np.ndarray,
    sigma_diff: np.ndarray,
    lam_bounds: tuple[float, float],
    use_mult_gaussians: bool = False,
    n_gaussians: int = 1
) -> tuple[float, float]:
    valid_mask = np.where((lam > lam_bounds[0]) & (lam < lam_bounds[1]))
    lam_portion = lam[valid_mask]
    flux_portion = flux_diff[valid_mask]
    sig_portion = sigma_diff[valid_mask]
    
    if use_mult_gaussians:
        #TODO: add stuff here
        pass
    else:
        #TODO: manually do trapezoidal rule so that errors can be calculated
            # see test_workspace.py
        flux = simpson(flux_portion, x=lam_portion)
        flux_err = None
        #

    return flux, flux_err

def integrate_flux(
    lam: np.ndarray,
    flux_diff: np.ndarray,
    sigma_diff: np.ndarray,
    lam_bounds: tuple[float, float],
    use_mult_gaussians: bool = False,
    n_gaussians: int = 1
) -> tuple[float, float]:
    valid_mask = np.where((lam > lam_bounds[0]) & (lam < lam_bounds[1]))
    lam_portion = lam[valid_mask]
    flux_portion = flux_diff[valid_mask]
    sig_portion = sigma_diff[valid_mask]
    
    if use_mult_gaussians:
        #TODO: add stuff here
        pass
    else:
        #TODO: manually do trapezoidal rule so that errors can be calculated
            # see test_workspace.py
        flux = simpson(flux_portion, x=lam_portion)
        flux_err = None
        #

    return flux, flux_err

def integrate_flux(
    lam: np.ndarray,
    flux_diff: np.ndarray,
    sigma_diff: np.ndarray,
    lam_bounds: tuple[float, float],
    use_mult_gaussians: bool = False,
    n_gaussians: int = 1,
    n_trials: int = 1000,  # Monte Carlo samples
    plot_fit: bool = False  # Plot the Gaussian fit
) -> tuple[float, float]:
    
    valid_mask = np.where((lam > lam_bounds[0]) & (lam < lam_bounds[1]))
    x = lam[valid_mask]
    y = flux_diff[valid_mask]
    sig = sigma_diff[valid_mask]
    sig[sig == 0] = np.min(sig[sig > 0])  # avoid division by zero
    weights = 1.0 / sig
    
    if use_mult_gaussians:
        # 1. Detect peaks and estimate initial params
        peaks, properties = find_peaks(y, height=np.std(y)*3, distance=3)
        peaks = peaks[:n_gaussians]
        mu_guesses = x[peaks]
        amp_guesses = y[peaks]  # Use y[peaks] instead of properties["peak_heights"] 
                                 # since peaks was shortened
        results_half = peak_widths(y, peaks, rel_height=0.5)
        sigma_guesses = results_half[0] * (x[1] - x[0]) / (2*np.sqrt(2*np.log(2)))
        
        # 2. Build multi-Gaussian model
        model, params = None, None
        for i, (amp, mu, sigma) in enumerate(zip(amp_guesses, mu_guesses, sigma_guesses)):
            prefix = f"g{i}_"
            g = GaussianModel(prefix=prefix)
            if model is None:
                model, params = g, g.make_params()
            else:
                model += g
                params.update(g.make_params())
            params[prefix+'amplitude'].set(value=amp*np.sqrt(2*np.pi)*sigma, min=0)
            params[prefix+'center'].set(value=mu)
            params[prefix+'sigma'].set(value=sigma, min=0.1)
        
        # Fit the model on original data (for plotting)
        result = model.fit(y, params, x=x, weights=weights)
        
        # 3. Monte Carlo integration (Tyler's approach)
        flux_list = []
        for _ in range(n_trials):
            y_perturbed = y + np.random.normal(0, sig)
            result_mc = model.fit(y_perturbed, params.copy(), x=x, weights=weights)
            flux_list.append(np.trapz(result_mc.best_fit, x))
        
        flux = np.mean(flux_list)
        flux_err = np.std(flux_list)
        
        # 4. Plot the fit if requested
        if plot_fit:
            redchisq = result.redchi  # lmfit stores reduced chisq here
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, label="Data", alpha=0.6)
            plt.plot(x, result.best_fit, 'r-', label="Total fit", linewidth=2)
            
            # Plot each component separately
            for name, comp in result.eval_components(x=x).items():
                plt.plot(x, comp, '--', label=name)
            
            # Plot uncertainty band (gray stripe)
            plt.fill_between(
                x,
                y - sig,
                y + sig,
                color="gray",
                alpha=0.3,
                label="Uncertainty"
            )
            plt.ylabel('Flux (10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)')
            plt.xlabel("Wavelength (Å)")
            plt.title("Multi-Gaussian Fit Difference Spectrum")
            
            plt.text(
                0.05, 0.95,
                f"Reduced χ² = {result.redchi:.2f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top'
            )
            plt.legend()
            plt.show()
    else:
        # Simple trapezoidal with error propagation
        flux = np.trapz(y, x)
        # Analytical error: sqrt(sum of (dx * sigma)^2) for trapezoid
        dx = np.diff(x)
        flux_err = np.sqrt(np.sum((0.5 * (sig[:-1] + sig[1:]) * dx)**2))
    
    return flux, flux_err