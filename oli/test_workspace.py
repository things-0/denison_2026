"""
Cleaned up integrate_flux function for spectral flux integration.
Supports simple trapezoidal integration, lmfit Gaussian fitting, and emcee MCMC fitting.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Union
from scipy.signal import find_peaks, peak_widths
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt

# Try to import my_emcee module
try:
    from my_emcee import fit_gaussians
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False

# =============================================================================
# Constants (these should match your main.ipynb)
# =============================================================================
SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))  # ≈ 2.355
c_km_s = 2.99792458e5  # Speed of light in km/s

# Default values (adjust as needed)
DEFAULT_Z = 0.0582  # Redshift
DEFAULT_MIN_FWHM_KMS = 350  # Minimum FWHM in km/s
DEFAULT_N_MC_TRIALS = 1000  # Number of Monte Carlo trials
DEFAULT_MIN_AMP_SIGMA = 1.0  # Minimum amplitude in units of std(y)
DEFAULT_PEAK_SIGMA_THRESHOLD = 2.9  # Minimum peak height in sigma above mean

# Plot settings
FIGURE_SIZE = (10, 6)
SFD_Y_AX_LABEL = r'Spectral flux density (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)'


# =============================================================================
# Helper functions
# =============================================================================
def convert_vel_to_lam(vel_kms: float, rest_lam: float) -> float:
    """Convert velocity (km/s) to wavelength (Å) using relativistic Doppler."""
    beta = vel_kms / c_km_s
    return rest_lam * np.sqrt((1 + beta) / (1 - beta))


def convert_fwhm_kms_to_ang(fwhm_kms: float, rest_lam: float) -> float:
    """Convert FWHM from km/s to Angstroms."""
    lam_plus = convert_vel_to_lam(fwhm_kms / 2, rest_lam)
    lam_minus = convert_vel_to_lam(-fwhm_kms / 2, rest_lam)
    return lam_plus - lam_minus


# =============================================================================
# Main function
# =============================================================================
def integrate_flux(
    lam: np.ndarray,
    spec_flux_density: np.ndarray,
    spec_flux_density_err: np.ndarray,
    lam_bounds: Tuple[float, float],
    # Gaussian fitting options
    n_gaussians: int = 0,
    use_emcee: bool = False,
    # FWHM constraints (provide one or none)
    min_fwhm_kms: Optional[float] = DEFAULT_MIN_FWHM_KMS,
    min_fwhm_ang: Optional[float] = None,
    max_fwhm_ang: Optional[float] = None,
    # Amplitude/height constraints
    min_amp_sigma: float = DEFAULT_MIN_AMP_SIGMA,
    # Peak detection settings
    peak_sigma_threshold: float = DEFAULT_PEAK_SIGMA_THRESHOLD,
    idx_dist_between_peaks: int = 3,
    # Monte Carlo settings (for lmfit method)
    n_mc_trials: int = DEFAULT_N_MC_TRIALS,
    # emcee settings
    emcee_nwalkers: int = 50,
    emcee_nsteps: int = 5000,
    emcee_burnin: int = 1000,
    # Output options
    calc_err: bool = True,
    plot_fit: bool = False,
    plot_corner: bool = False,
    plot_chains: bool = False,
    title: str = "",
    print_progress: bool = False,
    # Redshift (for velocity-wavelength conversion)
    z: float = DEFAULT_Z,
) -> Union[float, Tuple[float, float]]:
    """
    Integrate spectral flux density over a wavelength range.
    
    Supports three methods:
    1. Simple trapezoidal integration (n_gaussians=0)
    2. Multi-Gaussian fitting with lmfit + Monte Carlo error (n_gaussians>0, use_emcee=False)
    3. Multi-Gaussian fitting with emcee MCMC (n_gaussians>0, use_emcee=True)
    
    Parameters
    ----------
    lam : np.ndarray
        Wavelength array (Å)
    spec_flux_density : np.ndarray
        Spectral flux density array
    spec_flux_density_err : np.ndarray
        Uncertainties on spectral flux density
    lam_bounds : tuple
        (min_wavelength, max_wavelength) to integrate over
    n_gaussians : int
        Number of Gaussians to fit (0 for simple trapezoidal integration)
    use_emcee : bool
        If True, use emcee MCMC for fitting (requires n_gaussians > 0)
    min_fwhm_kms : float, optional
        Minimum FWHM constraint in km/s (converted to Å using central wavelength)
    min_fwhm_ang : float, optional
        Minimum FWHM constraint in Å (overrides min_fwhm_kms if provided)
    max_fwhm_ang : float, optional
        Maximum FWHM constraint in Å (defaults to wavelength range)
    min_amp_sigma : float
        Minimum amplitude in units of std(y)
    peak_sigma_threshold : float
        Minimum peak height for detection in sigma above mean
    idx_dist_between_peaks : int
        Minimum distance between peaks in indices
    n_mc_trials : int
        Number of Monte Carlo trials for error estimation (lmfit method)
    emcee_nwalkers, emcee_nsteps, emcee_burnin : int
        emcee MCMC settings
    calc_err : bool
        Whether to calculate flux uncertainty
    plot_fit : bool
        Whether to plot the Gaussian fit
    plot_corner : bool
        Whether to plot corner plot (emcee only)
    plot_chains : bool
        Whether to plot MCMC chains (emcee only)
    title : str
        Title for plots
    print_progress : bool
        Whether to print progress during Monte Carlo
    z : float
        Redshift (for velocity-wavelength conversion)
    
    Returns
    -------
    flux : float
        Integrated flux
    flux_err : float (if calc_err=True)
        Uncertainty on integrated flux
    """
    # ==========================================================================
    # 1. Extract data within bounds
    # ==========================================================================
    valid_mask = (lam > lam_bounds[0]) & (lam < lam_bounds[1]) & np.isfinite(lam)
    x = lam[valid_mask]
    y = spec_flux_density[valid_mask]
    yerr = spec_flux_density_err[valid_mask]
    
    # Handle zero uncertainties
    yerr[yerr == 0] = np.min(yerr[yerr > 0])
    
    # ==========================================================================
    # 2. Simple trapezoidal integration (no Gaussian fitting)
    # ==========================================================================
    if n_gaussians == 0:
        flux = np.trapezoid(y, x=x)
        
        if not calc_err:
            return flux
        
        # Error propagation for trapezoidal rule
        dx = np.diff(x)
        err_weights = np.zeros_like(x)
        err_weights[1:] += dx / 2
        err_weights[:-1] += dx / 2
        flux_err = np.sqrt(np.sum((err_weights * yerr)**2))
        
        return flux, flux_err
    
    # ==========================================================================
    # 3. Calculate FWHM constraints
    # ==========================================================================
    if min_fwhm_ang is not None:
        min_fwhm = min_fwhm_ang
    elif min_fwhm_kms is not None:
        lam_cent_rest = np.mean(lam_bounds) / (1 + z)
        min_fwhm = convert_fwhm_kms_to_ang(min_fwhm_kms, lam_cent_rest)
    else:
        min_fwhm = 0.0
        warnings.warn("No minimum FWHM constraint set", UserWarning)
    
    if max_fwhm_ang is None:
        max_fwhm = x[-1] - x[0]
    else:
        max_fwhm = max_fwhm_ang
    
    # Height constraints
    min_height = np.std(y) * min_amp_sigma
    max_height = np.max(y) + np.std(y)
    
    # ==========================================================================
    # 4. emcee MCMC fitting
    # ==========================================================================
    if use_emcee:
        if not EMCEE_AVAILABLE:
            raise ImportError("emcee fitting requires my_emcee module. Import failed.")
        
        result = fit_gaussians(
            x, y, yerr,
            n_gaussians=n_gaussians,
            min_fwhm=min_fwhm,
            max_fwhm=max_fwhm,
            min_peak_height=min_height,
            max_peak_height=max_height,
            nwalkers=emcee_nwalkers,
            nsteps=emcee_nsteps,
            burnin=emcee_burnin,
            plot_corner=plot_corner,
            plot_fit=plot_fit,
            plot_chains=plot_chains,
            title=title,
            progress=print_progress
        )
        
        return result.flux, result.flux_err
    
    # ==========================================================================
    # 5. lmfit Gaussian fitting
    # ==========================================================================
    weights = 1.0 / yerr
    
    # Detect peaks
    highest_peak_sigma = (np.max(y) - np.mean(y)) / np.std(y)
    if highest_peak_sigma < peak_sigma_threshold:
        warnings.warn(
            f"Highest peak ({highest_peak_sigma:.2f}σ) below threshold ({peak_sigma_threshold}σ). "
            f"Lowering threshold.",
            UserWarning
        )
        peak_sigma_threshold = highest_peak_sigma * 0.99
    
    peak_indices, _ = find_peaks(
        y,
        distance=idx_dist_between_peaks,
        height=np.std(y) * peak_sigma_threshold + np.mean(y)
    )
    
    if len(peak_indices) < n_gaussians:
        warnings.warn(f"Only {len(peak_indices)} peaks found, expected {n_gaussians}", UserWarning)
    
    peak_indices = peak_indices[:n_gaussians]
    
    # Get initial guesses
    mu_guesses = x[peak_indices]
    peak_height_guesses = y[peak_indices]
    
    idx_widths, _, _, _ = peak_widths(y, peak_indices, rel_height=0.5)
    dx_med = np.median(np.diff(x))
    fwhm_guesses = idx_widths * dx_med
    sigma_guesses = fwhm_guesses / SIGMA_TO_FWHM
    
    # Ensure minimum sigma
    min_sigma = min_fwhm / SIGMA_TO_FWHM
    max_sigma = max_fwhm / SIGMA_TO_FWHM
    sigma_guesses = np.maximum(sigma_guesses, min_sigma)
    
    # Build multi-Gaussian model
    model, params = None, None
    
    # Calculate amplitude bounds
    min_amp = min_height * np.sqrt(2 * np.pi) * max_sigma
    max_amp = max_height * np.sqrt(2 * np.pi) * max_sigma
    
    # Center bounds: where signal is above noise
    above_noise = y > np.std(y) * 1.5
    if np.any(above_noise):
        min_mu = x[np.argmax(above_noise)]
        max_mu = x[len(x) - 1 - np.argmax(above_noise[::-1])]
    else:
        min_mu, max_mu = x[0], x[-1]
    
    for i, (height, mu, sigma) in enumerate(zip(peak_height_guesses, mu_guesses, sigma_guesses)):
        prefix = f"Gaussian_{i+1}_"
        g = GaussianModel(prefix=prefix)
        
        if model is None:
            model, params = g, g.make_params()
        else:
            model += g
            params.update(g.make_params())
        
        # Set parameter values and bounds
        amp_guess = height * np.sqrt(2 * np.pi) * sigma
        params[prefix + 'amplitude'].set(value=amp_guess, min=min_amp, max=max_amp)
        params[prefix + 'center'].set(value=mu, min=min_mu, max=max_mu)
        params[prefix + 'sigma'].set(value=sigma, min=min_sigma, max=max_sigma)
    
    # Fit the model
    result = model.fit(y, params, x=x, weights=weights)
    best_fit = result.best_fit
    flux = np.trapezoid(best_fit, x)
    
    if not calc_err:
        if plot_fit:
            raise ValueError("plot_fit requires calc_err=True")
        return flux
    
    # Monte Carlo error estimation
    flux_list = np.zeros(n_mc_trials)
    for i in range(n_mc_trials):
        y_perturbed = y + np.random.normal(0, yerr)
        result_mc = model.fit(y_perturbed, params.copy(), x=x, weights=weights)
        flux_list[i] = np.trapezoid(result_mc.best_fit, x)
        
        if print_progress and (i + 1) % max(1, n_mc_trials // 10) == 0:
            print(f"MC trial {i + 1}/{n_mc_trials} completed")
    
    flux_err = np.std(flux_list)
    
    # Plot if requested
    if plot_fit:
        _plot_gaussian_fit(x, y, yerr, result, title)
    
    return flux, flux_err


def _plot_gaussian_fit(x, y, yerr, result, title=""):
    """Plot the Gaussian fit against data."""
    plt.figure(figsize=FIGURE_SIZE)
    
    # Data
    plt.plot(x, y, 'b-', alpha=0.6, label="Data")
    
    # Uncertainty band
    plt.fill_between(x, y - yerr, y + yerr, color="gray", alpha=0.3, label="Uncertainty")
    
    # Total fit
    plt.plot(x, result.best_fit, 'r-', linewidth=2, label="Total fit")
    
    # Individual components
    for name, comp in result.eval_components(x=x).items():
        plt.plot(x, comp, '--', label=name.strip("_"))
    
    # Reduced chi-squared
    plt.text(
        0.05, 0.95,
        f"Reduced χ² = {result.redchi:.2f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top'
    )
    
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(SFD_Y_AX_LABEL)
    plt.title(f"Multi-Gaussian Fit of {title}Difference Spectrum" if title else "Multi-Gaussian Fit")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Spectral line constants
# =============================================================================
H_ALPHA = 6564.61  # Å (rest frame)
H_BETA = 4862.68   # Å (rest frame)


# =============================================================================
# Balmer Decrement Calculation
# =============================================================================
def calculate_balmer_decrement(
    lam: np.ndarray,
    flux_diff: np.ndarray,
    sigma_diff: np.ndarray,
    vel_width: float,
    # Binning options (provide one or none)
    num_bins: Optional[int] = 1,
    bin_width: Optional[float] = None,
    adjust_vel_width: bool = False,
    # Gaussian fitting options
    n_gaussians: int = 0,
    use_emcee: bool = False,
    # FWHM constraints
    min_fwhm_kms: Optional[float] = DEFAULT_MIN_FWHM_KMS,
    min_fwhm_ang: Optional[float] = None,
    # Amplitude/height constraints
    min_amp_sigma: float = DEFAULT_MIN_AMP_SIGMA,
    peak_sigma_threshold: float = DEFAULT_PEAK_SIGMA_THRESHOLD,
    # Monte Carlo / emcee settings
    n_mc_trials: int = DEFAULT_N_MC_TRIALS,
    emcee_nwalkers: int = 50,
    emcee_nsteps: int = 5000,
    emcee_burnin: int = 1000,
    # Output options
    plot_fit: bool = False,
    plot_corner: bool = False,
    print_progress: bool = False,
    title_prefix: str = "",
    # Redshift
    z: float = DEFAULT_Z,
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculate the Balmer decrement (Hα/Hβ flux ratio).
    
    Supports two modes:
    1. Single bin (num_bins=1): Returns (balmer_decrement, balmer_decrement_err)
    2. Multiple bins: Returns (balmer_decrements, errors, velocity_bin_centers)
    
    Parameters
    ----------
    lam : np.ndarray
        Wavelength array (Å, observed frame)
    flux_diff : np.ndarray
        Difference spectrum flux
    sigma_diff : np.ndarray
        Uncertainties on difference spectrum
    vel_width : float
        Total velocity width to integrate over (km/s)
    num_bins : int, optional
        Number of velocity bins (default: 1 for single integrated value)
    bin_width : float, optional
        Width of each velocity bin in km/s (alternative to num_bins)
    adjust_vel_width : bool
        If True, adjust vel_width to be divisible by bin_width
    n_gaussians : int
        Number of Gaussians to fit (0 for trapezoidal integration)
    use_emcee : bool
        If True, use emcee MCMC for fitting
    min_fwhm_kms : float, optional
        Minimum FWHM constraint in km/s
    min_fwhm_ang : float, optional
        Minimum FWHM constraint in Å (overrides min_fwhm_kms)
    min_amp_sigma : float
        Minimum amplitude in units of std(y)
    peak_sigma_threshold : float
        Minimum peak height for detection in sigma above mean
    n_mc_trials : int
        Number of Monte Carlo trials for error estimation
    emcee_nwalkers, emcee_nsteps, emcee_burnin : int
        emcee MCMC settings
    plot_fit : bool
        Whether to plot Gaussian fits
    plot_corner : bool
        Whether to plot corner plots (emcee only)
    print_progress : bool
        Whether to print progress
    title_prefix : str
        Prefix for plot titles (e.g., "2021 ")
    z : float
        Redshift
    
    Returns
    -------
    If num_bins == 1:
        balmer_decrement : float
        balmer_decrement_err : float
    If num_bins > 1:
        balmer_decrements : np.ndarray
        balmer_decrements_err : np.ndarray
        vel_bin_centers : np.ndarray
    """
    # ==========================================================================
    # 1. Handle binning parameters
    # ==========================================================================
    if num_bins is None and bin_width is None:
        raise ValueError("Either num_bins or bin_width must be provided")
    
    if num_bins is not None and bin_width is not None:
        raise ValueError("Only one of num_bins or bin_width should be provided")
    
    if bin_width is not None:
        # Calculate num_bins from bin_width
        if vel_width % bin_width != 0:
            if adjust_vel_width:
                vel_width -= vel_width % bin_width
                print(f"Adjusted vel_width to {vel_width} km/s (divisible by bin_width)")
            else:
                raise ValueError(
                    f"vel_width ({vel_width}) is not divisible by bin_width ({bin_width}). "
                    f"Set adjust_vel_width=True to auto-adjust."
                )
        num_bins = int(vel_width / bin_width)
    else:
        bin_width = vel_width / num_bins
    
    # ==========================================================================
    # 2. Single bin calculation
    # ==========================================================================
    if num_bins == 1:
        # Calculate wavelength bounds for Hα and Hβ
        h_alpha_lam_bounds = (
            convert_vel_to_lam(-vel_width / 2, H_ALPHA),
            convert_vel_to_lam(vel_width / 2, H_ALPHA)
        )
        h_beta_lam_bounds = (
            convert_vel_to_lam(-vel_width / 2, H_BETA),
            convert_vel_to_lam(vel_width / 2, H_BETA)
        )
        
        # Common kwargs for integrate_flux
        integrate_kwargs = dict(
            n_gaussians=n_gaussians,
            use_emcee=use_emcee,
            min_fwhm_kms=min_fwhm_kms,
            min_fwhm_ang=min_fwhm_ang,
            min_amp_sigma=min_amp_sigma,
            peak_sigma_threshold=peak_sigma_threshold,
            n_mc_trials=n_mc_trials,
            emcee_nwalkers=emcee_nwalkers,
            emcee_nsteps=emcee_nsteps,
            emcee_burnin=emcee_burnin,
            plot_fit=plot_fit,
            plot_corner=plot_corner,
            print_progress=print_progress,
            z=z,
        )
        
        # Integrate Hα
        h_alpha_flux, h_alpha_flux_err = integrate_flux(
            lam, flux_diff, sigma_diff, h_alpha_lam_bounds,
            title=f"{title_prefix}Hα ",
            **integrate_kwargs
        )
        
        # Integrate Hβ
        h_beta_flux, h_beta_flux_err = integrate_flux(
            lam, flux_diff, sigma_diff, h_beta_lam_bounds,
            title=f"{title_prefix}Hβ ",
            **integrate_kwargs
        )
        
        # Calculate Balmer decrement and error
        balmer_decrement = h_alpha_flux / h_beta_flux
        balmer_decrement_err = balmer_decrement * np.sqrt(
            (h_alpha_flux_err / h_alpha_flux)**2 +
            (h_beta_flux_err / h_beta_flux)**2
        )
        
        return balmer_decrement, balmer_decrement_err
    
    # ==========================================================================
    # 3. Multiple bins calculation
    # ==========================================================================
    if n_gaussians > 0:
        raise ValueError(
            "Gaussian fitting (n_gaussians > 0) is not supported for multiple velocity bins. "
            "Use num_bins=1 for Gaussian fitting."
        )
    
    balmer_decrements = []
    balmer_decrements_err = []
    vel_bin_centers = []
    
    for i in range(num_bins):
        vel_left = -vel_width / 2 + i * bin_width
        vel_center = vel_left + bin_width / 2
        vel_right = vel_left + bin_width
        
        # Wavelength bounds for this bin
        cur_lam_bounds_alpha = (
            convert_vel_to_lam(vel_left, H_ALPHA),
            convert_vel_to_lam(vel_right, H_ALPHA)
        )
        cur_lam_bounds_beta = (
            convert_vel_to_lam(vel_left, H_BETA),
            convert_vel_to_lam(vel_right, H_BETA)
        )
        
        # Integrate (simple trapezoidal)
        h_alpha_flux, h_alpha_flux_err = integrate_flux(
            lam, flux_diff, sigma_diff, cur_lam_bounds_alpha, z=z
        )
        h_beta_flux, h_beta_flux_err = integrate_flux(
            lam, flux_diff, sigma_diff, cur_lam_bounds_beta, z=z
        )
        
        # Calculate Balmer decrement for this bin
        cur_balmer = h_alpha_flux / h_beta_flux
        cur_balmer_err = cur_balmer * np.sqrt(
            (h_alpha_flux_err / h_alpha_flux)**2 +
            (h_beta_flux_err / h_beta_flux)**2
        )
        
        balmer_decrements.append(cur_balmer)
        balmer_decrements_err.append(cur_balmer_err)
        vel_bin_centers.append(vel_center)
    
    return (
        np.array(balmer_decrements),
        np.array(balmer_decrements_err),
        np.array(vel_bin_centers)
    )


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    
    x = np.linspace(6500, 6700, 200)
    
    # True Gaussians
    y_true = (
        50 * np.exp(-0.5 * ((x - 6563) / 15)**2) +  # Broad component
        30 * np.exp(-0.5 * ((x - 6563) / 5)**2)     # Narrow component
    )
    
    yerr = np.ones_like(x) * 3
    y = y_true + np.random.normal(0, 3, len(x))
    
    print("Testing integrate_flux function...")
    print("=" * 50)
    
    # Test 1: Simple trapezoidal
    flux_trap, flux_trap_err = integrate_flux(x, y, yerr, (6500, 6700))
    print(f"Trapezoidal: {flux_trap:.2f} ± {flux_trap_err:.2f}")
    
    # Test 2: lmfit Gaussian fitting
    flux_lmfit, flux_lmfit_err = integrate_flux(
        x, y, yerr, (6500, 6700),
        n_gaussians=2,
        min_fwhm_ang=3.0,
        n_mc_trials=100,
        plot_fit=True,
        title="Test "
    )
    print(f"lmfit Gaussian: {flux_lmfit:.2f} ± {flux_lmfit_err:.2f}")
    
    # Test 3: emcee (if available)
    if EMCEE_AVAILABLE:
        flux_emcee, flux_emcee_err = integrate_flux(
            x, y, yerr, (6500, 6700),
            n_gaussians=2,
            use_emcee=True,
            min_fwhm_ang=3.0,
            emcee_nsteps=2000,
            emcee_burnin=500,
            plot_fit=True,
            print_progress=True,
            title="Test emcee "
        )
        print(f"emcee Gaussian: {flux_emcee:.2f} ± {flux_emcee_err:.2f}")
