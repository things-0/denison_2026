"""
Gaussian Model Fitting with emcee MCMC
======================================

A reusable module for fitting single or multiple Gaussians to spectral data
using the emcee MCMC sampler.

Usage:
    from my_emcee import fit_gaussians
    
    result = fit_gaussians(
        x, y, yerr,
        n_gaussians=2,
        min_fwhm=3.0,
        min_peak_height=1.0,
        plot_corner=True,
        plot_fit=True
    )
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Constants
SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))  # ≈ 2.355


@dataclass
class GaussianFitResult:
    """Container for Gaussian fit results."""
    # Best-fit parameters (median of posterior)
    amplitudes: np.ndarray  # Area under each Gaussian
    centers: np.ndarray  # Center positions
    sigmas: np.ndarray  # Standard deviations
    
    # Derived quantities
    peak_heights: np.ndarray
    fwhms: np.ndarray
    
    # Uncertainties (16th, 50th, 84th percentiles)
    amplitudes_err: np.ndarray  # Shape: (n_gaussians, 3)
    centers_err: np.ndarray
    sigmas_err: np.ndarray
    
    # Full samples and sampler
    samples: np.ndarray
    sampler: emcee.EnsembleSampler
    
    # Integrated flux
    flux: float
    flux_err: float
    
    # Model evaluation
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray
    best_fit: np.ndarray
    
    def __repr__(self):
        s = "GaussianFitResult:\n"
        for i in range(len(self.amplitudes)):
            s += f"  Gaussian {i+1}:\n"
            s += f"    Center: {self.centers[i]:.2f} (+{self.centers_err[i,2]-self.centers_err[i,1]:.2f} / -{self.centers_err[i,1]-self.centers_err[i,0]:.2f})\n"
            s += f"    FWHM: {self.fwhms[i]:.2f}\n"
            s += f"    Peak height: {self.peak_heights[i]:.2f}\n"
            s += f"    Amplitude (area): {self.amplitudes[i]:.2f}\n"
        s += f"  Total flux: {self.flux:.2f} ± {self.flux_err:.2f}\n"
        return s


def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """
    Single Gaussian function.
    
    Parameters
    ----------
    x : array
        Independent variable (e.g., wavelength)
    amplitude : float
        Area under the Gaussian (integrated flux)
    center : float
        Center position
    sigma : float
        Standard deviation (FWHM = sigma * 2.355)
    
    Returns
    -------
    y : array
        Gaussian evaluated at x
    """
    return (amplitude / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - center) / sigma)**2)


def multi_gaussian(x: np.ndarray, params: np.ndarray, n_gaussians: int) -> np.ndarray:
    """
    Sum of multiple Gaussians.
    
    Parameters
    ----------
    x : array
        Independent variable
    params : array
        Flattened array of [amp1, center1, sigma1, amp2, center2, sigma2, ...]
    n_gaussians : int
        Number of Gaussians
    
    Returns
    -------
    y : array
        Sum of Gaussians evaluated at x
    """
    model = np.zeros_like(x)
    for i in range(n_gaussians):
        amp = params[i * 3]
        center = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        model += gaussian(x, amp, center, sigma)
    return model


def log_likelihood(params: np.ndarray, x: np.ndarray, y: np.ndarray, yerr: np.ndarray, 
                   n_gaussians: int) -> float:
    """
    Log-likelihood function (chi-squared).
    
    Parameters
    ----------
    params : array
        Flattened parameter array
    x, y, yerr : arrays
        Data and uncertainties
    n_gaussians : int
        Number of Gaussians
    
    Returns
    -------
    ll : float
        Log-likelihood value
    """
    model = multi_gaussian(x, params, n_gaussians)
    chi2 = np.sum(((y - model) / yerr)**2)
    return -0.5 * chi2


def log_prior(params: np.ndarray, x: np.ndarray, y: np.ndarray, n_gaussians: int,
              min_fwhm: float = 0.0, max_fwhm: Optional[float] = None,
              min_peak_height: float = 0.0, max_peak_height: Optional[float] = None,
              min_amplitude: float = 0.0, max_amplitude: Optional[float] = None) -> float:
    """
    Log-prior function with constraints.
    
    Parameters
    ----------
    params : array
        Flattened parameter array
    x, y : arrays
        Data (used for bounds)
    n_gaussians : int
        Number of Gaussians
    min_fwhm, max_fwhm : float, optional
        FWHM constraints (in same units as x)
    min_peak_height, max_peak_height : float, optional
        Peak height constraints (in same units as y)
    min_amplitude, max_amplitude : float, optional
        Amplitude (area) constraints
    
    Returns
    -------
    lp : float
        Log-prior value (0 if valid, -inf if invalid)
    """
    # Default bounds
    if max_fwhm is None:
        max_fwhm = (x[-1] - x[0])
    if max_peak_height is None:
        max_peak_height = np.max(y) * 10
    if max_amplitude is None:
        max_amplitude = np.max(y) * (x[-1] - x[0])
    
    for i in range(n_gaussians):
        amp = params[i * 3]
        center = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        
        # Basic validity checks
        if amp <= 0 or sigma <= 0:
            return -np.inf
        
        # Center must be within data range
        if not (x[0] < center < x[-1]):
            return -np.inf
        
        # Calculate derived quantities
        fwhm = sigma * SIGMA_TO_FWHM
        peak_height = amp / (sigma * np.sqrt(2 * np.pi))
        
        # FWHM constraints
        if fwhm < min_fwhm or fwhm > max_fwhm:
            return -np.inf
        
        # Peak height constraints
        if peak_height < min_peak_height or peak_height > max_peak_height:
            return -np.inf
        
        # Amplitude constraints
        if amp < min_amplitude or amp > max_amplitude:
            return -np.inf
    
    return 0.0  # Uniform prior within bounds


def log_probability(params: np.ndarray, x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                    n_gaussians: int, prior_kwargs: dict) -> float:
    """
    Total log-probability = log-prior + log-likelihood.
    """
    lp = log_prior(params, x, y, n_gaussians, **prior_kwargs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x, y, yerr, n_gaussians)


def get_initial_guesses(x: np.ndarray, y: np.ndarray, n_gaussians: int,
                        min_fwhm: float = 0.0) -> np.ndarray:
    """
    Get initial parameter guesses from peak detection.
    
    Parameters
    ----------
    x, y : arrays
        Data
    n_gaussians : int
        Number of Gaussians to fit
    min_fwhm : float
        Minimum FWHM (used to set minimum peak width)
    
    Returns
    -------
    initial_params : array
        Initial guesses [amp1, center1, sigma1, amp2, ...]
    """
    # Detect peaks
    dx_med = np.median(np.diff(x))
    min_width_idx = max(1, int(min_fwhm / SIGMA_TO_FWHM / dx_med))
    
    peak_indices, properties = find_peaks(
        y, 
        height=np.std(y) * 2 + np.mean(y),
        distance=min_width_idx,
        width=min_width_idx
    )
    
    if len(peak_indices) == 0:
        # No peaks found, use maximum
        peak_indices = np.array([np.argmax(y)])
    
    # Sort by peak height (descending)
    sorted_idx = np.argsort(y[peak_indices])[::-1]
    peak_indices = peak_indices[sorted_idx][:n_gaussians]
    
    # Get initial guesses
    initial_params = []
    for i, peak_idx in enumerate(peak_indices):
        center = x[peak_idx]
        height = y[peak_idx]
        
        # Estimate sigma from peak width
        try:
            widths, _, _, _ = peak_widths(y, [peak_idx], rel_height=0.5)
            sigma = widths[0] * dx_med / SIGMA_TO_FWHM
        except:
            sigma = max(min_fwhm / SIGMA_TO_FWHM, 5 * dx_med)
        
        # Ensure minimum sigma
        sigma = max(sigma, min_fwhm / SIGMA_TO_FWHM if min_fwhm > 0 else 0.1)
        
        # Calculate amplitude from height
        amplitude = height * sigma * np.sqrt(2 * np.pi)
        
        initial_params.extend([amplitude, center, sigma])
    
    # If fewer peaks found than requested, add extra Gaussians
    while len(initial_params) < n_gaussians * 3:
        # Add a Gaussian at a random position
        center = np.random.uniform(x[0], x[-1])
        sigma = max(min_fwhm / SIGMA_TO_FWHM if min_fwhm > 0 else 0.1, 5 * dx_med)
        height = np.std(y) * 3
        amplitude = height * sigma * np.sqrt(2 * np.pi)
        initial_params.extend([amplitude, center, sigma])
    
    return np.array(initial_params)


def fit_gaussians(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    n_gaussians: int = 2,
    # Constraints
    min_fwhm: float = 0.0,
    max_fwhm: Optional[float] = None,
    min_peak_height: float = 0.0,
    max_peak_height: Optional[float] = None,
    min_amplitude: float = 0.0,
    max_amplitude: Optional[float] = None,
    # MCMC settings
    nwalkers: int = 50,
    nsteps: int = 5000,
    burnin: int = 1000,
    # Plotting options
    plot_corner: bool = False,
    plot_fit: bool = False,
    plot_chains: bool = False,
    title: str = "",
    # Progress
    progress: bool = False
) -> GaussianFitResult:
    """
    Fit multiple Gaussians to data using emcee MCMC.
    
    Parameters
    ----------
    x, y, yerr : arrays
        Data: independent variable, dependent variable, uncertainties
    n_gaussians : int
        Number of Gaussians to fit
    min_fwhm, max_fwhm : float, optional
        FWHM constraints (in same units as x)
    min_peak_height, max_peak_height : float, optional
        Peak height constraints (in same units as y)
    min_amplitude, max_amplitude : float, optional
        Amplitude (area) constraints
    nwalkers : int
        Number of MCMC walkers (default: 50)
    nsteps : int
        Number of MCMC steps (default: 5000)
    burnin : int
        Number of burn-in steps to discard (default: 1000)
    plot_corner : bool
        Whether to plot corner plot of posteriors
    plot_fit : bool
        Whether to plot the fit against data
    plot_chains : bool
        Whether to plot the MCMC chains
    title : str
        Title for plots
    progress : bool
        Whether to show progress bar
    
    Returns
    -------
    result : GaussianFitResult
        Container with fit results, uncertainties, and samples
    """
    # Number of parameters
    ndim = 3 * n_gaussians
    
    # Prior kwargs
    prior_kwargs = {
        'min_fwhm': min_fwhm,
        'max_fwhm': max_fwhm,
        'min_peak_height': min_peak_height,
        'max_peak_height': max_peak_height,
        'min_amplitude': min_amplitude,
        'max_amplitude': max_amplitude
    }
    
    # Get initial guesses
    initial_params = get_initial_guesses(x, y, n_gaussians, min_fwhm)
    
    # Initialize walkers with small scatter around initial guess
    p0 = np.array([
        initial_params * (1 + 0.01 * np.random.randn(ndim))
        for _ in range(nwalkers)
    ])
    
    # Ensure all initial positions are valid
    for i in range(nwalkers):
        while not np.isfinite(log_probability(p0[i], x, y, yerr, n_gaussians, prior_kwargs)):
            p0[i] = initial_params * (1 + 0.01 * np.random.randn(ndim))
    
    # Create sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability,
        args=(x, y, yerr, n_gaussians, prior_kwargs)
    )
    
    # Run MCMC
    sampler.run_mcmc(p0, nsteps, progress=progress)
    
    # Get samples (discard burn-in, flatten)
    samples = sampler.get_chain(discard=burnin, flat=True)
    
    # Extract best-fit parameters (median)
    amplitudes = np.array([np.median(samples[:, i*3]) for i in range(n_gaussians)])
    centers = np.array([np.median(samples[:, i*3+1]) for i in range(n_gaussians)])
    sigmas = np.array([np.median(samples[:, i*3+2]) for i in range(n_gaussians)])
    
    # Calculate uncertainties (16th, 50th, 84th percentiles)
    amplitudes_err = np.array([np.percentile(samples[:, i*3], [16, 50, 84]) for i in range(n_gaussians)])
    centers_err = np.array([np.percentile(samples[:, i*3+1], [16, 50, 84]) for i in range(n_gaussians)])
    sigmas_err = np.array([np.percentile(samples[:, i*3+2], [16, 50, 84]) for i in range(n_gaussians)])
    
    # Derived quantities
    peak_heights = amplitudes / (sigmas * np.sqrt(2 * np.pi))
    fwhms = sigmas * SIGMA_TO_FWHM
    
    # Best-fit model
    best_params = np.zeros(ndim)
    for i in range(n_gaussians):
        best_params[i*3] = amplitudes[i]
        best_params[i*3+1] = centers[i]
        best_params[i*3+2] = sigmas[i]
    best_fit = multi_gaussian(x, best_params, n_gaussians)
    
    # Integrated flux with uncertainty
    flux_samples = []
    for sample in samples[::10]:  # Subsample for speed
        model = multi_gaussian(x, sample, n_gaussians)
        flux_samples.append(np.trapezoid(model, x))
    flux = np.median(flux_samples)
    flux_err = np.std(flux_samples)
    
    # Create result object
    result = GaussianFitResult(
        amplitudes=amplitudes,
        centers=centers,
        sigmas=sigmas,
        peak_heights=peak_heights,
        fwhms=fwhms,
        amplitudes_err=amplitudes_err,
        centers_err=centers_err,
        sigmas_err=sigmas_err,
        samples=samples,
        sampler=sampler,
        flux=flux,
        flux_err=flux_err,
        x=x,
        y=y,
        yerr=yerr,
        best_fit=best_fit
    )
    
    # Plotting
    if plot_chains:
        _plot_chains(sampler, burnin, n_gaussians, title)
    
    if plot_corner:
        _plot_corner(samples, n_gaussians, title)
    
    if plot_fit:
        _plot_fit(result, n_gaussians, title)
    
    return result


def _plot_chains(sampler: emcee.EnsembleSampler, burnin: int, n_gaussians: int, title: str = ""):
    """Plot MCMC chains for convergence check."""
    ndim = 3 * n_gaussians
    fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim), sharex=True)
    samples = sampler.get_chain()
    
    labels = []
    for i in range(n_gaussians):
        labels.extend([f"amp_{i+1}", f"center_{i+1}", f"sigma_{i+1}"])
    
    for i in range(ndim):
        ax = axes[i] if ndim > 1 else axes
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.axvline(burnin, color='r', linestyle='--', label='Burn-in')
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("Step")
    if title:
        fig.suptitle(f"{title} - MCMC Chains")
    plt.tight_layout()
    plt.show()


def _plot_corner(samples: np.ndarray, n_gaussians: int, title: str = ""):
    """Plot corner plot of posterior distributions."""
    try:
        import corner
    except ImportError:
        print("Corner plot requires 'corner' package. Install with: pip install corner")
        return
    
    labels = []
    for i in range(n_gaussians):
        labels.extend([f"amp_{i+1}", f"center_{i+1}", f"σ_{i+1}"])
    
    fig = corner.corner(samples, labels=labels, show_titles=True, title_fmt=".2f")
    if title:
        fig.suptitle(f"{title} - Posterior Distributions", y=1.02)
    plt.show()


def _plot_fit(result: GaussianFitResult, n_gaussians: int, title: str = ""):
    """Plot the fit against data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data with uncertainty band
    ax.plot(result.x, result.y, 'b-', alpha=0.6, label='Data')
    ax.fill_between(
        result.x, 
        result.y - result.yerr, 
        result.y + result.yerr,
        color='gray', alpha=0.3, label='Uncertainty'
    )
    
    # Total fit
    ax.plot(result.x, result.best_fit, 'r-', linewidth=2, label='Total fit')
    
    # Individual components
    colors = plt.cm.tab10(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        component = gaussian(result.x, result.amplitudes[i], result.centers[i], result.sigmas[i])
        ax.plot(result.x, component, '--', color=colors[i], 
                label=f'Gaussian {i+1} (FWHM={result.fwhms[i]:.1f})')
    
    # Reduced chi-squared
    residuals = result.y - result.best_fit
    chi2 = np.sum((residuals / result.yerr)**2)
    dof = len(result.x) - 3 * n_gaussians
    reduced_chi2 = chi2 / dof
    
    ax.text(0.05, 0.95, f"Reduced χ² = {reduced_chi2:.2f}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Flux')
    ax.legend()
    
    if title:
        ax.set_title(f"{title}")
    
    plt.tight_layout()
    plt.show()


# Convenience function for quick fitting
def quick_fit(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
              n_gaussians: int = 1, min_fwhm: float = 3.0,
              plot: bool = True) -> GaussianFitResult:
    """
    Quick and easy Gaussian fitting with sensible defaults.
    
    Parameters
    ----------
    x, y, yerr : arrays
        Data
    n_gaussians : int
        Number of Gaussians (default: 1)
    min_fwhm : float
        Minimum FWHM constraint (default: 3.0)
    plot : bool
        Whether to show fit plot (default: True)
    
    Returns
    -------
    result : GaussianFitResult
    """
    return fit_gaussians(
        x, y, yerr,
        n_gaussians=n_gaussians,
        min_fwhm=min_fwhm,
        min_peak_height=np.std(y) * 2,  # 2 sigma above noise
        nwalkers=32,
        nsteps=3000,
        burnin=500,
        plot_fit=plot,
        progress=True
    )
