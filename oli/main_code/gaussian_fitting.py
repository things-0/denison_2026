import numpy as np
import scipy.signal as sps
import scipy.optimize as spo
import warnings
from typing import Any

from .constants import *
from .helpers import get_masked_diffs, get_default_bounds, get_vel_lam_mask
from .plotting import plot_gaussians

def get_gaussian_func(n: int, return_sum: bool = True):
    """Factory that creates a curve_fit-compatible function for n Gaussians."""
    def func(x, *params): # use * to pack params into a tuple (call with * to unpack tuple arguments)
        heights = np.array(params[:n])
        mus = np.array(params[n:2*n])
        sigmas = np.array(params[2*n:3*n])
        
        res = np.zeros((n, len(x)))
        for i in range(n):
            res[i, :] = heights[i] * np.exp(-(x - mus[i])**2 / (2 * sigmas[i]**2))
        
        if return_sum:
            return np.sum(res, axis=0)
        return res

    return func

def get_initial_guesses(
    num_of_gaussians: int,
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[list[float], list[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height_guesses = np.ones(num_of_gaussians) * np.max(y)
    mu_guesses = np.ones(num_of_gaussians) * x[np.argmax(y)] #TODO: adjust these a bit (rand noise maybe?)

    peak_indices = [np.argmax(y)] * num_of_gaussians
    fwhm_guesses_unitless, _, _, _ = sps.peak_widths(y, peak_indices, rel_height=0.5) # returns width in indices at half max
    peak_x_diffs = get_masked_diffs(x, peak_indices)
    fwhm_guesses = fwhm_guesses_unitless * peak_x_diffs
    sigma_guesses = fwhm_guesses / SIGMA_TO_FWHM

    min_height = bounds[0][0]
    min_mu = bounds[0][num_of_gaussians]
    min_sigma = bounds[0][2*num_of_gaussians]
    # min_height = bounds[0][:num_of_gaussians]
    # min_mu = bounds[0][num_of_gaussians:2*num_of_gaussians]
    # min_sigma = bounds[0][2*num_of_gaussians:]

    max_height = bounds[1][0]
    max_mu = bounds[1][num_of_gaussians]
    max_sigma = bounds[1][2*num_of_gaussians]
    # max_height = bounds[1][:num_of_gaussians]
    # max_mu = bounds[1][num_of_gaussians:2*num_of_gaussians]
    # max_sigma = bounds[1][2*num_of_gaussians:]

    clipped_height_guesses = np.clip(height_guesses, min_height, max_height)
    clipped_mu_guesses = np.clip(mu_guesses, min_mu, max_mu)
    clipped_sigma_guesses = np.clip(sigma_guesses, min_sigma, max_sigma)

    if (
        np.any(clipped_height_guesses != height_guesses) or
        np.any(clipped_mu_guesses != mu_guesses) or
        np.any(clipped_sigma_guesses != sigma_guesses)
    ):
        warn_msg = (
            "\nInitial guesses were outside of bounds given.\n" +
            "Values have been clipped to the bounds.\n" +
            f"Height bounds: ({min_height}, {max_height})\n" +
            f"Height guesses: {height_guesses}\n" +
            f"Clipped height guesses: {clipped_height_guesses}\n" +
            f"Mu bounds: ({min_mu}, {max_mu})\n" +
            f"Mu guesses: {mu_guesses}\n" +
            f"Clipped mu guesses: {clipped_mu_guesses}\n" +
            f"Sigma bounds: ({min_sigma}, {max_sigma})\n" +
            f"Sigma guesses: {sigma_guesses}\n" +
            f"Clipped sigma guesses: {clipped_sigma_guesses}\n"
        )
        warnings.warn(warn_msg)
    
    return clipped_height_guesses, clipped_mu_guesses, clipped_sigma_guesses

def fit_gaussians(
    x: np.ndarray,
    y: np.ndarray,
    num_of_gaussians: int = DEFAULT_NUM_GAUSSIANS,
    n_trials: int = NUM_MC_TRIALS,
    max_func_ev: int = MAXFEV,
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
    set_bounds_to_default: bool = True,
    mask_vel_width: float | None = VEL_PLOT_WIDTH,
    mask_lam_centre: float | None = None,
    plot_fit: bool = False,
    plot_params: dict[str, Any] = {}
        # plot_y_errs: bool,
        # y_errs: np.ndarray,
        # plot_summed_gaussian_errs: bool,
        # title: str,
        # y_axis_label: str,
        # x_axis_label: str,
        # colour_map: Colormap
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray[np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    if mask_vel_width is not None:
        if mask_lam_centre is None:
            raise ValueError("mask_lam_centre must be provided if mask_vel_width is provided")
        mask = get_vel_lam_mask(x, mask_vel_width, mask_lam_centre)
        x = x[mask]
        y = y[mask]
    else:
        mask = None

    default_bounds = get_default_bounds(x, y, num_of_gaussians)
    if lower_bounds is None:
        if set_bounds_to_default:
            lower_bounds = default_bounds[0]
        else:
            lower_bounds = -np.inf
    if upper_bounds is None:
        if set_bounds_to_default:
            upper_bounds = default_bounds[1]
        else:
            upper_bounds = np.inf
    

    height_guesses, mu_guesses, sigma_guesses = get_initial_guesses(num_of_gaussians, x, y, default_bounds)
    flat_params = np.concatenate((height_guesses, mu_guesses, sigma_guesses))
    calculate_n_gaussians_func = get_gaussian_func(num_of_gaussians)
    calculate_n_gaussians_func_sep = get_gaussian_func(num_of_gaussians, return_sum=False)
    #TD: remove testing
    # print(f"height bounds: [{lower_bounds[0]:.2f}, {upper_bounds[0]:.2f}]")
    # print(f"mu bounds: [{lower_bounds[num_of_gaussians]:.2f}, {upper_bounds[num_of_gaussians]:.2f}]")
    # print(f"sigma bounds: [{lower_bounds[2*num_of_gaussians]:.2f}, {upper_bounds[2*num_of_gaussians]:.2f}]")
    # for i in range(num_of_gaussians):
    #     print(f"Peak {i+1} guesses: A={height_guesses[i]:.2f}, μ={mu_guesses[i]:.2f}, σ={sigma_guesses[i]:.2f}")
    #
    best_fit_params, param_cov_matrix = spo.curve_fit(
        f=calculate_n_gaussians_func,
        xdata=x,
        ydata=y,
        p0=flat_params,
        maxfev=max_func_ev,
        bounds=(lower_bounds, upper_bounds)
    )
    heights, mus, sigmas = best_fit_params.reshape(3, num_of_gaussians)
    #TD: remove testing
    # for i in range(num_of_gaussians):
    #     print(f"Peak {i+1} best fit: A={heights[i]:.2f}, μ={mus[i]:.2f}, σ={sigmas[i]:.2f}")
    #
    height_errs, mu_errs, sigma_errs = (np.sqrt(np.diag(param_cov_matrix))).reshape(3, num_of_gaussians)
    summed_y_errs = get_mc_errs(x, best_fit_params, param_cov_matrix, num_of_gaussians, n_trials=n_trials)

    summed_y_vals = calculate_n_gaussians_func(x, *best_fit_params)
    sep_y_vals = calculate_n_gaussians_func_sep(x, *best_fit_params)

    if plot_fit:
        if plot_params.get("plot_y_errs") is None or False:
            y_errs = None
        else:
            y_errs = plot_params.get("y_errs")
            if mask is not None:
                y_errs = y_errs[mask]
        if plot_params.get("plot_summed_gaussian_errs") is None or False:
            summed_gaussian_errs = None
        else:
            summed_gaussian_errs = summed_y_errs
        if plot_params.get("error_opacity") is None:
            error_opacity = ERR_OPAC
        else:
            error_opacity = plot_params.get("error_opacity")
        plot_gaussians(
            x=x,
            y_data=y,
            mask_vel_width=mask_vel_width,
            mask_lam_centre=mask_lam_centre,
            y_data_errs=y_errs,
            summed_gaussian_errs=summed_gaussian_errs,
            sep_gaussian_vals=sep_y_vals,
            summed_gaussian_vals=summed_y_vals,
            title=plot_params.get("title"),
            y_axis_label=plot_params.get("y_axis_label"),
            x_axis_label=plot_params.get("x_axis_label"),
            error_opacity=error_opacity 
        )
    
    return summed_y_vals, summed_y_errs, sep_y_vals, (heights, mus, sigmas), (height_errs, mu_errs, sigma_errs)

def get_mc_errs(
    x: np.ndarray,
    best_fit_params: np.ndarray,
    param_cov_matrix: np.ndarray,
    num_of_gaussians: int,
    n_trials: int = NUM_MC_TRIALS
) -> tuple[np.ndarray, np.ndarray[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:

    # Sample parameters from multivariate normal using covariance matrix
    param_samples = np.random.multivariate_normal(best_fit_params, param_cov_matrix, size=n_trials)

    # Evaluate model for each parameter sample
    calculate_n_gaussians_func = get_gaussian_func(num_of_gaussians)
    y_samples = []
    for trial_params in param_samples:  # trial_params = [A1, A2, μ1, μ2, σ1, σ2] for one trial
        y_samples.append(calculate_n_gaussians_func(x, *trial_params))

    y_errs = np.std(y_samples, axis=0)  # element-wise y-uncertainties
    return y_errs
