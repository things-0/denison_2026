from matplotlib.pylab import not_equal
import numpy as np
import scipy.signal as sps
import scipy.optimize as spo
import warnings
from matplotlib.colors import Colormap

from .constants import *
from .helpers import (
    get_fwhm, get_masked_diffs,
    get_default_bounds, get_vel_lam_mask   
)
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
    num_gaussians: int,
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[list[float], list[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height_guesses = np.ones(num_gaussians) * np.max(y)
    mu_guesses = np.ones(num_gaussians) * x[np.argmax(y)] #TODO: adjust these a bit (rand noise maybe?)

    peak_indices = [np.argmax(y)] * num_gaussians
    fwhm_guesses_unitless, _, _, _ = sps.peak_widths(y, peak_indices, rel_height=0.5) # returns width in indices at half max
    peak_x_diffs = get_masked_diffs(x, peak_indices)
    fwhm_guesses = fwhm_guesses_unitless * peak_x_diffs
    sigma_guesses = fwhm_guesses / SIGMA_TO_FWHM

    min_height = bounds[0][0]
    min_mu = bounds[0][num_gaussians]
    min_sigma = bounds[0][2*num_gaussians]
    # min_height = bounds[0][:num_gaussians]
    # min_mu = bounds[0][num_gaussians:2*num_gaussians]
    # min_sigma = bounds[0][2*num_gaussians:]

    max_height = bounds[1][0]
    max_mu = bounds[1][num_gaussians]
    max_sigma = bounds[1][2*num_gaussians]
    # max_height = bounds[1][:num_gaussians]
    # max_mu = bounds[1][num_gaussians:2*num_gaussians]
    # max_sigma = bounds[1][2*num_gaussians:]

    clipped_height_guesses = np.clip(height_guesses, min_height, max_height)
    clipped_mu_guesses = np.clip(mu_guesses, min_mu, max_mu)
    clipped_sigma_guesses = np.clip(sigma_guesses, min_sigma, max_sigma)

    if (
        np.any(clipped_height_guesses != height_guesses) or
        np.any(clipped_mu_guesses != mu_guesses) or
        np.any(clipped_sigma_guesses != sigma_guesses)
    ):
        warn_msg = (
            "Initial guesses were outside of bounds given.\n" +
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
    x_untrimmed: np.ndarray,
    y_untrimmed: np.ndarray,
    y_errs_untrimmed: np.ndarray | None = None,
    num_gaussians: int = DEFAULT_NUM_GAUSSIANS,
    n_mc_trials: int = NUM_MC_TRIALS,
    max_func_ev: int = MAXFEV,
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
    set_bounds_to_default: bool = True,
    mask_vel_width: float | None = VEL_WIDTH_GAUSSIAN_FIT,
    mask_lam_centre: float | None = None,
    calculate_mean_fwhm: bool = False,
    use_best_fit_params_for_mc: bool = False, # faster and less accurate (smaller) errors if True
    # plot_fit: bool = True, #TODO: change back to False
    plot_fit: bool = False,
    plot_y_errs: bool = True,
    plot_summed_gaussian_errs: bool = False,
    colour_map: Colormap = COLOUR_MAP,
    error_opacity: float = ERR_OPAC,
    y_axis_label: str = SFD_Y_AX_LABEL,
    x_axis_label: str = ANG_LABEL,
    title: str | None = None
) -> tuple[
    np.ndarray, np.ndarray | None,
    np.ndarray[np.ndarray], float | None,
    float | None, float | None,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    if mask_vel_width is not None:
        if mask_lam_centre is None:
            raise ValueError("mask_lam_centre must be provided if mask_vel_width is provided")
        if x_untrimmed.shape != y_untrimmed.shape:
            raise ValueError("x_untrimmed and y_untrimmed must have the same shape")
        mask = get_vel_lam_mask(x_untrimmed, mask_vel_width, mask_lam_centre)
        x = x_untrimmed[mask]
        y = y_untrimmed[mask]
        y_errs = y_errs_untrimmed[mask] if y_errs_untrimmed is not None else None
    else:
        mask = None
        x = x_untrimmed
        y = y_untrimmed
        y_errs = y_errs_untrimmed

    default_bounds = get_default_bounds(x, y, num_gaussians)
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
    

    height_guesses, mu_guesses, sigma_guesses = get_initial_guesses(num_gaussians, x, y, default_bounds)
    flat_param_guesses = np.concatenate((height_guesses, mu_guesses, sigma_guesses))
    calculate_n_gaussians_func = get_gaussian_func(num_gaussians)
    calculate_n_gaussians_func_sep = get_gaussian_func(num_gaussians, return_sum=False)
    #TD: remove testing
    # print(f"height bounds: [{lower_bounds[0]:.2f}, {upper_bounds[0]:.2f}]")
    # print(f"mu bounds: [{lower_bounds[num_gaussians]:.2f}, {upper_bounds[num_gaussians]:.2f}]")
    # print(f"sigma bounds: [{lower_bounds[2*num_gaussians]:.2f}, {upper_bounds[2*num_gaussians]:.2f}]")
    # for i in range(num_gaussians):
    #     print(f"Peak {i+1} guesses: A={height_guesses[i]:.2f}, μ={mu_guesses[i]:.2f}, σ={sigma_guesses[i]:.2f}")
    #
    best_fit_params, param_cov_matrix = spo.curve_fit(
        f=calculate_n_gaussians_func,
        xdata=x,
        ydata=y,
        p0=flat_param_guesses,
        maxfev=max_func_ev,
        bounds=(lower_bounds, upper_bounds)
    )
    heights, mus, sigmas = best_fit_params.reshape(3, num_gaussians)
    
    param_errs = (np.sqrt(np.diag(param_cov_matrix))).reshape(3, num_gaussians)
    height_errs, mu_errs, sigma_errs = param_errs

    summed_y_vals = calculate_n_gaussians_func(x, *best_fit_params)
    sep_y_vals = calculate_n_gaussians_func_sep(x, *best_fit_params)

    if y_errs is None:
        warn_msg = (
            r"y_errs are not provided, so reduced $\chi^2$ and " +
            r"MC errors cannot be calculated"
        )
        warnings.warn(warn_msg)
        red_chi_sq = None
        summed_y_errs = None
    else:
        residuals = y - summed_y_vals
        chi_sq = np.sum((residuals / y_errs)**2)
        dof = len(x) - 3 * num_gaussians 
            # number of data points minus number of parameters
            # 3 because each Gaussian has 3 parameters (height, mu, sigma)
        red_chi_sq = chi_sq / dof
        
        #TD: remove
        # summed_y_errs = get_mc_errs_perturb_params(x, best_fit_params, param_cov_matrix, num_gaussians, n_trials=n_trials)
        #

        if use_best_fit_params_for_mc:
            mc_init_params = best_fit_params
        else:
            mc_init_params = flat_param_guesses

        summed_y_errs, fwhm_mean, fwhm_err, _, _ = get_mc_errs_perturb_y(
            x=x,
            y=y,
            y_errs=y_errs,
            num_gaussians=num_gaussians,
            initial_params=mc_init_params,
            max_func_ev=max_func_ev,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            n_mc_trials=n_mc_trials,
            best_fit_y=summed_y_vals,
            calculate_mean_fwhm=calculate_mean_fwhm,
            print_warnings=plot_fit
        )

    if plot_fit:
        plot_gaussians(
            x=x,
            y_data=y,
            mask_vel_width=mask_vel_width,
            mask_lam_centre=mask_lam_centre,
            y_data_errs=y_errs if plot_y_errs else None,
            summed_gaussian_errs=summed_y_errs if plot_summed_gaussian_errs else None,
            sep_gaussian_vals=sep_y_vals,
            summed_gaussian_vals=summed_y_vals,
            red_chi_sq=red_chi_sq,
            title=title,
            y_axis_label=y_axis_label,
            x_axis_label=x_axis_label,
            error_opacity=error_opacity,
            colour_map=colour_map
        )
    
    return (
        summed_y_vals, summed_y_errs, sep_y_vals,
        red_chi_sq, fwhm_mean, fwhm_err,
        (heights, mus, sigmas),
        (height_errs, mu_errs, sigma_errs)
    )

def get_mc_errs_perturb_y(
    x: np.ndarray,
    y: np.ndarray,
    y_errs: np.ndarray,
    num_gaussians: int,
    initial_params: np.ndarray,
    max_func_ev: int,
    lower_bounds: list[float],
    upper_bounds: list[float],
    best_fit_y: np.ndarray, # used for bias calculation
    n_mc_trials: int = NUM_MC_TRIALS,
    calculate_mean_fwhm: bool = False,
    print_warnings: bool = False,
) -> tuple[np.ndarray, float | None, float | None, np.ndarray, np.ndarray]:
    
    calculate_n_gaussians_func = get_gaussian_func(num_gaussians)
    
    fitted_y_samples = []
    fwhm_samples = []
    
    for i in range(n_mc_trials):
        y_perturbed = y + np.random.normal(0, y_errs)
        
        try:
            best_fit_params, _ = spo.curve_fit(
                f=calculate_n_gaussians_func,
                xdata=x,
                ydata=y_perturbed,  
                p0=initial_params,  
                maxfev=max_func_ev,
                bounds=(lower_bounds, upper_bounds)
            )
            ith_fitted_y = calculate_n_gaussians_func(x, *best_fit_params)
            fitted_y_samples.append(ith_fitted_y)
        except RuntimeError:
            #TD: remove testing
            print(f"Failed to fit Gaussian model on trial {i + 1}")
            #
            continue  # Skip failed fits
        
        if calculate_mean_fwhm:
            fwhm = get_fwhm(x, ith_fitted_y)
            fwhm_samples.append(fwhm)
    
    fitted_y_samples = np.array(fitted_y_samples)
    y_fit_mean = np.mean(fitted_y_samples, axis=0)
    y_fit_errs = np.std(fitted_y_samples, axis=0)
    fwhm_mean = np.mean(fwhm_samples) if calculate_mean_fwhm else None
    fwhm_err = np.std(fwhm_samples) if calculate_mean_fwhm else None

    # Bias in units of MC standard deviation
    bias_sigma = np.abs(y_fit_mean - best_fit_y) / y_fit_errs

    # Flag if bias > 0.5σ anywhere
    if print_warnings and np.any(bias_sigma > 0.5):
        mean_bias_sigma = np.mean(bias_sigma)
        warn_msg = (
            f"Potential bias: max deviation = {np.max(bias_sigma):.2f}σ\n" +
            f"Mean bias: {mean_bias_sigma:.3f}σ"
        )
        warnings.warn(warn_msg)

    
    return y_fit_errs, fwhm_mean, fwhm_err, bias_sigma, y_fit_mean


def get_mc_errs_perturb_params(
    x: np.ndarray,
    best_fit_params: np.ndarray,
    param_cov_matrix: np.ndarray,
    num_gaussians: int,
    n_trials: int = NUM_MC_TRIALS
) -> np.ndarray:

    # Sample parameters from multivariate normal using covariance matrix
    param_samples = np.random.multivariate_normal(best_fit_params, param_cov_matrix, size=n_trials)
    
    # Evaluate model for each parameter sample
    calculate_n_gaussians_func = get_gaussian_func(num_gaussians)
    y_samples = []
    for trial_params in param_samples:  # trial_params = [A1, A2, μ1, μ2, σ1, σ2] for one trial
        y_samples.append(calculate_n_gaussians_func(x, *trial_params))

    y_errs = np.std(y_samples, axis=0)  # element-wise y-uncertainties
    return y_errs

