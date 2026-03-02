from matplotlib.pylab import not_equal
import numpy as np
import scipy.signal as sps
import scipy.optimize as spo
import warnings
from matplotlib.colors import Colormap
from typing import Callable

from . import constants as const
from .helpers import (
    get_fwhm, get_masked_diffs,
    get_default_bounds, get_lam_mask   
)
from .plotting import plot_gaussians

def get_gaussian_func(n: int, return_sum: bool = True) -> Callable:
    """
    Makes a function that calculates the sum of n Gaussians for given x
    values and Gaussian parameters (heights, mus and sigmas). Note: the
    returned function does not take n as a parameter.

    Parameters
    ----------
    n: int
        Number of Gaussians to fit.
    return_sum: bool
        Whether to return the sum of the Gaussians or the individual Gaussians.

    Returns
    -------
    func: Callable
        The Gaussian calculation function.
    """

    def func(x: np.ndarray, *params: float) -> np.ndarray | np.ndarray[np.ndarray]:
        # use * to pack params into a tuple (call with * to unpack tuple arguments)
        """
        Calculates the sum of n Gaussians, where n is defined in
        :func:`get_gaussian_func`.

        Parameters
        ----------
        x: np.ndarray
            x values (e.g. wavelength array).
        *params: float
            The parameters of the Gaussians: [height_1, height_2, ..., height_n,
            mu_1, mu_2, ..., mu_n, sigma_1, sigma_2, ..., sigma_n]

        Returns
        -------
        y: np.ndarray
            The sum of the n Gaussians for the given x values and parameters,
            or a 2d array of each individual Gaussian if `return_sum` is False.
        """
        
        heights = np.array(params[:n])
        mus = np.array(params[n:2*n])
        sigmas = np.array(params[2*n:3*n])
        
        # store the result as a 2d array
        res = np.zeros((n, len(x)))
        for i in range(n):
            res[i, :] = heights[i] * np.exp(-(x - mus[i])**2 / (2 * sigmas[i]**2))
        
        if return_sum:
            # sum the Gaussians element-wise to get the total Gaussian fit
            return np.sum(res, axis=0)
        return res

    return func

def get_initial_guesses(
    num_gaussians: int,
    x: np.ndarray,
    y: np.ndarray,
    bounds: tuple[list[float], list[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets initial guesses for the parameters (heights, mus, sigmas)
    of n Gaussians.

    Parameters
    ----------
    num_gaussians: int
        Number of Gaussians to fit.
    x: np.ndarray
        x values (e.g. wavelength array).
    y: np.ndarray
        y values (e.g. flux).
    bounds: tuple[list[float], list[float]]
        The bounds on parameters.
        The first list is the lower bounds, and the second list is the upper bounds.
        In each list, the first `num_of_gaussians` elements are the height bounds,
        the next `num_of_gaussians` elements are the mu bounds, and the last
        `num_of_gaussians` elements are the sigma bounds. i.e.
        `([h_min] * n + [μ_min] * n + [σ_min] * n, [h_max] * n + [μ_max] * n + [σ_max] * n)`

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The initial guesses for the parameters: `[height_guesses, mu_guesses, sigma_guesses]`,
        where each array is of length `num_gaussians`.
    """
    height_guesses = np.ones(num_gaussians) * np.max(y)
    mu_guesses = np.ones(num_gaussians) * x[np.argmax(y)] #TODO: adjust these a bit (rand noise maybe?)

    peak_indices = [np.argmax(y)] * num_gaussians
    fwhm_guesses_unitless, _, _, _ = sps.peak_widths(y, peak_indices, rel_height=0.5) # returns width in indices at half max
    peak_x_diffs = get_masked_diffs(x, peak_indices)
    fwhm_guesses = fwhm_guesses_unitless * peak_x_diffs
    sigma_guesses = fwhm_guesses / const.SIGMA_TO_FWHM

    min_height = bounds[0][0]
    min_mu = bounds[0][num_gaussians]
    min_sigma = bounds[0][2*num_gaussians]

    max_height = bounds[1][0]
    max_mu = bounds[1][num_gaussians]
    max_sigma = bounds[1][2*num_gaussians]

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
    num_gaussians: int = const.DEFAULT_NUM_GAUSSIANS,
    n_mc_trials: int = const.NUM_MC_TRIALS,
    max_func_ev: int = const.MAXFEV,
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
    set_bounds_to_default: bool = True,
    mask_vel_width: float | None = const.VEL_WIDTH_GAUSSIAN_FIT,
    mask_lam_centre: float | None = None,
    calculate_mean_fwhm: bool = False,
    use_best_fit_params_for_mc: bool = False, # faster and less accurate (smaller) errors if True
    plot_fit: bool = False,
    plot_y_errs: bool = True,
    plot_summed_gaussian_errs: bool = False,
    colour_map: Colormap = const.COLOUR_MAP,
    error_opacity: float = const.ERR_OPAC,
    y_axis_label: str = const.SFD_Y_AX_LABEL,
    x_axis_label: str = const.REST_ANG_LABEL,
    title: str | None = None,
    save_fig_name: str | None = ""
) -> dict[str, np.ndarray | np.ndarray[np.ndarray] | None | float]:
    """
    Fits n Gaussians to a given dataset using :func:`scipy.optimize.curve_fit`
    with :func:`get_gaussian_func`.

    Parameters
    ----------
    x_untrimmed: np.ndarray
        x values (e.g. wavelength array).
    y_untrimmed: np.ndarray
        y values (e.g. flux).
    y_errs_untrimmed: np.ndarray | None
        y uncertainties (e.g. flux uncertainties).
    num_gaussians: int
        Number of Gaussians to fit.
    n_mc_trials: int
        Number of Monte Carlo trials to use for error estimation.
    max_func_ev: int
        Maximum number of function evaluations to use when optimising the curve fit.
    lower_bounds: list[float] | None
        Lower bounds on parameters, i.e. `([h_min] * n + [μ_min] * n + [σ_min] * n)`.
        If None, and `set_bounds_to_default` is True, the default bounds will be used
        (see :func:`get_default_bounds`), otherwise the parameters will be unbounded.
    upper_bounds: list[float] | None
        Upper bounds on parameters, i.e. `([h_max] * n + [μ_max] * n + [σ_max] * n)`.
        If None, and `set_bounds_to_default` is True, the default bounds will be used
        (see :func:`get_default_bounds`), otherwise the parameters will be unbounded.
    set_bounds_to_default: bool
        Whether to set the bounds to the default bounds (see :func:`get_default_bounds`).
        Note: this is ignored if `lower_bounds` or `upper_bounds` are provided.
    mask_vel_width: float | None
        Velocity width of data to include when fitting Gaussians. If None, no masking is applied.
    mask_lam_centre: float | None
        Zero velocity wavelength centre of the mask.
    calculate_mean_fwhm: bool
        Whether to calculate the mean FWHM of the fitted Gaussians across all MC trials.
    use_best_fit_params_for_mc: bool
        When running MC trials to calculate uncertainties, in each trial, the inital parameter
        guesses will be the best fitting parameters obtained from the *results* of the curve
        fit on the exact data (no noise perturbations) if this is True. Else, the initial guesses
        will be the same as what was used when fitting the exact data (i.e. calculated using
        :func:`get_initial_guesses`). 
    plot_fit: bool
        Whether to plot the fit.
    plot_y_errs: bool
        Whether to plot the y uncertainties on the original data (from y_errs_untrimmed).
    plot_summed_gaussian_errs: bool
        Whether to plot the errors on the fitted n Gaussians (calculated using MC trials).
    colour_map: Colormap
        The colour map to use when plotting each of the n Gaussians.
    error_opacity: float
        The opacity of the error regions.
    y_axis_label: str
        The label for the y axis.
    x_axis_label: str
        The label for the x axis.
    title: str | None
        The title of the plot.
    save_fig_name: str | None
        The name of the file to save the plot to.

    Returns
    -------
    dict[str, np.ndarray | np.ndarray[np.ndarray] | None | float]
        A dictionary containing the results of the fit:
        - "summed_y_vals": the fitted summed Gaussian values
        - "summed_y_errs": the errors on the fitted summed Gaussian values (from std of MC trials)
        - "sep_y_vals": the fitted individual Gaussian values (shape `(num_gaussians, len(x))`)
        - "red_chi_sq": the reduced chi squared of the fit (if y_errs are provided)
        - "fwhm_mean": the mean FWHM of the fitted Gaussians across all MC trials
        - "fwhm_err": the error on the mean FWHM (from std of MC trials)
        - "heights": the heights of the fitted Gaussians
        - "mus": the mus of the fitted Gaussians
        - "sigmas": the sigmas of the fitted Gaussians 
        - "height_errs": the errors on the heights of the fitted Gaussians (from curve fit's covariance matrix)
        - "mu_errs": the errors on the mus of the fitted Gaussians (from curve fit's covariance matrix)
        - "sigma_errs": the errors on the sigmas of the fitted Gaussians (from curve fit's covariance matrix)
    """
    # trim data
    if mask_vel_width is not None:
        if mask_lam_centre is None:
            raise ValueError("mask_lam_centre must be provided if mask_vel_width is provided")
        if x_untrimmed.shape != y_untrimmed.shape:
            raise ValueError("x_untrimmed and y_untrimmed must have the same shape")
        mask = get_lam_mask(x_untrimmed, mask_vel_width, mask_lam_centre)
        x = x_untrimmed[mask]
        y = y_untrimmed[mask]
        y_errs = y_errs_untrimmed[mask] if y_errs_untrimmed is not None else None
    else:
        mask = None
        x = x_untrimmed
        y = y_untrimmed
        y_errs = y_errs_untrimmed

    # change bounds if needed
    default_bounds = get_default_bounds(x, y, num_gaussians)
    if lower_bounds is None:
        if set_bounds_to_default:
            lower_bounds = default_bounds[0]
        else:
            # no lower bounds for all parameters
            lower_bounds = -np.inf
    if upper_bounds is None:
        if set_bounds_to_default:
            upper_bounds = default_bounds[1]
        else:
            # no upper bounds for all parameters
            upper_bounds = np.inf
    

    height_guesses, mu_guesses, sigma_guesses = get_initial_guesses(num_gaussians, x, y, default_bounds)
    flat_param_guesses = np.concatenate((height_guesses, mu_guesses, sigma_guesses))
    calculate_n_gaussians_func = get_gaussian_func(num_gaussians)
    calculate_n_gaussians_func_sep = get_gaussian_func(num_gaussians, return_sum=False)

    # optimise the gaussian parameters to fit n Gaussians to the data (according to the calculate_n_gaussians_func)
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

    # now use the optimised parameters to actually calculate the fitted summed and individual Gaussians
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
            # 3n because each Gaussian has 3 parameters (height, mu, sigma)
        red_chi_sq = chi_sq / dof
        
        if use_best_fit_params_for_mc:
            mc_init_params = best_fit_params
        else:
            mc_init_params = flat_param_guesses

        mc_results = get_mc_errs_perturb_y(
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
        summed_y_errs = mc_results["y_fit_errs"]
        fwhm_mean, fwhm_err = mc_results["fwhm_mean"], mc_results["fwhm_err"]

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
            colour_map=colour_map,
            save_fig_name=save_fig_name
        )
    
    return {
        "summed_y_vals": summed_y_vals,
        "summed_y_errs": summed_y_errs,
        "sep_y_vals": sep_y_vals,
        "red_chi_sq": red_chi_sq,
        "fwhm_mean": fwhm_mean,
        "fwhm_err": fwhm_err,
        "heights": heights,
        "mus": mus,
        "sigmas": sigmas,
        "height_errs": height_errs,
        "mu_errs": mu_errs,
        "sigma_errs": sigma_errs
    }

def get_mc_errs_perturb_y(
    x: np.ndarray,
    y: np.ndarray,
    y_errs: np.ndarray,
    num_gaussians: int,
    initial_params: np.ndarray,
    max_func_ev: int,
    lower_bounds: list[float] | float,
    upper_bounds: list[float] | float,
    best_fit_y: np.ndarray, # used for bias calculation
    n_mc_trials: int = const.NUM_MC_TRIALS,
    calculate_mean_fwhm: bool = False,
    print_warnings: bool = False,
) -> dict[str, np.ndarray | float | None]:
    """
    Calculates the errors on the fitted Gaussians using Monte Carlo trials.
    Each trial perturbs the y values randomly according to the y uncertainties.

    Parameters
    ----------
    x: np.ndarray
        x values (e.g. wavelength array).
    y: np.ndarray
        y values (e.g. flux).
    y_errs: np.ndarray
        y uncertainties (e.g. flux uncertainties).
    num_gaussians: int
        Number of Gaussians to fit.
    initial_params: np.ndarray
        Initial parameters for the curve fit to use in each MC trial. i.e.
        `[height_1, height_2, ..., height_n, mu_1, mu_2, ..., mu_n, sigma_1, sigma_2, ..., sigma_n]`
    max_func_ev: int
        Maximum number of function evaluations to use when optimising the curve fit in each MC trial.
    lower_bounds: list[float]
        Lower bounds on parameters, i.e. `([h_min] * n + [μ_min] * n + [σ_min] * n)`.
        If a float, all parameters will be bounded by this value (-np.inf will disable
        all lower bounds).
    upper_bounds: list[float]
        Upper bounds on parameters, i.e. `([h_max] * n + [μ_max] * n + [σ_max] * n)`.
        If a float, all parameters will be bounded by this value (np.inf will disable
        all upper bounds).
    best_fit_y: np.ndarray
        The fitted summed Gaussian values from the exact data curve fit. Used for bias calculation.
    n_mc_trials: int
        Number of Monte Carlo trials to run.
    calculate_mean_fwhm: bool
        Whether to calculate the mean FWHM of the fitted Gaussians across all MC trials.
    print_warnings: bool
        Whether to print warnings.

    Returns
    -------
    dict[str, np.ndarray | float | None]
        A dictionary containing the results of the MC trials:
        - "y_fit_errs": the errors on the fitted summed Gaussian values
        - "fwhm_mean": the mean FWHM of the fitted Gaussians across all MC trials
        - "fwhm_err": the error on the mean FWHM
        - "bias_sigma": the bias of the fitted summed Gaussian values as a fraction of the
           errors on the fitted y values
        - "y_fit_mean": the mean of the fitted summed Gaussian values across all MC trials
    """
    
    calculate_n_gaussians_func = get_gaussian_func(num_gaussians)
    
    fitted_y_samples = []
    fwhm_samples = []
    
    for i in range(n_mc_trials):
        y_perturbed = y + np.random.normal(loc=0, scale=y_errs) # mean (loc) and standard deviations (scale)
        
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
            if print_warnings:
                warn_msg = f"Failed to fit Gaussian model on trial {i + 1}"
                warnings.warn(warn_msg)

            continue  # Skip failed fits
        
        if calculate_mean_fwhm:
            fwhm = get_fwhm(x, ith_fitted_y)
            fwhm_samples.append(fwhm)
    
    fitted_y_samples = np.array(fitted_y_samples)
    y_fit_mean = np.mean(fitted_y_samples, axis=0)
    y_fit_errs = np.std(fitted_y_samples, axis=0)
    fwhm_mean = np.mean(fwhm_samples) if calculate_mean_fwhm else None
    fwhm_err = np.std(fwhm_samples) if calculate_mean_fwhm else None

    bias_sigma = np.abs(y_fit_mean - best_fit_y) / y_fit_errs

    # Flag if bias > 0.5σ anywhere
    if print_warnings and np.any(bias_sigma > 0.5):
        mean_bias_sigma = np.mean(bias_sigma)
        warn_msg = (
            f"Potential bias: max deviation = {np.max(bias_sigma):.2f}σ\n" +
            f"Mean bias: {mean_bias_sigma:.3f}σ"
        )
        warnings.warn(warn_msg)


    return {
        "y_fit_errs": y_fit_errs,
        "fwhm_mean": fwhm_mean,
        "fwhm_err": fwhm_err,
        "bias_sigma": bias_sigma,
        "y_fit_mean": y_fit_mean
    }


def get_mc_errs_perturb_params(
    x: np.ndarray,
    best_fit_params: np.ndarray,
    param_cov_matrix: np.ndarray,
    num_gaussians: int,
    n_trials: int = const.NUM_MC_TRIALS
) -> np.ndarray:
    """
    Calculates the errors on the fitted Gaussians using Monte Carlo trials.
    Each trial perturbs the best fit parameters randomly according to the
    covariance matrix of the exact data curve fit.

    Parameters
    ----------
    x: np.ndarray
        x values (e.g. wavelength array).
    best_fit_params: np.ndarray
        The best fit parameters from the exact data curve fit. i.e.
        `[height_1, height_2, ..., height_n, mu_1, mu_2, ...,
        mu_n, sigma_1, sigma_2, ..., sigma_n]`
    param_cov_matrix: np.ndarray
        The covariance matrix of the exact data curve fit. (from :func:`scipy.optimize.curve_fit`)
    num_gaussians: int
        Number of Gaussians to fit.
    n_trials: int
        Number of Monte Carlo trials to run.

    Returns
    -------
    np.ndarray
        The errors on the fitted Gaussians.
    """

    # Sample parameters from multivariate normal using covariance matrix
    param_samples = np.random.multivariate_normal(best_fit_params, param_cov_matrix, size=n_trials)
    
    # Evaluate model for each parameter sample
    calculate_n_gaussians_func = get_gaussian_func(num_gaussians)
    y_samples = []
    for trial_params in param_samples:  # trial_params = [A1, A2, μ1, μ2, σ1, σ2] for one trial
        y_samples.append(calculate_n_gaussians_func(x, *trial_params))

    y_errs = np.std(y_samples, axis=0)  # element-wise y-uncertainties
    return y_errs

