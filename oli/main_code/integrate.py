import numpy as np
import warnings

from . import constants as const
from .helpers import convert_vel_to_lam, get_lam_mask, get_masked_diffs
from .gaussian_fitting import fit_gaussians

def integrate_flux(
    lam: np.ndarray,
    spec_flux_density: np.ndarray,
    spec_flux_density_err: np.ndarray | None = None, # None doesn't work for num_gaussians > 0
    lam_bounds: tuple[float, float] | None = None, # None doesn't work for num_gaussians > 0
    num_gaussians: int = 0,
    vel_gaussian_fit_width: float = const.VEL_WIDTH_GAUSSIAN_FIT,
    lam_centre: float | None = None,
    n_mc_trials: int = const.NUM_MC_TRIALS,
    calculate_mean_fwhm: bool = False,
    plot_gaussians: bool = False,
    title: str | None = None
) -> tuple[float, float, float | None, float | None]:
    """
    Integrates the spectral flux density over a given wavelength range.

    Parameters
    ----------
    lam: np.ndarray
        Wavelength array (Å).
    spec_flux_density: np.ndarray
        Spectral flux density.
    spec_flux_density_err: np.ndarray | None
        Spectral flux density error. Must not be None if num_gaussians > 0.
    lam_bounds: tuple[float, float] | None
        Wavelength bounds (Å). Must not be None if num_gaussians > 0.
    num_gaussians: int
        Number of Gaussians to fit. Set to 0 to integrate exactly under the
        flux array.
    vel_gaussian_fit_width: float
        Velocity width of data to include when fitting Gaussians.
    lam_centre: float | None
        Zero velocity wavelength centre.
    n_mc_trials: int
        Number of Monte Carlo trials to use for error estimation.
    calculate_mean_fwhm: bool
        Whether to calculate the mean FWHM of the fitted Gaussians across all MC trials.
    plot_gaussians: bool
        Whether to plot the fitted Gaussians (if used).
    title: str | None
        Title of the Gaussian fit plot.

    Returns
    -------
    tuple[float, float, float | None, float | None]
        The integrated flux, flux error, mean FWHM and FWHM error.
    """
    if num_gaussians > 0:
        if lam_centre is None:
            raise ValueError("lam_centre must be provided if num_gaussians > 0")
        gauss_results = fit_gaussians(
            lam, spec_flux_density, spec_flux_density_err,
            num_gaussians=num_gaussians,
            mask_vel_width=vel_gaussian_fit_width,
            mask_lam_centre=lam_centre,
            n_mc_trials=n_mc_trials,
            calculate_mean_fwhm=calculate_mean_fwhm,
            plot_fit=plot_gaussians,
            title=title
        )
        gauss_sfd_vals, gauss_sfd_errs = gauss_results["summed_y_vals"], gauss_results["summed_y_errs"]
        fwhm_mean, fwhm_err = gauss_results["fwhm_mean"], gauss_results["fwhm_err"]

        # mask used for fitting Gaussians
        gaussian_trimmed_mask = get_lam_mask(lam, vel_gaussian_fit_width, lam_centre)
        gaussian_trimmed_lam = lam[gaussian_trimmed_mask]
        # mask used for integrating under the Gaussian values (smaller velocity width than the fit width)
        integrate_width_mask = np.where(
            (gaussian_trimmed_lam > lam_bounds[0]) & # apply the mask to the gauss trimmed lams because the mask
            (gaussian_trimmed_lam < lam_bounds[1]) & # will be applied to the gauss_sfd_vals, so their lengths
            (np.isfinite(gaussian_trimmed_lam))      # need to match
        )
        if gaussian_trimmed_lam.shape != gauss_sfd_vals.shape:
            raise ValueError("gaussian_trimmed_lam and gauss_sfd_vals should have the same shape")
        if gaussian_trimmed_lam.shape != gauss_sfd_errs.shape:
            raise ValueError("gaussian_trimmed_lam and gauss_sfd_errs should have the same shape")
        lam_trimmed = gaussian_trimmed_lam[integrate_width_mask]
        # integrate under gaussian values
        sfd_trimmed = gauss_sfd_vals[integrate_width_mask]
        sfd_err_trimmed = gauss_sfd_errs[integrate_width_mask]
    else:
        integrate_width_mask = np.where(
            (lam > lam_bounds[0]) & (lam < lam_bounds[1]) & (np.isfinite(spec_flux_density))
        ) if lam_bounds is not None else np.where(np.isfinite(spec_flux_density))

        if spec_flux_density.shape != lam.shape:
            raise ValueError("spec_flux_density and lam must have the same shape")
        if spec_flux_density_err is not None:
            if spec_flux_density_err.shape != lam.shape:
                raise ValueError("spec_flux_density_err and lam must have the same shape")
            sfd_err_trimmed = spec_flux_density_err[integrate_width_mask]
        else:
            sfd_err_trimmed = None

        lam_trimmed = lam[integrate_width_mask]
        # integrate under actual flux values
        sfd_trimmed = spec_flux_density[integrate_width_mask]
        fwhm_mean = None
        fwhm_err = None
    

    flux = np.trapezoid(sfd_trimmed, x=lam_trimmed)
    err_weights = get_masked_diffs(lam_trimmed, mask=None)
    flux_err = np.sqrt(np.sum((err_weights * sfd_err_trimmed)**2)) if sfd_err_trimmed is not None else None

    return flux, flux_err, fwhm_mean, fwhm_err

def calculate_balmer_decrement(
    lam: np.ndarray,
    sfd_diff: np.ndarray,
    sfd_diff_err: np.ndarray,
    vel_integration_width: float = const.VEL_TO_IGNORE_WIDTH,
    vel_gaussian_fit_width: float = const.VEL_WIDTH_GAUSSIAN_FIT,
    num_gaussians: int = const.DEFAULT_NUM_GAUSSIANS,
    n_mc_trials: int = const.NUM_MC_TRIALS,
    num_bins: int = 1,
    plot_curves: bool = False,
    year: int | str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the balmer decrement for a given wavelength range.

    Parameters
    ----------
    lam: np.ndarray
        Wavelength array (Å).
    sfd_diff: np.ndarray
        Spectral flux density difference from 2001 spectrum.
    sfd_diff_err: np.ndarray | None
        Spectral flux density difference error.
    vel_integration_width: float
        Velocity width of data to integrate over.
    vel_gaussian_fit_width: float
        Velocity width of data to include when fitting Gaussians.
    num_gaussians: int
        Number of Gaussians to fit. Set to 0 to integrate exactly under the
        flux array.
    n_mc_trials: int
        Number of Monte Carlo trials to use for error estimation.
    num_bins: int
        Number of bins to divide the velocity range into when calculating
        the balmer decrement. Set to 1 to calculate the balmer decrement for the
        entire velocity range.
    plot_curves: bool
        Whether to plot the fitted Gaussians (if used).
    year: int | str
        Epoch of the data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The balmer decrements, balmer decrement errors and velocity bin centres
        for each bin. If `num_bins` is 1, the values are still stored as arrays
        with length 1.
    """
    bin_width = vel_integration_width / num_bins

    # Hα and Hβ masks with `vel_gaussian_fit_width` velocity width
    gfw_alpha_mask = get_lam_mask(lam, vel_gaussian_fit_width, const.H_ALPHA)
    gfw_beta_mask = get_lam_mask(lam, vel_gaussian_fit_width, const.H_BETA)

    if num_gaussians > 0:
        gauss_results_alpha = fit_gaussians(
            lam, sfd_diff, sfd_diff_err,
            num_gaussians=num_gaussians,
            mask_vel_width=vel_gaussian_fit_width,
            mask_lam_centre=const.H_ALPHA,
            n_mc_trials=n_mc_trials,
            plot_fit=plot_curves,
            title=f"{year} Hα flux difference from 2001"
        )
        gauss_results_beta = fit_gaussians(
            lam, sfd_diff, sfd_diff_err,
            num_gaussians=num_gaussians,
            mask_vel_width=vel_gaussian_fit_width,
            mask_lam_centre=const.H_BETA,
            n_mc_trials=n_mc_trials,
            plot_fit=plot_curves,
            title=f"{year} Hβ flux difference from 2001"
        )
        y_alpha, y_alpha_err = gauss_results_alpha["summed_y_vals"], gauss_results_alpha["summed_y_errs"]
        y_beta, y_beta_err = gauss_results_beta["summed_y_vals"], gauss_results_beta["summed_y_errs"]
        if np.any(y_alpha_err < 0) or np.any(y_beta_err < 0):
            raise ValueError("y_alpha_err and y_beta_err should not be negative")
    else:
        # trim the data as if a mask was applied when fitting Gaussians to ensure the lengths match
        y_alpha = sfd_diff[gfw_alpha_mask]
        y_alpha_err = sfd_diff_err[gfw_alpha_mask]
        y_beta = sfd_diff[gfw_beta_mask]
        y_beta_err = sfd_diff_err[gfw_beta_mask]

    # assumes vel_integration_width is < vel_gaussian_fit_width
        # need to make separate masks for each num_gaussians case if assumption is incorrect
    x_alpha = lam[gfw_alpha_mask] 
    x_beta = lam[gfw_beta_mask]

    balmer_decrements = []
    balmer_decrements_err = []
    vel_bin_centres = []

    for i in range(num_bins):
        # get edges and centre of current bin
        vel_left = -vel_integration_width / 2 + i * bin_width
        vel_centre = vel_left + bin_width / 2
        vel_right = vel_left + bin_width

        #TODO: remove testing
        if num_bins == 1:
            print(f"vel_centre: {vel_centre:.3f}") # should be 0
        #

        cur_lam_bounds_alpha = (convert_vel_to_lam(vel_left, const.H_ALPHA), convert_vel_to_lam(vel_right, const.H_ALPHA))
        cur_lam_bounds_beta = (convert_vel_to_lam(vel_left, const.H_BETA), convert_vel_to_lam(vel_right, const.H_BETA))

        h_alpha_flux, h_alpha_flux_err, _, _ = integrate_flux(x_alpha, y_alpha, y_alpha_err, cur_lam_bounds_alpha)
        h_beta_flux, h_beta_flux_err, _, _ = integrate_flux(x_beta, y_beta, y_beta_err, cur_lam_bounds_beta)

        if h_alpha_flux < 0 or h_beta_flux < 0:
            warn_msg = "negative flux values in this bin. returning NaN"
            warnings.warn(warn_msg)
            balmer_decrements.append(np.nan)
            balmer_decrements_err.append(np.nan)
            vel_bin_centres.append(vel_centre)
            continue

        cur_balmer_decrement = h_alpha_flux / h_beta_flux
        cur_balmer_decrement_err = cur_balmer_decrement * np.sqrt(
            (h_alpha_flux_err / h_alpha_flux)**2 + 
            (h_beta_flux_err / h_beta_flux)**2
        )

        balmer_decrements_err.append(cur_balmer_decrement_err)
        balmer_decrements.append(cur_balmer_decrement)

        vel_bin_centres.append(vel_centre)

    return np.array(balmer_decrements), np.array(balmer_decrements_err), np.array(vel_bin_centres)

def get_bd_comparison_info(
    lam: np.ndarray,
    sfd_diff: np.ndarray,
    sfd_diff_err: np.ndarray,
    num_bins_bounds: tuple[int, int],
    num_gaussians_bounds: tuple[int, int],
    vel_integration_width: float = const.VEL_TO_IGNORE_WIDTH,
    vel_gaussian_fit_width: float = const.VEL_WIDTH_GAUSSIAN_FIT,
    n_mc_trials: int = const.NUM_MC_TRIALS,
    print_progress: bool = False,
) -> tuple[list[list[dict[str, np.ndarray]]], list[int], list[int]]:
    """
    Calculates the balmer decrement for a given wavelength range for different
    numbers of Gaussians and velocity bins.

    Parameters
    ----------
    lam: np.ndarray
        Wavelength array (Å).
    sfd_diff: np.ndarray
        Spectral flux density difference from 2001 spectrum.
    sfd_diff_err: np.ndarray | None
        Spectral flux density difference error.
    num_bins_bounds: tuple[int, int]
        Lower and upper bounds for the number of velocity bins (inclusive).
    num_gaussians_bounds: tuple[int, int]
        Lower and upper bounds for the number of Gaussians (inclusive).
    vel_integration_width: float
        Velocity width of data to integrate over.
    vel_gaussian_fit_width: float
        Velocity width of data to include when fitting Gaussians.
    n_mc_trials: int
        Number of Monte Carlo trials to use for error estimation.
    print_progress: bool
        Whether to print progress.

    Returns
    -------
    tuple[list[list[dict[str, np.ndarray]]], list[int], list[int]]
        A list of lists of dictionaries containing the keys "bd", "bd_err",
        "vel_centres", "num_bins" and "num_gaussians" corresponding to the
        balmer decrements, balmer decrement errors, velocity bin centres,
        number of bins and number of Gaussians used for each calculation.
        The inner list iterates over the number of bins, and the outer list
        iterates over the number of Gaussians.
    """
    
    # Store metadata for indexing
    num_gaussians_list = list(range(
        num_gaussians_bounds[0], num_gaussians_bounds[1] + 1
    ))
    num_bins_list = list(range(
        num_bins_bounds[0], num_bins_bounds[1] + 1
    ))

    # Results: [num_gaussians_idx][num_bins_idx] -> {"bd", "bd_err", "vel_centres", "num_bins", "num_gaussians"}
    all_results = []

    for num_gaussians in num_gaussians_list:
        one_gauss_results = []
        if print_progress:
            print(f"Calculating balmer decrements for {num_gaussians}/{num_gaussians_bounds[1]} Gaussians")

        for num_bins in num_bins_list:
            if print_progress:
                print(f"{num_bins}/{num_bins_bounds[1]} bins (of {num_gaussians} gaussians)")
            bd, bd_err, vel_centres = calculate_balmer_decrement(
                lam, sfd_diff, sfd_diff_err,
                num_bins=num_bins,
                num_gaussians=num_gaussians,
                n_mc_trials=n_mc_trials,
                vel_integration_width=vel_integration_width,
                vel_gaussian_fit_width=vel_gaussian_fit_width,
                plot_curves=False
            )
            if np.any(bd_err < 0):
                raise ValueError("Errors should be non-negative")
            if num_bins == 1:
                one_gauss_results.append({
                    'bd': bd[0],
                    'bd_err': bd_err[0],
                    'vel_centres': 0,
                    'num_bins': 1,
                    'num_gaussians': num_gaussians
                })
            else:
                one_gauss_results.append({
                    'bd': bd,
                    'bd_err': bd_err,
                    'vel_centres': vel_centres,
                    'num_bins': num_bins,
                    'num_gaussians': num_gaussians
                })
        
        all_results.append(one_gauss_results)

    return all_results, num_gaussians_list, num_bins_list
