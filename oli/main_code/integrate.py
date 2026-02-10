import numpy as np
import warnings

from . import constants as const
from .helpers import convert_vel_to_lam, get_vel_lam_mask, get_masked_diffs
from .gaussian_fitting import fit_gaussians

def integrate_flux(
    lam: np.ndarray,
    spec_flux_density: np.ndarray,
    spec_flux_density_err: np.ndarray,
    lam_bounds: tuple[float, float],
    num_gaussians: int = 0,
    vel_gaussian_fit_width: float = const.VEL_WIDTH_GAUSSIAN_FIT,
    lam_centre: float | None = None,
    n_mc_trials: int = const.NUM_MC_TRIALS,
    calculate_mean_fwhm: bool = False,
    # plot_gaussians: bool = True, #TODO: change back to False
    plot_gaussians: bool = False,
    title: str | None = None
) -> tuple[float, float, float | None, float | None]:

    

    if num_gaussians > 0:
        if lam_centre is None:
            raise ValueError("lam_centre must be provided if num_gaussians > 0")
        gauss_sfd_vals, gauss_sfd_errs, _, _, fwhm_mean, fwhm_err, _, _ = fit_gaussians(
            lam, spec_flux_density, spec_flux_density_err,
            num_gaussians=num_gaussians,
            mask_vel_width=vel_gaussian_fit_width,
            mask_lam_centre=lam_centre,
            n_mc_trials=n_mc_trials,
            calculate_mean_fwhm=calculate_mean_fwhm,
            plot_fit=plot_gaussians,
            title=title
        )
        gaussian_trimmed_mask = get_vel_lam_mask(lam, vel_gaussian_fit_width, lam_centre)
        gaussian_trimmed_lam = lam[gaussian_trimmed_mask]
        new_integrate_width_mask = np.where(
            (gaussian_trimmed_lam > lam_bounds[0]) &
            (gaussian_trimmed_lam < lam_bounds[1]) &
            (np.isfinite(gaussian_trimmed_lam))
        )
        if gaussian_trimmed_lam.shape != gauss_sfd_vals.shape:
            raise ValueError("gaussian_trimmed_lam and gauss_sfd_vals should have the same shape")
        if gaussian_trimmed_lam.shape != gauss_sfd_errs.shape:
            raise ValueError("gaussian_trimmed_lam and gauss_sfd_errs should have the same shape")
        lam_trimmed = gaussian_trimmed_lam[new_integrate_width_mask]
        sfd_trimmed = gauss_sfd_vals[new_integrate_width_mask]
        sfd_err_trimmed = gauss_sfd_errs[new_integrate_width_mask]
    else:
        if spec_flux_density.shape != lam.shape:
            raise ValueError("spec_flux_density and lam must have the same shape")
        if spec_flux_density_err.shape != lam.shape:
            raise ValueError("spec_flux_density_err and lam must have the same shape")

        integrate_width_mask = np.where((lam > lam_bounds[0]) & (lam < lam_bounds[1]) & (np.isfinite(lam)))
        lam_trimmed = lam[integrate_width_mask]
        sfd_trimmed = spec_flux_density[integrate_width_mask]
        sfd_err_trimmed = spec_flux_density_err[integrate_width_mask]
        fwhm_mean = None
        fwhm_err = None
    
    

    flux = np.trapezoid(sfd_trimmed, x=lam_trimmed)
    err_weights = get_masked_diffs(lam_trimmed, mask=None)
    flux_err = np.sqrt(np.sum((err_weights * sfd_err_trimmed)**2))

    return flux, flux_err, fwhm_mean, fwhm_err


#TODO: check bounds are correct for integration
#TODO: check result for no gaussians and for num_bins > 2
def calculate_balmer_decrement(
    lam: np.ndarray,
    sfd_diff: np.ndarray,
    sfd_diff_err: np.ndarray,
    vel_integration_width: float = const.VEL_TO_IGNORE_WIDTH,
    vel_gaussian_fit_width: float = const.VEL_WIDTH_GAUSSIAN_FIT,
    vel_plot_width: float = const.VEL_PLOT_WIDTH,
    num_gaussians: int = const.DEFAULT_NUM_GAUSSIANS,
    n_mc_trials: int = const.NUM_MC_TRIALS,
    num_bins: int = 1,
    plot_curves: bool = True, #TODO: create a new plotting function for this
    year: int | str = "",
) -> tuple[float, float]:
    bin_width = vel_integration_width / num_bins

    gfw_alpha_mask = get_vel_lam_mask(lam, vel_gaussian_fit_width, const.H_ALPHA)
    gfw_beta_mask = get_vel_lam_mask(lam, vel_gaussian_fit_width, const.H_BETA)

    if num_gaussians > 0:
        gauss_sfd_diff_alpha_vals, gauss_sfd_diff_alpha_errs, _, _, _, _, _, _ = fit_gaussians(
            lam, sfd_diff, sfd_diff_err,
            num_gaussians=num_gaussians,
            mask_vel_width=vel_gaussian_fit_width,
            mask_lam_centre=const.H_ALPHA,
            n_mc_trials=n_mc_trials,
            plot_fit=plot_curves,
            title=f"{year} Hα flux difference from 2001"
        )
        gauss_sfd_diff_beta_vals, gauss_sfd_diff_beta_errs, _, _, _, _, _, _ = fit_gaussians(
            lam, sfd_diff, sfd_diff_err,
            num_gaussians=num_gaussians,
            mask_vel_width=vel_gaussian_fit_width,
            mask_lam_centre=const.H_BETA,
            n_mc_trials=n_mc_trials,
            plot_fit=plot_curves,
            title=f"{year} Hβ flux difference from 2001"
        )
        y_alpha = gauss_sfd_diff_alpha_vals
        y_alpha_err = gauss_sfd_diff_alpha_errs
        y_beta = gauss_sfd_diff_beta_vals
        y_beta_err = gauss_sfd_diff_beta_errs
        if np.any(y_alpha_err < 0) or np.any(y_beta_err < 0):
            raise ValueError("y_alpha_err and y_beta_err should not be negative")
    else:
        y_alpha = sfd_diff[gfw_alpha_mask]
        y_alpha_err = sfd_diff_err[gfw_alpha_mask]
        y_beta = sfd_diff[gfw_beta_mask]
        y_beta_err = sfd_diff_err[gfw_beta_mask]

    # assumes vel_integration_width is < vel_gaussian_fit_width
        # make separate masks for each num_gaussians case if not the case
    x_alpha = lam[gfw_alpha_mask] 
    x_beta = lam[gfw_beta_mask]

    balmer_decrements = []
    balmer_decrements_err = []
    vel_bin_centres = []

    for i in range(num_bins):
        vel_left = -vel_integration_width / 2 + i * bin_width
        vel_centre = vel_left + bin_width / 2
        vel_right = vel_left + bin_width

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
    vel_plot_width: float = const.VEL_PLOT_WIDTH,
    n_mc_trials: int = const.NUM_MC_TRIALS,
    print_progress: bool = False,
) -> tuple[list[list[dict[str, np.ndarray]]], list[int], list[int]]:
    ...
    
    # Store metadata for indexing
    num_gaussians_list = list(range(
        num_gaussians_bounds[0], num_gaussians_bounds[1] + 1
    ))
    num_bins_list = list(range(
        num_bins_bounds[0], num_bins_bounds[1] + 1
    ))

    # Results: [num_gaussians_idx][num_bins_idx] -> (bd_arr, err_arr, vel_arr)
    all_results = []
    # one_bin_results = []

    for num_gaussians in num_gaussians_list:
        one_gauss_results = []
        if print_progress:
            print(f"Calculating balmer decrements for {num_gaussians}/{num_gaussians_bounds[1]} Gaussians")

        for num_bins in num_bins_list:
            if print_progress:
                print(f"{num_bins}/{num_bins_bounds[1]} bins (of {num_gaussians} gaussians)")
            bd, bd_err, vel_centres = calculate_balmer_decrement( #TODO: check why you're getting negative errors
                lam, sfd_diff, sfd_diff_err,
                num_bins=num_bins,
                num_gaussians=num_gaussians,
                n_mc_trials=n_mc_trials,
                vel_integration_width=vel_integration_width,
                vel_gaussian_fit_width=vel_gaussian_fit_width,
                vel_plot_width=vel_plot_width,
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
