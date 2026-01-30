import numpy as np
import warnings

from .constants import *
from .helpers import convert_vel_to_lam, get_vel_lam_mask, get_masked_diffs
from .gaussian_fitting import fit_gaussians

def integrate_flux(
    lam: np.ndarray,
    spec_flux_density: np.ndarray,
    spec_flux_density_err: np.ndarray,
    lam_bounds: tuple[float, float],
    num_gaussians: int = 0,
    lam_centre: float | None = None,
    n_mc_trials: int = NUM_MC_TRIALS,
    plot_gaussians: bool = True, #TODO: change back to False
    # plot_gaussians: bool = False,
    title: str | None = None
) -> tuple[float, float]:

    valid_mask = np.where((lam > lam_bounds[0]) & (lam < lam_bounds[1]) & (np.isfinite(lam)))
    lam_trimmed = lam[valid_mask]
    sfd_trimmed = spec_flux_density[valid_mask]
    sfd_err_trimmed = spec_flux_density_err[valid_mask]

    if num_gaussians > 0:
        if lam_centre is None:
            raise ValueError("lam_centre must be provided if num_gaussians > 0")
        gauss_sfd_vals, gauss_sfd_errs, _, _, _, _ = fit_gaussians(
            lam_trimmed, sfd_trimmed, sfd_err_trimmed,
            num_gaussians=num_gaussians,
            mask_vel_width=None,
            mask_lam_centre=lam_centre,
            n_trials=n_mc_trials,
            plot_fit=plot_gaussians,
            title=title
        )
        y = gauss_sfd_vals
        y_err = gauss_sfd_errs
    else:
        y = sfd_trimmed
        y_err = sfd_err_trimmed
    
    x = lam_trimmed

    flux = np.trapezoid(y, x=x)
    err_weights = get_masked_diffs(x, mask=None)
    flux_err = np.sqrt(np.sum((err_weights * y_err)**2))

    return flux, flux_err

#TODO: check bounds are correct for integration
#TODO: check result for no gaussians and for num_bins > 2
def calculate_balmer_decrement(
    lam: np.ndarray,
    sfd_diff: np.ndarray,
    sfd_diff_err: np.ndarray,
    vel_calculation_width: float = VEL_TO_IGNORE_WIDTH,
    vel_plot_width: float = VEL_PLOT_WIDTH,
    num_gaussians: int = DEFAULT_NUM_GAUSSIANS,
    n_mc_trials: int = NUM_MC_TRIALS,
    num_bins: int = 1,
    plot_curves: bool = True, #TODO: create a new plotting function for this
    year: int | str = "",
) -> tuple[float, float]:
    bin_width = vel_calculation_width / num_bins

    if num_gaussians > 0:
        gauss_sfd_diff_alpha_vals, gauss_sfd_diff_alpha_errs, _, _, _, _ = fit_gaussians(
            lam, sfd_diff, sfd_diff_err,
            num_gaussians=num_gaussians,
            mask_vel_width=vel_plot_width,
            mask_lam_centre=H_ALPHA,
            n_trials=n_mc_trials,
            plot_fit=plot_curves,
            title=f"{year} Hα flux difference from 2001"
        )
        gauss_sfd_diff_beta_vals, gauss_sfd_diff_beta_errs, _, _, _, _ = fit_gaussians(
            lam, sfd_diff, sfd_diff_err,
            num_gaussians=num_gaussians,
            mask_vel_width=vel_plot_width,
            mask_lam_centre=H_BETA,
            n_trials=n_mc_trials,
            plot_fit=plot_curves,
            title=f"{year} Hβ flux difference from 2001"
        )
        y_alpha = gauss_sfd_diff_alpha_vals
        y_alpha_err = gauss_sfd_diff_alpha_errs
        y_beta = gauss_sfd_diff_beta_vals
        y_beta_err = gauss_sfd_diff_beta_errs
    else:
        y_alpha = sfd_diff
        y_alpha_err = sfd_diff_err
        y_beta = sfd_diff
        y_beta_err = sfd_diff_err

    x_alpha = lam[get_vel_lam_mask(lam, vel_plot_width, H_ALPHA)]
    x_beta = lam[get_vel_lam_mask(lam, vel_plot_width, H_BETA)]

    balmer_decrements = []
    balmer_decrements_err = []
    vel_bin_centres = []

    for i in range(num_bins):
        vel_left = -vel_calculation_width / 2 + i * bin_width
        vel_centre = vel_left + bin_width / 2
        vel_right = vel_left + bin_width

        cur_lam_bounds_alpha = (convert_vel_to_lam(vel_left, H_ALPHA), convert_vel_to_lam(vel_right, H_ALPHA))
        cur_lam_bounds_beta = (convert_vel_to_lam(vel_left, H_BETA), convert_vel_to_lam(vel_right, H_BETA))

        h_alpha_flux, h_alpha_flux_err = integrate_flux(x_alpha, y_alpha, y_alpha_err, cur_lam_bounds_alpha)
        h_beta_flux, h_beta_flux_err = integrate_flux(x_beta, y_beta, y_beta_err, cur_lam_bounds_beta)

        cur_balmer_decrement = h_alpha_flux / h_beta_flux
        cur_balmer_decrement_err = cur_balmer_decrement * np.sqrt(
            (h_alpha_flux_err / h_alpha_flux)**2 + 
            (h_beta_flux_err / h_beta_flux)**2
        )

        balmer_decrements_err.append(cur_balmer_decrement_err)
        balmer_decrements.append(cur_balmer_decrement)

        vel_bin_centres.append(vel_centre)

    return np.array(balmer_decrements), np.array(balmer_decrements_err), np.array(vel_bin_centres)

# def get_bd_comparison_info(
#     lam: np.ndarray,
#     sfd_diff: np.ndarray,
#     sfd_diff_err: np.ndarray,
#     num_bins_bounds: tuple[int, int],
#     num_gaussians_bounds: tuple[int, int],
#     vel_calculation_width: float,
#     vel_plot_width: float,
#     n_mc_trials: int
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

#     if num_bins_bounds[0] == 1:
#         num_bins_range = range(2, num_bins_bounds[1] + 1)
#     else:
#         num_bins_range = range(num_bins_bounds[0], num_bins_bounds[1] + 1)

#     num_gaussians_range = range(num_gaussians_bounds[0], num_gaussians_bounds[1] + 1)

#     print(f"num_bins_range: {num_bins_range}")
#     print(f"num_gaussians_range: {num_gaussians_range}")

#     balmer_decrements_1_bin = []
#     balmer_decrements_1_bin_err = []
#     balmer_decrements_many_bins = []
#     balmer_decrements_many_bins_err = []
#     vel_bin_centres_all = []

#     for num_gaussians in num_gaussians_range:
#         balmer_decrement_1_bin_arr, balmer_decrement_1_bin_err_arr, _ = calculate_balmer_decrement(
#             lam, sfd_diff, sfd_diff_err,
#             num_bins=1,
#             num_gaussians=num_gaussians,
#             n_mc_trials=n_mc_trials,
#             vel_calculation_width=vel_calculation_width,
#             vel_plot_width=vel_plot_width,
#             plot_curves=False
#         )
#         balmer_decrements_1_bin.append(balmer_decrement_1_bin_arr[0])
#         balmer_decrements_1_bin_err.append(balmer_decrement_1_bin_err_arr[0])
        
#         one_gauss_results = []
#         for num_bins in num_bins_range:
#             balmer_decrement_many_bins, balmer_decrement_many_bins_err, vel_bin_centres = calculate_balmer_decrement(
#                 lam, sfd_diff, sfd_diff_err,
#                 num_bins=num_bins,
#                 num_gaussians=num_gaussians,
#                 n_mc_trials=n_mc_trials,
#                 vel_calculation_width=vel_calculation_width,
#                 vel_plot_width=vel_plot_width
#             )
#             one_gauss_results.append({
#                 'bd': balmer_decrement_many_bins,
#                 'bd_err': balmer_decrement_many_bins_err,
#                 'vel_centres': vel_bin_centres,
#                 'num_bins': num_bins,
#                 'num_gaussians': num_gaussians
#             })
#             #TODO: add dimension rather than just appending all into one flat list
#             # one_gauss_many_bins_bd.append(balmer_decrement_many_bins)
#             # one_gauss_many_bins_bd_err.append(balmer_decrement_many_bins_err)
#             # one_gauss_many_bins_vel_centres.append(vel_bin_centres)
#         balmer_decrements_many_bins.append(one_gauss_many_bins_bd)
#         balmer_decrements_many_bins_err.append(one_gauss_many_bins_bd_err)
#         vel_bin_centres_all.append(one_gauss_many_bins_vel_centres)
        
#     return (
#         (balmer_decrements_1_bin, balmer_decrements_1_bin_err),
#         (balmer_decrements_many_bins.T, balmer_decrements_many_bins_err.T),
#         vel_bin_centres_all.T
#     )

#     return (
#         np.array(balmer_decrements_1_bin),
#         np.array(balmer_decrements_many_bins),
#         np.array(vel_bin_centres_all)
#     )

def get_bd_comparison_info(
    lam: np.ndarray,
    sfd_diff: np.ndarray,
    sfd_diff_err: np.ndarray,
    num_bins_bounds: tuple[int, int],
    num_gaussians_bounds: tuple[int, int],
    vel_calculation_width: float,
    vel_plot_width: float,
    n_mc_trials: int
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
        
        for num_bins in num_bins_list:
            bd, bd_err, vel_centres = calculate_balmer_decrement(
                lam, sfd_diff, sfd_diff_err,
                num_bins=num_bins,
                num_gaussians=num_gaussians,
                n_mc_trials=n_mc_trials,
                vel_calculation_width=vel_calculation_width,
                vel_plot_width=vel_plot_width,
                plot_curves=False
            )
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
