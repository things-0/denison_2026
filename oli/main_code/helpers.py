import numpy as np
import os
import ast
import warnings
from typing import Any



from . import constants as const

# def get_res_data(
#     fwhm_spec_res: np.ndarray | None = None, # wresl
#     wdisp: np.ndarray | None = None,
#     resolving_power: np.ndarray | float | None = None, # RES_15_BLUE or RES_15_RED
#     sigma: np.ndarray | float | None = None,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     arr_len = 1
#     none_count = 0
#     for data in [fwhm_spec_res, wdisp, resolving_power, sigma]:
#         if data is None:
#             none_count += 1
#         elif isinstance(data, np.ndarray):
#             if arr_len != 1 and len(data) != arr_len:
#                 raise ValueError("all arrays must have the same length")
#     if none_count in [0, 4]:
#         raise ValueError("Must have 1-3 None arguments")
#     if isinstance(resolving_power, float):
#         resolving_power = np.full(arr_len, resolving_power)
#     if isinstance(sigma, float):
#         sigma = np.full(arr_len, sigma)

#     if fwhm_spec_res is not None:
#         return fwhm_spec_res, fwhm_spec_res, resolving_power
#     elif wdisp is not None:
#         return wdisp, wdisp, resolving_power
#     elif resolving_power is not None:
#         return resolving_power, resolving_power, resolving_power
#     else:
#         raise ValueError("No resolution data provided")

def remove_or_replace_bad_values(
    lam: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    fwhm_per_pix: np.ndarray,
    lam_bounds: tuple[float, float] | None,
    rm_or_replace_outside_lam_bounds: bool | float = True,
    rm_or_replace_other_bad_values: bool | float = np.nan,
) -> dict[str, np.ndarray]:
    """

    Parameters
    ----------
    lam: np.ndarray
        Wavelength (Å).
    flux: np.ndarray
        Flux.
    err: np.ndarray
        Flux error.
    fwhm_per_pix: np.ndarray
        FWHM per pixel.
    lam_bounds: tuple[float, float] | None
        Wavelength bounds.
    rm_or_replace_outside_lam_bounds: bool | float = True
        What to do with values outside the wavelength bounds. True to remove, float to
        replace with, or False to leave as is. Note: using a float will change the lam
        values.
    rm_or_replace_other_bad_values: bool | float = np.nan
        What to do with other bad values. True to remove, float to replace with, or False
        to leave as is. Note: using a float will not change any of the lam values.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary with the keys "lam", "flux", "flux_err", "fwhm_per_pix", "good_mask",
        and the corresponding removed/replaced values.
    """

    new_lam = lam.copy()
    new_flux = flux.copy()
    new_err = err.copy()
    new_fwhm_per_pix = fwhm_per_pix.copy()

    not_in_lam_bounds = (lam < lam_bounds[0]) | (lam > lam_bounds[1])
    if not isinstance(rm_or_replace_outside_lam_bounds, bool):
        if not rm_or_replace_outside_lam_bounds is np.nan:
            warn_msg = (
                f"rm_or_replace_outside_lam_bounds is a float but not nan. "
                f"This will replace all outside-of-bounds values (including lambdas) "
                f"with {rm_or_replace_outside_lam_bounds}.")
            warnings.warn(warn_msg)
        new_lam[not_in_lam_bounds] = rm_or_replace_outside_lam_bounds
        new_flux[not_in_lam_bounds] = rm_or_replace_outside_lam_bounds
        new_err[not_in_lam_bounds] = rm_or_replace_outside_lam_bounds
        new_fwhm_per_pix[not_in_lam_bounds] = rm_or_replace_outside_lam_bounds
        lam_bounds_mask = ~not_in_lam_bounds
    elif rm_or_replace_outside_lam_bounds == True:
        new_lam = new_lam[~not_in_lam_bounds]
        new_flux = new_flux[~not_in_lam_bounds]
        new_err = new_err[~not_in_lam_bounds]
        new_fwhm_per_pix = new_fwhm_per_pix[~not_in_lam_bounds]
        # bad values have been removed, so all values are good
        lam_bounds_mask = np.ones_like(new_lam, dtype=bool)
    else:
        lam_bounds_mask = ~not_in_lam_bounds

    bad_mask = (
        ~np.isfinite(new_flux) | ~np.isfinite(new_err) |
        (new_flux > const.MAX_FLUX) | (new_flux < const.MIN_FLUX) |
        (new_err > const.MAX_FLUX) | (new_err <= 0)
    )
    if not isinstance(rm_or_replace_other_bad_values, bool):
        # don't change lam values for other bad values
        new_flux[bad_mask] = rm_or_replace_other_bad_values
        new_err[bad_mask] = rm_or_replace_other_bad_values
        new_fwhm_per_pix[bad_mask] = rm_or_replace_other_bad_values
        good_mask = lam_bounds_mask & ~bad_mask
    elif rm_or_replace_other_bad_values == True:
        new_lam = new_lam[~bad_mask]
        new_flux = new_flux[~bad_mask]
        new_err = new_err[~bad_mask]
        new_fwhm_per_pix = new_fwhm_per_pix[~bad_mask]
        good_mask = lam_bounds_mask[~bad_mask]
    else:
        good_mask = lam_bounds_mask & ~bad_mask
    return {
        "lam": new_lam,
        "flux": new_flux,
        "flux_err": new_err,
        "fwhm_per_pix": new_fwhm_per_pix,
        "good_mask": good_mask
    }

def custom_showwarning(msg, category, filename, lineno, *args, **kwargs):
    """
    Custom warning handler.
    """
    print(f"WARNING ({filename}:{lineno}): {msg}", flush=True)

def get_first_valid_flux(flux: np.ndarray):
    """
    Get the first finite flux value.
    """
    return flux[np.where(np.isfinite(flux))[0][0]]

def convert_lam_to_vel(
    lam: np.ndarray | float,
    lam_centre: float
) -> np.ndarray | float:
    """
    Convert wavelength (Å) to velocity (km/s). 

    Parameters
    ----------
    lam: np.ndarray | float
        Wavelength (Å) to convert to velocity (km/s).
    lam_centre: float
        Centre wavelength (Å) of the line (0 km/s).

    Returns
    -------
    np.ndarray | float
        Velocity (km/s).
    """
    # v = c * Δλ / λ_cent
    return (lam - lam_centre) * const.C_KM_S / lam_centre

def convert_vel_to_lam(
    vel: np.ndarray | float,
    lam_centre: float
) -> np.ndarray | float:
    """
    Convert velocity (km/s) to wavelength (Å).

    Parameters
    ----------
    vel: np.ndarray | float
        Velocity (km/s) to convert to wavelength (Å).
    lam_centre: float
        Centre wavelength (Å) of the line (0 km/s).

    Returns
    -------
    np.ndarray | float
        Wavelength (Å).
    """
    # λ = λ_cent * (1 + v / c)
    return lam_centre * (1 + vel / const.C_KM_S)

def get_lam_bounds(
    lam_centre: float, width: float,
    width_is_vel: bool = False,
) -> tuple[float, float]:
    """
    Get the wavelength bounds for a given width and centre wavelength.

    Parameters
    ----------
    lam_centre: float
        Centre wavelength (Å) of the line (0 km/s).
    width: float
        Desired width of the lam bounds.
    width_is_vel: bool = False
        Whether the width is in velocity (km/s) or wavelength (Å).

    Returns
    -------
    tuple[float, float]
        The left and right wavelength bounds (Å).
    """
    if width_is_vel:
        left = convert_vel_to_lam(
            vel=-width / 2, lam_centre=lam_centre,
        )
        right = convert_vel_to_lam(
            vel=width / 2, lam_centre=lam_centre,
        )
    else:
        left = lam_centre - width / 2
        right = lam_centre + width / 2
    return left, right

def bin_data_by_median(
    x: np.ndarray, y: np.ndarray, bin_width: float,
    y_errs: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    step_size = np.median(np.diff(x))

    points_per_bin = int(bin_width / step_size)

    n_complete_bins = len(y) // points_per_bin
    n_points_to_keep = n_complete_bins * points_per_bin

    x_trimmed = x[:n_points_to_keep]
    y_trimmed = y[:n_points_to_keep]

    x_2d = x_trimmed.reshape(n_complete_bins, points_per_bin)
    y_2d = y_trimmed.reshape(n_complete_bins, points_per_bin)

    x_binned = np.median(x_2d, axis=1)
    y_binned = np.median(y_2d, axis=1)

    if y_errs is not None:
        y_errs_trimmed = y_errs[:n_points_to_keep]
        y_errs_2d = y_errs_trimmed.reshape(n_complete_bins, points_per_bin)
        y_errs_binned = np.median(y_errs_2d, axis=1)
    else:
        y_errs_binned = None

    return x_binned, y_binned, y_errs_binned

def convert_to_vel_data(
    lam: np.ndarray,
    flux: np.ndarray | None,
    flux_err: np.ndarray | None,
    lam_centres: float | list[float],
    vel_width: float | None
) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None, list[np.ndarray] | None]:
    """
    Gets the lams, fluxes and flux errors in velocity space for a given list of lam centres and widths.
    
    Parameters
    ----------
    lam: np.ndarray
        Wavelength (Å).
    flux: np.ndarray | None
        Flux.
    flux_err: np.ndarray | None
        Flux error.
    lam_centres: float | list[float]
        Centre wavelengths (Å) of the lines (0 km/s).
    vel_width: float | None
        Width of the velocity space (km/s).

    Returns
    -------
    tuple[list[np.ndarray] | None, list[np.ndarray] | None, list[np.ndarray] | None]
        The trimmed lams, fluxes and flux errors in velocity space, each with width `vel_width`.
    """

    if flux is None:
        return None, None, None
    
    trimmed_vels = []
    trimmed_fluxes = []
    trimmed_flux_errs = [] if flux_err is not None else None
    
    if not isinstance(lam_centres, list):
        lam_centres = [lam_centres]

    for lam_centre in lam_centres:
        vel = convert_lam_to_vel(lam, lam_centre=lam_centre)
        vel_width_mask = (vel >= -vel_width / 2) & (vel <= vel_width / 2)
        
        trimmed_vel = vel[vel_width_mask]
        trimmed_flux = flux[vel_width_mask]
        
        trimmed_vels.append(trimmed_vel)
        trimmed_fluxes.append(trimmed_flux)

        if flux_err is not None:
            trimmed_flux_err = flux_err[vel_width_mask]
            trimmed_flux_errs.append(trimmed_flux_err)
    

    # check if all trimmed_vels are the same
    # if not all(np.all(trimmed_vel == trimmed_vels[0]) for trimmed_vel in trimmed_vels):
    #     print(trimmed_vels)
    #     raise ValueError("All trimmed_vels are not the same")
    
    return trimmed_vels, trimmed_fluxes, trimmed_flux_errs

def get_radius_from_med(
    lam: np.ndarray,
    flux: np.ndarray,
    scale_factor: float = 100
) -> float:
    """
    Finds an appropriate y distance from the median of the flux by
    considering the difference between binned data points. Note:
    the median of the flux is not actually calculated in this function.

    Parameters
    ----------
    lam: np.ndarray
        Wavelength (Å).
    flux: np.ndarray
        Flux.
    scale_factor: float = 100
        Scale factor to apply to the difference between binned data points.

    Returns
    -------
    float
        The distance from the median of the flux.
    """
    _, flux_binned, _ = bin_data_by_median(lam, flux, 20)
    return np.nanmedian(np.abs(np.diff(flux_binned))) * scale_factor

def update_min_med_max_fluxes(
    flux: np.ndarray,
    median_fluxes: list[float],
    min_flux: float,
    max_flux: float,
    reasonable_min_flux: float,
    reasonable_max_flux: float,
    suggested_lower_bounds: list[tuple[float, float]],
    suggested_upper_bounds: list[tuple[float, float]],
    radius_from_med: int
) -> tuple[float, float, float, float]:
    """
    Updates the min, max, reasonable min and reasonable max fluxes, and appends
    to the suggested lower and upper bound lists. Fluxe bounds are considered
    reasonable if the flux is within the `radius_from_med` radius from the median.

    Parameters
    ----------
    flux: np.ndarray
        Flux.
    median_fluxes: list[float]
        List of median fluxes.
    min_flux: float
        Minimum flux.
    max_flux: float
        Maximum flux.
    reasonable_min_flux: float
        Reasonable minimum flux.
    reasonable_max_flux: float
        Reasonable maximum flux.
    suggested_lower_bounds: list[tuple[float, float]]
        Suggested lower bounds.
    suggested_upper_bounds: list[tuple[float, float]]
        Suggested upper bounds.
    radius_from_med: int
        Radius from the median.

    Returns
    -------
    tuple[float, float, float, float]
        The new min, max, reasonable min and reasonable max fluxes.
    """

    new_max = np.max((max_flux, np.nanmax(flux)))
    new_min = np.min((min_flux, np.nanmin(flux)))
    med = np.nanmedian(flux)
    median_fluxes.append(med)

    suggested_lower_bounds.append(med - radius_from_med)
    suggested_upper_bounds.append(med + radius_from_med)  

    if np.nanmin(flux) >= med - radius_from_med:
        new_reasonable_min_flux = np.min((reasonable_min_flux, np.nanmin(flux)))
    else:
        # min of flux is too small. don't update reasonable min
        new_reasonable_min_flux = reasonable_min_flux
    if np.nanmax(flux) <= med + radius_from_med:
        new_reasonable_max_flux = np.max((reasonable_max_flux, np.nanmax(flux)))
    else:
        # max of flux is too large. don't update reasonable max
        new_reasonable_max_flux = reasonable_max_flux

    suggested_lower_bounds.append(med - radius_from_med)
    suggested_upper_bounds.append(med + radius_from_med)
    return new_min, new_max, new_reasonable_min_flux, new_reasonable_max_flux


def get_better_y_bounds(
    y_bounds: tuple[float, float] | None,
    lams: list[np.ndarray | tuple[np.ndarray, np.ndarray]],
    fluxes: list[np.ndarray | tuple[np.ndarray, np.ndarray]],
    calculate_radius_from_med: bool = True,
    radius_from_med: int = 40,
    perc_rad_from_reasonable_bounds: float = 7.0
) -> tuple[float | None, float | None]:
    """
    Gets better y bounds for a given list of lams and fluxes to plot.

    Parameters
    ----------
    y_bounds: tuple[float, float] | None
        The current y bounds.
    lams: list[np.ndarray | tuple[np.ndarray, np.ndarray]]
        The lams to plot.
    fluxes: list[np.ndarray | tuple[np.ndarray, np.ndarray]]
        The fluxes to plot.
    calculate_radius_from_med: bool = True
        Whether to calculate the radius from the median. `radius_from_med` is used if False.
    radius_from_med: int = 40
        The radius from the median.
    perc_rad_from_reasonable_bounds: float = 7.0
        The percentage of the reasonable bounds to add to the y bounds.

    Returns
    -------
    tuple[float | None, float | None]
        The better y bounds.
    """

    if y_bounds is not None:
        return y_bounds

    median_fluxes = []
    max_flux = -np.inf
    min_flux = np.inf
    reasonable_max_flux = -np.inf
    reasonable_min_flux = np.inf
    suggested_lower_bounds = []
    suggested_upper_bounds = []
    for lam, flux in zip(lams, fluxes):
        if isinstance(flux, tuple):
            flux_blue, flux_red = flux
            lam_blue, lam_red = lam

            if calculate_radius_from_med:
                radius_blue = get_radius_from_med(lam_blue, flux_blue)
                radius_red = get_radius_from_med(lam_red, flux_red)
            else:
                radius_blue = radius_from_med
                radius_red = radius_from_med

            min_flux, max_flux, reasonable_min_flux, reasonable_max_flux = update_min_med_max_fluxes(
                flux_blue, median_fluxes, min_flux, max_flux,
                reasonable_min_flux, reasonable_max_flux,
                suggested_lower_bounds, suggested_upper_bounds,
                radius_blue
            )
            min_flux, max_flux, reasonable_min_flux, reasonable_max_flux = update_min_med_max_fluxes(
                flux_red, median_fluxes, min_flux, max_flux,
                reasonable_min_flux, reasonable_max_flux,
                suggested_lower_bounds, suggested_upper_bounds,
                radius_red
            )

        else:
            if calculate_radius_from_med:
                radius = get_radius_from_med(lam, flux)
            else:
                radius = radius_from_med
            min_flux, max_flux, reasonable_min_flux, reasonable_max_flux = update_min_med_max_fluxes(
                flux, median_fluxes, min_flux, max_flux,
                reasonable_min_flux, reasonable_max_flux,
                suggested_lower_bounds, suggested_upper_bounds,
                radius
            )
    
    y_lower = None
    y_upper = None

    if max_flux > np.max(suggested_upper_bounds):
        # max of flux is too large. use reasonable max (plus a little bit) if available, otherwise use suggested upper bound
        if np.isfinite(reasonable_max_flux):
            y_upper = reasonable_max_flux * (1 + perc_rad_from_reasonable_bounds / 100)
        else:
            y_upper = np.max(suggested_upper_bounds)
    if min_flux < np.min(suggested_lower_bounds):
        # min of flux is too small. use reasonable min (minus a little bit) if available, otherwise use suggested lower bound
        if np.isfinite(reasonable_min_flux) and np.isfinite(reasonable_max_flux):
            y_lower = reasonable_min_flux - reasonable_max_flux * (perc_rad_from_reasonable_bounds / 100)
        else:
            y_lower = np.min(suggested_lower_bounds)
    # if y_bounds are reasonable, return None (automatic scaling by matplotlib)
    return y_lower, y_upper

def convert_flux_to_mJy(
    flux: float,
    col_band_eff_width: float,
    col_band_eff_lam: float
) -> float:
    """
    Convert integrated line flux to equivalent broadband flux density in mJy.
    
    Parameters:
    -----------
    flux : float
        Integrated line flux in units of 10^-17 erg s^-1 cm^-2
        (i.e., the numerical value from integrate_flux)
    col_band_eff_width : float
        Effective width of the photometric band in Angstroms
    col_band_eff_lam : float
        Effective central wavelength of the photometric band in Angstroms
    
    Returns:
    --------
    float
        Flux density in mJy
    
    Notes:
    ------
    The conversion assumes the line flux is spread uniformly over the
    frequency width corresponding to the photometric band.
    """
    
    # Calculate frequency width in Hz
    # Using c_ang_s (speed of light in Å/s) for consistent units
    freq_eff_width = const.C_ANG_S * col_band_eff_width / (col_band_eff_lam**2)  # Hz
    
    # Calculate flux density
    flux_density = flux / freq_eff_width  # erg s^-1 cm^-2 Hz^-1
    
    # Convert to mJy 
    # 1 Jy = 10^-23 erg s^-1 cm^-2 Hz^-1
    # 10^-17 erg s^-1 cm^-2 = 10^9 mJy
    flux_density_mjy = flux_density * 1e9
    
    return flux_density_mjy

def get_lam_mask( # was get_vel_lam_mask
    lam: np.ndarray,
    width: float,
    lam_centre: float,
    width_is_vel: bool = True,
) -> np.ndarray:
    """
    Get a mask for the wavelength array based on the width and centre wavelength.

    Parameters
    ----------
    lam: np.ndarray
        Wavelength (Å).
    width: float
        Width of the True mask.
    lam_centre: float
        Centre wavelength (Å).
    width_is_vel: bool = True
        Whether the width is in velocity (km/s) or wavelength (Å).

    Returns
    -------
    np.ndarray
        A mask for the wavelength array.
    """
    lam_bounds = get_lam_bounds(lam_centre, width, width_is_vel=width_is_vel)
    return (lam >= lam_bounds[0]) & (lam <= lam_bounds[1])

def get_masked_diffs(
    x: np.ndarray,
    mask: np.ndarray | None,
    reduce_endpoint_weights: bool = True
) -> np.ndarray:
    """
    Get the average differences between adjacent elements in the array.

    Parameters
    ----------
    x: np.ndarray
        Array to get the average differences between adjacent elements of.
    mask: np.ndarray | None
        Mask to apply to the array.
    reduce_endpoint_weights: bool = True
        Whether to reduce the weights of the endpoints. Weights of endpoints
        are reduced by a factor of 2 if True (since only 1 adjacent element).

    Returns
    -------
    np.ndarray
        The average differences between adjacent elements of the array.
    """
    diffs = np.diff(x)
    av_diffs = np.zeros_like(x)
    av_diffs[1:] += diffs / 2
    av_diffs[:-1] += diffs / 2

    if not reduce_endpoint_weights:
        av_diffs[0] = av_diffs[0] * 2
        av_diffs[-1] = av_diffs[-1] * 2

    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    return av_diffs[mask]

def get_default_bounds(
    x: np.ndarray,
    y: np.ndarray,
    num_of_gaussians: int
) -> tuple[list[float], list[float]]:
    """
    Get the default bounds on parameters for the Gaussian fitting.

    Parameters
    ----------
    x: np.ndarray
        Wavelength (Å).
    y: np.ndarray
        Flux.
    num_of_gaussians: int
        Number of Gaussians to fit.

    Returns
    -------
    tuple[list[float], list[float]]
        The default bounds on parameters.
        The first list is the lower bounds, and the second list is the upper bounds.
        In each list, the first `num_of_gaussians` elements are the height bounds,
        the next `num_of_gaussians` elements are the mu bounds, and the last
        `num_of_gaussians` elements are the sigma bounds. i.e.
        `[h_min * n, μ_min * n, σ_min * n], [h_max * n, μ_max * n, σ_max * n]`
    """
    height_min = const.HEIGHT_MIN
    height_max = 2 * np.max(y)
    x_range = x[-1] - x[0]
    mu_min = x[0] + x_range * const.MIN_MU
    mu_max = x[-1] - x_range * const.MIN_MU
    sigma_min = const.PEAK_MIN_RANGE * x_range / const.SIGMA_TO_FWHM
    sigma_max = x_range / const.SIGMA_TO_FWHM

    lower_bounds = (
        [height_min] * num_of_gaussians +
        [mu_min] * num_of_gaussians +
        [sigma_min] * num_of_gaussians
    )
    upper_bounds = (
        [height_max] * num_of_gaussians +
        [mu_max] * num_of_gaussians +
        [sigma_max] * num_of_gaussians
    )
    return lower_bounds, upper_bounds

def make_row(values, widths, alignments):
    """
    Make a row of a table in :func:`pretty_print_flux_comparison`.

    Parameters
    ----------
    values: list[str]
        The values to make the row of.
    widths: list[int]
        The widths of the columns.
    alignments: list[str]
        The alignments of the columns.

    Returns
    -------
    str
        The row of the table as a string.
    """
    parts = []
    for val, w, align in zip(values, widths, alignments):
        if align == "center":
            parts.append(f"{val:^{w}}")
        elif align == "right":
            parts.append(f"{val:>{w}}")
        else:  # left
            parts.append(f"{val:<{w}}")
    return " | ".join(parts)

def pretty_print_flux_comparison(
    flux_alpha_21: float,
    flux_alpha_21_err: float,
    num_gaussians_alpha_21: int,
    flux_alpha_22: float,
    flux_alpha_22_err: float,
    num_gaussians_alpha_22: int,
    flux_beta_21: float,
    flux_beta_21_err: float,
    num_gaussians_beta_21: int,
    flux_beta_22: float,
    flux_beta_22_err: float,
    num_gaussians_beta_22: int
) -> None:
    """
    Pretty print the flux comparison of the photometric flux against the integrated spectroscopic flux for each survey.

    Parameters
    ----------
    flux_alpha_21: float
        The integrated spectroscopic flux for the Hα line in 2021.
    flux_alpha_21_err: float
        The error on the integrated spectroscopic flux for the Hα line in 2021.
    num_gaussians_alpha_21: int
        The number of Gaussians used to integrate the Hα line in 2021.
    flux_alpha_22: float
        The integrated spectroscopic flux for the Hα line in 2022.
    flux_alpha_22_err: float
        The error on the integrated spectroscopic flux for the Hα line in 2022.
    num_gaussians_alpha_22: int
        The number of Gaussians used to integrate the Hα line in 2022.
    flux_beta_21: float
        The integrated spectroscopic flux for the Hβ line in 2021.
    flux_beta_21_err: float
        The error on the integrated spectroscopic flux for the Hβ line in 2021.
    num_gaussians_beta_21: int
        The number of Gaussians used to integrate the Hβ line in 2022.
    flux_beta_22: float
        The integrated spectroscopic flux for the Hβ line in 2022.
    flux_beta_22_err: float
        The error on the integrated spectroscopic flux for the Hβ line in 2022.
    num_gaussians_beta_22: int
        The number of Gaussians used to integrate the Hβ line in 2022.
    """

    flux_alpha_asassn_g_21_mjy = convert_flux_to_mJy(flux_beta_21, const.ASASSN_G_BAND_WIDTH, const.ASASSN_G_BAND_LAM)
    flux_alpha_asassn_g_21_mjy_err = convert_flux_to_mJy(flux_beta_21_err, const.ASASSN_G_BAND_WIDTH, const.ASASSN_G_BAND_LAM)
    flux_alpha_atlas_o_21_mjy = convert_flux_to_mJy(flux_alpha_21, const.ATLAS_O_BAND_WIDTH, const.ATLAS_O_BAND_LAM)
    flux_alpha_atlas_o_21_mjy_err = convert_flux_to_mJy(flux_alpha_21_err, const.ATLAS_O_BAND_WIDTH, const.ATLAS_O_BAND_LAM)
    flux_alpha_ztf_r_21_mjy = convert_flux_to_mJy(flux_alpha_21, const.ZTF_R_BAND_WIDTH, const.ZTF_R_BAND_LAM)
    flux_alpha_ztf_r_21_mjy_err = convert_flux_to_mJy(flux_alpha_21_err, const.ZTF_R_BAND_WIDTH, const.ZTF_R_BAND_LAM)
    flux_alpha_ztf_g_21_mjy = convert_flux_to_mJy(flux_alpha_21, const.ZTF_G_BAND_WIDTH, const.ZTF_G_BAND_LAM)
    flux_alpha_ztf_g_21_mjy_err = convert_flux_to_mJy(flux_alpha_21_err, const.ZTF_G_BAND_WIDTH, const.ZTF_G_BAND_LAM)
    flux_alpha_ztf_i_21_mjy = convert_flux_to_mJy(flux_alpha_21, const.ZTF_I_BAND_WIDTH, const.ZTF_I_BAND_LAM)
    flux_alpha_ztf_i_21_mjy_err = convert_flux_to_mJy(flux_alpha_21_err, const.ZTF_I_BAND_WIDTH, const.ZTF_I_BAND_LAM)

    flux_alpha_asassn_g_22_mjy = convert_flux_to_mJy(flux_beta_22, const.ASASSN_G_BAND_WIDTH, const.ASASSN_G_BAND_LAM)
    flux_alpha_asassn_g_22_mjy_err = convert_flux_to_mJy(flux_beta_22_err, const.ASASSN_G_BAND_WIDTH, const.ASASSN_G_BAND_LAM)
    flux_alpha_atlas_o_22_mjy = convert_flux_to_mJy(flux_alpha_22, const.ATLAS_O_BAND_WIDTH, const.ATLAS_O_BAND_LAM)
    flux_alpha_atlas_o_22_mjy_err = convert_flux_to_mJy(flux_alpha_22_err, const.ATLAS_O_BAND_WIDTH, const.ATLAS_O_BAND_LAM)
    flux_alpha_ztf_r_22_mjy = convert_flux_to_mJy(flux_alpha_22, const.ZTF_R_BAND_WIDTH, const.ZTF_R_BAND_LAM)
    flux_alpha_ztf_r_22_mjy_err = convert_flux_to_mJy(flux_alpha_22_err, const.ZTF_R_BAND_WIDTH, const.ZTF_R_BAND_LAM)
    flux_alpha_ztf_g_22_mjy = convert_flux_to_mJy(flux_alpha_22, const.ZTF_G_BAND_WIDTH, const.ZTF_G_BAND_LAM)
    flux_alpha_ztf_g_22_mjy_err = convert_flux_to_mJy(flux_alpha_22_err, const.ZTF_G_BAND_WIDTH, const.ZTF_G_BAND_LAM)
    flux_alpha_ztf_i_22_mjy = convert_flux_to_mJy(flux_alpha_22, const.ZTF_I_BAND_WIDTH, const.ZTF_I_BAND_LAM)
    flux_alpha_ztf_i_22_mjy_err = convert_flux_to_mJy(flux_alpha_22_err, const.ZTF_I_BAND_WIDTH, const.ZTF_I_BAND_LAM)

    survey_names = (
        "ASASSN g band 2021 (Hβ)",
        "ASASSN g band 2022 (Hβ)",
        "Atlas o band 2021 (Hα)",
        "Atlas o band 2022 (Hα)",
        "ZTF r band 2021 (Hα)",
        "ZTF r band 2022 (Hα)",
        "ZTF g band 2021 (Hβ)",
        "ZTF g band 2022 (Hβ)",
        "ZTF i band 2021 (Hα)",
        "ZTF i band 2022 (Hα)"
    )
    int_flux_vals = [
        [flux_alpha_asassn_g_21_mjy, flux_alpha_asassn_g_21_mjy_err],
        [flux_alpha_asassn_g_22_mjy, flux_alpha_asassn_g_22_mjy_err],
        [flux_alpha_atlas_o_21_mjy, flux_alpha_atlas_o_21_mjy_err],
        [flux_alpha_atlas_o_22_mjy, flux_alpha_atlas_o_22_mjy_err],
        [flux_alpha_ztf_r_21_mjy, flux_alpha_ztf_r_21_mjy_err],
        [flux_alpha_ztf_r_22_mjy, flux_alpha_ztf_r_22_mjy_err],
        [flux_alpha_ztf_g_21_mjy, flux_alpha_ztf_g_21_mjy_err],
        [flux_alpha_ztf_g_22_mjy, flux_alpha_ztf_g_22_mjy_err],
        [flux_alpha_ztf_i_21_mjy, flux_alpha_ztf_i_21_mjy_err],
        [flux_alpha_ztf_i_22_mjy, flux_alpha_ztf_i_22_mjy_err]
    ]
    num_gaussians_vals = (
        num_gaussians_beta_21,      # ASASSN g band 2021
        num_gaussians_beta_22,      # ASASSN g band 2022
        num_gaussians_alpha_21,     # Atlas o band 2021
        num_gaussians_alpha_22,     # Atlas o band 2022
        num_gaussians_alpha_21,     # ZTF r band 2021
        num_gaussians_alpha_22,     # ZTF r band 2022
        num_gaussians_beta_21,      # ZTF g band 2021
        num_gaussians_beta_22,      # ZTF g band 2022
        num_gaussians_alpha_21,     # ZTF i band 2021
        num_gaussians_alpha_22      # ZTF i band 2022
    )
    photometric_flux_vals = [
        const.ASASSN_G_FLUX_21,
        const.ASASSN_G_FLUX_22,
        const.ATLAS_O_FLUX_21, 
        const.ATLAS_O_FLUX_22,
        const.ZTF_R_FLUX_21,
        const.ZTF_R_FLUX_22,
        const.ZTF_G_FLUX_21,
        const.ZTF_G_FLUX_22,
        const.ZTF_I_FLUX_21,
        const.ZTF_I_FLUX_22
    ]

    int_flux_vals_micro_jy = np.array(int_flux_vals) * 1e3
    photometric_flux_vals_micro_jy = np.array(photometric_flux_vals) * 1e3
 
    # Column headers (in desired order)
    headers = [
        "Survey",
        "Photometric Flux (μJy)",
        "Integrated Spectroscopic Flux (μJy)^",
        "Number of Gaussians used to integrate"
    ]

    # Build rows data (in same order as headers)
    rows = []
    for i, name in enumerate(survey_names):
        flux_val = int_flux_vals_micro_jy[i][0]
        flux_err = int_flux_vals_micro_jy[i][1]
        n_gauss = num_gaussians_vals[i]
        phot_flux = photometric_flux_vals_micro_jy[i]

        rows.append([
            name,
            f"{phot_flux: .4f}",                    # Space for sign alignment
            f"({flux_val:.3f} ± {flux_err:.3f})",
            str(n_gauss)
        ])

    # Calculate column widths based on headers and data
    col_widths = []
    for col_idx in range(len(headers)):
        header_len = len(headers[col_idx])
        data_lens = [len(row[col_idx]) for row in rows]
        col_widths.append(max(header_len, *data_lens))

    # Alignments for each column
    alignments = ["left", "left", "left", "center"]

    table_width = sum(col_widths) + 3 * len(col_widths) - 1  # " | " between columns

    # Print table
    print("=" * table_width)
    print(make_row(headers, col_widths, alignments))
    print("-" * table_width)
    for row in rows:
        print(make_row(row, col_widths, alignments))
    print("=" * table_width)
    print("^ assuming top-hat filter response")

def get_fwhm(
    x: np.ndarray, y_gaussian: np.ndarray,
    get_vel: bool = True,
    lam_centre: float | None = None
) -> float:
    """
    Get the full width at half maximum (FWHM) of a Gaussian-like array.

    Parameters
    ----------
    x: np.ndarray
        Wavelength (Å).
    y_gaussian: np.ndarray
        Gaussian-like array.
    get_vel: bool = True
        Whether to get the FWHM in velocity (km/s) or wavelength (Å).
    """
    half_max = np.nanmax(y_gaussian) / 2
    above_half = np.where(y_gaussian >= half_max)[0]
    if len(above_half) < 2:
        raise ValueError("No FWHM found - not enough points above half max")
    lam_left = x[above_half[0]]
    lam_right = x[above_half[-1]]
    fwhm_ang = lam_right - lam_left
    if get_vel is False:
        return fwhm_ang
    if lam_centre is None:
        lam_centre = (lam_right + lam_left) / 2
    #TD: remove testing
    # lam_centre_option_1 = (x[np.argmax(y_gaussian)])
    # lam_centre_option_2 = ((lam_right + lam_left) / 2)
    # lam_centre_option_3 = (x[(above_half[0] + above_half[-1]) // 2])
    # print(f"Hα: {const.H_ALPHA}, Hβ: {const.H_BETA}")
    # print(f"x[argmax(y)]: {lam_centre_option_1}")
    # print(f"average lam of lam_left and lam_right: {lam_centre_option_2}")
    # print(f"average index of lam_left and lam_right: {lam_centre_option_3}")
    #
    vel_left = convert_lam_to_vel(lam_left, lam_centre)
    vel_right = convert_lam_to_vel(lam_right, lam_centre)
    fwhm_vel = vel_right - vel_left
    return fwhm_vel

def compare_yasmeen_results(
    fwhm_alpha_15: tuple[float, float] | None = None,
    fwhm_alpha_21: tuple[float, float] | None = None,
    fwhm_beta_15: tuple[float, float] | None = None,
    fwhm_beta_21: tuple[float, float] | None = None,
    flux_alpha_15: tuple[float, float] | None = None,
    flux_alpha_21: tuple[float, float] | None = None,
    luminosity_alpha_15: tuple[float, float] | None = None,
    luminosity_alpha_21: tuple[float, float] | None = None,
    bd_15: tuple[float, float] | None = None,
    bd_21: tuple[float, float] | None = None,
    bh_mass_15: tuple[float, float] | None = None,
    bh_mass_21: tuple[float, float] | None = None
) -> None:
    """
    Compare the results of the YASMEEN code with the results of the SSP code,
    including the Balmer decrement, BH mass, FWHM, flux, and luminosity of 2015
    and 2021 Hα and Hβ.

    Parameters
    ----------
    fwhm_alpha_15: tuple[float, float] | None = None
        The FWHM of the Hα line in 2015.
    fwhm_alpha_21: tuple[float, float] | None = None
        The FWHM of the Hα line in 2021.
    fwhm_beta_15: tuple[float, float] | None = None
        The FWHM of the Hβ line in 2015.
    fwhm_beta_21: tuple[float, float] | None = None
        The FWHM of the Hβ line in 2021.
    flux_alpha_15: tuple[float, float] | None = None
        The flux of the Hα line in 2015.
    flux_alpha_21: tuple[float, float] | None = None
        The flux of the Hα line in 2021.
    luminosity_alpha_15: tuple[float, float] | None = None
        The luminosity of the Hα line in 2015.
    luminosity_alpha_21: tuple[float, float] | None = None
        The luminosity of the Hα line in 2021.
    bd_15: tuple[float, float] | None = None
        The Balmer decrement of the Hα line in 2015.
    bd_21: tuple[float, float] | None = None
        The Balmer decrement of the Hα line in 2021.
    bh_mass_15: tuple[float, float] | None = None
        The BH mass of the Hα line in 2015.
    bh_mass_21: tuple[float, float] | None = None
        The BH mass of the Hα line in 2021.
    """

    all_results = {
        "FWHM Hα 2015 (km/s)": (fwhm_alpha_15, const.YASMEEN_RESULTS["fwhm_alpha_15"]),
        "FWHM Hα 2021 (km/s)": (fwhm_alpha_21, const.YASMEEN_RESULTS["fwhm_alpha_21"]),
        "FWHM Hβ 2015 (km/s)": (fwhm_beta_15, const.YASMEEN_RESULTS["fwhm_beta_15"]),
        "FWHM Hβ 2021 (km/s)": (fwhm_beta_21, const.YASMEEN_RESULTS["fwhm_beta_21"]),
        "Flux Hα 2015 (1e-17 ergs/s/cm^2)": (flux_alpha_15, const.YASMEEN_RESULTS["flux_alpha_15"]),
        "Flux Hα 2021 (1e-17 ergs/s/cm^2)": (flux_alpha_21, const.YASMEEN_RESULTS["flux_alpha_21"]),
        "Luminosity Hα 2015 (1e40 ergs/s)": (luminosity_alpha_15, const.YASMEEN_RESULTS["luminosity_alpha_15"]),
        "Luminosity Hα 2021 (1e40 ergs/s)": (luminosity_alpha_21, const.YASMEEN_RESULTS["luminosity_alpha_21"]),
        "Balmer decrement Hα 2015": (bd_15, const.YASMEEN_RESULTS["bd_15"]),
        "Balmer decrement Hα 2021": (bd_21, const.YASMEEN_RESULTS["bd_21"]),
        "BH mass 2015 (1e6 M_sun)": (bh_mass_15, const.YASMEEN_RESULTS["bh_mass_15"]),
        "BH mass 2021 (1e6 M_sun)": (bh_mass_21, const.YASMEEN_RESULTS["bh_mass_21"])
    }
    
    col1_width = 35
    col2_width = 20
    col3_width = 20
    table_width = col1_width + col2_width + col3_width + 6

    print("=" * table_width)
    print("".ljust(col1_width), end=" | ")
    print("Most recent results".ljust(col2_width), end=" | ")
    print("SSP results".ljust(col3_width))
    print("-" * table_width)
    for name, (input_val, yasmeen_val) in all_results.items():
        if input_val is None:
            continue
        if input_val[1] is None:
            input_val_err = "?"
        else:
            input_val_err = f"{input_val[1]:.2f}"
        print(f"{name}".ljust(col1_width), end=" | ")
        print(f"{input_val[0]:.2f} ± {input_val_err}".ljust(col2_width), end=" | ")
        print(f"{yasmeen_val[0]:.2f} ± {yasmeen_val[1]:.2f}".ljust(col3_width))
    print("-" * table_width)

def get_new_qso_filename(
    filename: str = "qsopar0.fits",
    start: str = "qsopar",
    end: str = ".fits",
    folder_name: str = "pyqsofit_code/data/"
) -> str:
    """
    Get a new filename for the QSOFIT fits file.

    Parameters
    ----------
    filename: str = "qsopar0.fits"
        The filename of the QSOFIT fits file.
    start: str = "qsopar"
        The start of the filename.
    end: str = ".fits"
        The end of the filename.
    folder_name: str = "pyqsofit_code/data/"
        The folder name of the QSOFIT fits file.

    Returns
    -------
    str
        The new filename for the QSOFIT fits file.
    """
    if not (filename.startswith(start) and filename.endswith(end)):
        raise ValueError(f"Filename must start with '{start}' and end with '{end}'")
    num_str = filename[len(start) : -len(end)]
    if not num_str.isdigit():
        raise ValueError("Filename must contain a number")
    new_num = int(num_str) + 1
    while True:
        if os.path.exists(f"{folder_name}{start}{new_num}{end}"):
            new_num += 1
        else:
            break
    # while os.path.exists(f"{folder_name}{start}{new_num}{end}"): # God knows why this doesn't work
    #     new_num += 1
    return f"{start}{new_num}{end}"

def get_output_pyqsofit_file_name(
    fname: str, # without extension
    folder_name: str = "pyqsofit_code/output/",
) -> str:
    """
    Get the output filename for the QSOFIT code.

    Parameters
    ----------
    fname: str
        The filename of the spectrum data.
    folder_name: str = "pyqsofit_code/output/"
        The folder name of the output files.

    Returns
    -------
    str
        The output filename for the QSOFIT code.
    """
    if fname == const.FNAME_2001:
        output_name = "SDSS_2001"
    elif fname == const.FNAME_2021:
        output_name = "SDSS_2021"
    elif fname == const.FNAME_2022:
        output_name = "SDSS_2022"
    elif fname == const.FNAME_2015_BLUE_3_ARCSEC:
        output_name = "SAMI_BLUE_3_ARCSEC"
    elif fname == const.FNAME_2015_RED_3_ARCSEC:
        output_name = "SAMI_RED_3_ARCSEC"
    elif fname == const.FNAME_2015_BLUE_4_ARCSEC:
        output_name = "SAMI_BLUE_4_ARCSEC"
    elif fname == const.FNAME_2015_RED_4_ARCSEC:
        output_name = "SAMI_RED_4_ARCSEC"
    else:
        raise NotImplementedError(f"Invalid filename: {fname}")
    highest_num = 0
    for name in os.listdir(folder_name):
        if name.startswith(output_name):
            if name.endswith(".fits") or name.endswith(".pdf"):
                num = int(name.split("_v")[-1].split(".")[0])
                if num > highest_num:   
                    highest_num = num   
    return f"{output_name}_v{highest_num + 1}"

def log_kwargs(
    kwargs: dict,
    log_name: str = "pyqsofit_code/log.csv"
) -> None:
    """
    Log the keyword arguments of a PyQSOFit run to a CSV file.

    Parameters
    ----------
    kwargs: dict
        The keyword arguments to log.
    log_name: str = "pyqsofit_code/log.csv"
        The name of the log file.
    """
    with open(log_name, "a") as f:
        two_write = ""
        for val in kwargs.values():
            two_write += f"{val};"
        f.write(two_write.strip(";") + "\n")

def get_kwargs_from_log(
    output_file_name: str,
    log_name: str = "pyqsofit_code/log.csv",
    exclude_log_items: bool = True
) -> dict[str, Any]:
    """
    Get the keyword arguments from the log file of a PyQSOFit run.

    Parameters
    ----------
    output_file_name: str
        The output filename of the PyQSOFit run.
    log_name: str = "pyqsofit_code/log.csv"
        The name of the corresponding log file.
    exclude_log_items: bool = True
        Whether to exclude additional parameters that are shown in the
        log file but are not passed to the actual PyQSOFit call.

    Returns
    -------
    dict[str, Any]
        The keyword arguments from the log file.
    """
    with open(log_name, "r") as f:
        keys = f.readline().strip().split(";")
        if keys != const.COMBINED_KEYS:
            print(keys)
            print(const.COMBINED_KEYS)
            raise ValueError("Keys in log file do not match const.COMBINED_KEYS")
        output_file_name_index = const.COMBINED_KEYS.index("output_file_name") # should be 0
        for line in f:
            vals_str = line.strip().split(";")
            vals = []
            if output_file_name == vals_str[output_file_name_index]:
                if exclude_log_items:
                    new_keys = const.FIT_KEYS.copy()
                else:
                    new_keys = keys

                for i, val_str in enumerate(vals_str):
                    if (
                        exclude_log_items and
                        const.COMBINED_KEYS[i] in const.LOG_KEYS and
                        i != output_file_name_index
                    ):
                        continue
                    try:
                        val = ast.literal_eval(val_str)
                    except (ValueError, SyntaxError):
                        val = val_str
                    vals.append(val)
                return dict(zip(new_keys, vals))



def get_scaled_y_bounds(
    y1: np.ndarray | None = None,
    y2: np.ndarray | None = None,
    ax1_y_bounds: tuple[float, float] | None = None,
    ax2_y_bounds: tuple[float | None, float | None] | None = None,
    fix_y2_top: bool = True,
    y_val_line_up: float = 0.0,
    y_top_scale_factor: float = 1.25
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Get the scaled y bounds for two axes.

    Parameters
    ----------
    y1: np.ndarray | None = None
        The y values of the first axis.
    y2: np.ndarray | None = None
        The y values of the second axis.
    ax1_y_bounds: tuple[float, float] | None = None
        The y bounds of the first axis.
    ax2_y_bounds: tuple[float | None, float | None] | None = None
        The y bounds of the second axis.
    fix_y2_top: bool = True
        Whether to keep the top of the second axis constant.
    y_val_line_up: float = 0.0
        The y value that will line up on both axes.
    y_top_scale_factor: float = 1.25
        The factor that the max of y2 will be scaled by to create
        the top of the second axis.

    Returns
    -------
    tuple[tuple[float, float], tuple[float | None, float | None]]
        The y bounds of the first and second axes.
    """
    if ax1_y_bounds is None:
        if y1 is None:
            raise ValueError("y1 must be provided if ax1_y_bounds is None")
        y1_top = np.max(y1)*y_top_scale_factor
        y1_bottom = np.min(y1) - (y1_top - np.max(y1))
        ax1_y_bounds = (y1_bottom, y1_top)
    if ax2_y_bounds is None:
        if y2 is None:
            raise ValueError("y2 must be provided if ax2_y_bounds is None")
        y2_top = np.max(y2)*y_top_scale_factor
        if fix_y2_top:
            y2_bottom = None
        else:
            y2_bottom = np.min(y2) - (y2_top - np.max(y2))
            y2_top = None
        ax2_y_bounds = (y2_bottom, y2_top)

    ax1_y_range = ax1_y_bounds[1] - ax1_y_bounds[0]
    if ax1_y_range <= const.EPS:
        raise ValueError("ax1_y_bounds range too small")
    y_val_frac = (y_val_line_up - ax1_y_bounds[0]) / ax1_y_range
    if ax2_y_bounds[0] is None:
        if ax2_y_bounds[1] is None:
            raise ValueError("Only one of ax2_y_bounds can be None")
        ax2_low = (y_val_line_up - y_val_frac * ax2_y_bounds[1]) / (1 - y_val_frac)
        return ax1_y_bounds, (ax2_low, ax2_y_bounds[1])
    elif ax2_y_bounds[1] is None:
        ax2_high = (y_val_line_up + ax2_y_bounds[0] * (y_val_frac - 1)) / y_val_frac
        return ax1_y_bounds, (ax2_y_bounds[0], ax2_high)
    else:
        raise ValueError("One of ax2_y_bounds must be None")
