import numpy as np

from .constants import *

def convert_lam_to_vel(
    lam: np.ndarray | float,
    lam_centre_rest_frame: float = H_ALPHA,
    lam_is_rest_frame: bool = False
) -> np.ndarray | float:
    """
    Convert wavelength (Å) to velocity (km/s). 
    Set `lam_is_rest_frame` to `True` if the `lam`
    float or array to be converted is already in
    the rest frame.
    """
    # v = c * Δλ / λ_cent
    if lam_is_rest_frame:
        return (lam - lam_centre_rest_frame) * C_KM_S / lam_centre_rest_frame
    else:
        return (lam / (1 + Z_SPEC) - lam_centre_rest_frame) * C_KM_S / lam_centre_rest_frame

def convert_vel_to_lam(
    vel: np.ndarray | float,
    lam_centre_rest_frame: float = H_ALPHA,
    return_rest_frame: bool = False
) -> np.ndarray | float:
    """
    Convert velocity (km/s) to wavelength (Å).
    """
    # λ = λ_cent * (1 + v / c)
    if return_rest_frame:
        return lam_centre_rest_frame * (1 + vel / C_KM_S)
    else:
        return (lam_centre_rest_frame * (1 + vel / C_KM_S)) * (1 + Z_SPEC)

def get_lam_bounds(lam: float, width: float, is_rest_frame: bool = True, width_is_vel: bool = False) -> tuple[float, float]:
    if is_rest_frame:
        obs_lam = lam * (1+Z_SPEC)
        rest_lam = lam
    else:
        obs_lam = lam
        rest_lam = lam/(1+Z_SPEC)
    if width_is_vel:
        left = convert_vel_to_lam(-width / 2, lam_centre_rest_frame=rest_lam)
        right = convert_vel_to_lam(width / 2, lam_centre_rest_frame=rest_lam)
    else:
        left = obs_lam - width / 2
        right = obs_lam + width / 2
    return left, right

def get_min_res(
    res_01: np.ndarray,
    res_21: np.ndarray,
    res_22: np.ndarray
) -> np.ndarray:
    # res_min = np.minimum(np.minimum(
    #     res_21, 
    #     res_22
    # ), RES_15_BLUE)

    #TODO: check logic with Scott
    res_min = np.minimum(
        res_21, 
        res_22
    )
    #

    res_min =  res_min / SMOOTH_FACTOR

    return res_min

def bin_data_by_median(x: np.ndarray, y: np.ndarray, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
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

    return x_binned, y_binned

def convert_to_vel_data(
    lam: np.ndarray,
    flux: np.ndarray | None,
    flux_err: np.ndarray | None,
    centres_rest_ang: float | list[float],
    vel_width: float | None
) -> tuple[list[np.ndarray] | None, list[np.ndarray] | None, list[np.ndarray] | None]:

    if flux is None:
        return None, None, None
    
    trimmed_vels = []
    trimmed_fluxes = []
    trimmed_flux_errs = [] if flux_err is not None else None
    
    if not isinstance(centres_rest_ang, list):
        centres_rest_ang = [centres_rest_ang]

    for centre in centres_rest_ang:
        vel = convert_lam_to_vel(lam, lam_centre_rest_frame=centre)
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

def get_flux_bounds(
    lam01: np.ndarray,
    lam15: np.ndarray | tuple[np.ndarray, np.ndarray],
    lam21: np.ndarray,
    lam22: np.ndarray,
    flux01: np.ndarray,
    flux15: np.ndarray | tuple[np.ndarray, np.ndarray],
    flux21: np.ndarray,
    flux22: np.ndarray,
    x_bounds: tuple[float, float] | None
) -> tuple[float, float, float]:
    sami_is_split = True if isinstance(flux15, tuple) else False
    if x_bounds is not None:
        lam_01_plot_mask = np.where(np.isfinite(flux01) & (lam01 > x_bounds[0]) & (lam01 < x_bounds[1]))
        lam_21_plot_mask = np.where(np.isfinite(flux21) & (lam21 > x_bounds[0]) & (lam21 < x_bounds[1]))
        lam_22_plot_mask = np.where(np.isfinite(flux22) & (lam22 > x_bounds[0]) & (lam22 < x_bounds[1]))
    else:
        lam_01_plot_mask = np.isfinite(flux01)
        lam_21_plot_mask = np.isfinite(flux21)
        lam_22_plot_mask = np.isfinite(flux22)

    if sami_is_split:
        flux15_blue, flux15_red = flux15
        lam15_blue, lam15_red = lam15

        if x_bounds is not None:
            lam_15_blue_plot_mask = np.where(np.isfinite(flux15_blue) & (lam15_blue > x_bounds[0]) & (lam15_blue < x_bounds[1]))
            lam_15_red_plot_mask = np.where(np.isfinite(flux15_red) & (lam15_red > x_bounds[0]) & (lam15_red < x_bounds[1]))
        else:
            lam_15_blue_plot_mask = np.isfinite(flux15_blue)
            lam_15_red_plot_mask = np.isfinite(flux15_red)

        max_flux_15 = np.nanmax((
            np.nanmax(flux15_blue[lam_15_blue_plot_mask], initial=np.nan),
            np.nanmax(flux15_red[lam_15_red_plot_mask], initial=np.nan)
        ))
        min_flux_15 = np.nanmin((
            np.nanmin(flux15_blue[lam_15_blue_plot_mask], initial=np.nan),
            np.nanmin(flux15_red[lam_15_red_plot_mask], initial=np.nan)
        ))
        smallest_range_15 = np.nanmin((
            (
                np.nanmax(flux15_blue[lam_15_blue_plot_mask], initial=np.nan) -
                np.nanmin(flux15_blue[lam_15_blue_plot_mask], initial=np.nan)
            ), (
                np.nanmax(flux15_red[lam_15_red_plot_mask], initial=np.nan) -
                np.nanmin(flux15_red[lam_15_red_plot_mask], initial=np.nan)
            )
        ))
    else:
        lam_15_plot_mask = np.isfinite(flux15)

        max_flux_15 = np.nanmax(flux15[lam_15_plot_mask], initial=np.nan)
        min_flux_15 = np.nanmin(flux15[lam_15_plot_mask], initial=np.nan)
        smallest_range_15 = (
            np.nanmax(flux15[lam_15_plot_mask], initial=np.nan) -
            np.nanmin(flux15[lam_15_plot_mask], initial=np.nan)
        )

    max_flux = np.nanmax((
        np.nanmax(flux01[lam_01_plot_mask], initial=np.nan),
        max_flux_15,
        np.nanmax(flux21[lam_21_plot_mask], initial=np.nan),
        np.nanmax(flux22[lam_22_plot_mask], initial=np.nan)
    ))
    min_flux = np.nanmin((
        np.nanmin(flux01[lam_01_plot_mask]),
        min_flux_15,
        np.nanmin(flux21[lam_21_plot_mask]),
        np.nanmin(flux22[lam_22_plot_mask])
    ))
    total_range = max_flux - min_flux
    smallest_range = np.nanmin((
        (
            np.nanmax(flux01[lam_01_plot_mask], initial=np.nan) -
            np.nanmin(flux01[lam_01_plot_mask], initial=np.nan)
        ), smallest_range_15, (
            np.nanmax(flux21[lam_21_plot_mask], initial=np.nan) -
            np.nanmin(flux21[lam_21_plot_mask], initial=np.nan)
        ), (
            np.nanmax(flux22[lam_22_plot_mask], initial=np.nan) -
            np.nanmin(flux22[lam_22_plot_mask], initial=np.nan)
        )
    ))
    return min_flux, smallest_range, total_range

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
    freq_eff_width = C_ANG_S * col_band_eff_width / (col_band_eff_lam**2)  # Hz
    
    # Calculate flux density
    flux_density = flux / freq_eff_width  # erg s^-1 cm^-2 Hz^-1
    
    # Convert to mJy 
    # 1 Jy = 10^-23 erg s^-1 cm^-2 Hz^-1
    # 10^-17 erg s^-1 cm^-2 = 10^9 mJy
    flux_density_mjy = flux_density * 1e9
    
    return flux_density_mjy

def get_vel_lam_mask(
    lam: np.ndarray,
    vel_width: float,
    vel_centre_ang: float
) -> np.ndarray:
    """
    Get a mask for the wavelength array based on the velocity width and centre wavelength.
    """
    vel = convert_lam_to_vel(lam, lam_centre_rest_frame=vel_centre_ang)
    vel_width_mask = (vel >= -vel_width / 2) & (vel <= vel_width / 2)
    return vel_width_mask

def get_masked_diffs(
    x: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    diffs = np.diff(x)
    extended_diffs = np.zeros(len(x)+1)
    extended_diffs[1:-1] = diffs
    extended_diffs[0] = diffs[0]
    extended_diffs[-1] = diffs[-1]
    av_diffs = (extended_diffs[:-1] + extended_diffs[1:]) / 2
    return av_diffs[mask]

def get_default_bounds(
    x: np.ndarray,
    y: np.ndarray,
    num_of_gaussians: int
) -> tuple[list[float], list[float]]: # ([h_min * n, μ_min * n, σ_min * n], [h_max * n, μ_max * n, σ_max * n])
    height_min = HEIGHT_MIN
    height_max = 2 * np.max(y)
    x_range = x[-1] - x[0]
    mu_min = x[0] + x_range * MIN_MU
    mu_max = x[-1] - x_range * MIN_MU
    sigma_min = PEAK_MIN_RANGE * x_range / SIGMA_TO_FWHM
    sigma_max = x_range / SIGMA_TO_FWHM

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