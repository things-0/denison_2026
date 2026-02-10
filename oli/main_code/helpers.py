import numpy as np
import os
import ast

from . import constants as const

def convert_lam_to_vel(
    lam: np.ndarray | float,
    lam_centre_rest_frame: float = const.H_ALPHA,
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
        return (lam - lam_centre_rest_frame) * const.C_KM_S / lam_centre_rest_frame
    else:
        return (lam / (1 + const.Z_SPEC) - lam_centre_rest_frame) * const.C_KM_S / lam_centre_rest_frame

def convert_vel_to_lam(
    vel: np.ndarray | float,
    lam_centre_rest_frame: float = const.H_ALPHA,
    return_rest_frame: bool = False
) -> np.ndarray | float:
    """
    Convert velocity (km/s) to wavelength (Å).
    """
    # λ = λ_cent * (1 + v / c)
    if return_rest_frame:
        return lam_centre_rest_frame * (1 + vel / const.C_KM_S)
    else:
        return (lam_centre_rest_frame * (1 + vel / const.C_KM_S)) * (1 + const.Z_SPEC)

def get_lam_bounds(lam: float, width: float, is_rest_frame: bool = True, width_is_vel: bool = False) -> tuple[float, float]:
    if is_rest_frame:
        obs_lam = lam * (1+const.Z_SPEC)
        rest_lam = lam
    else:
        obs_lam = lam
        rest_lam = lam/(1+const.Z_SPEC)
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

    res_min =  res_min / const.SMOOTH_FACTOR

    return res_min

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
    freq_eff_width = const.C_ANG_S * col_band_eff_width / (col_band_eff_lam**2)  # Hz
    
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
    mask: np.ndarray | None,
    reduce_endpoint_weights: bool = True
) -> np.ndarray:
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
) -> tuple[list[float], list[float]]: # ([h_min * n, μ_min * n, σ_min * n], [h_max * n, μ_max * n, σ_max * n])
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
) -> str:

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

def get_fwhm(x: np.ndarray, y_gaussian: np.ndarray, get_vel: bool = True) -> float:
    half_max = np.max(y_gaussian) / 2
    above_half = np.where(y_gaussian >= half_max)[0]
    if len(above_half) < 2:
        raise ValueError("No FWHM found - not enough points above half max")
    lam_left = x[above_half[0]]
    lam_right = x[above_half[-1]]
    fwhm_ang = lam_right - lam_left
    if get_vel is False:
        return fwhm_ang
    lam_centre_rest_frame = ((lam_right + lam_left) / 2) / (1 + const.Z_SPEC)
    #TD: remove testing
    # lcrf_option_1 = (x[np.argmax(y_gaussian)]) / (1 + Z_SPEC)
    # lcrf_option_2 = ((lam_right + lam_left) / 2) / (1 + Z_SPEC)
    # lcrf_option_3 = (x[(above_half[0] + above_half[-1]) // 2]) / (1 + Z_SPEC)
    # print(f"Hα: {H_ALPHA}, Hβ: {H_BETA}")
    # print(f"x[argmax(y)]: {lcrf_option_1}")
    # print(f"average lam of lam_left and lam_right: {lcrf_option_2}")
    # print(f"average index of lam_left and lam_right: {lcrf_option_3}")
    #
    vel_left = convert_lam_to_vel(lam_left, lam_centre_rest_frame)
    vel_right = convert_lam_to_vel(lam_right, lam_centre_rest_frame)
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

def get_output_file_name(
    fname: str, # without extension
    folder_name: str = "pyqsofit_code/output/",
) -> str:
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
    with open(log_name, "a") as f:
        two_write = ""
        for val in kwargs.values():
            two_write += f"{val};"
        f.write(two_write.strip(";") + "\n")

def get_kwargs_from_log(
    output_file_name: str,
    log_name: str = "pyqsofit_code/log.csv",
    exclude_log_items: bool = True
) -> dict:
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

# def get_scaled_y_bounds(
#     ax1_y_bounds: tuple[float, float],
#     ax2_y_bounds: tuple[float | None, float | None],
#     y1_ref: float,
#     y2_ref: float,
# ) -> tuple[float, float]:
#     y1_min, y1_max = ax1_y_bounds
#     y2_min, y2_max = ax2_y_bounds

#     y1_range = y1_max - y1_min
#     if y1_range == 0:
#         raise ValueError("ax1_y_bounds has zero range")

#     f = (y1_ref - y1_min) / y1_range

#     if y2_min is None and y2_max is None:
#         raise ValueError("Only one of ax2_y_bounds can be None")

#     if y2_min is None:
#         # solve for y2_min
#         # (y2_ref - y2_min) / (y2_max - y2_min) = f
#         # y2_ref - y2_min = f*(y2_max - y2_min)
#         # y2_ref - y2_min = f*y2_max - f*y2_min
#         # y2_ref - f*y2_max = y2_min*(1 - f)
#         if f == 1:
#             raise ValueError("f=1 implies y2_min is undefined")
#         y2_min = (y2_ref - f * y2_max) / (1 - f)
#         return y2_min, y2_max

#     if y2_max is None:
#         # solve for y2_max
#         # (y2_ref - y2_min) / (y2_max - y2_min) = f
#         # y2_ref - y2_min = f*y2_max - f*y2_min
#         # y2_ref + y2_min*(f - 1) = f*y2_max
#         if f == 0:
#             raise ValueError("f=0 implies y2_max is undefined")
#         y2_max = (y2_ref + y2_min * (f - 1)) / f
#         return y2_min, y2_max

#     raise ValueError("One of ax2_y_bounds must be None")


def get_scaled_y_bounds(
    y1: np.ndarray | None = None,
    y2: np.ndarray | None = None,
    ax1_y_bounds: tuple[float, float] | None = None,
    ax2_y_bounds: tuple[float | None, float | None] | None = None,
    fix_y2_top: bool = True,
    y_val_line_up: float = 0.0,
    y_top_scale_factor: float = 1.25
) -> tuple[tuple[float, float], tuple[float, float]]:
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