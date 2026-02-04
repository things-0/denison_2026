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

    flux_alpha_assasn_g_21_mjy = convert_flux_to_mJy(flux_beta_21, ASSASN_G_BAND_WIDTH, ASSASN_G_BAND_LAM)
    flux_alpha_assasn_g_21_mjy_err = convert_flux_to_mJy(flux_beta_21_err, ASSASN_G_BAND_WIDTH, ASSASN_G_BAND_LAM)
    flux_alpha_atlas_o_21_mjy = convert_flux_to_mJy(flux_alpha_21, ATLAS_O_BAND_WIDTH, ATLAS_O_BAND_LAM)
    flux_alpha_atlas_o_21_mjy_err = convert_flux_to_mJy(flux_alpha_21_err, ATLAS_O_BAND_WIDTH, ATLAS_O_BAND_LAM)
    flux_alpha_ztf_r_21_mjy = convert_flux_to_mJy(flux_alpha_21, ZTF_R_BAND_WIDTH, ZTF_R_BAND_LAM)
    flux_alpha_ztf_r_21_mjy_err = convert_flux_to_mJy(flux_alpha_21_err, ZTF_R_BAND_WIDTH, ZTF_R_BAND_LAM)

    flux_alpha_assasn_g_22_mjy = convert_flux_to_mJy(flux_beta_22, ASSASN_G_BAND_WIDTH, ASSASN_G_BAND_LAM)
    flux_alpha_assasn_g_22_mjy_err = convert_flux_to_mJy(flux_beta_22_err, ASSASN_G_BAND_WIDTH, ASSASN_G_BAND_LAM)
    flux_alpha_atlas_o_22_mjy = convert_flux_to_mJy(flux_alpha_22, ATLAS_O_BAND_WIDTH, ATLAS_O_BAND_LAM)
    flux_alpha_atlas_o_22_mjy_err = convert_flux_to_mJy(flux_alpha_22_err, ATLAS_O_BAND_WIDTH, ATLAS_O_BAND_LAM)
    flux_alpha_ztf_r_22_mjy = convert_flux_to_mJy(flux_alpha_22, ZTF_R_BAND_WIDTH, ZTF_R_BAND_LAM)
    flux_alpha_ztf_r_22_mjy_err = convert_flux_to_mJy(flux_alpha_22_err, ZTF_R_BAND_WIDTH, ZTF_R_BAND_LAM)

    survey_names = (
        "ASASSN g band 2021 (Hβ)",
        "ASASSN g band 2022 (Hβ)",
        "Atlas o band 2021 (Hα)",
        "Atlas o band 2022 (Hα)",
        "ZTF r band 2021 (Hα)",
        "ZTF r band 2022 (Hα)"
    )
    int_flux_vals = [
        [flux_alpha_assasn_g_21_mjy, flux_alpha_assasn_g_21_mjy_err],
        [flux_alpha_assasn_g_22_mjy, flux_alpha_assasn_g_22_mjy_err],
        [flux_alpha_atlas_o_21_mjy, flux_alpha_atlas_o_21_mjy_err],
        [flux_alpha_atlas_o_22_mjy, flux_alpha_atlas_o_22_mjy_err],
        [flux_alpha_ztf_r_21_mjy, flux_alpha_ztf_r_21_mjy_err],
        [flux_alpha_ztf_r_22_mjy, flux_alpha_ztf_r_22_mjy_err]
    ]
    num_gaussians_vals = (
        num_gaussians_beta_21,      # ASASSN g band 2021
        num_gaussians_beta_22,      # ASASSN g band 2022
        num_gaussians_alpha_21,     # Atlas o band 2021
        num_gaussians_alpha_22,     # Atlas o band 2022
        num_gaussians_alpha_21,     # ZTF r band 2021
        num_gaussians_alpha_22      # ZTF r band 2022
    )
    photometric_flux_vals = [
        ASASSN_G_FLUX_21,
        ASASSN_G_FLUX_22,
        ATLAS_O_FLUX_21, 
        ATLAS_O_FLUX_22,
        ZTF_R_FLUX_21,
        ZTF_R_FLUX_22
    ]

    int_flux_vals_micro_jy = np.array(int_flux_vals) * 1e3
    photometric_flux_vals_micro_jy = np.array(photometric_flux_vals) * 1e3
 
    # Column headers (in desired order)
    headers = [
        "Survey",
        "Attenuated Photometric Flux (μJy)",
        "Integrated Spectroscopic Flux (μJy)",
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
    lam_centre_rest_frame = ((lam_right + lam_left) / 2) / (1 + Z_SPEC)
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
        "FWHM Hα 2015 (km/s)": (fwhm_alpha_15, YASMEEN_RESULTS["fwhm_alpha_15"]),
        "FWHM Hα 2021 (km/s)": (fwhm_alpha_21, YASMEEN_RESULTS["fwhm_alpha_21"]),
        "FWHM Hβ 2015 (km/s)": (fwhm_beta_15, YASMEEN_RESULTS["fwhm_beta_15"]),
        "FWHM Hβ 2021 (km/s)": (fwhm_beta_21, YASMEEN_RESULTS["fwhm_beta_21"]),
        "Flux Hα 2015 (1e-17 ergs/s/cm^2)": (flux_alpha_15, YASMEEN_RESULTS["flux_alpha_15"]),
        "Flux Hα 2021 (1e-17 ergs/s/cm^2)": (flux_alpha_21, YASMEEN_RESULTS["flux_alpha_21"]),
        "Luminosity Hα 2015 (1e40 ergs/s)": (luminosity_alpha_15, YASMEEN_RESULTS["luminosity_alpha_15"]),
        "Luminosity Hα 2021 (1e40 ergs/s)": (luminosity_alpha_21, YASMEEN_RESULTS["luminosity_alpha_21"]),
        "Balmer decrement Hα 2015": (bd_15, YASMEEN_RESULTS["bd_15"]),
        "Balmer decrement Hα 2021": (bd_21, YASMEEN_RESULTS["bd_21"]),
        "BH mass 2015 (1e6 M_sun)": (bh_mass_15, YASMEEN_RESULTS["bh_mass_15"]),
        "BH mass 2021 (1e6 M_sun)": (bh_mass_21, YASMEEN_RESULTS["bh_mass_21"])
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

def get_new_filename(
    filename: str,
    start: str = "qsopar",
    end: str = ".fits"
) -> str:
    if not (filename.startswith(start) and filename.endswith(end)):
        raise ValueError(f"Filename must start with '{start}' and end with '{end}'")
    num_str = filename[len(start) : -len(end)]
    if not num_str.isdigit():
        raise ValueError("Filename must contain a number")
    new_num = int(num_str) + 1
    return f"{start}{new_num}{end}"