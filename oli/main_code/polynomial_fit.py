import numpy as np
import matplotlib.pyplot as plt

from . import constants as const
from .helpers import bin_data_by_median, get_lam_bounds
from .plotting import plot_polynomial_ratio, plot_adjusted_spectrum
from .data_reading import get_adjusted_data

def get_polynom_fit(
    lambdas: np.ndarray,
    vals: np.ndarray,
    year_to_adjust: int,
    baseline_year: int = 2015,
    degree: float = 6,
    bin_width: float = const.POLY_FIT_BIN_WIDTH,
    bin_by_med: bool = True,
    plot_result: bool = True,
) -> tuple[np.poly1d, np.ndarray]:
    if bin_by_med:
        binned_lambdas, binned_vals, binned_val_errs = bin_data_by_median(lambdas, vals, bin_width)
        new_vals, new_lambdas = binned_vals, binned_lambdas
    else:
        binned_lambdas, binned_vals = None, None
        new_vals, new_lambdas = vals, lambdas

    valid_mask = np.isfinite(new_vals)
    coefficients = np.polyfit(new_lambdas[valid_mask], new_vals[valid_mask], degree)
    polynom = np.poly1d(coefficients)
    polynom_vals = polynom(new_lambdas)

    valid_indices = np.where(valid_mask)[0]
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    outer_mask = np.zeros_like(valid_mask, dtype=bool)
    outer_mask[:first_valid] = True
    outer_mask[last_valid + 1:] = True

    polynom_vals[outer_mask] = np.nan

    if plot_result:
        title = f"Spectral flux density ratio of {year_to_adjust} to {baseline_year}"
        plot_polynomial_ratio(
            lambdas, vals, polynom_vals,
            binned_lambdas=binned_lambdas,
            binned_vals=binned_vals,
            degree=degree,
            bin_width=bin_width,
            bin_by_med=bin_by_med,
            title=title,
            save_fig_name=f"sfd_ratio_{(year_to_adjust - 2000):02d}_to_{(baseline_year - 2000):02d}_med_binned"
        )
    return polynom, polynom_vals

def apply_poly_fit( #TODO: update other params as well (medflux, etc.)
    data: dict[str, dict[str, np.ndarray]] | None = None,
    year_to_adjust: int = 2022,
    baseline_year: int = 2015,
    lambdas_to_ignore_width: float = const.VEL_TO_IGNORE_WIDTH,
    width_is_vel: bool = True,
    poly_degree: float = 6,
    bin_by_med: bool = True,
    bin_width: float = const.POLY_FIT_BIN_WIDTH,
    plot_ratio_selection: bool = True, 
    plot_poly_ratio: bool = True,
    plot_adjusted: bool = True,
    adjusted_plot_lam_bounds: tuple[float] | None = None, # const.TOTAL_LAM_BOUNDS, # (3800, 8000),
    adjusted_flux_y_bounds: tuple[float] | None = None,
    ions: dict[str, float] | None = None,
    blur_step: int = 1,
    resample_step: int = 2,
    extrapolate_beyond_max_baseline_lam: bool = False,
) -> dict[str, np.ndarray | np.poly1d | int | None]:
    """
    Finds the least squares polynomial fit to the ratio of the flux of
    the year_to_adjust spectrum to the baseline_year spectrum, then normalises
    the year_to_adjust spectrum (and its errors) by dividing by the polynomial fit.

    Parameters
    ----------
    data: dict[str, dict[str, np.ndarray]] | None
        The data to adjust. If None, the data is created using :func:`data_reading.get_adjusted_data`.
    year_to_adjust: int
        The year to adjust. Must be one of 2001, 2015, 2021, or 2022.
    baseline_year: int
        The year to use as the baseline (calibration) flux. Must be one of 2001, 2015, 2021, or 2022.
    lambdas_to_ignore_width: float
        The width of wavelengths to ignore around the Balmer lines when finding the flux
        ratio and fitting the polynomial.
    width_is_vel: bool
        If True, the `lambdas_to_ignore_width` is in km/s, else in angstroms.
    poly_degree: float
        The degree of the polynomial to fit to the flux ratio.
    bin_by_med: bool = True
        If True, the flux ratio is binned by the median before fitting the polynomial.
    bin_width: float = const.POLY_FIT_BIN_WIDTH
        The width of the bins to use when calculating the median of the flux ratio.
    plot_ratio_selection: bool = True
        If True, the ratio of the flux of the year_to_adjust spectrum to the baseline_year
        spectrum is plotted, including the highlighted flux in Balmer regions to ignore.
    plot_poly_ratio: bool = True
        If True, the ratio of the flux of the year_to_adjust spectrum to the baseline_year
        spectrum is plotted, including the binned median. Note, the selection of Balmer
        regions to ignore is not plotted here.
    plot_adjusted: bool = True
        If True, the adjusted spectrum is plotted against the unadjusted and baseline spectra.
    adjusted_plot_lam_bounds: tuple[float] | None = None
        The wavelength bounds of the `plot_adjusted` plot.
    adjusted_flux_y_bounds: tuple[float] | None = None
        The flux bounds of the `plot_adjusted` plot.
    ions: dict[str, float] | None = None
        The ions to plot on the `plot_adjusted` plot. Keys represent the string labels seen in
        the legend, and values represent the rest wavelength of each line. E.g. `{"[OIII]": 5007}`.
    blur_step: int = 1
        The step to blur the unadjusted spectra. 0 to not blur, 1 to blur before resampling, 2 to
        blur after resampling. Note, this argument is irrelevant if `data` is provided.
    resample_step: int = 2
        The step to resample the unadjusted spectra. 0 to not resample, 1 to resample before blurring,
        2 to resample after blurring. Note, this argument is irrelevant if `data` is provided.
    extrapolate_beyond_max_baseline_lam: bool = False
        If True, the polynomial fit is extrapolated beyond the maximum wavelength of the baseline
        spectrum. This is likely to return absurd values since the polynomial fit is only applied
        to data within the wavelength range of the baseline spectrum.

    Returns
    -------
    to_return: dict[str, np.ndarray | np.poly1d | int | None]
        A dictionary with the keys "polynom", "flux", "flux_error",
        "last_valid_lam_idx", and values calculated from
        :func:`data_reading.get_adjusted_data`, including "fwhm_per_pix",
        "good_pixels", "velscale", "lam".
    """

    possible_years = [2001, 2015, 2021, 2022]
    if year_to_adjust not in possible_years or baseline_year not in possible_years:
        raise ValueError(f"year should be in {possible_years}")

    data = get_adjusted_data(
        plot_resampled_and_blurred=False, 
        blur_step=blur_step,
        resample_step=resample_step,
    ) if data is None else data

    # lam, (data01, data15, data21, data22) = data

    # data_map = {
    #     2022: data22,
    #     2021: data21,
    #     2015: data15,
    #     2001: data01
    # }

    # flux, err = data_map.get(year_to_adjust)
    # baseline_flux, baseline_err = data_map.get(baseline_year)

    lam = data["lam"]
    flux = data[f"{year_to_adjust}"]["flux"]
    err = data[f"{year_to_adjust}"]["flux_error"]
    baseline_flux = data[f"{baseline_year}"]["flux"]
    # baseline_err = data[f"{baseline_year}"]["flux_error"]

    to_return = {
        "lam": lam,
        "polynom": None,
        "flux": flux,
        "flux_error": err,
        "last_valid_lam_idx": np.nan,
        "fwhm_per_pix": data[f"{year_to_adjust}"]["fwhm_per_pix"],
        "good_pixels": data[f"{year_to_adjust}"]["good_pixels"],
        "velscale": data[f"{year_to_adjust}"]["velscale"],
    }

    if year_to_adjust == baseline_year:
        if extrapolate_beyond_max_baseline_lam:
            return to_return
        last_valid_lam_idx = int(np.where(np.isfinite(flux))[0][-1])
        to_return["flux"] = flux[:last_valid_lam_idx]
        to_return["flux_error"] = err[:last_valid_lam_idx]
        to_return["last_valid_lam_idx"] = last_valid_lam_idx
        to_return["lam"] = lam[:last_valid_lam_idx]
        to_return["fwhm_per_pix"] = to_return["fwhm_per_pix"][:last_valid_lam_idx]
        to_return["good_pixels"] = to_return["good_pixels"][to_return["good_pixels"] < last_valid_lam_idx]
        return to_return

    actual_ratio_flux = flux / baseline_flux
    balmer_mask = np.zeros_like(lam, dtype=bool)

    lambdas_to_ignore = [
        get_lam_bounds(const.H_ALPHA, lambdas_to_ignore_width, width_is_vel=width_is_vel),
        get_lam_bounds(const.H_BETA, lambdas_to_ignore_width, width_is_vel=width_is_vel)
    ]

    for start, end in lambdas_to_ignore:
        current_range_mask = (lam >= start) & (lam <= end)
        balmer_mask = balmer_mask | current_range_mask

    actual_ratio_flux_removed = np.copy(actual_ratio_flux)
    actual_ratio_flux_removed[~balmer_mask] = np.nan

    actual_ratio_flux[balmer_mask] = np.nan

    ratio_title = f"Spectral flux density ratio of {year_to_adjust} to {baseline_year}"

    if plot_ratio_selection:
        plot_polynomial_ratio(
            lambdas=lam,
            vals=actual_ratio_flux,
            vals_removed=actual_ratio_flux_removed,
            bin_by_med=False,
            plot_selection=True,
            title=ratio_title,
            save_fig_name=f"sfd_ratio_{(year_to_adjust - 2000):02d}_to_{(baseline_year - 2000):02d}_selection"
        )

    polynom, _ = get_polynom_fit(
        lambdas=lam, vals=actual_ratio_flux,
        year_to_adjust=year_to_adjust,
        baseline_year=baseline_year,
        degree=poly_degree, bin_by_med=bin_by_med,
        bin_width=bin_width, plot_result=plot_poly_ratio,
    )

    adjusted_flux = flux / polynom(lam)
    adjusted_err = err / polynom(lam)

    if not extrapolate_beyond_max_baseline_lam:
        last_valid_lam_idx = int(np.where(np.isfinite(actual_ratio_flux))[0][-1])
        adjusted_flux = adjusted_flux[:last_valid_lam_idx]
        adjusted_err = adjusted_err[:last_valid_lam_idx]
        adjusted_lam = lam[:last_valid_lam_idx]
    else:
        adjusted_lam = lam
        last_valid_lam_idx = len(lam)

    if plot_adjusted:
        adjusted_plot_title = f"Spectral flux density of {year_to_adjust} (polynomial fit to {baseline_year})"
        plot_adjusted_spectrum(
            lam=lam,
            adjusted_lam=adjusted_lam,
            baseline_flux=baseline_flux,
            unadjusted_flux=flux,
            adjusted_flux=adjusted_flux,
            year_to_adjust=year_to_adjust,
            baseline_year=baseline_year,
            ions=ions,
            lam_bounds=adjusted_plot_lam_bounds,
            flux_y_bounds=adjusted_flux_y_bounds,
            title=adjusted_plot_title,
            save_fig_name=f"sfd_{(year_to_adjust - 2000):02d}_to_{(baseline_year - 2000):02d}_poly_fit"
        )
    to_return["polynom"] = polynom
    to_return["flux"] = adjusted_flux
    to_return["flux_error"] = adjusted_err
    to_return["last_valid_lam_idx"] = last_valid_lam_idx
    to_return["lam"] = lam[:last_valid_lam_idx]
    to_return["fwhm_per_pix"] = to_return["fwhm_per_pix"][:last_valid_lam_idx]
    to_return["good_pixels"] = to_return["good_pixels"][to_return["good_pixels"] < last_valid_lam_idx]
    return to_return