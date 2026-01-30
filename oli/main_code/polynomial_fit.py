import numpy as np
import matplotlib.pyplot as plt

from .constants import *
from .helpers import bin_data_by_median, get_lam_bounds
from .plotting import plot_polynomial_ratio, plot_adjusted_spectrum
from .data_reading import get_adjusted_data

def get_polynom_fit(
    lambdas: np.ndarray, vals: np.ndarray,
    year_to_adjust: int,
    baseline_year: int = 2015,
    degree: float = 6, bin_width: float = 40,
    bin_by_med: bool = True, plot_result: bool = True,
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
            title=title
        )
    return polynom, polynom_vals

def apply_poly_fit(
    data: tuple[np.ndarray, tuple[tuple[np.ndarray, np.ndarray]]] | None = None,
    year_to_adjust: int = 2022,
    baseline_year: int = 2015,
    lambdas_to_ignore_width: float = VEL_TO_IGNORE_WIDTH,
    width_is_vel: bool = True,
    poly_degree: float = 6,
    bin_by_med: bool = True, bin_width: float = 50,
    plot_ratio_selection: bool = True, 
    plot_poly_ratio: bool = True,
    plot_adjusted: bool = True,
    adjusted_plot_lam_bounds: tuple[float] | None = (3800, 8000),
    adjusted_flux_y_bounds: tuple[float] | None = None,
    # adjusted_err_y_bounds: tuple[float] | None = None, #TD: remove?
    ions: dict[str, float] | None = None,
    blur_before_resampling: bool = True,
) -> tuple[np.poly1d | None, np.ndarray, np.ndarray]:
    possible_years = [2001, 2015, 2021, 2022]
    if year_to_adjust not in possible_years or baseline_year not in possible_years:
        raise ValueError(f"year should be in {possible_years}")

    data = get_adjusted_data(
        plot_resampled_and_blurred=False, 
        blur_before_resampling=blur_before_resampling
    ) if data is None else data

    lam, (data01, data15, data21, data22) = data

    data_map = {
        2022: data22,
        2021: data21,
        2015: data15,
        2001: data01
    }

    flux, err = data_map.get(year_to_adjust)
    baseline_flux, baseline_err = data_map.get(baseline_year)

    if year_to_adjust == baseline_year:
        return None, flux, err

    actual_ratio_flux = flux / baseline_flux
    balmer_mask = np.zeros(lam.shape, dtype=bool)

    lambdas_to_ignore = [
        get_lam_bounds(H_ALPHA, lambdas_to_ignore_width, width_is_vel=width_is_vel),
        get_lam_bounds(H_BETA, lambdas_to_ignore_width, width_is_vel=width_is_vel)
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
            title=ratio_title
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

    if plot_adjusted:
        adjusted_plot_title = f"Spectral flux density of {year_to_adjust} (polynomial fit to {baseline_year})"
        plot_adjusted_spectrum(
            lam=lam,
            baseline_flux=baseline_flux,
            unadjusted_flux=flux,
            adjusted_flux=adjusted_flux,
            year_to_adjust=year_to_adjust,
            baseline_year=baseline_year,
            ions=ions,
            lam_bounds=adjusted_plot_lam_bounds,
            flux_y_bounds=adjusted_flux_y_bounds,
            title=adjusted_plot_title
        )
    return polynom, adjusted_flux, adjusted_err