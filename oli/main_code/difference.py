import numpy as np

from . import constants as const
from .data_reading import get_adjusted_data
from .polynomial_fit import apply_poly_fit
from .plotting import plot_vert_emission_lines
from .helpers import get_lam_bounds

def get_diff_spectra(
    adjusted_fluxes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    return_lam: bool = False,
    arcsec: int = 3
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray]
] | tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray
]:
    if adjusted_fluxes is None:
        if arcsec == 3:
            data = get_adjusted_data(plot_resampled_and_blurred=False)
        elif arcsec == 4:
            data = get_adjusted_data(
                plot_resampled_and_blurred=False,
                fname_2015_blue=const.FNAME_2015_BLUE_4_ARCSEC,
                fname_2015_red=const.FNAME_2015_RED_4_ARCSEC
            )
        else:
            raise NotImplementedError("SAMI data must be 3 or 4 arc-seconds")
        lam = data[0]
        _, adjusted_01_flux_15, adjusted_01_err_15 = apply_poly_fit(
            data=data, year_to_adjust=2001, 
            plot_ratio_selection=False, plot_poly_ratio=False, plot_adjusted=False,
        )
        _, adjusted_15_flux_15, adjusted_15_err_15 = apply_poly_fit(
            data=data, year_to_adjust=2015,
            plot_ratio_selection=False, plot_poly_ratio=False, plot_adjusted=False,
        )
        _, adjusted_21_flux_15, adjusted_21_err_15 = apply_poly_fit(
            data=data, year_to_adjust=2021,
            plot_ratio_selection=False, plot_poly_ratio=False, plot_adjusted=False,
        )
        _, adjusted_22_flux_15, adjusted_22_err_15 = apply_poly_fit(
            data=data, year_to_adjust=2022,
            plot_ratio_selection=False, plot_poly_ratio=False, plot_adjusted=False,
        )
    else:
        (
            (adjusted_01_flux_15, adjusted_01_err_15),
            (adjusted_15_flux_15, adjusted_15_err_15),
            (adjusted_21_flux_15, adjusted_21_err_15),
            (adjusted_22_flux_15, adjusted_22_err_15)
        ) = adjusted_fluxes
        lam = None
    
    diff_15 = adjusted_15_flux_15 - adjusted_01_flux_15
    diff_21 = adjusted_21_flux_15 - adjusted_01_flux_15
    diff_22 = adjusted_22_flux_15 - adjusted_01_flux_15

    diff_15_err = np.sqrt(adjusted_01_err_15**2 + adjusted_15_err_15**2)
    diff_21_err = np.sqrt(adjusted_01_err_15**2 + adjusted_21_err_15**2)
    diff_22_err = np.sqrt(adjusted_01_err_15**2 + adjusted_22_err_15**2)

    if return_lam and lam is not None:
        to_return = (diff_15, diff_21, diff_22), (diff_15_err, diff_21_err, diff_22_err), lam
    elif return_lam and lam is None:
        raise ValueError(
            "lam can only be returned if adjusted_fluxes "
            "are calculated within this function."
        )
    else:
        to_return = (diff_15, diff_21, diff_22), (diff_15_err, diff_21_err, diff_22_err)
        
    return to_return
