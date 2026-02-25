import numpy as np

from . import constants as const
from .data_reading import get_adjusted_data
from .polynomial_fit import apply_poly_fit

def get_diff_spectra(
    adjusted_fluxes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    arcsec: int = 3,
    blur_step: int = 1,
    resample_step: int = 2,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    np.ndarray | None
]:
    """
    Subtract the 2001 spectral flux density from the flux of the other epochs
    (2015, 2021, and 2022).

    Parameters
    ----------
    adjusted_fluxes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        The fluxes of the 2001, 2015, 2021, and 2022 epochs.
    arcsec: int
        The arcsec of the SAMI data. (must be 3 or 4)
    blur_step: int
        The step to blur the spectra. 0 to not blur, 1 to blur before resampling,
        2 to blur after resampling. Note: this argument is irrelevant if `adjusted_fluxes`
        is provided.
    resample_step: int
        The step to resample the spectra. 0 to not resample, 1 to resample before
        blurring, 2 to resample after blurring. Note: this argument is irrelevant
        if `adjusted_fluxes` is provided. Additionally, to perform an element-wise
        subtraction, the spectra must be resampled onto the same wavelength grid
        (resample_step > 0).

    Returns
    -------
    diff_spectra: tuple[np.ndarray, np.ndarray, np.ndarray]
        The difference spectra of the 2015, 2021, and 2022 epochs.
    """
    if adjusted_fluxes is None:
        if resample_step == 0:
            raise ValueError("Data must be resampled to get_diff_spectra")
        if arcsec == 3:
            data = get_adjusted_data(
                plot_resampled_and_blurred=False,
                blur_step=blur_step,
                resample_step=resample_step,
            )
        elif arcsec == 4:
            data = get_adjusted_data(
                plot_resampled_and_blurred=False,
                fname_2015_blue=const.FNAME_2015_BLUE_4_ARCSEC,
                fname_2015_red=const.FNAME_2015_RED_4_ARCSEC,
                blur_step=blur_step,
                resample_step=resample_step,
            )
        else:
            raise NotImplementedError("SAMI data must be 3 or 4 arc-seconds")
        lam = data["lam"]
        adjusted_data_01 = apply_poly_fit(
            data=data, year_to_adjust=2001, 
            plot_ratio_selection=False, plot_poly_ratio=False, plot_adjusted=False,
        )
        adjusted_data_15 = apply_poly_fit(
            data=data, year_to_adjust=2015,
            plot_ratio_selection=False, plot_poly_ratio=False, plot_adjusted=False,
        )
        adjusted_data_21 = apply_poly_fit(
            data=data, year_to_adjust=2021,
            plot_ratio_selection=False, plot_poly_ratio=False, plot_adjusted=False,
        )
        adjusted_data_22 = apply_poly_fit(
            data=data, year_to_adjust=2022,
            plot_ratio_selection=False, plot_poly_ratio=False, plot_adjusted=False,
        )
        adjusted_01_flux_15, adjusted_15_flux_15, adjusted_21_flux_15, adjusted_22_flux_15 = (
            adjusted_data_01["flux"], adjusted_data_15["flux"],
            adjusted_data_21["flux"], adjusted_data_22["flux"]
        )
        adjusted_01_err_15, adjusted_15_err_15, adjusted_21_err_15, adjusted_22_err_15 = (
            adjusted_data_01["flux_error"], adjusted_data_15["flux_error"],
            adjusted_data_21["flux_error"], adjusted_data_22["flux_error"]
        )
        last_valid_lam_idx_01, last_valid_lam_idx_15, last_valid_lam_idx_21, last_valid_lam_idx_22 = (
            adjusted_data_01["last_valid_lam_idx"], adjusted_data_15["last_valid_lam_idx"],
            adjusted_data_21["last_valid_lam_idx"], adjusted_data_22["last_valid_lam_idx"]
        )
        # clip lam in case polynomial fit created invalid fluxes beyond max lam of baseline (2015)
        lam_adjusted = lam[:int(np.nanmin((
            last_valid_lam_idx_01, last_valid_lam_idx_15,
            last_valid_lam_idx_21, last_valid_lam_idx_22
        )))]
    else:
        (
            (adjusted_01_flux_15, adjusted_01_err_15),
            (adjusted_15_flux_15, adjusted_15_err_15),
            (adjusted_21_flux_15, adjusted_21_err_15),
            (adjusted_22_flux_15, adjusted_22_err_15)
        ) = adjusted_fluxes
        lam_adjusted = None
    
    diff_15 = adjusted_15_flux_15 - adjusted_01_flux_15
    diff_21 = adjusted_21_flux_15 - adjusted_01_flux_15
    diff_22 = adjusted_22_flux_15 - adjusted_01_flux_15

    diff_15_err = np.sqrt(adjusted_01_err_15**2 + adjusted_15_err_15**2)
    diff_21_err = np.sqrt(adjusted_01_err_15**2 + adjusted_21_err_15**2)
    diff_22_err = np.sqrt(adjusted_01_err_15**2 + adjusted_22_err_15**2)

    return (diff_15, diff_21, diff_22), (diff_15_err, diff_21_err, diff_22_err), lam_adjusted
