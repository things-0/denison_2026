import numpy as np
import spectres

import astropy.io.fits as fits
import ppxf.ppxf_util as util
from PyAstronomy import pyasl

from pathlib import Path
import warnings

from . import constants as const
from .adjust_calibration import (
    gaussian_blur_before_resampling,
    gaussian_blur_after_resampling, clip_sami_blue_edge
)
from .helpers import (
    remove_or_replace_bad_values, get_velscale,
    get_good_pixels, combine_sami_vals, set_all_values,
    assert_lengths_match
)
from .plotting import plot_min_res, plot_spectra

def get_sami_data(
    file_name: str,
    folder_path: Path = const.SAMI_DATA_DIR,
    flux_power_of_10: int = 17,
    lam_medium: tuple[str, str] = ("air", "vacuum"),
    lam_bounds: tuple[float, float] | None = const.TOTAL_LAM_BOUNDS,
    rm_or_replace_outside_lam_bounds: bool | float = True,
    rm_or_replace_other_bad_values: bool | float = np.nan,
    z: float = const.Z_SPEC,
    perform_log_rebin: bool = False,
    resolving_power: float | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Get relevant data from the SAMI spectrum.

    Parameters
    ----------
    file_name: str
        The name of the file to read.
    folder_path: Path
        The path to the folder containing the file.
    flux_power_of_10: int
        Desired magnitude of flux (in erg s⁻¹ cm⁻² Å⁻¹). Default is 17.
    lam_medium: tuple[str, str]
        The medium of the input wavelength and the desired medium of the output wavelength. "air" or "vacuum".
    lam_bounds: tuple[float, float] | None
        The bounds of the wavelength range to keep (rest-frame).
    rm_or_replace_outside_lam_bounds: bool | float
        What to do with values outside the wavelength bounds. True to remove, float to
        replace with, or False to leave as is. Note: using a float will change the lam
        values.
    rm_or_replace_other_bad_values: bool | float
        What to do with other bad values. True to remove, float to replace with, or False
        to leave as is. Note: using a float will not change any of the lam values.
    z: float
        The redshift of the spectrum. Set to 0 to prevent redshift correction to rest frame.
    perform_log_rebin: bool
        Resamples spectrum to logarithmic spacing if True.
    resolving_power: float | np.ndarray | None
        The resolving power of the spectrum.

    Returns
    -------
    data: dict[str, np.ndarray]
        A dictionary with the keys "lam", "flux", "flux_error", "fwhm_per_pix",
        "velscale", "good_pixels".
    """
    with fits.open(folder_path / file_name) as hdulist:
        header = hdulist["PRIMARY"].header
        coord_ref_val_axis_1 = header['CRVAL1']     # coordinate reference value for axis 1
        coord_ref_pix_axis_1 = header['CRPIX1']     # index location of reference value
        coord_delta_axis_1 = header['CDELT1']       # pixel width in Angstroms
        num_pixels_axis_1 = header['NAXIS1']        # number of pixels

        lam_indices = np.arange(1, num_pixels_axis_1+0.5)                           # (fits coordinates are 1-indexed)
        lam_0 = coord_ref_val_axis_1 - coord_ref_pix_axis_1 * coord_delta_axis_1    # λ_0 = λ_ref - num_pixels_from_start * dλ        
        lam_obs = lam_0 + lam_indices * coord_delta_axis_1                          # λ_obs = λ_0 + index * dλ
        lam_rest = lam_obs / (1 + z)  # lam_rest is actually lam_obs if you set z=0

        if np.any(~np.isfinite(lam_rest)):
            # In theory this should never happen
            raise ValueError("Lam has nans")

        flux = hdulist["PRIMARY"].data
        var = hdulist["VARIANCE"].data
    
    err = np.array(np.sqrt(var), dtype=float)

    if lam_medium[0] == "air" and lam_medium[1] == "vacuum":
        lam_rest = util.air_to_vac(lam_rest)
        # lam_rest = pyasl.airtovac2(lam_rest) # alternative function
    elif lam_medium[0] == "vacuum" and lam_medium[1] == "air":
        lam_rest = util.vac_to_air(lam_rest)
    elif not set(lam_medium).issubset({"air", "vacuum"}):
        raise ValueError(f"Invalid value for lam_medium: {lam_medium}")

    flux *= 10 ** (flux_power_of_10 - 16)
    err *= 10 ** (flux_power_of_10 - 16)

    if perform_log_rebin:       # necessary for pPXF fitting
        lam_range = [lam_rest[0], lam_rest[-1]]
        flux_resampled, ln_lam_rest_resampled, velscale = util.log_rebin(lam_range, flux)
        err_resampled, _, _ = util.log_rebin(lam_range, err)
        lam_resampled = np.array(np.exp(ln_lam_rest_resampled))
    else:
        flux_resampled = flux
        err_resampled = err
        lam_resampled = lam_rest
        velscale = None

    if resolving_power is not None:
        fwhm_per_pix = lam_rest / resolving_power
    else:
        warn_msg = "Resolving power not provided. Setting fwhm_per_pix to None."
        warnings.warn(warn_msg)
        fwhm_per_pix = None
    
    filtered_data = remove_or_replace_bad_values(
        lam=lam_resampled,
        flux=flux_resampled,
        err=err_resampled,
        fwhm_per_pix=fwhm_per_pix,
        lam_bounds=lam_bounds,
        rm_or_replace_outside_lam_bounds=rm_or_replace_outside_lam_bounds,
        rm_or_replace_other_bad_values=rm_or_replace_other_bad_values
    )

    good_pixels = np.where(filtered_data["good_mask"])[0]

    assert len(flux_resampled) == len(err_resampled) == len(lam_resampled) == len(fwhm_per_pix)
    assert np.max(good_pixels) < len(flux_resampled)

    return {
        "lam": filtered_data["lam"],
        "flux": filtered_data["flux"],
        "flux_error": filtered_data["flux_error"],
        "fwhm_per_pix": filtered_data["fwhm_per_pix"],
        "good_pixels": good_pixels,
        "velscale": velscale
    }

def get_sdss_data(
    file_name: str,
    folder_path: Path = const.SDSS_DATA_DIR,
    flux_power_of_10: int = 17,
    lam_medium: tuple[str, str] = ("vacuum", "vacuum"),
    lam_bounds: tuple[float, float] | None = const.TOTAL_LAM_BOUNDS,
    rm_or_replace_outside_lam_bounds: bool | float = True,
    rm_or_replace_other_bad_values: bool | float = np.nan,
    z: float = const.Z_SPEC,
) -> dict[str, np.ndarray]:
    """
    Get relevant data from the SDSS spectrum.

    Parameters
    ----------
    file_name: str
        The name of the file to read.
    folder_path: Path
        The path to the folder containing the file.
    flux_power_of_10: int
        Desired magnitude of flux (in erg s⁻¹ cm⁻² Å⁻¹). Default is 17.
    lam_medium: tuple[str, str]
        The medium of the input wavelength and the desired medium of the output wavelength. "air" or "vacuum".
    lam_bounds: tuple[float, float] | None
        The bounds of the wavelength range to keep (rest-frame).
    rm_or_replace_outside_lam_bounds: bool | float
        What to do with values outside the wavelength bounds. True to remove, float to
        replace with, or False to leave as is. Note: using a float will change the lam
        values.
    rm_or_replace_other_bad_values: bool | float
        What to do with other bad values. True to remove, float to replace with, or False
        to leave as is. Note: using a float will not change any of the lam values.
    z: float
        The redshift of the spectrum. Set to 0 to prevent redshift correction to rest frame.
    perform_log_rebin: bool
        Resamples spectrum to logarithmic spacing if True.
    resolution: float | np.ndarray | None
        The resolution of the spectrum.

    Returns
    -------
    data: dict[str, np.ndarray]
        A dictionary with the keys "lam", "flux", "flux_error", "fwhm_per_pix",
        "velscale", "good_pixels".
    """
    with fits.open(folder_path / file_name) as hdulist:
        image_data = hdulist['COADD'].data


        flux = image_data['flux']
        inv_var = image_data['ivar']
        log_lam_obs = image_data['loglam']
        wdisp = image_data['wdisp']             # Instrumental dispersion of every pixel (pixels units)
        try:
            wresl = image_data['wresl']         # FWHM resolution accounting for instrumental + other dispersion (Angstroms)
            mjd = hdulist['SPALL'].data['MJD'][0]
        except KeyError: # wresl unavailable for 2001 spectrum
            warn_msg = f"Wavelength resolution data not available in {file_name}"
            warnings.warn(warn_msg)
            wresl = None
            mjd = hdulist['SPZLINE'].data['MJD'][0]

    flux *= 10 ** (flux_power_of_10 - 17)
    flux_err = np.sqrt(1 / inv_var)

    lam_obs = 10**log_lam_obs
    lam_rest = lam_obs / (1 + z)

    if np.any(~np.isfinite(lam_rest)):
        raise ValueError("Lam has nans")

    approx_conv_fac = 1.0
    if lam_medium[0] == "air" and lam_medium[1] == "vacuum":
        # use median to get constant scale factor to conserve log spacing
        approx_conv_fac = np.median(util.air_to_vac(lam_rest)/lam_rest)
    elif lam_medium[0] == "vacuum" and lam_medium[1] == "air":
        approx_conv_fac = np.median(util.vac_to_air(lam_rest)/lam_rest)
    elif not set(lam_medium).issubset({"air", "vacuum"}):
        raise ValueError(f"Invalid value for lam_medium: {lam_medium}")
    
    lam_rest *= approx_conv_fac

    d_lam_rest = np.gradient(lam_rest)                              # Size of every pixel in Angstroms
    if wresl is None:
        fwhm_per_pix_unscaled = wdisp * d_lam_rest * const.SIGMA_TO_FWHM     # Resolution FWHM of every pixel, in Angstroms
        fwhm_per_pix = fwhm_per_pix_unscaled * const.WDISP_SCALE_FACTOR # scale to account for non instrumental dispersion
    else:
        fwhm_per_pix = wresl

    filtered_data = remove_or_replace_bad_values(
        lam=lam_rest,
        flux=flux,
        err=flux_err,
        fwhm_per_pix=fwhm_per_pix,
        lam_bounds=lam_bounds,
        rm_or_replace_outside_lam_bounds=rm_or_replace_outside_lam_bounds,
        rm_or_replace_other_bad_values=rm_or_replace_other_bad_values
    )

    velscale = get_velscale(filtered_data["lam"])
    good_pixels = np.where(filtered_data["good_mask"])[0]

    assert len(flux) == len(flux_err) == len(lam_rest) == len(fwhm_per_pix)
    assert np.max(good_pixels) < len(flux)

    return {
        "lam": filtered_data["lam"],
        "flux": filtered_data["flux"],
        "flux_error": filtered_data["flux_error"],
        "fwhm_per_pix": filtered_data["fwhm_per_pix"],
        "good_pixels": good_pixels,
        "velscale": velscale
    }

def get_adjusted_data(
    blur_step: int = 1,
    resample_step: int = 2,
    plot_res_coverage: bool = False,
    plot_as_is: bool = False,
    plot_clipped: bool = False,
    plot_just_blurred: bool = False,
    plot_just_resampled: bool = False,
    plot_resampled_and_blurred: bool = True,
    plot_errors: bool = False,
    as_is_xlim: tuple[float, float] | None = None,
    clipped_xlim: tuple[float, float] | None = None,
    blurred_xlim: tuple[float, float] | None = None,
    resampled_xlim: tuple[float, float] | None = None,
    resampled_and_blurred_xlim: tuple[float, float] | None = None,
    resampled_and_blurred_vlines: dict[str, float] | None = None,
    sdss_folder_path: Path = const.SDSS_DATA_DIR,
    sami_folder_path: Path = const.SAMI_DATA_DIR,
    fname_2001: str = const.FNAME_2001,
    fname_2015_blue: str = const.FNAME_2015_BLUE_3_ARCSEC,
    fname_2015_red: str = const.FNAME_2015_RED_3_ARCSEC,
    fname_2021: str = const.FNAME_2021,
    fname_2022: str = const.FNAME_2022,
    z: float = const.Z_SPEC
) -> dict[str, dict[str, np.ndarray]]:
    """
    Get the wavelength, flux, and flux error arrays for the spectra.

    Parameters
    ----------
    blur_step: int
        The step to blur the spectra. 0 to not blur, 1 to blur before resampling, 2 to blur after resampling.
    resample_step: int
        The step to resample the spectra. 0 to not resample, 1 to resample before blurring, 2 to resample after blurring.
    plot_res_coverage: bool
        If True, plot the resolving power coverage
    plot_as_is: bool
        If True, plot the spectra as is.
    plot_clipped: bool
        If True, plot the spectra after clipping.
    plot_just_blurred: bool
        If True, plot the spectra after blurring.
    plot_just_resampled: bool
        If True, plot the spectra after resampling.
    plot_resampled_and_blurred: bool
        If True, plot the spectra after resampling and blurring.
    plot_errors: bool
        If True, plot the flux errors as ± shaded regions.
    as_is_xlim: tuple[float, float] | None
        The x-axis limits to plot the spectra as is.
    clipped_xlim: tuple[float, float] | None
        The x-axis limits to plot the spectra after clipping.
    blurred_xlim: tuple[float, float] | None
        The x-axis limits to plot the spectra after blurring.
    resampled_xlim: tuple[float, float] | None
        The x-axis limits to plot the spectra after resampling.
    resampled_and_blurred_xlim: tuple[float, float] | None
        The x-axis limits to plot the spectra after resampling and blurring.
    resampled_and_blurred_vlines: dict[str, float] | None
        The vertical lines to plot on the spectra after resampling and blurring.
        Each key is the line label, and each value is the rest wavelength of the line.
    sdss_folder_path: Path
        The path to the folder containing the SDSS spectra.
    sami_folder_path: Path
        The path to the folder containing the SAMI spectra.
    fname_2001: str
        The name of the 2001 SDSS spectrum file.
    fname_2015_blue: str
        The name of the 2015 blue SAMI spectrum file.
    fname_2015_red: str
        The name of the 2015 red SAMI spectrum file.
    fname_2021: str
        The name of the 2021 SDSS spectrum file.
    fname_2022: str
        The name of the 2022 SDSS spectrum file.
    z: float
        The redshift of the spectrum. Set to 0 to prevent redshift correction.

    Returns
    -------
    data: dict[str, dict[str, np.ndarray]]
        A dictionary with the keys "2001", "2015" (or "2015_blue" and "2015_red"),
        "2021", and "2022", each with the keys "lam", "flux", "flux_error", "fwhm_per_pix",
        "good_pixels", and "velscale", unless these values are the same across all epochs,
        in which case they exist as key value pairs in the outer dictionary.
    """
    if not {blur_step, resample_step}.issubset({0, 1, 2}):
        raise ValueError("Blur_step and resample_step must be 0, 1, or 2")
    if np.abs(blur_step - resample_step) > 1:
        raise ValueError("Must have step 1 if step 2 is provided")
    if blur_step == resample_step and blur_step != 0:
        raise ValueError("Blur and resample steps cannot be the same (unless 0)")
    if plot_resampled_and_blurred and np.max((blur_step, resample_step)) != 2:
        raise ValueError("Step 2 must be provided to plot_resampled_and_blurred")
    if plot_just_blurred and blur_step == 0:
        raise ValueError("Blur step must be provided to plot_just_blurred")
    if plot_just_resampled and resample_step == 0:
        raise ValueError("Resample step must be provided to plot_just_resampled")

    plot_as_is = plot_as_is or as_is_xlim is not None
    plot_clipped = plot_clipped or clipped_xlim is not None
    plot_just_blurred = plot_just_blurred or blurred_xlim is not None
    plot_just_resampled = plot_just_resampled or resampled_xlim is not None
    plot_resampled_and_blurred = plot_resampled_and_blurred or resampled_and_blurred_xlim is not None
    
    input_data01 = get_sdss_data(fname_2001, folder_path=sdss_folder_path, z=z)
    input_data15_blue = get_sami_data(fname_2015_blue, folder_path=sami_folder_path, z=z, resolving_power=const.RES_15_BLUE, perform_log_rebin=True)
    input_data15_red = get_sami_data(fname_2015_red, folder_path=sami_folder_path, z=z, resolving_power=const.RES_15_RED, perform_log_rebin=True)
    input_data21 = get_sdss_data(fname_2021, folder_path=sdss_folder_path, z=z)
    input_data22 = get_sdss_data(fname_2022, folder_path=sdss_folder_path, z=z)

    lam01, lam15_blue, lam15_red, lam21, lam22 = (
        input_data01["lam"], input_data15_blue["lam"],
        input_data15_red["lam"], input_data21["lam"], input_data22["lam"]
    )
    flux01, flux15_blue, flux15_red, flux21, flux22 = (
        input_data01["flux"], input_data15_blue["flux"],
        input_data15_red["flux"], input_data21["flux"], input_data22["flux"]
    )
    err01, err15_blue, err15_red, err21, err22 = (
        input_data01["flux_error"], input_data15_blue["flux_error"],
        input_data15_red["flux_error"], input_data21["flux_error"], input_data22["flux_error"]
    )
    fwhm01, fwhm15_blue, fwhm15_red, fwhm21, fwhm22 = (
        input_data01["fwhm_per_pix"], input_data15_blue["fwhm_per_pix"],
        input_data15_red["fwhm_per_pix"], input_data21["fwhm_per_pix"], input_data22["fwhm_per_pix"]
    )
    good_pixels_01, good_pixels_15_blue, good_pixels_15_red, good_pixels_21, good_pixels_22 = (
        input_data01["good_pixels"], input_data15_blue["good_pixels"],
        input_data15_red["good_pixels"], input_data21["good_pixels"], input_data22["good_pixels"]
    )
    velscale01, velscale15_blue, velscale15_red, velscale21, velscale22 = (
        input_data01["velscale"], input_data15_blue["velscale"],
        input_data15_red["velscale"], input_data21["velscale"], input_data22["velscale"]
    )

    target_sdss_len = len(lam01)
    for sdss_arr in [lam01, lam21, lam22, flux01, flux21, flux22, err01, err21, err22, fwhm01, fwhm21, fwhm22]:
        if len(sdss_arr) != target_sdss_len:
            raise ValueError("Mismatch in number of SDSS wavelength points. Try adjusting TOTAL_LAM_BOUNDS.")
    target_sami_blue_len = len(lam15_blue)
    for sami_blue_arr in [lam15_blue, flux15_blue, err15_blue, fwhm15_blue]:
        if len(sami_blue_arr) != target_sami_blue_len:
            raise ValueError("Mismatch in number of SAMI blue wavelength points")
    target_sami_red_len = len(lam15_red)
    for sami_red_arr in [lam15_red, flux15_red, err15_red, fwhm15_red]:
        if len(sami_red_arr) != target_sami_red_len:
            raise ValueError("Mismatch in number of SAMI red wavelength points")

    resolving_power01 = lam01 / fwhm01
    resolving_power15_blue = const.RES_15_BLUE
    resolving_power15_red = const.RES_15_RED
    resolving_power21 = lam21 / fwhm21
    resolving_power22 = lam22 / fwhm22

    min_resolving_power = np.min((
        resolving_power01, resolving_power21,
        # np.full_like(lam01, resolving_power15_blue), # we already know these have better resolving powers
        # np.full_like(lam01, resolving_power15_red),  # we already know these have better resolving powers
        resolving_power22
    ), axis=0)

    if plot_res_coverage:
        plot_min_res(
            lam_sdss=lam01,
            lam15_blue=lam15_blue,
            lam15_red=lam15_red,
            res_min=min_resolving_power,
            res01=resolving_power01,
            res21=resolving_power21,
            res22=resolving_power22,
            plot_RES_15_RED=False
        )

    if plot_as_is:
        plot_spectra(
            lam01,
            (lam15_blue, lam15_red),
            lam21,
            lam22,
            flux01,
            (flux15_blue, flux15_red),
            flux21,
            flux22,
            title="Spectra from 2001 to 2022 (as is)",
            x_bounds=as_is_xlim
        )
    
    data01 = {
        "lam": lam01,
        "flux": flux01,
        "flux_error": err01,
        "fwhm_per_pix": fwhm01,
        "good_pixels": good_pixels_01,
        "velscale": velscale01
    }
    data15_blue = {
        "lam": lam15_blue,
        "flux": flux15_blue,
        "flux_error": err15_blue,
        "fwhm_per_pix": fwhm15_blue,
        "good_pixels": good_pixels_15_blue,
        "velscale": velscale15_blue
    }
    data15_red = {
        "lam": lam15_red,
        "flux": flux15_red,
        "flux_error": err15_red,
        "fwhm_per_pix": fwhm15_red,
        "good_pixels": good_pixels_15_red,
        "velscale": velscale15_red
    }
    data21 = {
        "lam": lam21,
        "flux": flux21,
        "flux_error": err21,
        "fwhm_per_pix": fwhm21,
        "good_pixels": good_pixels_21,
        "velscale": velscale21
    }
    data22 = {
        "lam": lam22,
        "flux": flux22,
        "flux_error": err22,
        "fwhm_per_pix": fwhm22,
        "good_pixels": good_pixels_22,
        "velscale": velscale22
    }
    all_data = {
        "2001": data01,
        "2015_blue": data15_blue,
        "2015_red": data15_red,
        "2021": data21,
        "2022": data22
    }

    if blur_step == 0 and resample_step == 0:
        # don't blur or resample
        assert_lengths_match(all_data)
        return all_data

    if blur_step == 1:
        min_ssd_lam = np.min(lam01)
        lam15_blue_clipped, flux15_blue_clipped, err15_blue_clipped, good_pixels_15_blue_clipped = clip_sami_blue_edge(
            flux15_blue, lam15_blue, err15_blue, min_ssd_lam, good_pixels_15_blue # we know fwhm15_blue is not None because RES_15_BLUE was passed in to get_sami_data
        )
        velscale15_blue_clipped = get_velscale(lam15_blue_clipped)

        if plot_clipped:
            plot_spectra(
                lam01,
                (lam15_blue_clipped, lam15_red),
                lam21,
                lam22,
                flux01,
                (flux15_blue_clipped, flux15_red),
                flux21,
                flux22,
                plot_errors=plot_errors,
                flux01_err=err01,
                flux15_err=(err15_blue_clipped, err15_red),
                flux21_err=err21,
                flux22_err=err22,
                title=f"Spectra from 2001 to 2022 (SAMI clipped to ~{min_ssd_lam:.0f} Å)",
                x_bounds=clipped_xlim
            )
        
        flux01_blurred, fwhm01_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power01, lam01, lam01, flux01)
        flux21_blurred, fwhm21_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power21, lam01, lam21, flux21)
        flux22_blurred, fwhm22_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power22, lam01, lam22, flux22)
        flux15_red_blurred, fwhm15_red_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power15_red, lam01, lam15_red, flux15_red)
        flux15_blue_blurred, fwhm15_blue_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power15_blue, lam01, lam15_blue_clipped, flux15_blue_clipped)


        if plot_just_blurred:
            plot_spectra(
                lam01,
                (lam15_blue_clipped, lam15_red),
                lam21,
                lam22,
                flux01_blurred,
                (flux15_blue_blurred, flux15_red_blurred),
                flux21_blurred,
                flux22_blurred,
                plot_errors=plot_errors,
                flux01_err=err01,
                flux15_err=(err15_blue_clipped, err15_red),
                flux21_err=err21,
                flux22_err=err22,
                title="Spectra from 2001 to 2022 (blurred before resampling)",
                x_bounds=blurred_xlim
            )
                
        if resample_step == 0:            
            all_data["2015_blue"]["good_pixels"] = good_pixels_15_blue_clipped
            all_data["2015_blue"]["velscale"] = velscale15_blue_clipped
            
            all_data["2001"]["flux"] = flux01_blurred
            all_data["2015_blue"]["flux"] = flux15_blue_blurred
            all_data["2015_red"]["flux"] = flux15_red_blurred
            all_data["2021"]["flux"] = flux21_blurred
            all_data["2022"]["flux"] = flux22_blurred

            all_data["2001"]["fwhm_per_pix"] = fwhm01_blurred
            all_data["2015_blue"]["fwhm_per_pix"] = fwhm15_blue_blurred
            all_data["2015_red"]["fwhm_per_pix"] = fwhm15_red_blurred
            all_data["2021"]["fwhm_per_pix"] = fwhm21_blurred
            all_data["2022"]["fwhm_per_pix"] = fwhm22_blurred

            assert_lengths_match(all_data)
            return all_data

        # else resample_step == 2

        flux01_blurred_resampled, err01_resampled = flux01_blurred, err01

        flux15_blue_blurred_resampled, err15_blue_resampled = spectres.spectres(
            lam01, lam15_blue_clipped, flux15_blue_blurred, spec_errs=err15_blue_clipped, fill=np.nan
        )
        flux15_red_blurred_resampled, err15_red_resampled = spectres.spectres(
            lam01, lam15_red, flux15_red_blurred, spec_errs=err15_red, fill=np.nan
        )
        flux21_blurred_resampled, err21_resampled = spectres.spectres(
            lam01, lam21, flux21_blurred, spec_errs=err21, fill=np.nan
        )
        flux22_blurred_resampled, err22_resampled = spectres.spectres(
            lam01, lam22, flux22_blurred, spec_errs=err22, fill=np.nan
        )

        flux15_blurred_resampled, err15_resampled = combine_sami_vals(
            [flux15_blue_blurred_resampled, err15_blue_resampled],
            [flux15_red_blurred_resampled, err15_red_resampled],
        )


        all_data.pop("2015_blue")
        all_data.pop("2015_red")


        all_data["2015"] = {
            "flux": flux15_blurred_resampled,
            "flux_error": err15_resampled,
            "good_pixels": get_good_pixels(flux15_blurred_resampled, err15_resampled)
        }

        fwhm_per_pix_all = lam01 / min_resolving_power #TODO: check that no resampling is required here
        velscale_all = get_velscale(lam01)

        all_data["lam"] = lam01
        all_data["fwhm_per_pix"] = fwhm_per_pix_all
        all_data["velscale"] = velscale_all

        set_all_values(
            all_data, [
                ("lam", lam01),
                ("fwhm_per_pix", fwhm_per_pix_all),
                ("velscale", velscale_all)
            ]
        )

        all_data["2001"]["flux"] = flux01_blurred_resampled
        all_data["2001"]["flux_error"] = err01_resampled
        all_data["2001"]["good_pixels"] = get_good_pixels(flux01_blurred_resampled, err01_resampled)

        all_data["2021"]["flux"] = flux21_blurred_resampled
        all_data["2021"]["flux_error"] = err21_resampled
        all_data["2021"]["good_pixels"] = get_good_pixels(flux21_blurred_resampled, err21_resampled)
        
        all_data["2022"]["flux"] = flux22_blurred_resampled
        all_data["2022"]["flux_error"] = err22_resampled
        all_data["2022"]["good_pixels"] = get_good_pixels(flux22_blurred_resampled, err22_resampled)

        resampled_and_blurred_title = "Spectra from 2001 to 2022 (blurred then resampled to 2001 grid)"

    elif resample_step == 1:

        flux01_resampled, err01_resampled = flux01, err01

        flux15_blue_resampled, err15_blue_resampled = spectres.spectres(
            lam01, lam15_blue, flux15_blue, spec_errs=err15_blue, fill=np.nan
        )
        flux15_red_resampled, err15_red_resampled = spectres.spectres(
            lam01, lam15_red, flux15_red, spec_errs=err15_red, fill=np.nan
        )
        flux21_resampled, err21_resampled = spectres.spectres(
            lam01, lam21, flux21, spec_errs=err21, fill=np.nan
        )
        flux22_resampled, err22_resampled = spectres.spectres(
            lam01, lam22, flux22, spec_errs=err22, fill=np.nan
        )

        flux15_resampled, err15_resampled = combine_sami_vals(
            [flux15_blue_resampled, err15_blue_resampled],
            [flux15_red_resampled, err15_red_resampled],
        )

        if plot_just_resampled:
            plot_spectra(
                lam01,
                (lam01, lam01),
                lam01,
                lam01,
                flux01_resampled,
                (flux15_blue_resampled, flux15_red_resampled),
                flux21_resampled,
                flux22_resampled,
                plot_errors=plot_errors,
                flux01_err=err01_resampled,
                flux15_err=(err15_blue_resampled, err15_red_resampled),
                flux21_err=err21_resampled,
                flux22_err=err22_resampled,
                title="Spectra from 2001 to 2022 (resampled to 2001 grid)",
                x_bounds=resampled_xlim
            )

        all_data.pop("2015_blue")
        all_data.pop("2015_red")

        all_data["2015"] = {
            "flux": flux15_resampled,
            "flux_error": err15_resampled,
            "good_pixels": get_good_pixels(flux15_resampled, err15_resampled),
        }

        velscale_all = get_velscale(lam01)
        all_data["velscale"] = velscale_all
        all_data["lam"] = lam01

        set_all_values(all_data, [("lam", lam01), ("velscale", velscale_all)])

        all_data["2001"]["flux"] = flux01_resampled
        all_data["2001"]["flux_error"] = err01_resampled
        all_data["2001"]["good_pixels"] = get_good_pixels(flux01_resampled, err01_resampled)

        all_data["2021"]["flux"] = flux21_resampled
        all_data["2021"]["flux_error"] = err21_resampled
        all_data["2021"]["good_pixels"] = get_good_pixels(flux21_resampled, err21_resampled)
        
        all_data["2022"]["flux"] = flux22_resampled
        all_data["2022"]["flux_error"] = err22_resampled
        all_data["2022"]["good_pixels"] = get_good_pixels(flux22_resampled, err22_resampled)

        if blur_step == 0:
            fwhm01_resampled = fwhm01
            fwhm15_blue_resampled = spectres.spectres(
                lam01, lam15_blue, fwhm15_blue, fill=np.nan
            )
            fwhm15_red_resampled = spectres.spectres(
                lam01, lam15_red, fwhm15_red, fill=np.nan
            )
            fwhm15_resampled = combine_sami_vals([fwhm15_blue_resampled], [fwhm15_red_resampled])[0]
            fwhm21_resampled = spectres.spectres(
                lam01, lam21, fwhm21, fill=np.nan
            )
            fwhm22_resampled = spectres.spectres(
                lam01, lam22, fwhm22, fill=np.nan
            )
            all_data["2001"]["fwhm_per_pix"] = fwhm01_resampled
            all_data["2015"]["fwhm_per_pix"] = fwhm15_resampled
            all_data["2021"]["fwhm_per_pix"] = fwhm21_resampled
            all_data["2022"]["fwhm_per_pix"] = fwhm22_resampled

            assert_lengths_match(all_data)
            return all_data

        flux01_resampled_blurred, fwhm01_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power01, lam01, flux01_resampled)
        flux21_resampled_blurred, fwhm21_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power21, lam01, flux21_resampled)
        flux22_resampled_blurred, fwhm22_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power22, lam01, flux22_resampled)
        flux15_red_resampled_blurred, fwhm15_red_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power15_red, lam01, flux15_red_resampled)
        flux15_blue_resampled_blurred, fwhm15_blue_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power15_blue, lam01, flux15_blue_resampled)

        flux15_resampled_blurred, err15_resampled, fwhm15_resampled_blurred = combine_sami_vals(
            [flux15_blue_resampled_blurred, err15_blue_resampled, fwhm15_blue_resampled_blurred],
            [flux15_red_resampled_blurred, err15_red_resampled, fwhm15_red_resampled_blurred],
        )

        # fwhm should all be very similar now
        mean_fwhm_after_resampling_and_blurring = np.nanmean(
            [fwhm01_resampled_blurred, fwhm15_resampled_blurred,
            fwhm21_resampled_blurred, fwhm22_resampled_blurred],
            axis=0
        )

        all_data["fwhm_per_pix"] = mean_fwhm_after_resampling_and_blurring

        all_data["2001"]["fwhm_per_pix"] = fwhm01_resampled_blurred
        all_data["2015"]["fwhm_per_pix"] = fwhm15_resampled_blurred
        all_data["2021"]["fwhm_per_pix"] = fwhm21_resampled_blurred
        all_data["2022"]["fwhm_per_pix"] = fwhm22_resampled_blurred

        all_data["2001"]["flux"] = flux01_resampled_blurred
        all_data["2015"]["flux"] = flux15_resampled_blurred
        all_data["2021"]["flux"] = flux21_resampled_blurred
        all_data["2022"]["flux"] = flux22_resampled_blurred

        resampled_and_blurred_title = "Spectra from 2001 to 2022 (resampled to 2001 grid then blurred)"
    # else:
        # this should never happen

    if plot_resampled_and_blurred:
        plot_spectra(
            all_data["lam"],
            all_data["lam"],
            all_data["lam"],
            all_data["lam"],
            all_data["2001"]["flux"],
            all_data["2015"]["flux"],
            all_data["2021"]["flux"],
            all_data["2022"]["flux"],
            plot_errors=plot_errors,
            flux01_err=all_data["2001"]["flux_error"],
            flux15_err=all_data["2015"]["flux_error"],
            flux21_err=all_data["2021"]["flux_error"],
            flux22_err=all_data["2022"]["flux_error"],
            title=resampled_and_blurred_title,
            ions=resampled_and_blurred_vlines,
            x_bounds=resampled_and_blurred_xlim
        )

    assert_lengths_match(all_data)
    return all_data