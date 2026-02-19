import numpy as np
import spectres

import astropy.io.fits as fits
import ppxf.ppxf_util as util
from PyAstronomy import pyasl

#TODO: remove testing
import matplotlib.pyplot as plt
#

from pathlib import Path
import warnings

from . import constants as const
from .adjust_calibration import gaussian_blur_before_resampling, gaussian_blur_after_resampling, clip_sami_blue_edge
from .helpers import remove_or_replace_bad_values
from .plotting import plot_min_res, plot_spectra

def get_sami_data(
    file_name: str,
    folder_path: Path = const.SAMI_DATA_DIR,
    flux_power_of_10: int = 17,
    lam_medium: tuple[str, str] = ("air", "vacuum"),
    lam_bounds: tuple[float, float] | None = const.TOTAL_LAM_BOUNDS,
    rm_or_replace_outside_lam_bounds: bool | float = True,
    rm_or_replace_other_bad_values: bool | float = np.nan,
    z: float = const.Z_SPEC, # use 0 to get observed frame data
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
        A dictionary with the keys "lam", "flux", "flux_err", "fwhm_per_pix", "median_flux",
        "velscale", "good_mask".
    """
    with fits.open(folder_path / file_name) as hdulist:
        header = hdulist["PRIMARY"].header
        coord_ref_val_axis_1 = header['CRVAL1']     # coordinate reference value for axis 1
        coord_ref_pix_axis_1 = header['CRPIX1']     # index location of reference value
        coord_delta_axis_1 = header['CDELT1']       # pixel width in Angstroms
        num_pixels_axis_1 = header['NAXIS1']        # number of pixels

        lam_indices = np.arange(1, num_pixels_axis_1+0.5)                           # (fits coordinates are 1-indexed)
        lam_0 = coord_ref_val_axis_1 - coord_ref_pix_axis_1 * coord_delta_axis_1    # λ_0 = λ_ref - num_pixels_from_start * dλ        
        lam_obs = lam_0 + lam_indices * coord_delta_axis_1                            # λ_obs = λ_0 + index * dλ
        lam_rest = lam_obs / (1 + z)  # lam_rest is actually lam_obs if you set z=0

        if np.any(~np.isfinite(lam_rest)):
            # In theory this should never happen
            raise ValueError("ERROR: lam has nans")

        flux = hdulist["PRIMARY"].data
        var = hdulist["VARIANCE"].data
    
    err = np.array(np.sqrt(var), dtype=float)

    if lam_medium[0] == "air" and lam_medium[1] == "vacuum":
        lam_rest =util.air_to_vac(lam_rest)
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
        warn_msg = "Resolving power data not provided. Setting fwhm_per_pix to None."
        warnings.warn(warn_msg)
        fwhm_per_pix = None
    
    median_flux = np.nanmedian(flux_resampled)
    filtered_data = remove_or_replace_bad_values(
        lam=lam_resampled,
        flux=flux_resampled,
        err=err_resampled,
        fwhm_per_pix=fwhm_per_pix,
        lam_bounds=lam_bounds,
        rm_or_replace_outside_lam_bounds=rm_or_replace_outside_lam_bounds,
        rm_or_replace_other_bad_values=rm_or_replace_other_bad_values
    )

    return {
        "lam": filtered_data["lam"],
        "flux": filtered_data["flux"],
        "flux_err": filtered_data["flux_err"],
        "fwhm_per_pix": filtered_data["fwhm_per_pix"],
        "good_pixels": np.where(filtered_data["good_mask"])[0],
        "median_flux": median_flux,
        "velscale": velscale
    }

def get_sdss_data(
    file_name: str,
    folder_path: Path = const.SDSS_DATA_DIR,
    flux_power_of_10: int = 17,
    lam_medium: tuple[str, str] = ("air", "vacuum"),
    lam_bounds: tuple[float, float] | None = const.TOTAL_LAM_BOUNDS,
    rm_or_replace_outside_lam_bounds: bool | float = True,
    rm_or_replace_other_bad_values: bool | float = np.nan,
    z: float = const.Z_SPEC, # use 0 to get observed frame data
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
        A dictionary with the keys "lam", "flux", "flux_err", "fwhm_per_pix", "median_flux",
        "velscale", "good_mask".
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
    median_flux = np.nanmedian(flux)
    flux_err = np.sqrt(1 / inv_var)

    lam_obs = 10**log_lam_obs
    lam_rest = lam_obs / (1 + z)

    ln_lam_obs = log_lam_obs * np.log(10)
    ln_lam_rest = ln_lam_obs - np.log(1 + z)

    if np.any(~np.isfinite(lam_rest)):
        raise ValueError("ERROR: lam has nans")

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

    d_ln_lam_rest = (ln_lam_rest[-1] - ln_lam_rest[0])/(len(ln_lam_rest) - 1) # average spacing between ln wavelengths
    velscale = const.C_KM_S*d_ln_lam_rest  

    filtered_data = remove_or_replace_bad_values(
        lam=lam_rest,
        flux=flux,
        err=flux_err,
        fwhm_per_pix=fwhm_per_pix,
        lam_bounds=lam_bounds,
        rm_or_replace_outside_lam_bounds=rm_or_replace_outside_lam_bounds,
        rm_or_replace_other_bad_values=rm_or_replace_other_bad_values
    )

    return {
        "lam": filtered_data["lam"],
        "flux": filtered_data["flux"],
        "flux_err": filtered_data["flux_err"],
        "fwhm_per_pix": filtered_data["fwhm_per_pix"],
        "good_pixels": np.where(filtered_data["good_mask"])[0],
        "median_flux": median_flux,
        "velscale": velscale
    }


# def get_sami_lam_flux_err(
#     fname: str,
#     folder_name: str = const.SAMI_FOLDER_NAME,
#     filter_bad_values: bool = True,
#     interpolate_bad_values: bool = False,
#     lam_bounds: tuple[float, float] | None = const.TOTAL_LAM_BOUNDS,
#     flux_power_of_10: int = 17,
#     lam_in_vacuum: bool = True,
#     doareacorr: bool = False,
#     bugfix: bool = True
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

#     with fits.open(folder_name + fname) as hdulist:
#         header = hdulist["PRIMARY"].header
#         crval1=header['CRVAL1']
#         cdelt1=header['CDELT1']
#         crpix1=header['CRPIX1']
#         naxis1=header['NAXIS1'] 
#         x=np.arange(naxis1)+1
#         L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
#         lam = L0+x*cdelt1
        
#         flux_uncorrected = hdulist["PRIMARY"].data
#         var = hdulist["VARIANCE"].data
#         err = np.array(np.sqrt(var), dtype=float)

#         #TD: remove
#         if doareacorr:
#             if bugfix:
#                 areacorr=areacorr/2.0
#             areacorr = header['AREACORR']
#             flux = flux_uncorrected * areacorr *10 ** (flux_power_of_10 - 16)
#         else:
#             flux = flux_uncorrected * 10 ** (flux_power_of_10 - 16) #TODO: apply this to errors as well (in *all* functions)
#         #
        
#     # flux = flux_uncorrected * 10 ** (flux_power_of_10 - 16)

#     if lam_in_vacuum:
#         lam = pyasl.airtovac2(lam)

#     lam_mask = np.isfinite(lam)
#     if np.any(~lam_mask):
#         warn_msg = f"{np.sum(~lam_mask)} values of lam are not finite. These will be ignored."
#         warnings.warn(warn_msg)     #TODO: raise error instead
#         lam_valid = lam[lam_mask]
#         flux_valid = flux[lam_mask]
#         err_valid = err[lam_mask]
#     else:
#         lam_valid = lam
#         flux_valid = flux
#         err_valid = err

#     if not filter_bad_values:
#         # if np.any(~np.isfinite(var) | (var <= 0)):
#             # raise ValueError("ERROR: bad values in variance array. Cannot take the square root")
#         return flux, lam_valid, err

#     if lam_bounds is None:
#         not_in_lam_bounds = np.zeros_like(lam_valid, dtype=bool)
#     else:
#         not_in_lam_bounds = (lam_valid < lam_bounds[0]) | (lam_valid > lam_bounds[1])
#     bad_mask = (
#         # ~np.isfinite(lam) | #TODO: remove?
#         not_in_lam_bounds |
#         ~np.isfinite(flux_valid) | (flux_valid > const.MAX_FLUX) | (flux_valid < const.MIN_FLUX) |
#         ~np.isfinite(err_valid) | (err_valid <= 0) | (err_valid > const.MAX_FLUX)
#     )
#     good_mask = ~bad_mask

#     if interpolate_bad_values:
#         flux_interpolated = np.interp(lam_valid, lam_valid[good_mask], flux_valid[good_mask])
#         err_interpolated = np.interp(lam_valid, lam_valid[good_mask], err_valid[good_mask])
#         return lam_valid, flux_interpolated, err_interpolated
#     else:
#         lam_masked = lam_valid[good_mask]
#         flux_masked = flux_valid[good_mask]
#         err_masked = err_valid[good_mask]
#         return lam_masked, flux_masked, err_masked


# def get_sdss_lam_flux_err(
#     fname: str,
#     folder_name: str = const.SDSS_FOLDER_NAME,
#     filter_bad_values: bool = True,
#     interpolate_bad_values: bool = False,
#     lam_bounds: tuple[float, float] | None = const.TOTAL_LAM_BOUNDS,
#     flux_power_of_10: int = 17,
#     get_other_data: bool = False
# ) -> (
#     tuple[np.ndarray, np.ndarray, np.ndarray] |
#     tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# ):
#     with fits.open(folder_name + fname) as hdulist:
#         # read spectrum from COADD extension:
#         # spec_table = hdulist["COADD"].data
#         spec_table = hdulist[1].data

#         flux = spec_table['flux']
#         flux *= 10 ** (flux_power_of_10 - 17)

#         loglam = spec_table['loglam']
#         # SDSS spectra are in log wavelength bins, to convert to linear:
#         lam = 10.0**loglam

#         # inverse variance
#         ivar = spec_table['ivar']
#         err = np.array(np.sqrt(1 / ivar), dtype=float)

#         if get_other_data:
#             header_info = hdulist[0].header

#             try:
#                 # wavelength resolution
#                 wresl = spec_table['wresl']
#             except KeyError:
#                 warn_msg = f"Wavelength resolution data not available in {fname}"
#                 warnings.warn(warn_msg)
#                 wresl = None
            
#             plug_ra = header_info['plug_ra']
#             plug_dec = header_info['plug_dec']
#             plateid = header_info['plateid']
#             mjd = header_info['mjd']
#             fiberid = header_info['fiberid']
#             z = hdulist[2].data['z'][0]

#             other_data = {
#                 "wresl": wresl,
#                 "plug_ra": plug_ra,
#                 "plug_dec": plug_dec,
#                 "plateid": plateid,
#                 "mjd": mjd,
#                 "fiberid": fiberid,
#                 "z": z
#             }


#     lam_mask = np.isfinite(lam)
#     if np.any(~lam_mask):
#         warn_msg = f"{np.sum(~lam_mask)} values of lam are not finite. These will be ignored."
#         warnings.warn(warn_msg)     #TODO: raise error instead
#         lam_valid = lam[lam_mask]
#         flux_valid = flux[lam_mask]
#         err_valid = err[lam_mask]
#     else:
#         lam_valid = lam
#         flux_valid = flux
#         err_valid = err

#     # return flux, lam, 1 / np.sqrt(ivar)
#     if filter_bad_values: 
#         if lam_bounds is None:
#             not_in_lam_bounds = np.zeros_like(lam_valid, dtype=bool)
#         else:
#             not_in_lam_bounds = (lam_valid < lam_bounds[0]) | (lam_valid > lam_bounds[1])
#         bad_mask = (
#             not_in_lam_bounds |
#             ~np.isfinite(flux_valid) | (flux_valid > const.MAX_FLUX) | (flux_valid < const.MIN_FLUX) |
#             ~np.isfinite(err_valid) | (err_valid <= 0) | (err_valid > const.MAX_FLUX)
#         )
#         good_mask = ~bad_mask
#         if interpolate_bad_values:
#             # lam = np.interp(lam, lam[good_mask], lam[good_mask])
#             flux_interpolated = np.interp(lam_valid, lam_valid[good_mask], flux_valid[good_mask])
#             err_interpolated = np.interp(lam_valid, lam_valid[good_mask], err_valid[good_mask])
#             lfe = lam_valid, flux_interpolated, err_interpolated
#         else:
#             lam_masked = lam_valid[good_mask]
#             flux_masked = flux_valid[good_mask]
#             err_masked = err_valid[good_mask]
#             lfe = lam_masked, flux_masked, err_masked

#     if get_other_data:
#         return *lfe, other_data
#     else:
#         return lfe


# def sami_read_apspec(
#     filename: str,
#     extname: str,
#     var_extname: str,
#     doareacorr: bool = False,
#     bugfix: bool = True
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     hdulist = fits.open(filename)
#     sami_flux = hdulist[extname].data
#     header = hdulist[extname].header
#     crval1=header['CRVAL1']
#     cdelt1=header['CDELT1']
#     crpix1=header['CRPIX1']
#     naxis1=header['NAXIS1'] 
#     x=np.arange(naxis1)+1
#     L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
#     sami_lam = L0+x*cdelt1

#     if (doareacorr):
#         # fix bug in ap spec code:
#         if (bugfix):
#             areacorr=areacorr/2.0
#         areacorr = header['AREACORR']
#         sami_flux = sami_flux * areacorr

#     #TD: remove testing
#     # print(header['BUNIT'])
#     #

#     #convert SAMI (wavelength in air) to wavelength in vacuum By default, the conversion specified by Ciddor 1996 are used.
#     samivac_lam = pyasl.airtovac2(sami_lam)
    
#     sami_flux *= 10 # flux unit conversion from 10e16 to 10e17 to match sdss

    
#     sami_var = hdulist[var_extname].data #extract the variance array
#     hdulist.close()
#     return sami_flux, samivac_lam, sami_var

# def sdss_read(
#     infile: str, return_wresl: bool = True
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

#     # get basic info on file:
    
#     # opening file
#     hdulist = fits.open(infile)
#     hdr0 = fits.getheader(infile)
#     hdr1 = hdulist['COADD'].header

#     #TD: remove testing
#     # try:
#     #     print(hdr0['BUNIT'])
#     # except:
#     #     print(f"no BUNIT in file {infile}")
#     #

#     # read spectrum from COADD extension:
#     sdss_spec_table = hdulist['COADD'].data
#     sdss_flux = sdss_spec_table['flux']
#     sdss_loglam = sdss_spec_table['loglam']

#     # SDSS spectra are in log wavelength bins, to convert to linear:
#     sdss_lam = 10.0**sdss_loglam
#     # inverse variance
#     sdss_ivar = sdss_spec_table['ivar']

#     # close fits file:
#     hdulist.close()

#     # define parts of spectrum where data is likely to be good:
#     idx = np.where((sdss_lam>const.TOTAL_LAM_BOUNDS[0]) & (sdss_lam<const.TOTAL_LAM_BOUNDS[1]))
#     sdss_flux = sdss_flux[idx]
#     sdss_lam = sdss_lam[idx]
#     sdss_ivar = sdss_ivar[idx]
#     if return_wresl:
#         wresl = sdss_spec_table['wresl']
#         wresl = wresl[idx]
#         return sdss_flux, sdss_lam, sdss_ivar, wresl
#     else:
#         return sdss_flux, sdss_lam, sdss_ivar


#TODO: return all of this (resampled/blurred if necessary - maybe use dictionary instead?)
#     data["flux"],
#     data["flux_err"],
#     data["median_flux"],
#     data["lam"],
#     data["fwhm_per_pix"],
#     data["good_pixels"],
#     data["velscale"]

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
) -> tuple[np.ndarray, tuple[
        np.ndarray, np.ndarray
    ], tuple[
        np.ndarray, np.ndarray
    ], tuple[
        np.ndarray, np.ndarray
    ], tuple[
        np.ndarray, np.ndarray
    ]
]:
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
    data: tuple[np.ndarray, tuple[
            np.ndarray, np.ndarray
        ], tuple[
            np.ndarray, np.ndarray
        ], tuple[
            np.ndarray, np.ndarray
        ], tuple[
            np.ndarray, np.ndarray
        ]
    ]
    First array is the wavelength array, the rest are the flux and flux error pairs for
    each epoch (2001, 2015, 2021, 2022). 
    """
    if not {blur_step, resample_step}.issubset({0, 1, 2}):
        raise ValueError("ERROR: blur_step and resample_step must be 0, 1, or 2")
    if np.abs(blur_step - resample_step) > 1:
        raise ValueError("ERROR: must have step 1 if step 2 is provided")
    if blur_step == resample_step and blur_step != 0:
        raise ValueError("ERROR: blur and resample steps cannot be the same (unless 0)")
    if plot_resampled_and_blurred and np.max((blur_step, resample_step)) != 2:
        raise ValueError("ERROR: step 2 must be provided to plot_resampled_and_blurred")
    if plot_just_blurred and blur_step == 0:
        raise ValueError("ERROR: blur step must be provided to plot_just_blurred")
    if plot_just_resampled and resample_step == 0:
        raise ValueError("ERROR: resample step must be provided to plot_just_resampled")

    plot_as_is = plot_as_is or as_is_xlim is not None
    plot_clipped = plot_clipped or clipped_xlim is not None
    plot_just_blurred = plot_just_blurred or blurred_xlim is not None
    plot_just_resampled = plot_just_resampled or resampled_xlim is not None
    plot_resampled_and_blurred = plot_resampled_and_blurred or resampled_and_blurred_xlim is not None
    
    input_data01 = get_sdss_data(fname_2001, folder_path=sdss_folder_path, z=z)
    input_data15_blue = get_sami_data(fname_2015_blue, folder_path=sami_folder_path, z=z, resolving_power=const.RES_15_BLUE)
    input_data15_red = get_sami_data(fname_2015_red, folder_path=sami_folder_path, z=z, resolving_power=const.RES_15_RED)
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
        input_data01["flux_err"], input_data15_blue["flux_err"],
        input_data15_red["flux_err"], input_data21["flux_err"], input_data22["flux_err"]
    )
    fwhm01, fwhm15_blue, fwhm15_red, fwhm21, fwhm22 = (
        input_data01["fwhm_per_pix"], input_data15_blue["fwhm_per_pix"],
        input_data15_red["fwhm_per_pix"], input_data21["fwhm_per_pix"], input_data22["fwhm_per_pix"]
    )

    target_sdss_len = len(lam01)
    for sdss_arr in [lam01, lam21, lam22, flux01, flux21, flux22, err01, err21, err22, fwhm01, fwhm21, fwhm22]:
        if len(sdss_arr) != target_sdss_len:
            raise ValueError("ERROR: mismatch in number of wavelength points. Try adjusting TOTAL_LAM_BOUNDS.")
    target_sami_blue_len = len(lam15_blue)
    for sami_blue_arr in [lam15_blue, flux15_blue, err15_blue, fwhm15_blue]:
        if len(sami_blue_arr) != target_sami_blue_len:
            raise ValueError("ERROR: mismatch in number of wavelength points")
    target_sami_red_len = len(lam15_red)
    for sami_red_arr in [lam15_red, flux15_red, err15_red, fwhm15_red]:
        if len(sami_red_arr) != target_sami_red_len:
            raise ValueError("ERROR: mismatch in number of wavelength points")

    resolving_power01 = lam01 / fwhm01
    resolving_power15_blue = const.RES_15_BLUE
    resolving_power15_red = const.RES_15_RED
    resolving_power21 = lam21 / fwhm21
    resolving_power22 = lam22 / fwhm22

    min_resolving_power = np.min((
        resolving_power01, resolving_power21,
        # np.full_like(lam01, resolving_power15_blue), # we already know these have better resolving powers
        # np.full_like(lam01, resolving_power15_red),
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
    
    if blur_step == 0 and resample_step == 0:
    # if return_as_is:
        return (
            (lam01, flux01, err01),
            (lam15_blue, flux15_blue, err15_blue),
            (lam15_red, flux15_red, err15_red),
            (lam21, flux21, err21),
            (lam22, flux22, err22)
        )

    if blur_step == 1:
    # if blur_before_resampling:
        min_ssd_lam = np.min(lam01)
        flux15_blue_clipped, lam15_blue_clipped, err15_blue_clipped = clip_sami_blue_edge(
            flux15_blue, lam15_blue, err15_blue, min_ssd_lam
        )

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
            #TD: remove testing
            # print(f"SAMI 2015 mean red flux error: {np.nanmean(err15_red)}")
            #
        
        flux01_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power01, lam01, lam01, flux01)
        flux21_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power21, lam01, lam21, flux21)
        flux22_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power22, lam01, lam22, flux22)
        flux15_red_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power15_red, lam01, lam15_red, flux15_red)
        flux15_blue_blurred = gaussian_blur_before_resampling(min_resolving_power, resolving_power15_blue, lam01, lam15_blue_clipped, flux15_blue_clipped)

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
            return (
                (lam01, flux01_blurred, err01),
                (lam15_blue_clipped, flux15_blue_blurred, err15_blue),
                (lam15_red, flux15_red_blurred, err15_red),
                (lam21, flux21_blurred, err21),
                (lam22, flux22_blurred, err22)
            )
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

        flux15_blurred_resampled = np.fmax(flux15_blue_blurred_resampled, flux15_red_blurred_resampled)
        if (
            np.sum(np.isfinite(flux15_blue_blurred_resampled)) +
            np.sum(np.isfinite(flux15_red_blurred_resampled)) !=
            np.sum(np.isfinite(flux15_blurred_resampled))
        ):
            raise ValueError("ERROR: mismatch in number of non-nan values")

        err15_resampled = np.fmax(err15_blue_resampled, err15_red_resampled)
        if (
            np.sum(np.isfinite(err15_blue_resampled)) +
            np.sum(np.isfinite(err15_red_resampled)) !=
            np.sum(np.isfinite(err15_resampled))
        ):
            raise ValueError("ERROR: mismatch in number of non-nan values")
        
        data01 = (flux01_blurred_resampled, err01_resampled)
        data15 = (flux15_blurred_resampled, err15_resampled)
        data21 = (flux21_blurred_resampled, err21_resampled)
        data22 = (flux22_blurred_resampled, err22_resampled)

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

        if blur_step == 0:
            flux15_resampled = np.fmax(flux15_blue_resampled, flux15_red_resampled)
            err15_resampled = np.fmax(err15_blue_resampled, err15_red_resampled)
            data01 = (flux01_resampled, err01_resampled)
            data15 = (flux15_resampled, err15_resampled)
            data21 = (flux21_resampled, err21_resampled)
            data22 = (flux22_resampled, err22_resampled)
            return lam01, (data01, data15, data21, data22)


        wavelength_step = np.median(np.diff(lam01))

        flux01_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power01, lam01, flux01_resampled, wavelength_step)
        flux21_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power21, lam01, flux21_resampled, wavelength_step)
        flux22_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power22, lam01, flux22_resampled, wavelength_step)
        flux15_red_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power15_red, lam01, flux15_red_resampled, wavelength_step)
        flux15_blue_resampled_blurred = gaussian_blur_after_resampling(min_resolving_power, resolving_power15_blue, lam01, flux15_blue_resampled, wavelength_step)

        flux15_resampled_blurred = np.fmax(flux15_blue_resampled_blurred, flux15_red_resampled_blurred)
        if (
            np.sum(np.isfinite(flux15_blue_resampled_blurred)) +
            np.sum(np.isfinite(flux15_red_resampled_blurred)) !=
            np.sum(np.isfinite(flux15_resampled_blurred))
        ):
            raise ValueError("ERROR: mismatch in number of non-nan values")

        err15_resampled = np.fmax(err15_blue_resampled, err15_red_resampled)
        if (
            np.sum(np.isfinite(err15_blue_resampled)) +
            np.sum(np.isfinite(err15_red_resampled)) !=
            np.sum(np.isfinite(err15_resampled))
        ):
            raise ValueError("ERROR: mismatch in number of non-nan values")
        
        data01 = (flux01_resampled_blurred, err01_resampled)
        data15 = (flux15_resampled_blurred, err15_resampled)
        data21 = (flux21_resampled_blurred, err21_resampled)
        data22 = (flux22_resampled_blurred, err22_resampled)

        resampled_and_blurred_title = "Spectra from 2001 to 2022 (resampled to 2001 grid then blurred)"
    # else:
        # this should never happen

    if plot_resampled_and_blurred:
        plot_spectra(
            lam01,
            lam01,
            lam01,
            lam01,
            data01[0],
            data15[0],
            data21[0],
            data22[0],
            plot_errors=plot_errors,
            flux01_err=data01[1],
            flux15_err=data15[1],
            flux21_err=data21[1],
            flux22_err=data22[1],
            title=resampled_and_blurred_title,
            ions=resampled_and_blurred_vlines,
            x_bounds=resampled_and_blurred_xlim
        )

    return lam01, (data01, data15, data21, data22)