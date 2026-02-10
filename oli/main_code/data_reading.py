import numpy as np
import astropy.io.fits as fits
from PyAstronomy import pyasl
import spectres
import warnings

from .adjust_calibration import gaussian_blur_before_resampling, gaussian_blur_after_resampling, clip_sami_blue_edge
from . import constants as const
from .helpers import get_min_res
from .plotting import plot_min_res, plot_spectra, plot_vert_emission_lines

def get_sami_lam_flux_err(
    fname: str,
    folder_name: str = const.SAMI_FOLDER_NAME,
    filter_bad_values: bool = True,
    interpolate_bad_values: bool = False,
    lam_bounds: tuple[float, float] | None = const.TOTAL_LAM_BOUNDS,
    flux_power_of_10: int = 17,
    lam_in_vacuum: bool = True,
    doareacorr: bool = False,
    bugfix: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    with fits.open(folder_name + fname) as hdulist:
        header = hdulist["PRIMARY"].header
        crval1=header['CRVAL1']
        cdelt1=header['CDELT1']
        crpix1=header['CRPIX1']
        naxis1=header['NAXIS1'] 
        x=np.arange(naxis1)+1
        L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
        lam = L0+x*cdelt1
        
        flux_uncorrected = hdulist["PRIMARY"].data
        var = hdulist["VARIANCE"].data
        err = np.array(np.sqrt(var), dtype=float)

        #TD: remove
        if doareacorr:
            if bugfix:
                areacorr=areacorr/2.0
            areacorr = header['AREACORR']
            flux = flux_uncorrected * areacorr *10 ** (flux_power_of_10 - 16)
        else:
            flux = flux_uncorrected * 10 ** (flux_power_of_10 - 16)
        #
        
    # flux = flux_uncorrected * 10 ** (flux_power_of_10 - 16)

    if lam_in_vacuum:
        lam = pyasl.airtovac2(lam)

    lam_mask = np.isfinite(lam)
    if np.any(~lam_mask):
        warn_msg = f"{np.sum(~lam_mask)} values of lam are not finite. These will be ignored."
        warnings.warn(warn_msg)     #TODO: raise error instead
        lam_valid = lam[lam_mask]
        flux_valid = flux[lam_mask]
        err_valid = err[lam_mask]
    else:
        lam_valid = lam
        flux_valid = flux
        err_valid = err

    if not filter_bad_values:
        # if np.any(~np.isfinite(var) | (var <= 0)):
            # raise ValueError("ERROR: bad values in variance array. Cannot take the square root")
        return flux, lam_valid, err

    if lam_bounds is None:
        not_in_lam_bounds = np.zeros_like(lam_valid, dtype=bool)
    else:
        not_in_lam_bounds = (lam_valid < lam_bounds[0]) | (lam_valid > lam_bounds[1])
    bad_mask = (
        # ~np.isfinite(lam) | #TODO: remove?
        not_in_lam_bounds |
        ~np.isfinite(flux_valid) | (flux_valid > const.MAX_FLUX) | (flux_valid < const.MIN_FLUX) |
        ~np.isfinite(err_valid) | (err_valid <= 0) | (err_valid > const.MAX_FLUX)
    )
    good_mask = ~bad_mask

    if interpolate_bad_values:
        flux_interpolated = np.interp(lam_valid, lam_valid[good_mask], flux_valid[good_mask])
        err_interpolated = np.interp(lam_valid, lam_valid[good_mask], err_valid[good_mask])
        return lam_valid, flux_interpolated, err_interpolated
    else:
        lam_masked = lam_valid[good_mask]
        flux_masked = flux_valid[good_mask]
        err_masked = err_valid[good_mask]
        return lam_masked, flux_masked, err_masked


def get_sdss_lam_flux_err(
    fname: str,
    folder_name: str = const.SDSS_FOLDER_NAME,
    filter_bad_values: bool = True,
    interpolate_bad_values: bool = False,
    lam_bounds: tuple[float, float] | None = const.TOTAL_LAM_BOUNDS,
    flux_power_of_10: int = 17,
    get_other_data: bool = False
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray] |
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    with fits.open(folder_name + fname) as hdulist:
        # read spectrum from COADD extension:
        # spec_table = hdulist["COADD"].data
        spec_table = hdulist[1].data

        flux = spec_table['flux']
        flux *= 10 ** (flux_power_of_10 - 17)

        loglam = spec_table['loglam']
        # SDSS spectra are in log wavelength bins, to convert to linear:
        lam = 10.0**loglam

        # inverse variance
        ivar = spec_table['ivar']
        err = np.array(np.sqrt(1 / ivar), dtype=float)

        if get_other_data:
            header_info = hdulist[0].header

            try:
                # wavelength resolution
                wresl = spec_table['wresl']
            except KeyError:
                warn_msg = f"Wavelength resolution data not available in {fname}"
                warnings.warn(warn_msg)
                wresl = None
            
            plug_ra = header_info['plug_ra']
            plug_dec = header_info['plug_dec']
            plateid = header_info['plateid']
            mjd = header_info['mjd']
            fiberid = header_info['fiberid']
            z = hdulist[2].data['z'][0]

            other_data = {
                "wresl": wresl,
                "plug_ra": plug_ra,
                "plug_dec": plug_dec,
                "plateid": plateid,
                "mjd": mjd,
                "fiberid": fiberid,
                "z": z
            }


    lam_mask = np.isfinite(lam)
    if np.any(~lam_mask):
        warn_msg = f"{np.sum(~lam_mask)} values of lam are not finite. These will be ignored."
        warnings.warn(warn_msg)     #TODO: raise error instead
        lam_valid = lam[lam_mask]
        flux_valid = flux[lam_mask]
        err_valid = err[lam_mask]
    else:
        lam_valid = lam
        flux_valid = flux
        err_valid = err

    # return flux, lam, 1 / np.sqrt(ivar)
    if filter_bad_values: 
        if lam_bounds is None:
            not_in_lam_bounds = np.zeros_like(lam_valid, dtype=bool)
        else:
            not_in_lam_bounds = (lam_valid < lam_bounds[0]) | (lam_valid > lam_bounds[1])
        bad_mask = (
            not_in_lam_bounds |
            ~np.isfinite(flux_valid) | (flux_valid > const.MAX_FLUX) | (flux_valid < const.MIN_FLUX) |
            ~np.isfinite(err_valid) | (err_valid <= 0) | (err_valid > const.MAX_FLUX)
        )
        good_mask = ~bad_mask
        if interpolate_bad_values:
            # lam = np.interp(lam, lam[good_mask], lam[good_mask])
            flux_interpolated = np.interp(lam_valid, lam_valid[good_mask], flux_valid[good_mask])
            err_interpolated = np.interp(lam_valid, lam_valid[good_mask], err_valid[good_mask])
            lfe = lam_valid, flux_interpolated, err_interpolated
        else:
            lam_masked = lam_valid[good_mask]
            flux_masked = flux_valid[good_mask]
            err_masked = err_valid[good_mask]
            lfe = lam_masked, flux_masked, err_masked

    if get_other_data:
        return *lfe, other_data
    else:
        return lfe


def sami_read_apspec(
    filename: str,
    extname: str,
    var_extname: str,
    doareacorr: bool = False,
    bugfix: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hdulist = fits.open(filename)
    sami_flux = hdulist[extname].data
    header = hdulist[extname].header
    crval1=header['CRVAL1']
    cdelt1=header['CDELT1']
    crpix1=header['CRPIX1']
    naxis1=header['NAXIS1'] 
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    sami_lam = L0+x*cdelt1

    if (doareacorr):
        # fix bug in ap spec code:
        if (bugfix):
            areacorr=areacorr/2.0
        areacorr = header['AREACORR']
        sami_flux = sami_flux * areacorr

    #TD: remove testing
    # print(header['BUNIT'])
    #

    #convert SAMI (wavelength in air) to wavelength in vacuum By default, the conversion specified by Ciddor 1996 are used.
    samivac_lam = pyasl.airtovac2(sami_lam)
    
    sami_flux *= 10 # flux unit conversion from 10e16 to 10e17 to match sdss

    
    sami_var = hdulist[var_extname].data #extract the variance array
    hdulist.close()
    return sami_flux, samivac_lam, sami_var

def sdss_read(
    infile: str, return_wresl: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # get basic info on file:
    
    # opening file
    hdulist = fits.open(infile)
    hdr0 = fits.getheader(infile)
    hdr1 = hdulist['COADD'].header

    #TD: remove testing
    # try:
    #     print(hdr0['BUNIT'])
    # except:
    #     print(f"no BUNIT in file {infile}")
    #

    # read spectrum from COADD extension:
    sdss_spec_table = hdulist['COADD'].data
    sdss_flux = sdss_spec_table['flux']
    sdss_loglam = sdss_spec_table['loglam']

    # SDSS spectra are in log wavelength bins, to convert to linear:
    sdss_lam = 10.0**sdss_loglam
    # inverse variance
    sdss_ivar = sdss_spec_table['ivar']

    # close fits file:
    hdulist.close()

    # define parts of spectrum where data is likely to be good:
    idx = np.where((sdss_lam>const.TOTAL_LAM_BOUNDS[0]) & (sdss_lam<const.TOTAL_LAM_BOUNDS[1]))
    sdss_flux = sdss_flux[idx]
    sdss_lam = sdss_lam[idx]
    sdss_ivar = sdss_ivar[idx]
    if return_wresl:
        wresl = sdss_spec_table['wresl']
        wresl = wresl[idx]
        return sdss_flux, sdss_lam, sdss_ivar, wresl
    else:
        return sdss_flux, sdss_lam, sdss_ivar

def get_adjusted_data(
    return_as_is: bool = False,
    blur_before_resampling: bool = True,
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
    sdss_folder_name: str = const.SDSS_FOLDER_NAME,
    sami_folder_name: str = const.SAMI_FOLDER_NAME,
    fname_2001: str = const.FNAME_2001,
    fname_2015_blue: str = const.FNAME_2015_BLUE_3_ARCSEC,
    fname_2015_red: str = const.FNAME_2015_RED_3_ARCSEC,
    fname_2021: str = const.FNAME_2021,
    fname_2022: str = const.FNAME_2022
) -> (
    tuple[np.ndarray, tuple[
        np.ndarray, np.ndarray
    ], tuple[
        np.ndarray, np.ndarray
    ], tuple[
        np.ndarray, np.ndarray
    ], tuple[
        np.ndarray, np.ndarray
    ]
    ] | tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
):
    """
    Get the flux, wavelength, and variance arrays for the spectra.
    """
    plot_as_is = plot_as_is or as_is_xlim is not None
    plot_clipped = plot_clipped or clipped_xlim is not None
    plot_just_blurred = plot_just_blurred or blurred_xlim is not None
    plot_just_resampled = plot_just_resampled or resampled_xlim is not None
    plot_resampled_and_blurred = plot_resampled_and_blurred or resampled_and_blurred_xlim is not None
    
    flux01, lam01, ivar01 = sdss_read(sdss_folder_name + fname_2001, return_wresl=False) # 2001 file
    flux21, lam21, ivar21, fwhm21 = sdss_read(sdss_folder_name + fname_2021) # 2021 file
    flux22, lam22, ivar22, fwhm22 = sdss_read(sdss_folder_name + fname_2022) # 2022 file
    
    if len(lam21) != len(lam22) or len(lam01) != len(lam21):
        raise ValueError("ERROR: mismatch in number of wavelength points. Try adjusting TOTAL_LAM_BOUNDS.")

    flux15_blue, lam15_blue, var15_blue = sami_read_apspec(
        sami_folder_name + fname_2015_blue,
        "PRIMARY", "VARIANCE"
    )
    flux15_red, lam15_red, var15_red = sami_read_apspec(
        sami_folder_name + fname_2015_red,
        "PRIMARY", "VARIANCE"
    )

    # inverse variance -> variance
    var01 = 1 / ivar01
    var21 = 1 / ivar21
    var22 = 1 / ivar22

    res_21 = lam21 / fwhm21
    res_22 = lam22 / fwhm22
    # Note: fwhm01 (wresl) not available
    res_01 = (res_21 + res_22) / 2
    #

    res_min = get_min_res(
        res_01,
        res_21,
        res_22,
    )
    if plot_res_coverage:
        plot_min_res(
            lam_sdss=lam21,
            lam15_blue=lam15_blue,
            lam15_red=lam15_red,
            res_min=res_min,
            res_01=res_01,
            res_21=res_21,
            res_22=res_22,
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

    err01 = np.sqrt(var01)
    err15_blue = np.sqrt(var15_blue)
    err15_red = np.sqrt(var15_red)
    err21 = np.sqrt(var21)
    err22 = np.sqrt(var22)
    
    if return_as_is:
        return (
            (lam01, flux01, err01),
            (lam15_blue, flux15_blue, err15_blue),
            (lam15_red, flux15_red, err15_red),
            (lam21, flux21, err21),
            (lam22, flux22, err22)
        )

    if blur_before_resampling:
        min_ssd_lam = np.min(lam01)
        flux15_blue_clipped, lam15_blue_clipped, var15_blue_clipped = clip_sami_blue_edge(
            flux15_blue, lam15_blue, var15_blue, min_ssd_lam
        )
        err15_blue_clipped = np.sqrt(var15_blue_clipped)

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
        
        flux01_blurred = gaussian_blur_before_resampling(res_min, res_01, lam01, lam01, flux01)
        flux21_blurred = gaussian_blur_before_resampling(res_min, res_21, lam01, lam21, flux21)
        flux22_blurred = gaussian_blur_before_resampling(res_min, res_22, lam01, lam22, flux22)
        flux15_red_blurred = gaussian_blur_before_resampling(res_min, const.RES_15_RED, lam01, lam15_red, flux15_red)
        flux15_blue_blurred = gaussian_blur_before_resampling(res_min, const.RES_15_BLUE, lam01, lam15_blue_clipped, flux15_blue_clipped)

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

    else:
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

        wavelength_step = np.median(np.diff(lam01))

        flux01_resampled_blurred = gaussian_blur_after_resampling(res_min, res_01, lam01, flux01_resampled, wavelength_step)
        flux21_resampled_blurred = gaussian_blur_after_resampling(res_min, res_21, lam01, flux21_resampled, wavelength_step)
        flux22_resampled_blurred = gaussian_blur_after_resampling(res_min, res_22, lam01, flux22_resampled, wavelength_step)
        flux15_red_resampled_blurred = gaussian_blur_after_resampling(res_min, const.RES_15_RED, lam01, flux15_red_resampled, wavelength_step)
        flux15_blue_resampled_blurred = gaussian_blur_after_resampling(res_min, const.RES_15_BLUE, lam01, flux15_blue_resampled, wavelength_step)

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