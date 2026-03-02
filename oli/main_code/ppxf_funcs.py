"""
Functions for analysis of changing look AGN
"""

from time import perf_counter as clock
from importlib import resources
from pathlib import Path
from urllib import request
import os
import warnings

from . import constants as const
from .data_reading import get_sdss_data, get_sami_data, get_adjusted_data
from .helpers import get_lam_mask, get_velscale
from .polynomial_fit import apply_poly_fit

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

def assign_narrow_components(
    gas_names: list[str],
) -> tuple[list[int], int]:
    """
    Uses a grouping scheme to assign component numbers to emission lines.

    Parameters
    ----------
    gas_names: list[str]
        The names of the gas lines to assign components to.

    Returns
    -------
    component_list: list[int]
        A list of component numbers (one per gas line)
    max_component: int
        The highest component number used
    """
    
    # Define groupings
    narrow_balmer = ['H10', 'H9', 'H8', 'Heps', 'Hdelta', 'Hgamma', 'Hbeta', 'Halpha']
    oii_doublet = ['[OII]3726', '[OII]3729']
    sii_doublet = ['[SII]6716', '[SII]6731']
    neiii_doublet = ['[NeIII]3968', '[NeIII]3869']
    
    # Map individual lines to components
    other_lines = [
        'HeII4687',
        'HeI5876',
        '[OIII]5007_d',
        '[OI]6300_d',
        '[NII]6583_d'
    ]
    
    component_list = []
    
    groups_line_nums = {}
    for line in gas_names:
        if line in narrow_balmer:
            groups_line_nums.setdefault("narrow_balmer", len(groups_line_nums)+1)
            component_list.append(groups_line_nums["narrow_balmer"])
        elif line in oii_doublet:
            groups_line_nums.setdefault("oii_doublet", len(groups_line_nums)+1)
            component_list.append(groups_line_nums["oii_doublet"])
        elif line in sii_doublet:
            groups_line_nums.setdefault("sii_doublet", len(groups_line_nums)+1)
            component_list.append(groups_line_nums["sii_doublet"])
        elif line in neiii_doublet:
            groups_line_nums.setdefault("neiii_doublet", len(groups_line_nums)+1)
            component_list.append(groups_line_nums["neiii_doublet"])
        elif line in other_lines:
            groups_line_nums.setdefault("other_lines", len(groups_line_nums)+1)
            component_list.append(groups_line_nums["other_lines"])
        else:
            raise ValueError(f"Unknown emission line: {line}")
    
    max_component = max(component_list) if component_list else 0
    
    return component_list, max_component

def fit_agn(
    data: dict[str, np.ndarray] | None = None,
    baseline_year: int | None = 2015,
    year_to_adjust: int | None = None,
    is_sdss: bool | None = None,
    sami_res: float | None = None,
    infile_name: str | None = None,
    outfile_suffix: str = "",
    outfile_dir: Path = const.PPXF_DATA_DIR,
    z: float = const.Z_SPEC,
    normalise_flux: bool = True,
    filter_bad_pixels: bool = True,
) -> None:
    """
    Fits a pPXF model to a spectrum and saves the results to a FITS file.

    Parameters
    ----------
    data: dict[str, np.ndarray] | None
        The data to fit the pPXF model to. If None, the data is created using
        :func:`data_reading.get_adjusted_data` or (:func:`data_reading.get_sdss_data`
        or :func:`data_reading.get_sami_data`).
    baseline_year: int | None
        The year to use as the baseline (calibration) flux. Must be one of 2001, 2015,
        2021, or 2022. Should be None if `data` is provided or data is read in raw
        using `infile_name` (no polynomial fit).
    year_to_adjust: int | None
        The year to adjust. Must be one of 2001, 2015, 2021, or 2022. Should be None if `data`
        is provided or data is read in raw using `infile_name` (no polynomial fit).
    is_sdss: bool | None
        Specifies whether the data read in from `infile_name` is SDSS or SAMI data.
    sami_res: float | None
        The resolving power of the SAMI data.
    infile_name: str | None
        The name of the file to read the data from. Should be None if `data` is provided
        or a polynomial fit is to be applied to the data (by specifying `year_to_adjust`
        and `baseline_year`).
    outfile_suffix: str
        The suffix to add to the output file name: "ppxf_components_<outfile_suffix>.fits".
    outfile_dir: Path
        The directory to save the output file to.
    z: float
        The redshift of the galaxy.
    normalise_flux: bool
        If True, the flux is normalised to the median flux of the galaxy.
    filter_bad_pixels: bool
        If True, bad pixels are filtered out of the data (and `goodpixels` is set to
        `np.arange(len(lam))`). See :func:`helpers.get_good_pixels` for more details.
    """
    if data is None:
        if baseline_year is not None:
            if infile_name is not None:
                raise ValueError("baseline_year provided but infile_name is not None.")
            if year_to_adjust is None:
                raise ValueError("year_to_adjust is required if using baseline_year to get data")
            all_epochs_data = get_adjusted_data(
                blur_step=0,
                resample_step=1,
                plot_resampled_and_blurred=False,
                z=z
            )
            data = apply_poly_fit(
                data=all_epochs_data,
                year_to_adjust=year_to_adjust,
                baseline_year=baseline_year,
                plot_ratio_selection=False,
                plot_poly_ratio=False,
                plot_adjusted=False,
            )
        else:
            # raw data without applying polynomial fit
            if year_to_adjust is not None:
                raise ValueError("baseline_year is required if year_to_adjust is provided.")
            if infile_name is None:
                raise ValueError("infile_name is required if data is not provided and baseline_year is None.")
            if is_sdss is None:
                raise ValueError("is_sdss is required if data is not provided and baseline_year is None.")
            if is_sdss:
                data = get_sdss_data( 
                    file_name=infile_name, folder_path=const.SDSS_DATA_DIR,
                    z=z, rm_or_replace_other_bad_values=True,
                )
            else:
                if sami_res is None:
                    raise ValueError("sami_res is required if data is not provided, baseline_year is None, and is_sdss is False")
                data = get_sami_data(
                    file_name=infile_name, folder_path=const.SAMI_DATA_DIR,
                    z=z, perform_log_rebin=True, resolving_power=sami_res,
                    rm_or_replace_other_bad_values=True,
                )
    galaxy, noise, lam_gal, fwhm_gal, goodpixels, velscale = (
        data["flux"],
        data["flux_error"],
        data["lam"],
        data["fwhm_per_pix"],
        data["good_pixels"],
        data["velscale"]
    )
    medflux = np.nanmedian(galaxy)
    if normalise_flux:
        galaxy /= medflux
        noise /= medflux

    print(f"goodpixels: {goodpixels}")

    for arr in (galaxy, noise, lam_gal, fwhm_gal):
        if np.any(~np.isfinite(arr[goodpixels])):
            raise ValueError("NaN values found in 'goodpixels' data")
    if filter_bad_pixels:
        # sps_lib raises an error if fwhm_gal_dic or noise contains nans
        lam_gal = lam_gal[goodpixels]
        galaxy = galaxy[goodpixels]
        noise = noise[goodpixels]
        fwhm_gal = fwhm_gal[goodpixels]
        goodpixels = np.arange(len(lam_gal))
        velscale = get_velscale(lam_gal)

    # decide which templates to use:
    # sps_name = 'fsps'
    # sps_name = 'galaxev'
    sps_name = 'emiles'
    # sps_name = 'xsl'

    # set ppxf_dir (path for various files, templates etc):
    ppxf_dir = resources.files('ppxf')
    #ppxf_dir='/Users/scroom/code/ppxf/files/ppxf_data'

    basename = f"spectra_{sps_name}_9.0.npz"
    filename = ppxf_dir / 'sps_models' / basename
    if not filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, filename)


    fwhm_gal_dic = {"lam": lam_gal, "fwhm": fwhm_gal}
    sps = lib.sps_lib(filename, velscale, fwhm_gal_dic)
    
    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

    # set up a set of gaussian emission line templates:
    lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)]) # /(1 + z) lam_gal is already rest frame?
    gas_templates, gas_names, gas_wave = \
      util.emission_lines(sps.ln_lam_temp, lam_range_gal, fwhm_gal_dic)

    # find Balmer lines:
    balmer_names = {'Heps','Hdelta','Hgamma','Hbeta','Halpha'}
    is_balmer = np.array([name in balmer_names for name in gas_names])
    
    n_balmer = np.sum(is_balmer)
    
    # cope ony Balmer lines to a broad comp:
    gas_templates_b1 = gas_templates[:, is_balmer]
    gas_names_b1 = np.array(gas_names)[is_balmer]

    gas_templates_b2 = gas_templates_b1.copy()
    gas_names_b2 = gas_names_b1.copy()
    
    gas_templates_b3 = gas_templates_b1.copy()
    gas_names_b3 = gas_names_b1.copy()
    
    gas_templates_b4 = gas_templates_b1.copy()
    gas_names_b4 = gas_names_b1.copy()
    
    print("Just balmer lines:")
    print(gas_names_b1)
    
    gas_names_all = np.concatenate([gas_names,gas_names_b1,gas_names_b2,gas_names_b3])
    #gas_names_all = np.concatenate([gas_names,gas_names_b1,gas_names_b2,gas_names_b3,gas_names_b4])
    
    # combine the templates:
    #templates = np.column_stack([stars_templates, gas_templates,gas_templates_b1,gas_templates_b2,gas_templates_b3,gas_templates_b4])
    templates = np.column_stack([stars_templates, gas_templates,gas_templates_b1,gas_templates_b2,gas_templates_b3])

    n_temps = stars_templates.shape[1]
    n_temps_all = templates.shape[1]

    print('n_temps=',n_temps)
    print('n_temps_all=',n_temps_all)

    # set up fitting:
    #vel0 = c*np.log(1 + redshift)
    sol = [0.0, 200]

    component = []
    
    # Component 0: All stellar templates tied together
    component += [0] * n_temps
    
    # Narrow gas components: Use the original grouping scheme
    narrow_components, max_narrow_component = assign_narrow_components(gas_names)
    component += narrow_components
    
    # Broad Balmer components: All Balmer lines in each set tied together
    next_component = max_narrow_component + 1
    component += [next_component] * n_balmer        # Broad component 1
    component += [next_component + 1] * n_balmer    # Broad component 2
    component += [next_component + 2] * n_balmer    # Broad component 3
    
    component = np.array(component)
    
    n_components = len(np.unique(component))
    moments = [2] * n_components

    # set bounds for components:
    bounds = []
    
    # Component 0: Stellar (narrow velocity range)
    bounds.append([(-500, 500), (10, 400)])
    
    # Components 1-9: Narrow emission lines
    for i in range(1, max_narrow_component + 1):
        bounds.append([(-500, 500), (50, 400)])
    
    # Broad Balmer components (wider velocity and dispersion ranges)
    bounds.append([(-2000, 2000), (500, 2000)])     # Broad 1 # sigma was (500, 5000)
    bounds.append([(-2000, 2000), (1000, 3500)])   # Broad 2  # sigma was (1000, 10000)
    bounds.append([(-2000, 2000), (1000, 3500)])   # Broad 3  # sigma was (1000, 10000)
 

        
    start = [sol for j in range(len(moments))]  # adopt the same starting value for both gas and stars

    print(start)
    print(np.shape(start))
        
    start[-1] = [0.0,2000.0]
    start[-2] = [0.0,2000.0]
    start[-2] = [0.0,500.0]
    
    print(start)

    #tied = [['', ''] for j in range(len(moments))]
    #for j in range(3, len(moments)):
    #    tied[j][0] = 'p[4]'

    # inequality limits for gas (may not want these):
    #          V0 s0 V1 s1 V2 s2 V3 s3 V4 s4 V5 s5 V6 s6 V7 s7 V8 s8
    #A_ineq = [[0, -2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -2*s0 + s1 < 0 => s1 < 2*s0
    #      [0, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # s2 < 2*s0
    #      [0, -2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # s3 < 2*s0
    #      [0, -2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # s4 < 2*s0
    #      [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # s5 < 2*s0
    #      [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # s6 < 2*s0
    #      [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # s7 < 2*s0
    #      [0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]  # s8 < 2*s0
    #b_ineq = [0, 0, 0, 0, 0, 0, 0, 0]
    #constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

    # do the fit:
    degree= -1
    mdegree = 10
    t = clock()

    print(f"\n\ncomponent shape: {component.shape}")
    print(f"templates shape: {templates.shape}\n\n")
    pp = ppxf(templates, galaxy, noise, velscale, start, plot=False,
            moments=moments, degree=degree, mdegree=mdegree,
            lam=lam_gal, component=component,bounds=bounds, 
            gas_component=component > 0, gas_names=gas_names_all,
            lam_temp=sps.lam_temp,goodpixels=goodpixels)
    # don't use complicated constraints:
    #            constr_kinem=constr_kinem, lam_temp=sps.lam_temp)
    print(f"Elapsed time in pPXF: {(clock() - t):.2f}")

    
    pp.plot()
    plt.title(f"pPXF fit with {sps_name} SPS templates")
    plt.pause(5)

    # write the best fit out:
    hdus = []

    hdus.append(fits.PrimaryHDU())

    stellar_mask = (pp.component == 0)
    stellar = pp.matrix @ (pp.weights * stellar_mask)
    gas_mask = pp.gas_component   # boolean per template
    gas = pp.matrix @ (pp.weights * gas_mask)
    
    hdus.append(fits.ImageHDU(lam_gal, name='WAVELENGTH'))
    
    gal_hdu = fits.ImageHDU(galaxy, name='GALAXY')
    gal_hdu.header['MEDFLUX'] = (medflux, 'Median flux of the galaxy spectrum')
    hdus.append(gal_hdu)
    
    hdus.append(fits.ImageHDU(noise, name='NOISE'))
    hdus.append(fits.ImageHDU(goodpixels, name='GOODPIXELS'))
    hdus.append(fits.ImageHDU(pp.bestfit, name='BESTFIT'))
    hdus.append(fits.ImageHDU(stellar, name='STELLAR'))

    gas_hdu = fits.ImageHDU(gas, name='GAS_ALL')
    gas_hdu.header['MAX_NL_COMP'] = max_narrow_component
    hdus.append(gas_hdu)
    
    hdus.append(fits.ImageHDU(galaxy - pp.bestfit, name='RESIDUALS'))

    # individual gas comp:
    for k in np.unique(pp.component):
        mask = (pp.component == k) & pp.gas_component
        if np.any(mask):
            comp = pp.matrix @ (pp.weights * mask)
            hdu = fits.ImageHDU(comp, name=f'GAS_COMP_{k}')
            hdus.append(hdu)

    # check if outfile_dir exists, if not create it
    if not outfile_dir.exists():
        warn_msg = f"outfile_dir {outfile_dir} does not exist. Creating it."
        warnings.warn(warn_msg)
        outfile_dir.mkdir(parents=True, exist_ok=True)

    outfile_suffix = f"_{outfile_suffix}" if outfile_suffix != "" else ""
    outfile_name = f"ppxf_components{outfile_suffix}.fits"
    outfile_path = outfile_dir / outfile_name
    # check if the outfile already exists, if so create a copy with a different name
    while outfile_path.is_file():
        warnings.warn(f"file {outfile_name} already exists. Attempting to create copy.")
        outfile_name = outfile_name[:-5] + "_cpy.fits"
        outfile_path = outfile_dir / outfile_name
    fits.HDUList(hdus).writeto(outfile_path, overwrite=False)
    print(f"\n\nFits file written to {outfile_path}")

def get_nl_and_stell_cont(
    infile: str = "ppxf_components",
    infile_suffix: str = "",
    infile_path: Path = const.PPXF_DATA_DIR,
    nl_gas_comp_ids: list[int] = [1, 2, 3, 4], #TODO: generalise to arbitrary number of narrow line components
    data_is_normalised: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gets the narrow line and stellar continuum components from a pPXF fit.

    Parameters
    ----------
    infile: str
        The name of the file to read the data from: "<infile>_<infile_suffix>.fits".
    infile_suffix: str
        The suffix of the infile name: "<infile>_<infile_suffix>.fits".
    infile_path: Path
        The path to the directory containing the infile.
    nl_gas_comp_ids: list[int]
        The IDs of the narrow line components to get.
    data_is_normalised: bool
        If True, the data is normalised to the median flux of the galaxy.

    Returns
    -------
    nl: np.ndarray
        The narrow line components.
    stell_cont: np.ndarray
        The stellar continuum components.
    """
    if infile_suffix != "":
        infile_suffix = "_" + infile_suffix
    actual_infile = infile_path / (infile + infile_suffix + ".fits")
    with fits.open(actual_infile) as hdul:
        lam = hdul['WAVELENGTH'].data
        nl = np.zeros_like(lam)

        for comp_id in nl_gas_comp_ids:
            nl += hdul[f'GAS_COMP_{comp_id}'].data
        stell_cont = hdul['STELLAR'].data

        if data_is_normalised:
            medflux = hdul['GALAXY'].header.get('MEDFLUX')
            if not isinstance(medflux, float):
                raise ValueError("could not find MEDFLUX value in GALAXY header")
            print(f"Re-scaling by median flux: {medflux:.2f}")
            nl *= medflux
            stell_cont *= medflux

    return nl, stell_cont

def get_ha_hb_comps(
    infile: str = "ppxf_components",
    infile_suffix: str = "",
    infile_path: Path = const.PPXF_DATA_DIR,
    br_gas_comp_ids: list[int] = [5, 6, 7], #TODO: generalise to arbitrary number of narrow line components
    data_is_normalised: bool = True,
    vel_width: float = const.VEL_WIDTH_GAUSSIAN_FIT
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Gets the broad line components from a pPXF fit.

    Parameters
    ----------
    infile: str
        The name of the file to read the data from: "<infile>_<infile_suffix>.fits".
    infile_suffix: str
        The suffix of the infile name: "<infile>_<infile_suffix>.fits".
    infile_path: Path
        The path to the directory containing the infile.
    br_gas_comp_ids: list[int]
        The IDs of the broad line components to get.
    data_is_normalised: bool
        If True, the data is normalised to the median flux of the galaxy.
    vel_width: float
        The velocity width to use when masking the broad line flux for Hα and Hβ.

    Returns
    -------
    ha_broad: np.ndarray
        The summed Hα broad line components.
    hb_broad: np.ndarray
        The summed Hβ broad line components.
    lam: np.ndarray
        The wavelength array.
    all_broad: np.ndarray
        The broad line flux for each broad line component. Has shape
        (`len(br_gas_comp_ids), len(lam)`).
    """
    if infile_suffix != "":
        infile_suffix = "_" + infile_suffix
    actual_infile = infile_path / (infile + infile_suffix + ".fits")

    with fits.open(actual_infile) as hdul:
        lam = hdul['WAVELENGTH'].data

        all_broad = np.zeros((len(br_gas_comp_ids), len(lam)))
        summed_broad = np.zeros_like(lam)


        for i, comp_id in enumerate(br_gas_comp_ids):
            comp_flux = hdul[f'GAS_COMP_{comp_id}'].data
            all_broad[i, :] = comp_flux
            summed_broad += comp_flux

        if data_is_normalised:
            medflux = hdul['GALAXY'].header.get('MEDFLUX')
            if not isinstance(medflux, float):
                raise ValueError("could not find MEDFLUX value in GALAXY header")
            print(f"Re-scaling by median flux: {medflux:.2f}")
            all_broad *= medflux
            summed_broad *= medflux
            
    
    ha_mask = get_lam_mask(lam, vel_width, const.H_ALPHA, width_is_vel=True)
    hb_mask = get_lam_mask(lam, vel_width, const.H_BETA, width_is_vel=True)
    ha_broad = np.full_like(lam, np.nan)
    ha_broad[ha_mask] = summed_broad[ha_mask]
    hb_broad = np.full_like(lam, np.nan)
    hb_broad[hb_mask] = summed_broad[hb_mask]

    return ha_broad, hb_broad, (lam, all_broad)
