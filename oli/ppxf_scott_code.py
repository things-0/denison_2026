"""
Functions for analysis of changing look AGN

"""

from time import perf_counter as clock
from importlib import resources
from urllib import request
import os

from main_code import constants as const
from main_code import helpers

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

def get_clean_sami_data(
    infile: str,
    resolution: float,  # const.RES_15_BLUE or const.RES_15_RED
    infile_dir: str = const.SAMI_FOLDER_NAME,
    z: float = const.Z_SPEC,
    normalise_flux: bool = True,
    flux_power_of_10: int = 17,
    apply_good_pixels_mask: bool = False,
    set_nan_noise_to: str = "large positive"
) -> dict[str, np.ndarray]:
    with fits.open(infile_dir + infile) as hdulist:
        header = hdulist["PRIMARY"].header
        crval1 = header['CRVAL1']   # reference value
        crpix1 = header['CRPIX1']   # index location of reference value
        cdelt1 = header['CDELT1']   # pixel width in Angstroms
        naxis1 = header['NAXIS1']   # number of pixels

        x = np.arange(1, naxis1+0.5)        # (fits coordinates are 1-indexed I think)
        lam_0 = crval1 - crpix1 * cdelt1    # λ_0 = λ_ref - num_pixels_from_start * dλ        
        lam_obs = lam_0 + x*cdelt1          # λ_obs = λ_0 + index * dλ
        lam_rest = lam_obs / (1 + z)
        
        flux = hdulist["PRIMARY"].data
        var = hdulist["VARIANCE"].data
        err = np.array(np.sqrt(var), dtype=float)

    flux *= 10 ** (flux_power_of_10 - 16)
    err *= 10 ** (flux_power_of_10 - 16)

    medflux = np.nanmedian(flux)


    lam_range = [lam_rest[0], lam_rest[-1]]
    flux_resampled, ln_lam_rest_resampled, velscale = util.log_rebin(lam_range, flux)
    err_resampled, _, _ = util.log_rebin(lam_range, err)
    lam_resampled = np.exp(ln_lam_rest_resampled)

    fwhm_per_pix = lam_rest / resolution
    good_pixels_mask = np.where(((lam_obs > 3700) & (err > 0) & np.isfinite(err)))[0]

    if set_nan_noise_to == "large positive":
        err[~np.isfinite(err) | (err <= 0)] = 1.0e10
    elif set_nan_noise_to == "nan":
        err[~np.isfinite(err) | (err <= 0)] = np.nan
    else:
        raise ValueError(f"Invalid value for set_nan_noise_to: {set_nan_noise_to}")

    if apply_good_pixels_mask:
        assert (len(flux) == len(err) == len(lam_rest) == len(fwhm_per_pix))
        flux = flux[good_pixels_mask]
        err = err[good_pixels_mask]
        lam_rest = lam_rest[good_pixels_mask]
        fwhm_per_pix = fwhm_per_pix[good_pixels_mask]

        good_pixels_mask = None

    return {
        "flux": flux,
        "flux_error": err,
        "median_flux": medflux,
        "lam": lam_rest,
        "fwhm_per_pix": fwhm_per_pix,
        "good_pixels_mask": good_pixels_mask,
        "velscale": velscale
    }

def get_clean_sdss_data(
    infile: str,
    infile_dir: str = const.SDSS_FOLDER_NAME,
    z: float = const.Z_SPEC,
    normalise_flux: bool = True,
    flux_power_of_10: int = 17,
    apply_good_pixels_mask: bool = False,
    set_nan_noise_to: str = "large positive"
) -> dict[str, np.ndarray | None]:

    with fits.open(infile_dir + infile) as hdu:
        t = hdu['COADD'].data

    galaxy = t['flux']
    medflux = np.nanmedian(galaxy)
    ln_lam_gal = t['loglam']*np.log(10)-np.log(1+z)       # Convert lg --> ln and also de-redshift
    ln_lam_gal_obs = t['loglam']*np.log(10)       # Convert lg --> ln and keep an version of the observed frame
    lam_gal = np.exp(ln_lam_gal)
    lam_gal_obs = np.exp(ln_lam_gal_obs)

    # could use the variance array:
    #noise = np.full_like(galaxy, 0.01635)  # Assume constant noise per pixel here
    # get errors from data:
    ivar = np.array(t['ivar'], dtype=float)
    good = ivar > 0
    noise = np.sqrt(ivar[good])

    if normalise_flux:
        galaxy /= medflux
        noise /= medflux
    
    good_pixels_mask = np.where(((lam_gal_obs > 3700) & (ivar > 0) & np.isfinite(ivar)))[0]
    
    # print('Redshift:',redshift)

    #convert to air wavelength:
    lam_gal *= np.median(util.vac_to_air(lam_gal)/lam_gal)

    dlam_gal = np.gradient(lam_gal)                 # Size of every pixel in Angstroms
    wdisp = t['wdisp']                              # Instrumental dispersion of every pixel, in pixels units
    fwhm_gal = wdisp*dlam_gal*const.SIGMA_TO_FWHM   # Resolution FWHM of every pixel, in Angstroms

    d_ln_lam_gal = (ln_lam_gal[-1] - ln_lam_gal[0])/(ln_lam_gal.size - 1)  # Use full lam range for accuracy
    velscale = const.C_KM_S*d_ln_lam_gal                   # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)

    if set_nan_noise_to == "large positive":
        noise[~np.isfinite(noise) | (noise <= 0)] = 1.0e10
    elif set_nan_noise_to == "nan":
        noise[~np.isfinite(noise) | (noise <= 0)] = np.nan
    else:
        raise ValueError(f"Invalid value for set_nan_noise_to: {set_nan_noise_to}")

    if apply_good_pixels_mask:
        assert (
            len(good_pixels_mask) == len(galaxy) == \
            len(noise) == len(lam_gal) == len(fwhm_gal)
        )
        galaxy = galaxy[good_pixels_mask]
        noise = noise[good_pixels_mask]
        lam_gal = lam_gal[good_pixels_mask]
        fwhm_gal = fwhm_gal[good_pixels_mask]

        good_pixels_mask = None


    return {
        "flux": galaxy,
        "flux_error": noise,
        "median_flux": medflux,
        "lam": lam_gal,
        "fwhm_per_pix": fwhm_gal,
        "good_pixels_mask": good_pixels_mask,
        "velscale": velscale
    }

def assign_gas_components(gas_names):
    """
    Assign component numbers to emission lines following the original grouping scheme.
    
    Grouping rules:
    - Component 1: All narrow Balmer lines (H10, H9, H8, Heps, Hdelta, Hgamma, Hbeta, Halpha)
    - Component 2: [OII] doublet ([OII]3726, [OII]3729)
    - Component 3: [SII] doublet ([SII]6716, [SII]6731)
    - Component 4: [NeIII] doublet ([NeIII]3968, [NeIII]3869)
    - Component 5: HeII4687
    - Component 6: HeI5876
    - Component 7: [OIII]5007_d
    - Component 8: [OI]6300_d
    - Component 9: [NII]6583_d
    
    Returns:
        component_list: List of component numbers (one per gas line)
        max_component: Highest component number used
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

def fit_ppxf_sdss_agn(
    data: dict[str, np.ndarray] | None = None,
    infile: str | None = None,
    infile_dir: str = const.SDSS_FOLDER_NAME,
    outfile_suffix: str = "",
    outfile_dir: str = const.PPXF_FOLDER_NAME,
    z: float = const.Z_SPEC,
):

    if data is None:
        if infile is None:
            raise ValueError("infile is required if data is not provided")
        print(f"Assuming data is from SDSS spectrum:\n{infile}")
        data = get_clean_sdss_data(infile=infile, infile_dir=infile_dir, z=z, normalise_flux=True)
    galaxy, noise, medflux, lam_gal, fwhm_gal, goodpixels, velscale = (
        data["flux"],
        data["flux_error"],
        data["median_flux"],
        data["lam"],
        data["fwhm_per_pix"],
        data["good_pixels_mask"],
        data["velscale"]
    )

    print(f"goodpixels: {goodpixels}")

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
    lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)])/(1 + z)
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
    narrow_components, max_narrow_component = assign_gas_components(gas_names)
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
    
    # Components 1-9: Narrow emission lines (use max_narrow_component to determine count)
    for i in range(1, max_narrow_component + 1):
        bounds.append([(-500, 500), (50, 400)])
    
    # Broad Balmer components (wider velocity and dispersion ranges)
    bounds.append([(-2000, 2000), (500, 5000)])     # Broad 1
    bounds.append([(-2000, 2000), (1000, 10000)])   # Broad 2  
    bounds.append([(-2000, 2000), (1000, 10000)])   # Broad 3
 

        
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
    hdus.append(fits.ImageHDU(gas, name='GAS_ALL'))
    hdus.append(fits.ImageHDU(galaxy - pp.bestfit, name='RESIDUALS'))

    # individual gas comp:
    for k in np.unique(pp.component):
        mask = (pp.component == k) & pp.gas_component
        if np.any(mask):
            comp = pp.matrix @ (pp.weights * mask)
            hdu = fits.ImageHDU(comp, name=f'GAS_COMP_{k}')
            hdus.append(hdu)

    # check if file already exists using os:
    outfile_suffix = f"_{outfile_suffix}" if outfile_suffix != "" else ""
    outfile_name = f"ppxf_components{outfile_suffix}.fits"
    outfile_path = os.path.join(outfile_dir, outfile_name)
    while os.path.exists(outfile_path):
        warnings.warn(f"file {outfile_path} already exists. Attempting to create copy.")
        outfile_name = outfile_name[:-5] + "_cpy.fits"
        outfile_path = os.path.join(outfile_dir, outfile_name)
    fits.HDUList(hdus).writeto(outfile_path, overwrite=False)

def get_nl_and_stell_cont(
    infile: str = "ppxf_components",
    infile_suffix: str = "",
    infile_folder: str = const.PPXF_FOLDER_NAME,
    nl_gas_comp_ids: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9],
    data_is_normalised: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if infile_suffix != "":
        infile_suffix = "_" + infile_suffix
    actual_infile = infile_folder + infile + infile_suffix + ".fits"
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

    return nl, stell_cont, (lam)

def get_ha_hb_comps(
    infile: str = "ppxf_components",
    infile_suffix: str = "",
    infile_folder: str = const.PPXF_FOLDER_NAME,
    br_gas_comp_ids: list[int] = [10, 11, 12],
    data_is_normalised: bool = True,
    vel_width: float = const.VEL_WIDTH_GAUSSIAN_FIT
) -> tuple[np.ndarray, np.ndarray]:
    if infile_suffix != "":
        infile_suffix = "_" + infile_suffix
    actual_infile = infile_folder + infile + infile_suffix + ".fits"
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
            
    
    ha_mask = helpers.get_vel_lam_mask(lam, vel_width, const.H_ALPHA, lam_is_rest_frame=True)
    hb_mask = helpers.get_vel_lam_mask(lam, vel_width, const.H_BETA, lam_is_rest_frame=True)
    ha_broad = np.full_like(lam, np.nan)
    ha_broad[ha_mask] = summed_broad[ha_mask]
    hb_broad = np.full_like(lam, np.nan)
    hb_broad[hb_mask] = summed_broad[hb_mask]

    return ha_broad, hb_broad, (lam, all_broad)

##########################################################
# plot components:
#
def plot_ppxf_fit_comp(
    infile: str = "ppxf_components",
    infile_suffix: str = "",
    infile_folder: str = const.PPXF_FOLDER_NAME,
    actual_lam: np.ndarray | None = None,
    actual_flux: np.ndarray | None = None,
    z: float = 0.0581892766058445
):

    if infile_suffix != "":
        infile_suffix = "_" + infile_suffix
    actual_infile = infile_folder + infile + infile_suffix + ".fits"
    fits.info(actual_infile)
    hdul = fits.open(actual_infile)
    lam = hdul['WAVELENGTH'].data
    galaxy = hdul['GALAXY'].data
    try:
        edflux = hdul['GALAXY'].header.get('MEDFLUX')
    except KeyError:
        warnings.warn("MEDFLUX not found in GALAXY header. Using 1 instead.")
        edflux = 1
    stellar = hdul['STELLAR'].data
    bestfit = hdul['BESTFIT'].data
    goodpixels = hdul['GOODPIXELS'].data

    print(goodpixels)

    # get gas components:
    gas_components = {}
    for hdu in hdul:
        name = hdu.name
        if name.startswith('GAS_COMP_'):
            k = int(name.split('_')[-1])
            gas_components[k] = hdu.data

    
    gas_components = dict(sorted(gas_components.items()))

    # plot indiviual gas components:
    for k, spec in gas_components.items():
        plt.figure(figsize=const.FIG_SIZE, layout="constrained")
        plt.plot(lam, spec, 'k')
        plt.title(f'Gas component {k}')
        plt.xlabel(r'Wavelength [$\AA$]')
        plt.ylabel('Flux')
        # plt.tight_layout()
        plt.show()

    # plot indiviual gas components:
    plt.figure(figsize=const.FIG_SIZE, layout="constrained")
    for i, (k, spec) in enumerate(gas_components.items()):
        plt.plot(lam, spec, 'k', color=const.ALT_COLOUR_MAP(i), label=f'Gas component {k}')
        plt.xlabel(r'Wavelength [$\AA$]')
        plt.ylabel('Flux')
        # plt.tight_layout()
    plt.title("All Gas components")
    plt.legend()
    plt.show()
        
    # get the narrow IDs:
    narrow_ids = [k for k in gas_components if k < 10]

    narrow_gas = np.zeros_like(galaxy)
    for k in narrow_ids:
        narrow_gas += gas_components[k]
    
    galaxy_sub_host = galaxy - stellar - narrow_gas

    # plot spectrum with host removed:
    fig1 = plt.figure(figsize=const.FIG_SIZE, layout="constrained")

    ax1 = fig1.add_subplot(2,1,1)
    if actual_lam is not None and actual_flux is not None:
        if z is not None:
            # actual_lam /= (1+z)
            # actual_flux /= (1+z)
            actual_lam *= (1+z)
            actual_flux *= (1+z)
        ax1.plot(actual_lam, actual_flux / np.median(actual_flux), linestyle="--", label="actual data")
    ax1.plot(lam, galaxy, 'k', label="galaxy")
    ax1.plot(lam, bestfit, 'm', label="bestfit")
    ax1.plot(lam, stellar, 'r', label="stellar")
    ax1.set(xlabel=r'Wavelength [$\AA$]',ylabel='Flux',title=f'full spectrum')
    ymax = galaxy[lam > 3700].max()
    ymin = galaxy[lam > 3700].min()
    yrange = ymax-ymin
    ymin=0.0
    ax1.set(ylim=[ymin-0.05*yrange,ymax+0.05*yrange]) #, xlim=[6450, 6800])
    ax1.legend()
    
    
    ax2 = fig1.add_subplot(2,1,2)
    ax2.plot(lam, galaxy_sub_host, 'k')
    ax2.set(xlabel=r'Wavelength [$\AA$]',ylabel='Flux',title=f'host subtracted')
    # ax2.axvline(const.OIII_STRONG, linestyle="--")

    ymax = galaxy_sub_host[lam > 3700].max()
    ymin = galaxy_sub_host[lam > 3700].min()
    yrange = ymax-ymin
    ax2.set(ylim=[ymin-0.05*yrange,ymax+0.05*yrange])
    # plt.tight_layout()
    plt.show()
        




    
