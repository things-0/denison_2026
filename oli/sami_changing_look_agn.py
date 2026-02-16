"""
Functions for analysis of changing look AGN

"""

from time import perf_counter as clock
from importlib import resources
from urllib import request

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

#############################################################################
# read in an SDSS spectrum and do a ppxf fit (based on https://github.com/micappe/ppxf_examples/blob/main/ppxf_example_gas_sdss_tied.ipynb

def fit_ppxf_sdss_agn(infile,z=0.0581892766058445):


    # set ppxf_dir (path for various files, templates etc):
    ppxf_dir = resources.files('ppxf')
    #ppxf_dir='/Users/scroom/code/ppxf/files/ppxf_data'
    
    # read in sdss format file:
    #infile = ppxf_dir / 'spectra/NGC3073_SDSS_DR18.fits'
    hdu = fits.open(infile)
    t = hdu['COADD'].data
    #redshift = hdu['SPECOBJ'].data['z'].item()       # SDSS redshift estimate
    redshift = z

    medflux = np.nanmedian(t['flux'])
    galaxy = t['flux']/medflux   # Normalize spectrum to avoid numerical issues
    ln_lam_gal = t['loglam']*np.log(10)-np.log(1+redshift)       # Convert lg --> ln and also de-redshift
    ln_lam_gal_obs = t['loglam']*np.log(10)       # Convert lg --> ln and keep an version of the observed frame
    lam_gal = np.exp(ln_lam_gal)
    lam_gal_obs = np.exp(ln_lam_gal_obs)

    # could use the variance array:
    #noise = np.full_like(galaxy, 0.01635)  # Assume constant noise per pixel here
    # get errors from data:
    ivar = t['ivar']
    good = ivar > 0
    noise = np.empty_like(ivar, dtype=float)
    noise[good] = (1 / np.sqrt(ivar[good]))/medflux
    noise[~good] = 1.0e10
    
    goodpixels = np.where(((lam_gal_obs > 3700) & (ivar > 0)))[0]
    
    print('Redshift:',redshift)

    #convert to air wavelength:
    lam_gal *= np.median(util.vac_to_air(lam_gal)/lam_gal)

    c = 299792.458 # speed of light in km/s
    d_ln_lam_gal = (ln_lam_gal[-1] - ln_lam_gal[0])/(ln_lam_gal.size - 1)  # Use full lam range for accuracy
    velscale = c*d_ln_lam_gal                   # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)
    print ('velscale=',velscale)

    dlam_gal = np.gradient(lam_gal)             # Size of every pixel in Angstroms
    wdisp = t['wdisp']                          # Instrumental dispersion of every pixel, in pixels units
    fwhm_gal = 2.355*wdisp*dlam_gal             # Resolution FWHM of every pixel, in Angstroms

    # decide which templates to use:
    # sps_name = 'fsps'
    # sps_name = 'galaxev'
    sps_name = 'emiles'
    # sps_name = 'xsl'

    basename = f"spectra_{sps_name}_9.0.npz"
    filename = ppxf_dir / 'sps_models' / basename
    if not filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, filename)

    fwhm_gal_dic = {"lam": lam_gal, "fwhm": fwhm_gal}
    sps = lib.sps_lib(filename, velscale, fwhm_gal_dic)
    
    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

    # set up a set of gaussian emission line templates:
    lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)])/(1 + redshift)
    gas_templates, gas_names, gas_wave = \
      util.emission_lines(sps.ln_lam_temp, lam_range_gal, fwhm_gal_dic)

    # find Balmer lines:
    balmer_names = {'Heps','Hdelta','Hgamma','Hbeta','Halpha'}
    is_balmer = np.array([name in balmer_names for name in gas_names])
    
    
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

    # need to add in extra component for [OII] compared to example:
    component = [0]*n_temps  # Single stellar kinematic component=0 for all templates
    component += [1]*8
    component += [2, 2, 3, 3, 4,4,5, 6, 7, 8, 9]
    component += [10]*5 # broad balmer lines 1
    component += [11]*5 # broad balmer lines 2
    component += [12]*5 # broad balmer lines 3
    #component += [13]*5 # broad balmer lines 4
    #component += [2, 2, 3, 3,4, 5, 6, 7, 8]
    component = np.array(component)

    print(len(component))
    
    moments = [2]*13
    #moments = [2]*9

    # set bounds for components:
    #bounds = [
    #[(-500+vel0, 500+vel0), (10, 400)],       # stars
    #[(-500+vel0, 500+vel0), (10, 400)],       # narrow gas
    #[(-500+vel0, 500+vel0), (10, 400)],       # narrow gas
    #[(-500+vel0, 500+vel0), (10, 400)],       # narrow gas
    #[(-500+vel0, 500+vel0), (10, 400)],       # narrow gas
    #[(-500+vel0, 500+vel0), (10, 400)],       # narrow gas
    #[(-500+vel0, 500+vel0), (10, 400)],       # narrow gas
    #[(-500+vel0, 500+vel0), (10, 400)],       # narrow gas
    #[(-500+vel0, 500+vel0), (10, 400)],       # narrow gas
    #[(-500+vel0, 500+vel0), (10, 400)],       # narrow gas
    #[(-2000+vel0, 2000+vel0), (100, 5000)],   # broad Balmer 1
    #[(-2000+vel0, 2000+vel0), (500, 5000)],   # broad Balmer 2
    #[(-2000+vel0, 2000+vel0), (1000, 10000)],   # broad Balmer 3
    #[(-2000+vel0, 2000+vel0), (1000, 10000)]   # broad Balmer 4
    #    ]
    bounds = [
    [(-500, 500), (10, 400)],       # stars
    [(-500, 500), (50, 400)],       # narrow gas
    [(-500, 500), (50, 400)],       # narrow gas
    [(-500, 500), (50, 400)],       # narrow gas
    [(-500, 500), (50, 400)],       # narrow gas
    [(-500, 500), (50, 400)],       # narrow gas
    [(-500, 500), (50, 400)],       # narrow gas
    [(-500, 500), (50, 400)],       # narrow gas
    [(-500, 500), (50, 400)],       # narrow gas
    [(-500, 500), (50, 400)],       # narrow gas
    #[(-2000, 2000), (500, 5000)],   # broad Balmer 1
    [(-2000, 2000), (500, 5000)],   # broad Balmer 2
    [(-2000, 2000), (1000, 10000)],   # broad Balmer 3
    [(-2000, 2000), (1000, 10000)]   # broad Balmer 4
        ]
 

        
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

    print(component.size)
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
    hdus.append(fits.ImageHDU(galaxy, name='GALAXY'))
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

    fits.HDUList(hdus).writeto('ppxf_components.fits', overwrite=True)
            
##########################################################
# plot components:
#
def plot_ppxf_fit_comp(infile):


    fits.info(infile)
    hdul = fits.open(infile)
    lam = hdul['WAVELENGTH'].data
    galaxy = hdul['GALAXY'].data
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
        plt.figure()
        plt.plot(lam, spec, 'k')
        plt.title(f'Gas component {k}')
        plt.xlabel(r'Wavelength [$\AA$]')
        plt.ylabel('Flux')
        plt.tight_layout()
        plt.show()
        
    # get the narrow IDs:
    narrow_ids = [k for k in gas_components if k < 10]

    narrow_gas = np.zeros_like(galaxy)
    for k in narrow_ids:
        narrow_gas += gas_components[k]
    
    galaxy_sub_host = galaxy - stellar - narrow_gas

    # plot spectrum with host removed:
    fig1 = plt.figure()

    ax1 = fig1.add_subplot(2,1,1)
    ax1.plot(lam, galaxy, 'k')
    ax1.plot(lam, bestfit, 'm')
    ax1.plot(lam, stellar, 'r')
    ax1.set(xlabel=r'Wavelength [$\AA$]',ylabel='Flux',title=f'full spectrum')
    ymax = galaxy[lam > 3700].max()
    ymin = galaxy[lam > 3700].min()
    yrange = ymax-ymin
    ymin=0.0
    ax1.set(ylim=[ymin-0.05*yrange,ymax+0.05*yrange])
    
    
    ax2 = fig1.add_subplot(2,1,2)
    ax2.plot(lam, galaxy_sub_host, 'k')
    ax2.set(xlabel=r'Wavelength [$\AA$]',ylabel='Flux',title=f'host subtracted')

    ymax = galaxy_sub_host[lam > 3700].max()
    ymin = galaxy_sub_host[lam > 3700].min()
    yrange = ymax-ymin
    ax2.set(ylim=[ymin-0.05*yrange,ymax+0.05*yrange])
    plt.tight_layout()
    plt.show()
        




    
