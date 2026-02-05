import glob, os, sys, timeit
import matplotlib
import numpy as np

sys.path.append('../')
from pyqsofit.PyQSOFit import QSOFit
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

QSOFit.set_mpl_style()

# Show the versions so we know what works
import astropy
import lmfit
import pyqsofit

print(astropy.__version__)
print(lmfit.__version__)
print(pyqsofit.__version__)

import emcee  # optional, for MCMC

print(emcee.__version__)
print(pyqsofit.__path__)

path_ex = '.' #os.path.join(pyqsofit.__path__[0], '..', 'example')
path_out = os.path.join(pyqsofit.__path__[0], '../', 'example/data/')

# create a header
hdr0 = fits.Header()
hdr0['Author'] = 'Hengxiao Guo'
primary_hdu = fits.PrimaryHDU(header=hdr0)
"""
In this table, we specify the priors / initial conditions and boundaries for the line fitting parameters.
"""

line_priors = np.rec.array([
    (6564.61, 'Ha', 6400, 6800, 'Ha_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05, 1),
    (6564.61, 'Ha', 6400, 6800, 'Ha_na', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    (6549.85, 'Ha', 6400, 6800, 'NII6549', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3, 1, 1, 1, 0.001, 1),
    (6585.28, 'Ha', 6400, 6800, 'NII6585', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3, 1, 1, 1, 0.003, 1),
    (6718.29, 'Ha', 6400, 6800, 'SII6718', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3, 1, 1, 2, 0.001, 1),
    (6732.67, 'Ha', 6400, 6800, 'SII6732', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 5e-3, 1, 1, 2, 0.001, 1),

    (4862.68, 'Hb', 4640, 5100, 'Hb_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.01, 0, 0, 0, 0.01, 1),
    (4862.68, 'Hb', 4640, 5100, 'Hb_na', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    (4960.30, 'Hb', 4640, 5100, 'OIII4959c', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    (5008.24, 'Hb', 4640, 5100, 'OIII5007c', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.01, 1, 1, 0, 0.004, 1),
    (4960.30, 'Hb', 4640, 5100, 'OIII4959w',   1, 0.0, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.001, 1),
    (5008.24, 'Hb', 4640, 5100, 'OIII5007w',   1, 0.0, 0.0, 1e10, 3e-3, 2.3e-4, 0.004,  0.01,  2, 2, 0, 0.002, 1),
    #(4687.02, 'Hb', 4640, 5100, 'HeII4687_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.004,  0.05,   0.005, 0, 0, 0, 0.001, 1),
    #(4687.02, 'Hb', 4640, 5100, 'HeII4687_na', 1, 0.0, 0.0, 1e10, 1e-3, 2.3e-4, 0.00169, 0.005, 1, 1, 0, 0.001, 1),

    #(3934.78, 'CaII', 3900, 3960, 'CaII3934' , 2, 0.0, 0.0, 1e10, 1e-3, 3.333e-4, 0.00169, 0.01, 99, 0, 0, -0.001, 1),

    #(3728.48, 'OII', 3650, 3800, 'OII3728', 1, 0.0, 0.0, 1e10, 1e-3, 3.333e-4, 0.00169, 0.01, 1, 1, 0, 0.001, 1),

    #(3426.84, 'NeV', 3380, 3480, 'NeV3426',    1, 0.0, 0.0, 1e10, 1e-3, 3.333e-4, 0.00169, 0.01, 0, 0, 0, 0.001, 1),
    #(3426.84, 'NeV', 3380, 3480, 'NeV3426_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.0025,   0.02,   0.01, 0, 0, 0, 0.001, 1),

    (2798.75, 'MgII', 2700, 2900, 'MgII_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05, 1),
    (2798.75, 'MgII', 2700, 2900, 'MgII_na', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),

    (1908.73, 'CIII', 1700, 1970, 'CIII_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 99, 0, 0, 0.01, 1),
    #(1908.73, 'CIII', 1700, 1970, 'CIII_na',   1, 0.0, 0.0, 1e10, 1e-3, 5e-4,  0.00169, 0.01,  1, 1, 0, 0.002, 1),
    #(1892.03, 'CIII', 1700, 1970, 'SiIII1892', 1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1857.40, 'CIII', 1700, 1970, 'AlIII1857', 1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.003, 1, 1, 0, 0.005, 1),
    #(1816.98, 'CIII', 1700, 1970, 'SiII1816',  1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1786.7,  'CIII', 1700, 1970, 'FeII1787',  1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.0002, 1),
    #(1750.26, 'CIII', 1700, 1970, 'NIII1750',  1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),
    #(1718.55, 'CIII', 1700, 1900, 'NIV1718',   1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015,  0.01,  1, 1, 0, 0.001, 1),

    (1549.06, 'CIV', 1500, 1700, 'CIV_br', 2, 0.0, 0.0, 1e10, 5e-3, 0.004, 0.05, 0.015, 0, 0, 0, 0.05, 1),
    # (1549.06, 'CIV', 1500, 1700, 'CIV_na', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 1, 1, 0, 0.002, 1),
    #(1640.42, 'CIV', 1500, 1700, 'HeII1640',    1, 0.0, 0.0, 1e10, 1e-3, 5e-4,   0.00169, 0.008, 1, 1, 0, 0.002, 1),
    #(1663.48, 'CIV', 1500, 1700, 'OIII1663',    1, 0.0, 0.0, 1e10, 1e-3, 5e-4,   0.00169, 0.008, 1, 1, 0, 0.002, 1),
    #(1640.42, 'CIV', 1500, 1700, 'HeII1640_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),
    #(1663.48, 'CIV', 1500, 1700, 'OIII1663_br', 1, 0.0, 0.0, 1e10, 5e-3, 0.0025, 0.02,   0.008, 1, 1, 0, 0.002, 1),

    #(1402.06, 'SiIV', 1290, 1450, 'SiIV_OIV1', 1, 0.0, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1396.76, 'SiIV', 1290, 1450, 'SiIV_OIV2', 1, 0.0, 0.0, 1e10, 5e-3, 0.002, 0.05,  0.015, 1, 1, 0, 0.05, 1),
    #(1335.30, 'SiIV', 1290, 1450, 'CII1335',   1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),
    #(1304.35, 'SiIV', 1290, 1450, 'OI1304',    1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.015, 0.01,  1, 1, 0, 0.001, 1),

    (1215.67, 'Lya', 1150, 1290, 'Lya_br', 3, 0.0, 0.0, 1e10, 5e-3, 0.002, 0.05, 0.02, 0, 0, 0, 0.05, 1),
    (1240.14, 'Lya', 1150, 1290, 'NV1240', 1, 0.0, 0.0, 1e10, 2e-3, 0.001, 0.01, 0.005, 0, 0, 0, 0.002, 1),
    # (1215.67, 'Lya', 1150, 1290, 'Lya_na', 1, 0.0, 0.0, 1e10, 1e-3, 5e-4, 0.00169, 0.01, 0, 0, 0, 0.002, 1),
    ],

    formats='float32,    a20,  float32, float32,      a20,  int32, float32, float32, float32, float32, float32, float32, float32,   int32,  int32,  int32, float32, int32',
    names=' lambda, compname,   minwav,  maxwav, linename, ngauss,  inisca,  minsca,  maxsca,  inisig,  minsig,  maxsig,    voff,  vindex, windex, findex,  fvalue,  vary')

# Header
hdr1 = fits.Header()
hdr1['lambda'] = 'Vacuum Wavelength in Ang'
hdr1['minwav'] = 'Lower complex fitting wavelength range'
hdr1['maxwav'] = 'Upper complex fitting wavelength range'
hdr1['ngauss'] = 'Number of Gaussians for the line'

# Can be set to negative for absorption lines if you want
hdr1['inisca'] = 'Initial guess of line scale [flux]'
hdr1['minsca'] = 'Lower range of line scale [flux]'
hdr1['maxsca'] = 'Upper range of line scale [flux]'

hdr1['inisig'] = 'Initial guess of linesigma [lnlambda]'
hdr1['minsig'] = 'Lower range of line sigma [lnlambda]'
hdr1['maxsig'] = 'Upper range of line sigma [lnlambda]'

hdr1['voff  '] = 'Limits on velocity offset from the central wavelength [lnlambda]'
hdr1['vindex'] = 'Entries w/ same NONZERO vindex constrained to have same velocity'
hdr1['windex'] = 'Entries w/ same NONZERO windex constrained to have same width'
hdr1['findex'] = 'Entries w/ same NONZERO findex have constrained flux ratios'
hdr1['fvalue'] = 'Relative scale factor for entries w/ same findex'

hdr1['vary'] = 'Whether or not to vary the parameter (set to 0 to fix the line parameter to initial values)'

# Save line info
hdu1 = fits.BinTableHDU(data=line_priors, header=hdr1, name='line_priors')
"""
In this table, we specify the windows and priors / initial conditions and boundaries for the continuum fitting parameters.
"""

conti_windows = np.rec.array([
    (1150., 1170.), 
    (1275., 1290.),
    (1350., 1360.),
    (1445., 1465.),
    (1690., 1705.),
    (1770., 1810.),
    (1970., 2400.),
    (2480., 2675.),
    (2925., 3400.),
    (3775., 3832.),
    (4000., 4050.),
    (4200., 4230.),
    (4435., 4640.),
    (5100., 5535.),
    (6005., 6035.),
    (6110., 6250.),
    (6800., 7000.),
    (7160., 7180.),
    (7500., 7800.),
    (8050., 8150.), # Continuum fitting windows (to avoid emission line, etc.)  [AA]
    ], 
    formats = 'float32,  float32',
    names =    'min,     max')

hdu2 = fits.BinTableHDU(data=conti_windows, name='conti_windows')

conti_priors = np.rec.array([
    ('Fe_uv_norm',  0.0,   0.0,   1e10,  1), # Normalization of the MgII Fe template [flux]
    ('Fe_uv_FWHM',  3000,  1200,  18000, 1), # FWHM of the MgII Fe template [AA]
    ('Fe_uv_shift', 0.0,   -0.01, 0.01,  1), # Wavelength shift of the MgII Fe template [lnlambda]
    ('Fe_op_norm',  0.0,   0.0,   1e10,  1), # Normalization of the Hbeta/Halpha Fe template [flux]
    ('Fe_op_FWHM',  3000,  1200,  18000, 1), # FWHM of the Hbeta/Halpha Fe template [AA]
    ('Fe_op_shift', 0.0,   -0.01, 0.01,  1), # Wavelength shift of the Hbeta/Halpha Fe template [lnlambda]
    ('PL_norm',     1.0,   0.0,   1e10,  1), # Normalization of the power-law (PL) continuum f_lambda = (lambda/3000)^-alpha
    ('PL_slope',    -1.5,  -5.0,  3.0,   1), # Slope of the power-law (PL) continuum
    ('Blamer_norm', 0.0,   0.0,   1e10,  1), # Normalization of the Balmer continuum at < 3646 AA [flux] (Dietrich et al. 2002)
    ('Balmer_Te',   15000, 10000, 50000, 1), # Te of the Balmer continuum at < 3646 AA [K?]
    ('Balmer_Tau',  0.5,   0.1,   2.0,   1), # Tau of the Balmer continuum at < 3646 AA
    ('conti_a_0',   0.0,   None,  None,  1), # 1st coefficient of the polynomial continuum
    ('conti_a_1',   0.0,   None,  None,  1), # 2nd coefficient of the polynomial continuum
    ('conti_a_2',   0.0,   None,  None,  1), # 3rd coefficient of the polynomial continuum
    # Note: The min/max bounds on the conti_a_0 coefficients are ignored by the code,
    # so they can be determined automatically for numerical stability.
    ],

    formats = 'a20,  float32, float32, float32, int32',
    names = 'parname, initial,   min,     max,     vary')

hdr3 = fits.Header()
hdr3['ini'] = 'Initial guess of line scale [flux]'
hdr3['min'] = 'FWHM of the MgII Fe template'
hdr3['max'] = 'Wavelength shift of the MgII Fe template'

hdr3['vary'] = 'Whether or not to vary the parameter (set to 0 to fix the continuum parameter to initial values)'


hdu3 = fits.BinTableHDU(data=conti_priors, header=hdr3, name='conti_priors')
"""
In this table, we allow user to customized some key parameters in our result measurements.
"""

measure_info = Table(
    [
        [[1350, 1450, 3000, 4200, 5100]],
        [[
            # [2240, 2650], 
            [4435, 4685],
        ]]
    ],
    names=([
        'cont_loc',
        'Fe_flux_range'
    ]),
    dtype=([
        'float32',
        'float32'
    ])
)
hdr4 = fits.Header()
hdr4['cont_loc'] = 'The wavelength of continuum luminosity in results'
hdr4['Fe_flux_range'] = 'Fe emission wavelength range calculated in results'

hdu4 = fits.BinTableHDU(data=measure_info, header=hdr4, name='measure_info')

#region unnecessary code

# hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4])
# hdu_list.writeto(os.path.join(path_ex, 'qsopar.fits'), overwrite=True)
# print(Table(line_priors))


# # Requried
# data = fits.open(os.path.join(path_ex, 'data/spec-0332-52367-0639.fits'))
# lam = 10 ** data[1].data['loglam']  # OBS wavelength [A]
# flux = data[1].data['flux']  # OBS flux [erg/s/cm^2/A]
# err = 1 / np.sqrt(data[1].data['ivar'])  # 1 sigma error
# z = data[2].data['z'][0]  # Redshift

# # Optional
# ra = data[0].header['plug_ra']  # RA
# dec = data[0].header['plug_dec']  # DEC
# plateid = data[0].header['plateid']  # SDSS plate ID
# mjd = data[0].header['mjd']  # SDSS MJD
# fiberid = data[0].header['fiberid']  # SDSS fiber ID


# # Prepare data
# q_mle = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, fiberid=fiberid, path=path_ex)

# # Double check the installation path with the PCA / Fe template files
# # print('install path:', q_mle.install_path)

# # Change it if you installed them somewhere else
# #q_mle.install_path = '...'

# start = timeit.default_timer()
# # Do the fitting

# q_mle.Fit(name=None,  # customize the name of given targets. Default: plate-mjd-fiber
#           # prepocessing parameters
#           nsmooth=1,  # do n-pixel smoothing to the raw input flux and err spectra
#           and_mask=False,  # delete the and masked pixels
#           or_mask=False,  # delete the or masked pixels
#           reject_badpix=True,  # reject 10 most possible outliers by the test of pointDistGESD
#           deredden=True,  # correct the Galactic extinction
#           wave_range=None,  # trim input wavelength
#           wave_mask=None,  # 2-D array, mask the given range(s)

#           # host decomposition parameters
#           decompose_host=True,  # If True, the host galaxy-QSO decomposition will be applied
#           host_prior=False, # If True, the code will adopt prior-informed method to assist decomposition. Currently, only 'CZBIN1' and 'DZBIN1' model for QSO PCA are available. And the model for galaxy must be PCA too.
#           host_prior_scale=0.2, # scale of prior panelty. Usually, 0.2 works fine for SDSS spectra. Adjust it smaller if you find the prior affect the fitting results too much.

#           host_line_mask=True, # If True, the line region of galaxy will be masked when subtracted from original spectra.
#           decomp_na_mask=True, # If True, the narrow line region will be masked when perform decomposition
#           qso_type='CZBIN1', # PCA template name for quasar
#           npca_qso=10, # numebr of quasar templates
#           host_type='PCA', # template name for galaxy
#           npca_gal=5, # number of galaxy templates
          
#           # continuum model fit parameters
#           Fe_uv_op=True,  # If True, fit continuum with UV and optical FeII template
#           poly=True,  # If True, fit continuum with the polynomial component to account for the dust reddening
#           BC=False,  # If True, fit continuum with Balmer continua from 1000 to 3646A
#           initial_guess=None,  # Initial parameters for continuum model, read the annotation of this function for detail
#           rej_abs_conti=False,  # If True, it will iterately reject 3 sigma outlier absorption pixels in the continuum
#           n_pix_min_conti=100,  # Minimum number of negative pixels for host continuuum fit to be rejected.

#           # emission line fit parameters
#           linefit=True,  # If True, the emission line will be fitted
#           rej_abs_line=False,
#           # If True, it will iterately reject 3 sigma outlier absorption pixels in the emission lines

#           # fitting method selection
#           MC=False,
#           # If True, do Monte Carlo resampling of the spectrum based on the input error array to produce the MC error array
#           MCMC=False,
#           # If True, do Markov Chain Monte Carlo sampling of the posterior probability densities to produce the error array
#           nsamp=2,
#           # The number of trials of the MC process (if MC=True) or number samples to run MCMC chain (if MCMC=True)

#           # advanced fitting parameters
#           param_file_name='qsopar.fits',  # Name of the qso fitting parameter FITS file.
#           nburn=20,  # The number of burn-in samples to run MCMC chain
#           nthin=10,  # To set the MCMC chain returns every n samples
#           epsilon_jitter=0.,
#           # Initial jitter for every initial guass to avoid local minimum. (Under test, not recommanded to change)

#           # customize the results
#           save_result=False,  # If True, all the fitting results will be saved to a fits file
#           save_fits_name=None,  # The output name of the result fits
#           save_fits_path=path_out,  # The output path of the result fits
#           plot_fig=True,  # If True, the fitting results will be plotted
#           save_fig=False,  # If True, the figure will be saved
#           plot_corner=True,  # Whether or not to plot the corner plot results if MCMC=True

#           # debugging mode
#           verbose=True,  # turn on (True) or off (False) debugging output

#           # sublevel parameters for figure plot and emcee
#           kwargs_plot={
#               'save_fig_path': '.',  # The output path of the figure
#               'broad_fwhm'   : 1200  # km/s, lower limit that code decide if a line component belongs to broad component
#           },
#           kwargs_conti_emcee={},
#           kwargs_line_emcee={})

# end = timeit.default_timer()

# print(f'Fitting finished in {np.round(end - start, 1)}s')

# # Prepare data
# q_mle = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, fiberid=fiberid, path=path_ex)

# # Double check the installation path with the PCA / Fe template files
# # print('install path:', q_mle.install_path)

# # Change it if you installed them somewhere else
# #q_mle.install_path = '...'

# start = timeit.default_timer()
# # Do the fitting

# q_mle.Fit(name=None,  # customize the name of given targets. Default: plate-mjd-fiber
#           # prepocessing parameters
#           nsmooth=1,  # do n-pixel smoothing to the raw input flux and err spectra
#           and_mask=False,  # delete the and masked pixels
#           or_mask=False,  # delete the or masked pixels
#           reject_badpix=True,  # reject 10 most possible outliers by the test of pointDistGESD
#           deredden=True,  # correct the Galactic extinction
#           wave_range=None,  # trim input wavelength
#           wave_mask=None,  # 2-D array, mask the given range(s)

#           # host decomposition parameters
#           decompose_host=True,  # If True, the host galaxy-QSO decomposition will be applied
#           host_prior=True, # If True, the code will adopt prior-informed method to assist decomposition. Currently, only 'CZBIN1' and 'DZBIN1' model for QSO PCA are available. And the model for galaxy must be PCA too.
#           host_prior_scale=0.2, # scale of prior panelty. Usually, 0.2 works fine for SDSS spectra. Adjust it smaller if you find the prior affect the fitting results too much.

#           host_line_mask=True, # If True, the line region of galaxy will be masked when subtracted from original spectra.
#           decomp_na_mask=True, # If True, the narrow line region will be masked when perform decomposition
#           qso_type='CZBIN1', # PCA template name for quasar
#           npca_qso=10, # numebr of quasar templates
#           host_type='PCA', # template name for galaxy
#           npca_gal=5, # number of galaxy templates
          
#           # continuum model fit parameters
#           Fe_uv_op=True,  # If True, fit continuum with UV and optical FeII template
#           poly=True,  # If True, fit continuum with the polynomial component to account for the dust reddening
#           BC=False,  # If True, fit continuum with Balmer continua from 1000 to 3646A
#           initial_guess=None,  # Initial parameters for continuum model, read the annotation of this function for detail
#           rej_abs_conti=False,  # If True, it will iterately reject 3 sigma outlier absorption pixels in the continuum
#           n_pix_min_conti=100,  # Minimum number of negative pixels for host continuuum fit to be rejected.

#           # emission line fit parameters
#           linefit=True,  # If True, the emission line will be fitted
#           rej_abs_line=False,
#           # If True, it will iterately reject 3 sigma outlier absorption pixels in the emission lines

#           # fitting method selection
#           MC=False,
#           # If True, do Monte Carlo resampling of the spectrum based on the input error array to produce the MC error array
#           MCMC=False,
#           # If True, do Markov Chain Monte Carlo sampling of the posterior probability densities to produce the error array
#           nsamp=2,
#           # The number of trials of the MC process (if MC=True) or number samples to run MCMC chain (if MCMC=True)

#           # advanced fitting parameters
#           param_file_name='qsopar.fits',  # Name of the qso fitting parameter FITS file.
#           nburn=20,  # The number of burn-in samples to run MCMC chain
#           nthin=10,  # To set the MCMC chain returns every n samples
#           epsilon_jitter=0.,
#           # Initial jitter for every initial guass to avoid local minimum. (Under test, not recommanded to change)

#           # customize the results
#           save_result=False,  # If True, all the fitting results will be saved to a fits file
#           save_fits_name=None,  # The output name of the result fits
#           save_fits_path=path_out,  # The output path of the result fits
#           plot_fig=True,  # If True, the fitting results will be plotted
#           save_fig=False,  # If True, the figure will be saved
#           plot_corner=True,  # Whether or not to plot the corner plot results if MCMC=True

#           # debugging mode
#           verbose=True,  # turn on (True) or off (False) debugging output

#           # sublevel parameters for figure plot and emcee
#           kwargs_plot={
#               'save_fig_path': '.',  # The output path of the figure
#               'broad_fwhm'   : 1200  # km/s, lower limit that code decide if a line component belongs to broad component
#           },
#           kwargs_conti_emcee={},
#           kwargs_line_emcee={})

# end = timeit.default_timer()

# print(f'Fitting finished in {np.round(end - start, 1)}s')



# q_mcmc = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, fiberid=fiberid, path=path_ex)

# start = timeit.default_timer()
# # Do the fitting
# q_mcmc.Fit(name=None, nsmooth=1, deredden=True, reject_badpix=False, wave_range=None, host_type="PCA", \
#            wave_mask=None, decompose_host=True, host_prior=False, decomp_na_mask=True, npca_gal=5, npca_qso=10, qso_type='CZBIN1',
#            Fe_uv_op=True, poly=True, rej_abs_conti=False, \
#            MCMC=True, epsilon_jitter=0, nburn=10, nsamp=4, nthin=10, linefit=True, save_result=True, \
#            plot_fig=True, save_fig=False, plot_corner=True, kwargs_plot={'save_fig_path': '.'}, save_fits_name=None,
#            verbose=False)

# end = timeit.default_timer()

# print(f'Fitting finished in {np.round(end - start, 1)}s')


# fig, ax = plt.subplots(figsize=(15, 5))

# # Plot the quasar rest frame spectrum after removed the host galaxy component
# ax.plot(q_mcmc.wave, q_mcmc.flux, 'grey', label='QSO Data')
# ax.plot(q_mcmc.wave, q_mcmc.err, 'r', label='Error')

# # Skip the error results before plotting
# if q_mcmc.MCMC == True:
#     gauss_result = q_mcmc.gauss_result[::2]
# else:
#     gauss_result = q_mcmc.gauss_result

# # To plot the whole model, we use Manygauss to show the line fitting results saved in gauss_result
# ax.plot(q_mcmc.wave, q_mcmc.Manygauss(np.log(q_mcmc.wave), gauss_result) + q_mcmc.f_conti_model, 'b', label='Line',
#         lw=2)
# ax.plot(q_mcmc.wave, q_mcmc.f_conti_model, 'c', lw=2, label='Continuum+FeII')
# ax.plot(q_mcmc.wave, q_mcmc.PL_poly_BC, 'orange', lw=2, label='Continuum')
# ax.plot(q_mcmc.wave, q_mcmc.host, 'm', lw=2, label='Host')
# plt.legend()

# ax.set_xlim(3500, 8000)
# ax.set_xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
# ax.set_ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20)

# #print('optical Fe flux (10^(-17) erg/s/cm^2): ' + q_mcmc.conti_result[q_mcmc.conti_result_name=='Fe_flux_4435_4685'][0])

# Fe_flux_result, Fe_flux_type, Fe_flux_name = q_mcmc.Get_Fe_flux(np.array([4400, 4900]))
# print('Fe flux within a specific range: \n' + Fe_flux_name[0] + ': ' + str(Fe_flux_result[0]))


# fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# if q_mcmc.MCMC:
#     gauss_result = q_mcmc.gauss_result[::2]
# else:
#     gauss_result = q_mcmc.gauss_result

# # Plot individual line components
# for p in range(int(len(gauss_result) / 3)):
#     if q_mcmc.CalFWHM(gauss_result[3 * p + 2]) < 1200:  # < 1200 km/s narrow
#         color = 'g'  # narrow
#     else:
#         color = 'r'  # broad
#     ax.plot(q_mcmc.wave, q_mcmc.Onegauss(np.log(q_mcmc.wave), gauss_result[p * 3:(p + 1) * 3]), color=color)

# # Plot total line model
# ax.plot(q_mcmc.wave, q_mcmc.Manygauss(np.log(q_mcmc.wave), gauss_result), 'b', lw=2)
# ax.plot(q_mcmc.wave, q_mcmc.line_flux, 'k')
# ax.set_xlim(4640, 5100)
# ax.set_xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
# ax.set_ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20)

# """
# Line properties
# """

# # The line_prop function is used to calculate the broad line properties
# # (defined, by default, as ln sigma > 0.00169 (1200 km/s) )

# # OLD WAY: If you want to calculate the paramters of broad Hb
# # then find all the broad Hb component, i.e., Hb_br_[1,2,3]_[scale,centerwave,sigma]
# # for here q.line_result_name[12:15], q.line_result[12:15] is the broad Hb
# # If MCMC=False, this would be:
# # fwhm, sigma, ew, peak, area = q.line_prop(q.linelist[6][0], q.line_result[12:15], 'broad')

# # NEW WAY: using line_prop_from_name convenience function
# fwhm, sigma, ew, peak, area, snr = q_mcmc.line_prop_from_name('Hb_br', 'broad')

# print("Broad Hb:")
# print("FWHM (km/s)", np.round(fwhm, 1))
# print("Sigma (km/s)", np.round(sigma, 1))
# print("EW (A)", np.round(ew, 1))
# print("Peak (A)", np.round(peak, 1))
# print("Area (10^(-17) erg/s/cm^2)", np.round(area, 1))
# print("")

# # OLD WAY: If you want to calculate the  the narrow [OIII]5007
# # If MCMC=False, this would be:
# # the coresponding parameters are  q.line_result_name[21:24], q.line_result[21:24]
# # fwhm, sigma, ew, peak, area = q.line_prop(q.linelist[6][0], q.line_result[21:24], 'narrow')

# fwhm, sigma, ew, peak, area, snr = q_mcmc.line_prop_from_name('OIII5007c', 'narrow')

# print("Narrow [OIII]5007:")
# print("FWHM (km/s)", np.round(fwhm, 1))
# print("Sigma (km/s)", np.round(sigma, 1))
# print("EW (A)", np.round(ew, 1))
# print("Peak (A)", np.round(peak, 1))
# print("Area (10^(-17) erg/s/cm^2)", np.round(area, 1))

# # Required
# data = fits.open(os.path.join(path_ex, 'data/spec-0332-52367-0639.fits'))
# lam = 10 ** data[1].data['loglam']  # OBS wavelength [A]
# flux = data[1].data['flux']  # OBS flux [erg/s/cm^2/A]
# err = 1 / np.sqrt(data[1].data['ivar'])  # 1 sigma error
# z = data[2].data['z'][0]  # Redshift

# # Optional
# ra = data[0].header['plug_ra']  # RA
# dec = data[0].header['plug_dec']  # DEC
# plateid = data[0].header['plateid']  # SDSS plate ID
# mjd = data[0].header['mjd']  # SDSS MJD
# fiberid = data[0].header['fiberid']  # SDSS fiber ID

# ngauss_max = 5  # stop at 5 components
# bic_last = np.inf

# # Number of Gaussians to try loop
# for ngauss in range(1, ngauss_max):

#     print(fr'Fitting broad H$\alpha$ with {ngauss} components.')

#     # Change the number of Gaussians for the Ha line in the line parameter file

#     hdu1_new = hdu1.copy()
#     if 'Ha_br' in hdu1.data['linename']:
#         hdu1.data['ngauss'][hdu1.data['linename'] == 'Ha_br'] = ngauss
#     hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4])
#     hdu_list.writeto(os.path.join(path_ex, 'qsopar.fits'), overwrite=True)

#     # Do the fitting
#     q = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, fiberid=fiberid, path=path_ex)

#     q.Fit(name=None, nsmooth=1, deredden=True, reject_badpix=False, wave_range=None, host_type="PCA", \
#           wave_mask=None, decompose_host=True, host_prior=True, decomp_na_mask=True, npca_gal=5, npca_qso=10, qso_type='CZBIN1',\
#           Fe_uv_op=True, poly=True, BC=False, rej_abs_conti=False, rej_abs_line=False, \
#           initial_guess=None, MCMC=False, nburn=20, nsamp=2, nthin=10, linefit=True, \
#           save_result=False, plot_fig=False, verbose=False)

#     mask_Ha_bic = q.line_result_name == '2_line_min_chi2'

#     bic = float(q.line_result[mask_Ha_bic][0])

#     print(f'Delta BIC = {np.round(bic_last - bic, 1)}')

#     # Stop condition of Delta BIC = 10 is a good rule of thumb
#     if bic_last - bic < 10:
#         print(f'{ngauss-1} components is prefered')
#         break

#     bic_last = bic

#     # Plot the result
#     if q.MCMC:
#         gauss_result = q.gauss_result[::2]
#     else:
#         gauss_result = q.gauss_result

#     fig, ax = plt.subplots(1, 1, figsize=(8, 5))

#     # Plot individual line components
#     for p in range(len(gauss_result) // 3):
#         if q.CalFWHM(gauss_result[3 * p + 2]) < 1200:  # < 1200 km/s narrow
#             color = 'g'  # narrow
#         else:
#             color = 'r'  # broad
#         ax.plot(q.wave, q.Onegauss(np.log(q.wave), gauss_result[p * 3:(p + 1) * 3]), color=color)

#     # Plot total line model
#     ax.plot(q.wave, q.Manygauss(np.log(q.wave), gauss_result), 'b', lw=2)
#     ax.plot(q.wave, q.line_flux, 'k')
#     ax.set_xlim(6400, 6800)
#     ax.set_xlabel(r'$\rm Rest \, Wavelength$ ($\rm \AA$)', fontsize=20)
#     ax.set_ylabel(r'$\rm f_{\lambda}$ ($\rm 10^{-17} erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)', fontsize=20)

#     c = 1  # Ha
#     ax.text(0.02, 0.80, r'$\chi ^2_\nu=$' + str(np.round(float(q.comp_result[c * 7 + 4]), 2)),
#             fontsize=16, transform=ax.transAxes)

#     plt.show()

# def job(file):
#     print(file + '\n')

#     data = fits.open(file)

#     lam = 10 ** data[1].data['loglam']
#     flux = data[1].data['flux']
#     err = 1 / np.sqrt(data[1].data['ivar'])
#     ra = data[0].header['plug_ra']
#     dec = data[0].header['plug_dec']
#     z = data[2].data['z'][0]
#     plateid = data[0].header['plateid']
#     mjd = data[0].header['mjd']
#     fiberid = data[0].header['fiberid']

#     q = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, fiberid=fiberid, path=path_ex)

#     q.Fit(name=None, nsmooth=1, deredden=True, reject_badpix=False, wave_range=None, wave_mask=None,
#           decompose_host=True, host_prior=False, decomp_na_mask=True, npca_gal=5, npca_qso=20, Fe_uv_op=True, poly=True, BC=False, rej_abs_conti=False,
#           rej_abs_line=False, host_type="PCA",
#           MCMC=False, plot_fig=False, save_fig=False, save_result=True,
#           kwargs_plot={'save_fig_path': '.', 'broad_fwhm': 1200},
#           save_fits_path=path_out, save_fits_name=None, verbose=False)

# # Edit the directory before use
# from multiprocess import Pool # Use multiprocessing instead if you are running in a script!
# import os, timeit, glob

# if __name__ == '__main__':
#     start = timeit.default_timer()

#     files = glob.glob(os.path.join(path_ex, 'data/spec-*.fits'))

#     pool = Pool(3)  # Create a multiprocessing Pool
#     pool.imap(func=job, iterable=files)

#     end = timeit.default_timer()
#     print(f'Fitting finished in : {np.round(end - start)}s')
#endregion

# Remove the broad components from the line parameter file 
mask_na = [True if not '_br' in str(row['linename']) else False for row in hdu1.data]

# Or you can set broad component number to zero

# Save line info
hdu1 = fits.BinTableHDU(data=hdu1.data[mask_na], header=hdr1, name='data')
hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4])
hdu_list.writeto(os.path.join(path_ex, 'qsopar.fits'), overwrite=True)

data = fits.open(os.path.join(path_ex, 'data/spec-0266-51602-0107.fits'))
lam = 10 ** data[1].data['loglam']  # OBS wavelength [A]
flux = data[1].data['flux']  # OBS flux [erg/s/cm^2/A]
err = 1 / np.sqrt(data[1].data['ivar'])  # 1 sigma error
z = data[2].data['z'][0]  # Redshift

# Optional
ra = data[0].header['plug_ra']  # RA
dec = data[0].header['plug_dec']  # DEC
plateid = data[0].header['plateid']  # SDSS plate ID
mjd = data[0].header['mjd']  # SDSS MJD
fiberid = data[0].header['fiberid']  # SDSS fiber ID

q = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, fiberid=fiberid, path=path_ex)

q.Fit(name="example_fit_spec-0266-51602-0107", nsmooth=1, deredden=True, reject_badpix=False, wave_range=None, host_type="PCA", \
      wave_mask=None, decompose_host=True, host_prior=True, decomp_na_mask=True, npca_gal=5, npca_qso=10, qso_type='CZBIN1',\
      Fe_uv_op=False, poly=False, BC=False, rej_abs_conti=False, rej_abs_line=False, MCMC=False, \
      save_result=False, plot_fig=True, save_fig=True, save_fits_path=".", save_fits_name=None)

# data = fits.open(os.path.join(path_ex, 'data/spec-0266-51602-0013.fits'))
# lam = 10 ** data[1].data['loglam']  # OBS wavelength [A]
# flux = data[1].data['flux']  # OBS flux [erg/s/cm^2/A]
# err = 1 / np.sqrt(data[1].data['ivar'])  # 1 sigma error
# z = data[2].data['z'][0]  # Redshift

# # Optional
# ra = data[0].header['plug_ra']  # RA
# dec = data[0].header['plug_dec']  # DEC
# plateid = data[0].header['plateid']  # SDSS plate ID
# mjd = data[0].header['mjd']  # SDSS MJD
# fiberid = data[0].header['fiberid']  # SDSS fiber ID

# q = QSOFit(lam, flux, err, z, ra=ra, dec=dec, plateid=plateid, mjd=mjd, fiberid=fiberid, path=path_ex)

# q.Fit(name=None, nsmooth=1, deredden=True, reject_badpix=False, wave_range=None, \
#       wave_mask=None, decompose_host=True, host_prior=True, decomp_na_mask=True, npca_gal=5, npca_qso=10, qso_type='CZBIN1', linefit=False, \
#       Fe_uv_op=False, poly=False, BC=False, MCMC=False, host_type="PCA", \
#       save_result=False, plot_fig=True, save_fig=False, save_fits_path=path_out, save_fits_name=None)