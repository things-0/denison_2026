import numpy as np
from astropy.cosmology import Planck18 as cosmo 
import astropy.units as u

from .constants import *


def fwhm_from_fit(x, y, sig, model, params, n_trials):
    # n_trials = no. of monte carlo samples
  fwhm_list = []
  weights = 1.0 / sig
  for _ in range(n_trials):
    y_perturbed = y + np.random.normal(0, sig) # perturb the data with noise from uncertainty arr

    result_mc = model.fit(y_perturbed, params.copy(), x=x, weights=weights) # refit model

    #compute fwhm
    y_fit_mc = result_mc.best_fit
    half_max = np.max(y_fit_mc) / 2
    above_half = np.where(y_fit_mc >= half_max)[0]
    if len(above_half) > 1:
      fwhm = x[above_half[-1]] - x[above_half[0]]
      fwhm_list.append(fwhm)

  # extract values
  fwhm_mean = np.mean(fwhm_list) * u.AA
  fwhm_std = np.std(fwhm_list) * u.AA

  # convert to km/s
  rest_wavelength = 6562.8 * u.AA  # Hα rest wavelength
  fwhm_vel = (fwhm_mean / rest_wavelength) * C_KM_S
  fwhm_vel_err = (fwhm_std / rest_wavelength)* C_KM_S

  return fwhm_vel, fwhm_vel_err

def lum_from_fit(x, y, sig, model, params, n_trials):
  trapflux_list = []
  weights = 1.0 / sig
  for _ in range(n_trials):
      y_perturbed = y + np.random.normal(0, sig) # perturb the data with noise from uncertainty arr

      result_mc = model.fit(y_perturbed, params.copy(), x=x, weights=weights) #refit model

      y_fit_mc = result_mc.best_fit
      trapflux = np.trapz(y_fit_mc, x)  #integrate with trapezoidal rule
      trapflux_list.append(trapflux)

  # extract values: mean and sd of the sampled integrals
  trap_mean = np.mean(trapflux_list) * (10**(-17)) * u.erg / (u.s * u.cm**2)
  trap_std = np.std(trapflux_list) * (10**(-17)) * u.erg / (u.s * u.cm**2)
  
  # Luminosity distance
  z = 0.0582
  d = cosmo.luminosity_distance(z)
  d = d.to(u.cm)

  # compute luminosity
  luminosity = trap_mean * 4 * np.pi * (d**2)
  lum_err = trap_std * 4 * np.pi * (d**2)

  return luminosity, lum_err


def get_bh_mass(l_ha, l_ha_err, fwhm_ha, fwhm_ha_err):
    """
    Estimate BH mass from Halpha line. Using the equation from Greene & Ho 2005
    
    Parameters:
    l_ha : Halpha luminosity in erg/s (with error l_ha_err)
    fwhm_ha : Halpha line FWHM in km/s (with error fwhm_ha_err)

    Returns:
    mbh : Black hole mass in solar masses (with error mbh_err)
    Note: mbh_err calculated using log(mass) for easier calculation
    """
    coeff = 2.0e6  
    exp_l = 0.55
    exp_fwhm = 2.06

    # Normalize inputs
    l_norm = l_ha / 1e42
    fwhm_norm = fwhm_ha / 1e3 
    mbh = coeff * (l_norm**exp_l) * (fwhm_norm**exp_fwhm)

    # uncertainty (using log(mass))
    sig_loglum = (1/np.log(10)) * (l_ha_err / l_ha)
    sig_logfwhm = (1/np.log(10)) * (fwhm_ha_err / fwhm_ha)
    sig_loga = (1/np.log(10)) * (0.35 / 2)
    sig_b = 0.02
    sig_c = 0.06
    b = 0.55
    c = 2.06

    
    var_logm = (sig_loga)**2 + (np.log10(l_ha/(10**42)) * sig_b)**2 + (b*sig_loglum)**2 + (np.log10(fwhm_ha/(10**3)) * sig_c)**2 + (c*sig_logfwhm)**2

    sig_logm = np.sqrt(var_logm)
    mbh_err = np.log(10) * mbh * sig_logm

    return mbh, mbh_err