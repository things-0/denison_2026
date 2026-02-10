import numpy as np
from astropy.cosmology import Planck18 as cosmo 
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from . import constants as const


def get_luminosity(flux: float, flux_err: float, z: float = const.Z_SPEC) -> float:
    dist = cosmo.luminosity_distance(z)
    dist_cm = dist.to(u.cm)

    flux_actual = flux * u.erg / (u.s * u.cm**2) * 1e-17
    flux_err_actual = flux_err * u.erg / (u.s * u.cm**2) * 1e-17

    luminosity = flux_actual * 4 * np.pi * (dist_cm**2)
    lum_err = flux_err_actual * 4 * np.pi * (dist_cm**2)

    return luminosity.value, lum_err.value

def get_bh_mass(
    lum_alpha: float,
    lum_alpha_err: float,
    fwhm_alpha: float,
    fwhm_alpha_err: float
) -> tuple[float, float]:
    """
    Estimate BH mass from Halpha line. Using the equation from Greene & Ho 2005
    
    Parameters:
    l_ha : Halpha luminosity in erg/s (with error l_ha_err)
    fwhm_ha : Halpha line FWHM in km/s (with error fwhm_ha_err)

    Returns:
    mbh : Black hole mass in solar masses (with error mbh_err)
    Note: mbh_err calculated using log(mass) for easier calculation
    """
    coeff = 2.0e6   # a
    exp_lum = 0.55  # b
    exp_fwhm = 2.06 # c

    # Normalize inputs
    lum_norm = lum_alpha / 1e42     # L
    fwhm_norm = fwhm_alpha / 1e3    # V

    # M = a * L^b * V^c

    mbh = coeff * (lum_norm**exp_lum) * (fwhm_norm**exp_fwhm)

    # uncertainty (using log(mass))
    sig_log_lum = lum_alpha_err / lum_alpha     # error of log(luminosity) = δL/L
    sig_log_fwhm = fwhm_alpha_err / fwhm_alpha  # error of log(fwhm) = δV/V
    sig_log_coeff = 0.35 / 2 # +0.4 -0.3
    sig_exp_lum = 0.02
    sig_exp_fwhm = 0.06

    var_log_mbh = (
        sig_log_coeff**2 +
        (np.log(lum_norm) * sig_exp_lum)**2 +   # derivative * uncertainty...?
        (exp_lum * sig_log_lum)**2 +
        (np.log(fwhm_norm) * sig_exp_fwhm)**2 +
        (exp_fwhm*sig_log_fwhm)**2
    )

    sig_log_mbh = np.sqrt(var_log_mbh)
    mbh_err = mbh * sig_log_mbh

    return mbh, mbh_err