import numpy as np
from astropy.cosmology import Planck18 as cosmo 
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from . import constants as const


def get_luminosity(flux: float, flux_err: float, z: float = const.Z_LUM) -> float:
    """
    Get the luminosity of a given line flux.

    Parameters
    ----------
    flux: float
        The flux of the line in 10^-17 erg/s/cm^2.
    flux_err: float
        The error on the flux of the line in 10^-17 erg/s/cm^2.
    z: float = const.Z_LUM
        The redshift of the object (used for luminosity distance calculation).

    Returns
    -------
    luminosity: float
        The luminosity of the line in erg/s.
    lum_err: float
        The error on the luminosity of the line in erg/s.
    """
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
    Estimate BH mass from Hα line using the equation from Greene & Ho (2005).
    
    Parameters
    ----------
    lum_alpha: float
        The luminosity of the Hα line in erg/s.
    lum_alpha_err: float
        The error on the luminosity of the Hα line in erg/s.
    fwhm_alpha: float
        The FWHM of the Hα line in km/s.
    fwhm_alpha_err: float
        The error on the FWHM of the Hα line in km/s.

    Returns
    -------
    mbh: float
        The BH mass in solar masses.
    mbh_err: float
        The error on the BH mass in solar masses.
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
    sig_log_coeff_upper = 0.4e6 / coeff         # +ve error of log(coeff) = δa/a
    sig_log_coeff_lower = 0.3e6 / coeff         # -ve error of log(coeff) = δa/a
    sig_exp_lum = 0.02
    sig_exp_fwhm = 0.06

    var_mbh_wout_coeff = (
        (np.log(lum_norm) * sig_exp_lum)**2 +
        (exp_lum * sig_log_lum)**2 +
        (np.log(fwhm_norm) * sig_exp_fwhm)**2 +
        (exp_fwhm*sig_log_fwhm)**2
    )
    var_mbh_upper = (
        sig_log_coeff_upper**2 +
        var_mbh_wout_coeff
    )
    var_mbh_lower = (
        sig_log_coeff_lower**2 +
        var_mbh_wout_coeff
    )

    sig_mbh_upper = np.sqrt(var_mbh_upper)
    sig_mbh_lower = np.sqrt(var_mbh_lower)
    mbh_err = (mbh * sig_mbh_lower, mbh * sig_mbh_upper)

    return mbh, mbh_err