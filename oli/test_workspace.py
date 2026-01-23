import numpy as np
from astropy.table import Table
import sys
import argparse as ap
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    EarthLocation
    )
import astropy.units as u

# =============================================================================
# Flux measurements from different telescopes for SAMI 323854
# =============================================================================

# ATLAS difference imaging fluxes (in micro-Jansky)
atlas_2021_fluxes = np.array([17, 5, 16, 38])
atlas_2021_fluxes_errs = np.array([8, 9, 8, 10])

atlas_2022_fluxes = np.array([-2, -21, 12, 15])
atlas_2022_fluxes_errs = np.array([13, 11, 12, 11])

def get_weighted_average(flux, flux_err):
    """
    Calculate inverse-variance weighted mean and its uncertainty.
    
    Parameters
    ----------
    flux : np.ndarray
        Array of flux measurements
    flux_err : np.ndarray
        Array of flux uncertainties
        
    Returns
    -------
    weighted_mean : float
        The weighted mean of the flux values
    uncertainty : float
        The uncertainty on the weighted mean
    """
    weights = 1.0 / (flux_err**2)
    weighted_mean = np.sum(flux * weights) / np.sum(weights)
    uncertainty = 1.0 / np.sqrt(np.sum(weights))
    return weighted_mean, uncertainty

# =============================================================================
# ATLAS Weighted Averages
# =============================================================================
print("=" * 70)
print("ATLAS Difference Imaging Flux Weighted Averages (micro-Jansky)")
print("=" * 70)

atlas_2021_mean, atlas_2021_err = get_weighted_average(atlas_2021_fluxes, atlas_2021_fluxes_errs)
atlas_2022_mean, atlas_2022_err = get_weighted_average(atlas_2022_fluxes, atlas_2022_fluxes_errs)

print(f"ATLAS 2021: {atlas_2021_mean:.3f} ± {atlas_2021_err:.3f} μJy")
print(f"  Input fluxes: {atlas_2021_fluxes} μJy")
print(f"  Input errors: {atlas_2021_fluxes_errs} μJy")
print()
print(f"ATLAS 2022: {atlas_2022_mean:.3f} ± {atlas_2022_err:.3f} μJy")
print(f"  Input fluxes: {atlas_2022_fluxes} μJy")
print(f"  Input errors: {atlas_2022_fluxes_errs} μJy")
print()

# =============================================================================
# ZTF Magnitude to Flux Conversion
# =============================================================================
print("=" * 70)
print("ZTF Magnitude to Flux Conversion")
print("=" * 70)

ztf_2021_mag = np.array([16.5218, 16.5421, 16.5487])
ztf_2021_mag_err = np.array([0.014664, 0.01477, 0.014804])

ztf_2022_mag = np.array([16.6011])
ztf_2022_mag_err = np.array([0.01509])

# Convert AB magnitude to flux in milli-Jansky
# Formula: F_Jy = 3631 * 10^(-0.4 * mag), then multiply by 1000 for mJy
ztf_2022_flux_mJy = 3631 * 10**(-0.4 * ztf_2022_mag) * 1000
# Flux error propagation: δF/F = 0.921 * δm (where 0.921 ≈ ln(10)/2.5)
ztf_2022_flux_err_mJy = ztf_2022_flux_mJy * 0.921 * ztf_2022_mag_err

ztf_2022_mean, ztf_2022_err = get_weighted_average(ztf_2022_flux_mJy, ztf_2022_flux_err_mJy)

print(f"ZTF 2022 magnitude: {ztf_2022_mag[0]:.4f} ± {ztf_2022_mag_err[0]:.5f} mag")
print(f"ZTF 2022 flux:      {ztf_2022_mean:.4f} ± {ztf_2022_err:.4f} mJy")
print(f"  Conversion: F_mJy = 3631 × 10^(-0.4 × mag) × 1000")
print()

# Convert observation time
t_mjd = Time(2459252.77, format='jd').mjd
print(f"Reference observation time: JD 2459252.77 = MJD {t_mjd:.2f}")
print()

# =============================================================================
# ASAS-SN Weighted Averages
# =============================================================================
print("=" * 70)
print("ASAS-SN Flux Weighted Averages (milli-Jansky)")
print("=" * 70)

asn_2021_flux = np.array([1.036])
asn_2021_flux_err = np.array([0.033])

asn_2022_flux = np.array([0.944, 0.974])
asn_2022_flux_err = np.array([0.022, 0.029])

asn_2021_mean, asn_2021_err = get_weighted_average(asn_2021_flux, asn_2021_flux_err)
asn_2022_mean, asn_2022_err = get_weighted_average(asn_2022_flux, asn_2022_flux_err)

print(f"ASAS-SN 2021: {asn_2021_mean:.3f} ± {asn_2021_err:.3f} mJy")
print(f"  Input fluxes: {asn_2021_flux} mJy")
print()
print(f"ASAS-SN 2022: {asn_2022_mean:.3f} ± {asn_2022_err:.3f} mJy")
print(f"  Input fluxes: {asn_2022_flux} mJy")
print()

# =============================================================================
# Attenuation Corrections
# =============================================================================
print("=" * 70)
print("Attenuation-Corrected Fluxes (milli-Jansky)")
print("=" * 70)
print("Attenuation factors (fraction of flux from AGN):")

# Attenuation factors - fraction of total flux from AGN
# These represent how much the AGN variability is "diluted" by host galaxy light
ztf_atten = 0.137  # ZTF r-band
at_atten = 0.7299  # ATLAS orange band (less host contamination)
as_atten = 0.36    # ASAS-SN V-band

print(f"  ZTF (r-band):     {ztf_atten:.3f} (heavily diluted by host)")
print(f"  ATLAS (o-band):   {at_atten:.4f} (moderate host contribution)")
print(f"  ASAS-SN (V-band): {as_atten:.2f}")
print()

# Summary values [flux, uncertainty] in mJy
ztf_2021 = [0.881, 0.0069]  # mJy
ztf_2022 = [0.83, 0.0012]   # mJy
at_2021 = [17.86/1000, 4.32/1000]  # Convert μJy to mJy
at_2022 = [0.75/1000, 5.83/1000]   # Convert μJy to mJy
as_2021 = [1.036, 0.033]    # mJy
as_2022 = [0.955, 0.17]     # mJy

print("Corrected AGN fluxes (F_AGN = F_observed / attenuation_factor):")
print("-" * 70)
print(f"{'Telescope':<12} {'Year':<6} {'Observed (mJy)':<20} {'Corrected AGN (mJy)':<25}")
print("-" * 70)

ztf_2021_corr = np.array(ztf_2021) / ztf_atten
ztf_2022_corr = np.array(ztf_2022) / ztf_atten
at_2021_corr = np.array(at_2021) / at_atten
at_2022_corr = np.array(at_2022) / at_atten
as_2021_corr = np.array(as_2021) / as_atten
as_2022_corr = np.array(as_2022) / as_atten

print(f"{'ZTF':<12} {'2021':<6} {ztf_2021[0]:.4f} ± {ztf_2021[1]:.4f}       {ztf_2021_corr[0]:.4f} ± {ztf_2021_corr[1]:.4f}")
print(f"{'ZTF':<12} {'2022':<6} {ztf_2022[0]:.4f} ± {ztf_2022[1]:.4f}       {ztf_2022_corr[0]:.4f} ± {ztf_2022_corr[1]:.4f}")
print(f"{'ATLAS':<12} {'2021':<6} {at_2021[0]:.5f} ± {at_2021[1]:.5f}     {at_2021_corr[0]:.5f} ± {at_2021_corr[1]:.5f}")
print(f"{'ATLAS':<12} {'2022':<6} {at_2022[0]:.5f} ± {at_2022[1]:.5f}     {at_2022_corr[0]:.5f} ± {at_2022_corr[1]:.5f}")
print(f"{'ASAS-SN':<12} {'2021':<6} {as_2021[0]:.3f} ± {as_2021[1]:.3f}         {as_2021_corr[0]:.4f} ± {as_2021_corr[1]:.4f}")
print(f"{'ASAS-SN':<12} {'2022':<6} {as_2022[0]:.3f} ± {as_2022[1]:.2f}          {as_2022_corr[0]:.4f} ± {as_2022_corr[1]:.4f}")
print("-" * 70)
print()
print("Note: The corrected flux represents the intrinsic AGN flux after")
print("      removing the diluting effect of host galaxy light.")