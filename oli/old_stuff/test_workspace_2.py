import numpy as np
from astropy.time import Time

# =============================================================================
# Flux measurements from different telescopes for SAMI 323854
# =============================================================================
# This script calculates weighted averages of flux/variability measurements
# from ATLAS, ZTF, and ASAS-SN, then corrects for host galaxy dilution.
# =============================================================================

# -----------------------------------------------------------------------------
# ATLAS difference imaging fluxes (in micro-Jansky)
# NOTE: These are DIFFERENCE fluxes - they measure variability, not total flux
# -----------------------------------------------------------------------------
atlas_2021_fluxes = np.array([17, 5, 16, 38])
atlas_2021_fluxes_errs = np.array([8, 9, 8, 10])

atlas_2022_fluxes = np.array([-2, -21, 12, 15])
atlas_2022_fluxes_errs = np.array([13, 11, 12, 11])

# -----------------------------------------------------------------------------
# ZTF magnitudes (AB system, r-band)
# -----------------------------------------------------------------------------
ztf_2021_mag = np.array([16.5218, 16.5421, 16.5487])
ztf_2021_mag_err = np.array([0.014664, 0.01477, 0.014804])

ztf_2022_mag = np.array([16.6011])
ztf_2022_mag_err = np.array([0.01509])

# -----------------------------------------------------------------------------
# ASAS-SN fluxes (milli-Jansky)
# -----------------------------------------------------------------------------
asn_2021_flux = np.array([1.036])
asn_2021_flux_err = np.array([0.033])

asn_2022_flux = np.array([0.944, 0.974])
asn_2022_flux_err = np.array([0.022, 0.029])

# -----------------------------------------------------------------------------
# Dilution factors - fraction of observed flux/variability from the AGN
# These account for host galaxy light within the aperture/PSF
# -----------------------------------------------------------------------------
ztf_dilution = 0.137   # ZTF r-band: only 13.7% of flux is from AGN
atlas_dilution = 0.7299  # ATLAS o-band: 73% of flux is from AGN  
asn_dilution = 0.36    # ASAS-SN V-band: 36% of flux is from AGN


def get_weighted_average(values, errors):
    """
    Calculate inverse-variance weighted mean and its uncertainty.
    
    Uses weights w_i = 1/σ_i², which is optimal for combining independent
    measurements with Gaussian uncertainties.
    
    Parameters
    ----------
    values : np.ndarray
        Array of measurements
    errors : np.ndarray
        Array of measurement uncertainties (1-sigma)
        
    Returns
    -------
    weighted_mean : float
        The weighted mean: Σ(x_i * w_i) / Σ(w_i)
    uncertainty : float
        The uncertainty on the weighted mean: 1 / √(Σ w_i)
    """
    weights = 1.0 / (errors ** 2)
    weighted_mean = np.sum(values * weights) / np.sum(weights)
    uncertainty = 1.0 / np.sqrt(np.sum(weights))
    return weighted_mean, uncertainty


def mag_to_flux_mJy(mag, mag_err):
    """
    Convert AB magnitude to flux in milli-Jansky.
    
    The AB magnitude system is defined as:
        m_AB = -2.5 * log10(F_ν / 3631 Jy)
    
    Therefore:
        F_ν = 3631 Jy * 10^(-0.4 * m_AB)
    
    Error propagation:
        δF/F = |d(ln F)/dm| * δm = 0.4 * ln(10) * δm ≈ 0.9210 * δm
    
    Parameters
    ----------
    mag : float or np.ndarray
        AB magnitude(s)
    mag_err : float or np.ndarray
        Magnitude uncertainty(ies)
        
    Returns
    -------
    flux_mJy : float or np.ndarray
        Flux in milli-Jansky
    flux_err_mJy : float or np.ndarray
        Flux uncertainty in milli-Jansky
    """
    flux_Jy = 3631 * 10 ** (-0.4 * mag)
    flux_mJy = flux_Jy * 1000  # Convert to mJy
    
    # Error propagation: δF = F * 0.4 * ln(10) * δm
    flux_err_mJy = flux_mJy * 0.4 * np.log(10) * mag_err
    
    return flux_mJy, flux_err_mJy


def correct_for_dilution(flux, flux_err, dilution_factor, measurement_type="variability"):
    """
    Correct flux/variability for host galaxy dilution.
    
    For VARIABILITY measurements (e.g., difference imaging, or comparing epochs):
        The observed variability is diluted: ΔF_obs = dilution × ΔF_AGN
        Therefore: ΔF_AGN = ΔF_obs / dilution
        
    For TOTAL FLUX measurements (if you want to extract the AGN component):
        The AGN contributes a fraction of total: F_AGN = dilution × F_total
        Therefore: F_AGN = dilution × F_obs (MULTIPLY, not divide!)
    
    Parameters
    ----------
    flux : float
        Observed flux or variability amplitude
    flux_err : float
        Uncertainty on observed flux
    dilution_factor : float
        Fraction of observed light coming from the AGN (0 < dilution ≤ 1)
    measurement_type : str
        Either "variability" or "total_flux"
        
    Returns
    -------
    corrected_flux : float
        Intrinsic AGN flux or variability
    corrected_err : float
        Uncertainty on corrected value
    """
    if measurement_type == "variability":
        # Divide to recover intrinsic variability amplitude
        corrected_flux = flux / dilution_factor
        corrected_err = flux_err / dilution_factor
    elif measurement_type == "total_flux":
        # Multiply to extract AGN component from total
        corrected_flux = flux * dilution_factor
        corrected_err = flux_err * dilution_factor
    else:
        raise ValueError(f"measurement_type must be 'variability' or 'total_flux', got '{measurement_type}'")
    
    return corrected_flux, corrected_err


# =============================================================================
# MAIN CALCULATIONS
# =============================================================================

print("=" * 75)
print("PHOTOMETRIC ANALYSIS OF SAMI 323854 (AGN)")
print("=" * 75)
print()

# -----------------------------------------------------------------------------
# 1. ATLAS Difference Imaging (Variability)
# -----------------------------------------------------------------------------
print("-" * 75)
print("1. ATLAS DIFFERENCE IMAGING FLUXES")
print("-" * 75)
print("   These are VARIABILITY measurements from difference imaging (μJy)")
print()

atlas_2021_mean, atlas_2021_err = get_weighted_average(atlas_2021_fluxes, atlas_2021_fluxes_errs)
atlas_2022_mean, atlas_2022_err = get_weighted_average(atlas_2022_fluxes, atlas_2022_fluxes_errs)

print(f"   ATLAS 2021:")
print(f"     Input measurements: {atlas_2021_fluxes} μJy")
print(f"     Input uncertainties: {atlas_2021_fluxes_errs} μJy")
print(f"     Weighted mean: {atlas_2021_mean:.2f} ± {atlas_2021_err:.2f} μJy")
print()
print(f"   ATLAS 2022:")
print(f"     Input measurements: {atlas_2022_fluxes} μJy")
print(f"     Input uncertainties: {atlas_2022_fluxes_errs} μJy")
print(f"     Weighted mean: {atlas_2022_mean:.2f} ± {atlas_2022_err:.2f} μJy")
print(f"     (Consistent with zero variability: {abs(atlas_2022_mean):.1f} < {atlas_2022_err:.1f})")
print()

# -----------------------------------------------------------------------------
# 2. ZTF Magnitude to Flux Conversion
# -----------------------------------------------------------------------------
print("-" * 75)
print("2. ZTF PHOTOMETRY (r-band)")
print("-" * 75)
print("   Converting AB magnitudes to flux using: F_mJy = 3631 × 10^(-0.4m) × 1000")
print()

# ZTF 2021
ztf_2021_flux, ztf_2021_flux_err = mag_to_flux_mJy(ztf_2021_mag, ztf_2021_mag_err)
ztf_2021_mean, ztf_2021_err = get_weighted_average(ztf_2021_flux, ztf_2021_flux_err)

print(f"   ZTF 2021:")
print(f"     Input magnitudes: {ztf_2021_mag}")
print(f"     Input mag errors: {ztf_2021_mag_err}")
print(f"     Converted fluxes: {ztf_2021_flux.round(4)} mJy")
print(f"     Weighted mean: {ztf_2021_mean:.4f} ± {ztf_2021_err:.4f} mJy")
print()

# ZTF 2022
ztf_2022_flux, ztf_2022_flux_err = mag_to_flux_mJy(ztf_2022_mag, ztf_2022_mag_err)
ztf_2022_mean, ztf_2022_err = get_weighted_average(ztf_2022_flux, ztf_2022_flux_err)

print(f"   ZTF 2022:")
print(f"     Input magnitudes: {ztf_2022_mag}")
print(f"     Input mag errors: {ztf_2022_mag_err}")
print(f"     Converted flux: {ztf_2022_flux[0]:.4f} mJy")
print(f"     Weighted mean: {ztf_2022_mean:.4f} ± {ztf_2022_err:.4f} mJy")
print()

# ZTF variability (difference)
ztf_variability = ztf_2021_mean - ztf_2022_mean
ztf_variability_err = np.sqrt(ztf_2021_err**2 + ztf_2022_err**2)
print(f"   ZTF Variability (2021 - 2022): {ztf_variability*1000:.2f} ± {ztf_variability_err*1000:.2f} μJy")
print()

# -----------------------------------------------------------------------------
# 3. ASAS-SN Fluxes
# -----------------------------------------------------------------------------
print("-" * 75)
print("3. ASAS-SN PHOTOMETRY (V-band)")
print("-" * 75)

asn_2021_mean, asn_2021_err = get_weighted_average(asn_2021_flux, asn_2021_flux_err)
asn_2022_mean, asn_2022_err = get_weighted_average(asn_2022_flux, asn_2022_flux_err)

print(f"   ASAS-SN 2021:")
print(f"     Input measurements: {asn_2021_flux} mJy")
print(f"     Weighted mean: {asn_2021_mean:.4f} ± {asn_2021_err:.4f} mJy")
print()
print(f"   ASAS-SN 2022:")
print(f"     Input measurements: {asn_2022_flux} mJy")
print(f"     Input uncertainties: {asn_2022_flux_err} mJy")
print(f"     Weighted mean: {asn_2022_mean:.4f} ± {asn_2022_err:.4f} mJy")
print()

# ASAS-SN variability
asn_variability = asn_2021_mean - asn_2022_mean
asn_variability_err = np.sqrt(asn_2021_err**2 + asn_2022_err**2)
print(f"   ASAS-SN Variability (2021 - 2022): {asn_variability*1000:.2f} ± {asn_variability_err*1000:.2f} μJy")
print()

# -----------------------------------------------------------------------------
# 4. Dilution Corrections
# -----------------------------------------------------------------------------
print("-" * 75)
print("4. DILUTION CORRECTIONS")
print("-" * 75)
print("   The AGN light is diluted by host galaxy light within the aperture.")
print("   Dilution factor = fraction of observed light from AGN")
print()
print(f"   Dilution factors:")
print(f"     ZTF (r-band):     {ztf_dilution:.3f} (heavily diluted)")
print(f"     ATLAS (o-band):   {atlas_dilution:.4f}")
print(f"     ASAS-SN (V-band): {asn_dilution:.2f}")
print()

print("   CORRECTED VARIABILITY AMPLITUDES:")
print("   (Intrinsic AGN variability = Observed variability / dilution)")
print()

# ATLAS corrections (already variability measurements)
atlas_2021_corr, atlas_2021_corr_err = correct_for_dilution(
    atlas_2021_mean, atlas_2021_err, atlas_dilution, "variability"
)
atlas_2022_corr, atlas_2022_corr_err = correct_for_dilution(
    atlas_2022_mean, atlas_2022_err, atlas_dilution, "variability"
)

# ZTF corrections (variability = 2021 - 2022 total flux difference)
ztf_var_corr, ztf_var_corr_err = correct_for_dilution(
    ztf_variability, ztf_variability_err, ztf_dilution, "variability"
)

# ASAS-SN corrections (variability)
asn_var_corr, asn_var_corr_err = correct_for_dilution(
    asn_variability, asn_variability_err, asn_dilution, "variability"
)

print("   " + "-" * 71)
print(f"   {'Telescope':<12} {'Year/Type':<15} {'Observed (μJy)':<22} {'Corrected AGN (μJy)':<22}")
print("   " + "-" * 71)
print(f"   {'ATLAS':<12} {'2021 var':<15} {atlas_2021_mean:>8.2f} ± {atlas_2021_err:<8.2f}   {atlas_2021_corr:>8.2f} ± {atlas_2021_corr_err:<8.2f}")
print(f"   {'ATLAS':<12} {'2022 var':<15} {atlas_2022_mean:>8.2f} ± {atlas_2022_err:<8.2f}   {atlas_2022_corr:>8.2f} ± {atlas_2022_corr_err:<8.2f}")
print(f"   {'ZTF':<12} {'2021-2022 Δ':<15} {ztf_variability*1000:>8.2f} ± {ztf_variability_err*1000:<8.2f}   {ztf_var_corr*1000:>8.2f} ± {ztf_var_corr_err*1000:<8.2f}")
print(f"   {'ASAS-SN':<12} {'2021-2022 Δ':<15} {asn_variability*1000:>8.2f} ± {asn_variability_err*1000:<8.2f}   {asn_var_corr*1000:>8.2f} ± {asn_var_corr_err*1000:<8.2f}")
print("   " + "-" * 71)
print()

# -----------------------------------------------------------------------------
# 5. Summary and Comparison
# -----------------------------------------------------------------------------
print("-" * 75)
print("5. SUMMARY: INTRINSIC AGN VARIABILITY (2021 vs 2022)")
print("-" * 75)
print()
print("   All values in micro-Jansky (μJy):")
print()
print(f"   ATLAS 2021 difference flux:       {atlas_2021_corr:>8.2f} ± {atlas_2021_corr_err:.2f} μJy")
print(f"   ATLAS 2022 difference flux:       {atlas_2022_corr:>8.2f} ± {atlas_2022_corr_err:.2f} μJy")
print(f"   ZTF flux decrease (2021→2022):    {ztf_var_corr*1000:>8.2f} ± {ztf_var_corr_err*1000:.2f} μJy")
print(f"   ASAS-SN flux decrease (2021→2022):{asn_var_corr*1000:>8.2f} ± {asn_var_corr_err*1000:.2f} μJy")
print()

# Significance of detections
atlas_2021_sigma = abs(atlas_2021_corr / atlas_2021_corr_err)
atlas_2022_sigma = abs(atlas_2022_corr / atlas_2022_corr_err)
ztf_sigma = abs(ztf_var_corr / ztf_var_corr_err)
asn_sigma = abs(asn_var_corr / asn_var_corr_err)

print("   Detection significance:")
print(f"     ATLAS 2021 variability: {atlas_2021_sigma:.1f}σ {'(detected)' if atlas_2021_sigma > 3 else '(marginal)' if atlas_2021_sigma > 2 else '(not significant)'}")
print(f"     ATLAS 2022 variability: {atlas_2022_sigma:.1f}σ {'(detected)' if atlas_2022_sigma > 3 else '(marginal)' if atlas_2022_sigma > 2 else '(not significant)'}")
print(f"     ZTF 2021→2022 change:   {ztf_sigma:.1f}σ {'(detected)' if ztf_sigma > 3 else '(marginal)' if ztf_sigma > 2 else '(not significant)'}")
print(f"     ASAS-SN 2021→2022 change: {asn_sigma:.1f}σ {'(detected)' if asn_sigma > 3 else '(marginal)' if asn_sigma > 2 else '(not significant)'}")
print()
print("=" * 75)

# Reference observation time
t_mjd = Time(2459252.77, format='jd').mjd
print(f"\nReference observation time: JD 2459252.77 = MJD {t_mjd:.2f}")
