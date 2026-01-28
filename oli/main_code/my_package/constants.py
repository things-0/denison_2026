import numpy as np

Z_SPEC = 0.0582 # for spectroscopy
Z_LUM = 0.05938

# Rest frame wavelengths
OIII_1 = 4960.30
OIII_2 = 5008.24
NII_1 = 6549.85
NII_2 = 6585.28
SII_1 = 6718.29
SII_2 = 6732.67
H_ALPHA = 6564.61
H_BETA = 4862.68

# Bandpass effective central wavelengths and widths
ATLAS_O_BAND_LAM = 6629.82
ATLAS_O_BAND_WIDTH = 2368.06
ZTF_I_BAND_LAM = 7503.03
ZTF_I_BAND_WIDTH = 1206.62
ZTF_R_BAND_LAM = 6155.47
ZTF_R_BAND_WIDTH = 1252.41
ASSASN_V_BAND_LAM = 5467.57
ASSASN_V_BAND_WIDTH = 889.80
ASSASN_G_BAND_LAM = 4671.78
ASSASN_G_BAND_WIDTH = 1064.68

"""
ASAS-SN uses Sloan g-band filters with an effective central
wavelength of 480.3 nm, and a FWHM of 140.9 nm.
"""

C_KM_S = 2.99792458e5 # km/s
C_M_S = 2.99792458e8 # m/s
C_ANG_S = 2.99792458e18 # Ang/s

SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
EPS = 1e-8
LAMBDAS_TO_IGNORE_WIDTH = 180
VEL_TO_IGNORE_WIDTH = 7000
# BALMER_DECREMENT_VEL_WIDTH = 4000
FIND_PEAK_MIN_WIDTH_KMS = 1000
FIND_PEAK_MIN_WIDTH_ANG = 15
# FIND_PEAK_MIN_WIDTH_IDX = 7

#TD: remove testing
SMOOTH_FACTOR = 1.0
#

# Plotting parameters
LINEWIDTH = 0.5
FIG_SIZE = (10,4)


SDSS_FOLDER_NAME = "../data/sami323854/sdss_data/"
SAMI_FOLDER_NAME = "../data/sami323854/sami_data/"

FNAME_2001 = "spec-0469-51913-0338.fits"
FNAME_2021 = "spec-015167-59252-6747964707.fits"
FNAME_2022 = "spec-104405-59664-27021600108375953.fits"

FNAME_2015_BLUE_3_ARCSEC = "323854_A_spectrum_3-arcsec_blue.fits"
FNAME_2015_RED_3_ARCSEC = "323854_A_spectrum_3-arcsec_red.fits"

FNAME_2015_BLUE_4_ARCSEC = "323854_A_spectrum_4-arcsec_blue.fits"
FNAME_2015_RED_4_ARCSEC = "323854_A_spectrum_4-arcsec_red.fits"

SFD_UNITS_NOT_LATEX = "10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹"
FLUX_UNITS_NOT_LATEX = "10⁻¹⁷ erg s⁻¹ cm⁻²"
FLUX_UNITS = r"$10^{-17}$ erg $\text{s}^{-1}$ $\text{cm}^{-2}$"
SFD_UNITS = FLUX_UNITS + r" $\AA^{-1}$"
SFD_Y_AX_LABEL = f"Spectral flux density ({SFD_UNITS})"

# From 2021 "The SAMI Galaxy Survey: the third and final data release"
"""
                                For the SAMI survey, we used the
580V and 1000R gratings, delivering a wavelength range of 3750-
5750 and 6300-7400 Å for the blue and red arms, respectively. The
spectral resolutions are R = 1808 and 4304 for the blue and red
arms, equivalent to an effective velocity dispersion of σ of 70.4 and
29.6 km s^{-1}, respectively
"""

RES_15_BLUE = 1808
RES_15_RED = 4304


# From 2014 "The SAMI Galaxy Survey: instrument specification and target selection"
    # Out of date --> don't use
"""
                                SAMI feeds the AAOmega spectrograph
(Sharp et al. 2006), which for the survey is set up to have resolu-
tions of R = 1730 in the blue arm and R = 4500 in the red arm.
"""

# R_15_blue = 1730
# R_15_red = 4500

# Average across all epochs and wavelength ranges
# SDSS_res = 2000