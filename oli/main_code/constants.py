import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Z_SPEC = 0.0582 # (from Tyler's report)
Z_SPEC = 0.0581892766058445 # for spectroscopy (from Scott's pPXF code)
Z_LUM = 0.05938

# From SAMI header
RA = 133.40525
DEC = 1.7176667

# Rest frame wavelengths from https://iopscience.iop.org/article/10.1086/321167/fulltext/
OIII_BLUE = 4960.30
OIII_RED = 5008.24
NII_BLUE = 6549.85
NII_RED = 6585.28
SII_BLUE = 6718.29
SII_RED = 6732.67
H_ALPHA = 6564.61
H_BETA = 4862.68


# Bandpass effective central wavelengths and widths

#https://svo2.cab.inta-csic.es/svo/theory/fps/index.php?id=Misc/Atlas.orange
ATLAS_O_BAND_LAM = 6629.82
ATLAS_O_BAND_WIDTH = 2368.06
# https://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=PAN-STARRS&asttype=
ZTF_I_BAND_LAM = 7503.03
ZTF_I_BAND_WIDTH = 1206.62
ZTF_R_BAND_LAM = 6155.47
ZTF_R_BAND_WIDTH = 1252.41
ZTF_G_BAND_LAM = 4810.16
ZTF_G_BAND_WIDTH = 1053.08
# https://svo2.cab.inta-csic.es/svo/theory/fps/index.php?id=Generic/Johnson.V
ASASSN_V_BAND_LAM = 5467.57
ASASSN_V_BAND_WIDTH = 889.80
# https://svo2.cab.inta-csic.es/svo/theory/fps/index.php?id=SLOAN/SDSS.g
ASASSN_G_BAND_LAM = 4671.78
ASASSN_G_BAND_WIDTH = 1064.68

C_KM_S = 2.99792458e5 # km/s
C_M_S = 2.99792458e8 # m/s
C_ANG_S = 2.99792458e18 # Ang/s

# Photometry values - in mJy (from Shaun's code)
ZTF_R_FLUX_21 = 0.0049
ZTF_R_FLUX_22 = -0.0229
ZTF_G_FLUX_21 = 0.0052
ZTF_G_FLUX_22 = -0.0155
ZTF_I_FLUX_21 = -0.0203 # don't trust these values too much
ZTF_I_FLUX_22 = -0.0502 # don't trust these values too much
ATLAS_O_FLUX_21 = 0.0526
ATLAS_O_FLUX_22 = -0.0061
ASASSN_G_FLUX_21 = -0.0172
ASASSN_G_FLUX_22 = -0.0301

# old photometry values (from Shaun's code)
# BAD_ZTF_R_FLUX_21 = 0.0358
# BAD_ZTF_R_FLUX_22 = -0.0187
# BAD_ATLAS_O_FLUX_21 = 0.0452
# BAD_ATLAS_O_FLUX_22 = 0.0023

YASMEEN_RESULTS = { # (value, error)
    "fwhm_alpha_15": (2143, 60),            # km/s
    "fwhm_alpha_21": (2097, 365),           # km/s
    "fwhm_beta_15": (1731, 394),            # km/s
    "fwhm_beta_21": (1759, 292),            # km/s
    "flux_alpha_15": (635, 22),             # 10^-17 ergs/s/cm^2
    "flux_alpha_21": (2330, 40),            # 10^-17 ergs/s/cm^2
    "luminosity_alpha_15": (5.49, 0.19),    # 10^40 ergs/s
    "luminosity_alpha_21": (20.1, 0.3),     # 10^40 ergs/s
    "bd_15": (4.58, 1.69),                  # dimensionless
    "bd_21": (7.68, 2.57),                  # dimensionless
    "bh_mass_15": (1.95, 0.43),             # 1e6 M_sun
    "bh_mass_21": (3.81, 1.58)              # 1e6 M_sun
}

SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
WDISP_SCALE_FACTOR = 1.15
EPS = 1e-8
MAX_FLUX = 500
MIN_FLUX = -1
NUM_MC_TRIALS = 1000
TEST_NUM_MC_TRIALS = 50

TOTAL_LAM_BOUNDS = (3900, 8000)
VEL_TO_IGNORE_WIDTH = 7000
POLY_FIT_BIN_WIDTH = 50 # angstrom
VEL_BLUR_BIN_WIDTH = 1500 # km/s

# Gaussian fitting parameters
MAXFEV = 2100 # default is 1600
MIN_MU = 1/5 # of total range of x plus minimum x value
PEAK_MIN_RANGE = 1/35 # of total range of x - decrease for sharper peaks
HEIGHT_MIN = 0
HEIGHT_MAX = 10
DEFAULT_NUM_GAUSSIANS = 2

# Plotting parameters
SAVE_FIGS = True
PLOT_TITLES = not SAVE_FIGS
VEL_PLOT_WIDTH = 3 * VEL_TO_IGNORE_WIDTH
LINEWIDTH = 1.5
FIG_SIZE = (10,6)
DOUBLE_FIG_SIZE = (18,6)
TEXT_SCALE_FACTOR = 1.8
LEGEND_SCALE_FACTOR = 0.8
TEXT_SIZE = FIG_SIZE[0] * TEXT_SCALE_FACTOR
LATEX_SINGLE_TO_DOUBLE_COL_WIDTH_RATIO = 2.05712
DOUBLE_TEXT_SIZE = DOUBLE_FIG_SIZE[0] * TEXT_SCALE_FACTOR / LATEX_SINGLE_TO_DOUBLE_COL_WIDTH_RATIO
ERR_OPAC = 0.1
FILL_BETWEEN_OPAC = 0.5
COLOUR_MAP = plt.cm.tab10
ALT_COLOUR_MAP = plt.cm.tab20
FIG_LAYOUT = "constrained"

VEL_WIDTH_GAUSSIAN_FIT = VEL_PLOT_WIDTH


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
DATA_DIR = ROOT_DIR / "data"
PPXF_DATA_DIR = DATA_DIR / "ppxf_results"
SAMI323854_DIR = DATA_DIR / "sami323854"
SAMI_DATA_DIR = SAMI323854_DIR / "sami_data"
SDSS_DATA_DIR = SAMI323854_DIR / "sdss_data"

FNAME_2001 = "spec-0469-51913-0338.fits"
FNAME_2021 = "spec-015167-59252-6747964707.fits"
FNAME_2022 = "spec-104405-59664-27021600108375953.fits"

FNAME_2015_BLUE_3_ARCSEC = "323854_A_spectrum_3-arcsec_blue.fits"
FNAME_2015_RED_3_ARCSEC = "323854_A_spectrum_3-arcsec_red.fits"

FNAME_2015_BLUE_4_ARCSEC = "323854_A_spectrum_4-arcsec_blue.fits"
FNAME_2015_RED_4_ARCSEC = "323854_A_spectrum_4-arcsec_red.fits"

VEL_LABEL = r"Velocity (km$\text{s}^{-1}$)"
BASE_ANG_LABEL = r"wavelength ($\mathrm{\AA}$)"
OBS_ANG_LABEL = "Observed " + BASE_ANG_LABEL
REST_ANG_LABEL = "Rest frame " + BASE_ANG_LABEL
FLUX_UNITS_NOT_LATEX = "10⁻¹⁷ erg s⁻¹ cm⁻²"
SFD_UNITS_NOT_LATEX = FLUX_UNITS_NOT_LATEX + " Å⁻¹"
FLUX_UNITS = r"$10^{-17}$ erg $\text{s}^{-1}$ $\text{cm}^{-2}$"
SFD_UNITS = FLUX_UNITS + r" ${\mathrm{\AA}}^{-1}$"
SFD_Y_AX_LABEL = r"$F_{\lambda}$ " + f"({SFD_UNITS})"
HA_LATEX = r"H$\alpha$"
HB_LATEX = r"H$\beta$"
OIII_BLUE_LATEX = r"[OIII]$\lambda$4959"
OIII_RED_LATEX = r"[OIII]$\lambda$5007"
SII_BLUE_LATEX = r"[SII]$\lambda$6718"
SII_RED_LATEX = r"[SII]$\lambda$6733"
NII_BLUE_LATEX = r"[NII]$\lambda$6549"
NII_RED_LATEX = r"[NII]$\lambda$6584"

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


# PyQSOFit parameters
FIT_KEYS = [
    "name", "nsmooth", "and_mask", "or_mask", "reject_badpix",
    "deredden", "wave_range", "wave_mask", "decompose_host",
    "host_prior", "host_prior_scale", "host_line_mask",
    "decomp_na_mask", "qso_type", "npca_gal", "host_type",
    "npca_qso", "Fe_uv_op", "poly", "BC", "rej_abs_conti",
    "rej_abs_line", "initial_guess", "n_pix_min_conti",
    "param_file_name", "MC", "MCMC", "nburn", "nsamp", "nthin",
    "epsilon_jitter", "linefit", "save_result", "plot_fig",
    "save_fig", "plot_corner", "save_fits_path", "save_fits_name",
    "verbose", "kwargs_plot", "kwargs_conti_emcee", "kwargs_line_emcee"
]
LOG_KEYS = [
    "output_file_name", "data_fname", "params_file_name",
    "lam_bounds", "filter_bad_values", "interpolate_bad_values"
]
COMBINED_KEYS = LOG_KEYS + FIT_KEYS[1:]

