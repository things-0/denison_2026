# CL-AGN Spectroscopy Analysis

The code in this folder reads in SAMI and SDSS spectroscopy data from different epochs and attempts to analyze the broad line components to determine the properties of the corresponding AGN. Most of the functions are defined in python files under [`main_code/`](main_code/) and are executed in [`main.ipynb`](main.ipynb).

### Data and file structure
Check that you have the spectroscopy .fits files and ensure the correct directory/file names are used in [`constants.py`](main_code/constants.py). Otherwise, most data reading functions can adjust the file names when they are called.

### Main steps

The difference spectrum method in [`main.ipynb`](main.ipynb) follows these steps approximately:

- Read in the data, filter out bad values, blur all spectra to the lowest resolution, and resample all data onto a common wavelength array using [`get_adjusted_data`](main_code/data_reading.py#get_adjusted_data)
- Find the ratio of flux between all epochs and the 2015 spectrum, find a best fitting polynomial to the ratio and multiply its inverse back on to the origigal spectra using [`apply_poly_fit`](main_code/polynomial_fit.py#apply_poly_fit)
- Subtract the 2001 flux from the flux of all other epochs using [`get_diff_spectra`](main_code/difference.py#get_diff_spectra)
- Fit a sum of $n$ Gaussians to the difference spectra using [`fit_gaussians`](main_code/gaussian_fitting.py#fit_gaussians)
- Integrate the flux under the best fit Gaussian curves using [`integrate_flux`](main_code/integrate.py#integrate_flux)
- From the integrated flux, calculate the Balmer decrements for each epoch with (`num_bins > 1`) or without (`num_bins = 1`) binning using [`calculate_balmer_decrement`](main_code/integrate.py#calculate_balmer_decrement).
- From the integrated $H\alpha$ flux, calculate the $H\alpha$ luminosity and then the black hole mass for each epoch using [`get_luminosity`](main_code/bh_mass.py#get_luminosity) and [`get_bh_mass`](main_code/bh_mass.py#get_bh_mass)

The pPXF method reads in the data as above, and fits the broad and narrow line components using the modified [`fit_agn`](main_code/ppxf_funcs.py#fit_agn) function (from Scott Croom). The fit results are saved to [`PPXF_DATA_DIR`](main_code/constants.py#PPXF_DATA_DIR). The components are then extracted from the results and analyzed using the same functions above. 

### Plotting and saving figures
Depending on the data to plot, different functions are required within [`plotting.py`](main_code/plotting.py). Most plotting parameters (figure size, axis labels, etc.) can be adjusted within the function call, or within the default values used in [`constants.py`](main_code/constants.py). To save the figures to an `output/` directory, set [`SAVE_FIGS`](main_code/constants.py#SAVE_FIGS) to `True`. By default, figure titles will be omitted (shown) if the figures are saved (not saved).


### Other code
- [`helpers.py`](main_code/helpers.py) contains various functions that assist functions defined in other main python files.
- [`d400_lick_indices.py`](main_code/d4000_lick_indices.py) calculates the D4000 (and other) lick indices for a spectrum.
- [`adjust_calibration.py`](main_code/adjust_calibration.py) contains functions that help recalibrate and adjust the data as it is being read in via [`get_adjusted_data`](main_code/data_reading.py#get_adjusted_data)
- [`qsofit.py`](main_code/qsofit.py) is an old file used to help with the QSOFit analysis (obsolete)

### Notes
- Any remaining tasks I haven't got round to completing are marked with `#TODO:`
- [`test_workspace.ipynb`](test_workspace.ipynb) is a (messy) file used to test various code. Feel free to ignore it.