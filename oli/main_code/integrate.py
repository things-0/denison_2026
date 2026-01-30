import numpy as np
import warnings

from .constants import *

# def integrate_flux(
#     lam: np.ndarray,
#     spec_flux_density: np.ndarray,
#     spec_flux_density_err: np.ndarray,
#     lam_bounds: tuple[float, float],
#     n_gaussians: int = 0,
#     n_trials: int = 6,  # Monte Carlo samples #TODO: change back to 1000
#     plot_fit: bool = False,
#     calc_err: bool = True,
#     title_info: str = "",
#     peak_min_sigma_threshold: float = 2.9,
#     idx_dist_between_peaks: int = 3,
#     print_progress: bool = False,
#     find_peak_min_width_kms: float | None = FIND_PEAK_MIN_WIDTH_KMS,
#     find_peak_min_width_ang: float | None = None,
#     # find_peak_min_width_idx: int | None = FIND_PEAK_MIN_WIDTH_IDX,
#     min_amp_std: float = 1.0
# ) -> tuple[float, float]:

#     valid_mask = np.where((lam > lam_bounds[0]) & (lam < lam_bounds[1]) & (np.isfinite(lam)))
#     x = lam[valid_mask]
#     y = spec_flux_density[valid_mask]
#     sig = spec_flux_density_err[valid_mask]
#     sig[sig == 0] = np.min(sig[sig > 0])  # avoid division by zero
#     weights = 1.0 / sig
    
#     #TODO: run bayesian or mc to find best starting params? (might be a bit overkill)
#     #TODO: make plots with varying params and show Scott

#     if n_gaussians > 0:
#         if find_peak_min_width_ang is None:
#             if find_peak_min_width_kms is None:
#                 min_fwhm = 0
#                 warn_msg = (
#                     f"\nboth find_peak_min_width_ang and find_peak_min_width_kms are None.\n"
#                     f"Setting min_fwhm (minimum dispersion width of peaks) to 0"
#                 )
#                 warnings.warn(warn_msg, UserWarning)
#             lam_cent_rest = np.mean(lam_bounds) / (1 + z)
#             min_fwhm = (
#                 convert_vel_to_lam(find_peak_min_width_kms/2, lam_cent_rest) -
#                 convert_vel_to_lam(-find_peak_min_width_kms/2, lam_cent_rest)
#             )
#         elif find_peak_min_width_kms is None:
#             min_fwhm = find_peak_min_width_ang
#         else:
#             raise ValueError("find_peak_min_width_ang and find_peak_min_width_kms cannot both be provided")
        

#         max_fwhm = (x[-1] - x[0])

#         max_height = np.max(y) + np.std(y)
#         min_height = np.std(y) * min_amp_std

#         min_mu = x[np.argwhere(y > np.std(y)*1.5)[0][0]]
#         max_mu = x[np.argwhere(y > np.std(y)*1.5)[-1][0]]

#     if emcee_fit:
#         if n_gaussians < 1:
#             raise ValueError("emcee_fit requires at least one Gaussian")


#         result = fit_gaussians(
#             x, y, sig,
#             n_gaussians=2,
#             # Constraints
#             min_fwhm=min_fwhm,           # Minimum FWHM in Å
#             max_fwhm=max_fwhm,         # Maximum FWHM
#             min_peak_height=min_height,    # Minimum peak height
#             max_peak_height=max_height,    # Maximum peak height
#             # MCMC settings
#             # nwalkers=50,
#             # nsteps=5000,
#             # burnin=1000,
#             # Plotting
#             plot_corner=True,       # Corner plot of posteriors
#             plot_fit=True,          # Fit vs data plot
#             plot_chains=True,       # MCMC chains (for convergence)
#             progress=True,          # Print progress
#             title=f"{title_info}Difference Spectrum"
#         )

#         print(result)

#         flux, flux_err = result.flux, result.flux_err

#     elif n_gaussians > 0:

#         # 1. Detect peaks and estimate initial params
#         highest_peak_sigma = (np.max(y) - np.mean(y)) / np.std(y)
#         if highest_peak_sigma < peak_min_sigma_threshold:
#             warn_msg = (
#                 f"\nhighest peak is less than {peak_min_sigma_threshold} "
#                 f"sigma above mean (only {highest_peak_sigma:.2f} sigma above).\n"
#                 f"Setting peak_min_sigma_threshold to {highest_peak_sigma:.2f}"
#             )
#             warnings.warn(warn_msg, UserWarning)
#             peak_min_sigma_threshold = highest_peak_sigma * 0.99

#         #TODO: set all to central wavelength (as a starting guess)
#         peak_indices, properties = sps.find_peaks(  #TODO: set min dispersion width and/or smooth first
#             y, distance = idx_dist_between_peaks,
#             height = np.std(y) * peak_min_sigma_threshold + np.mean(y)
#             # width = min_width #TODO: put back in
#         )
#         if len(peak_indices) < n_gaussians:
#             warn_msg = f"\nonly {len(peak_indices)} peaks found, expected {n_gaussians}"
#             warnings.warn(warn_msg, UserWarning)
        
#         peak_indices = peak_indices[:n_gaussians]
#         mu_guesses = x[peak_indices]
#         peak_height_guesses = y[peak_indices]
#         # amp_guesses = properties["peak_heights"][peak_indices]
#         idx_width_at_half_max_guesses, _, _, _ = peak_widths(y, peak_indices, rel_height=0.5) # returns width in indices at half max
#         dx_med = np.median(np.diff(x))
#         fwhm_guesses = idx_width_at_half_max_guesses * dx_med
#         sigma_guesses = fwhm_guesses / SIGMA_TO_FWHM

#         # 2. Build multi-Gaussian model
#         model, params = None, None
#         for i, (height, mu, sigma) in enumerate(
#             zip(peak_height_guesses, mu_guesses, sigma_guesses)
#         ):
#             prefix = f"Gaussian_model_{i+1}_"
#             g = GaussianModel(prefix=prefix)
#             if model is None:
#                 model, params = g, g.make_params()
#             else:
#                 model += g
#                 params.update(g.make_params())
#             #TODO: check these values

#             max_fwhm = (x[-1] - x[0])

#             min_sig = min_fwhm / SIGMA_TO_FWHM 
#             max_sig = max_fwhm / SIGMA_TO_FWHM

#             min_height_to_amp = np.sqrt(2*np.pi)*min_sig
#             height_to_amp = np.sqrt(2*np.pi)*sigma
#             max_height_to_amp = np.sqrt(2*np.pi)*max_sig

#             max_height = np.max(y) + np.std(y)
#             min_height = np.std(y) * min_amp_std
            
#             max_amp = max_height*max_height_to_amp
#             min_amp = min_height*max_height_to_amp
#             # min_amp = min_height*min_height_to_amp
#             # min_amp = min_height*height_to_amp
#             # max_amp = max_height*height_to_amp

#             min_mu = x[np.argwhere(y > np.std(y)*1.5)[0][0]]
#             max_mu = x[np.argwhere(y > np.std(y)*1.5)[-1][0]]
#             #

#             params[prefix+'amplitude'].set(
#                 value=height*height_to_amp,
#                 min=min_amp,
#                 max=max_amp
#             )
#             params[prefix+'center'].set(
#                 value=mu, min=min_mu,
#                 max=max_mu
#             )
#             params[prefix+'sigma'].set(
#                 value=sigma, min=min_sig,
#                 max=max_sig
#             )

#             #TD: remove testing
#             # print(f"{prefix} height bounds: ({min_height}, {max_height}) {SFD_UNITS_NOT_LATEX}")
#             # print(f"{prefix} amplitude bounds: ({min_amp}, {max_amp}) {SFD_UNITS_NOT_LATEX}")
#             # print(f"{prefix} FWHM bounds: ({min_fwhm:.2f}, {max_fwhm:.2f}) Å")
#             # print(f"{prefix} mu bounds: ({min_mu:.2f}, {max_mu:.2f}) Å")

#             # print(f"(before fitting) peak height for {prefix}: {(params[prefix+'amplitude'].value / (params[prefix+'sigma'].value * np.sqrt(2 * np.pi))):.2f} {SFD_UNITS_NOT_LATEX}")
#             # print(f"(before fitting) amplitude for {prefix}: {(params[prefix+'amplitude'].value):.2f} {SFD_UNITS_NOT_LATEX}")
#             # print(f"(before fitting) FWHM for {prefix}: {(params[prefix+'sigma'].value * SIGMA_TO_FWHM):.2f} Å")
#             # print(f"(before fitting) mu for {prefix}: {(params[prefix+'center'].value):.2f} Å")
#             #
        
#         result = model.fit(y, params, x=x, weights=weights)
#         best_fit = result.best_fit


#         flux = np.trapezoid(best_fit, x)

#         #TD: remove testing
#         for i, name in enumerate(result.eval_components(x=x).keys()):
#             amp_fitted = result.params[name + 'amplitude'].value  # This is AREA
#             sigma_fitted = result.params[name + 'sigma'].value
#             mu_fitted = result.params[name + 'center'].value
#             fwhm_fitted = sigma_fitted * SIGMA_TO_FWHM
            
#             # Calculate peak height
#             peak_height = amp_fitted / (sigma_fitted * np.sqrt(2 * np.pi))
            
#             # print(f"(after fitting) peak height for {name}: {peak_height:.2f} {SFD_UNITS_NOT_LATEX}")
#             # print(f"(after fitting) amplitude for {name}: {amp_fitted:.2f} {SFD_UNITS_NOT_LATEX}")
#             # print(f"(after fitting) FWHM for {name}: {fwhm_fitted:.2f} Å")
#             # print(f"(after fitting) mu for {name}: {mu_fitted:.2f} Å")
#         #

#         if not calc_err:
#             if plot_fit:
#                 raise ValueError("plot_fit must be False if calc_err is False")
#             return flux

#         # 3. Monte Carlo integration (Tyler's approach)
#         flux_list = np.zeros(n_trials)
#         for i in range(n_trials):
#             y_perturbed = y + np.random.normal(0, sig)
#             result_mc = model.fit(y_perturbed, params.copy(), x=x, weights=weights)
#             flux_list[i] = np.trapezoid(result_mc.best_fit, x)
#             if print_progress and i % (n_trials // 10) == 0:
#                 print(f"MC trial {i} of {n_trials} completed")
        
#         flux_mc = np.mean(flux_list)
        
#         #TD: remove testing
#         # print(f"{title_info}flux_mc: {flux_mc:.2f}")
#         # print(f"{title_info}flux: {flux:.2f}")
#         # print(f"{title_info}flux - flux_mc: {(flux - flux_mc):.3f}")
#         #

#         flux_err = np.std(flux_list)

#         if plot_fit:
#             redchisq = result.redchi    # lmfit stores reduced chisq here
#             plt.figure(figsize=FIGURE_SIZE)
#             plt.plot(x, y, label="Data", alpha=0.6)
#             plt.plot(x, best_fit, 'r-', label="Total fit", linewidth= 4 * LINEWIDTH)

#             # Plot each component separately
#             for name, comp in result.eval_components(x=x).items():
#                 plt.plot(x, comp, '--', label=name.strip("_"))

#             # Plot uncertainty band (gray stripe)
#             plt.fill_between(
#                 x,
#                 y - sig,
#                 y + sig,
#                 color="gray",
#                 alpha=0.3,
#                 label="Uncertainty"
#             )
#             plt.ylabel(SFD_Y_AX_LABEL)
#             plt.xlabel(r"Wavelength ($\AA$)")
#             plt.title(f"Multi-Gaussian Fit of {title_info}Difference Spectrum")
                
#             plt.text(
#                 0.05, 0.95,         
#                 f"Reduced χ² = {redchisq:.2f}",
#                 transform=plt.gca().transAxes,
#                 fontsize=12,
#                 verticalalignment='top'
#             )
#             plt.legend(loc='upper right')
#             plt.show()

#     else:
#         dx = np.diff(x)
        
#         flux = np.trapezoid(y, x=x)
#         # flux = np.sum((flux[:-1] + flux[1:]) / 2 * dx)

#         if not calc_err:
#             return flux

#         # Weights for interior points are 1.0 (0.5 from left trap + 0.5 from right trap)
#         # Weights for endpoints are 0.5
        
#         # General case for non-uniform spacing:
#         # Each sigma_i is multiplied by (dx_{i-1} + dx_i) / 2
#         err_weights = np.zeros_like(x)
#         err_weights[1:] += dx / 2
#         err_weights[:-1] += dx / 2
        
#         flux_err = np.sqrt(np.sum((err_weights * spec_flux_density_err[valid_mask])**2))
    
#     return flux, flux_err

