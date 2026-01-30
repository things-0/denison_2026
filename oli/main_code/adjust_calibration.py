import numpy as np
from scipy.ndimage import gaussian_filter1d

from .constants import *

def clip_sami_blue_edge(
    unclipped_sami_flux: np.ndarray,
    unclipped_sami_lam: np.ndarray,
    unclipped_sami_var: np.ndarray,
    min_ssd_lam: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clip the SAMI blue edge to the minimum SDSS wavelength.
    """
    # Find the index of the first wavelength greater than min_ssd_lam
    clip_idx = np.where(unclipped_sami_lam > min_ssd_lam)[0][0]
    
    # Clip the flux and variance arrays
    clipped_sami_flux = unclipped_sami_flux[clip_idx:]
    clipped_sami_lam = unclipped_sami_lam[clip_idx:]
    clipped_sami_var = unclipped_sami_var[clip_idx:]

    return clipped_sami_flux, clipped_sami_lam, clipped_sami_var


def gaussian_blur_before_resampling(
    low_res: np.ndarray | float, 
    high_res: np.ndarray | float, 
    lam_low_res: np.ndarray,            # should always be lam01?
    lam_high_res: np.ndarray,
    flux_high_res: np.ndarray,
    n_chunks: int = 100
) -> np.ndarray:
    """
    Gaussian blur using resolution (R = λ/FWHM) arrays.
    Degrades high_res spectrum to match low_res spectrum.
    """
    # Compute sigma arrays on their native grids
    sigma_low_res_arr = (lam_low_res / low_res) / 2.355
    
    sigma_high_res_arr = (lam_high_res / high_res) / 2.355

    #TD: remove testing
    # diffs = np.diff(lam_high_res)
    # print(f"Min step: {diffs.min()}, Max step: {diffs.max()}, Std/Mean: {diffs.std()/diffs.mean()}")
    n_smoothed = 0
    n_ignored = 0
    #

    blurred = flux_high_res.copy()
    chunk_size = len(lam_high_res) // n_chunks
    
    for i in range(n_chunks):
        high_res_chunk_start_idx = i * chunk_size
        high_res_chunk_end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(lam_high_res)
        
        # Wavelength range for this chunk (on the high_res grid)
        lam_high_res_chunk = lam_high_res[high_res_chunk_start_idx:high_res_chunk_end_idx]
        lam_min = lam_high_res_chunk[0]
        lam_max = lam_high_res_chunk[-1]
        
        # Find low_res wavelengths that fall in this range
        low_res_lam_chunk_mask = (lam_low_res >= lam_min) & (lam_low_res <= lam_max)
        


        if np.any(low_res_lam_chunk_mask):
            # Compute median sigma in this chunk for both spectra
            sigma_low_res_chunk = sigma_low_res_arr[low_res_lam_chunk_mask]
            sigma_high_res_chunk = sigma_high_res_arr[high_res_chunk_start_idx:high_res_chunk_end_idx]
            sigma_low_res_median = np.median(sigma_low_res_chunk)
            sigma_high_res_median = np.median(sigma_high_res_chunk)

            #TD: remove testing
            # diffs = np.diff(lam_high_res_chunk)
            # print(f"\nChunk {i+1}\nMin step: {diffs.min()}, Max step: {diffs.max()}, Std/Mean: {diffs.std()/diffs.mean()}")
            #


            wavelength_step = np.median(np.diff(lam_high_res_chunk))

            # Note: should be median(sqrt(sigma_low^2 - sigma_high^2)) but
            # sqrt(median(sigma_low)^2 - median(sigma_high)^2) is a good approximation
            # since sigma_low and sigma_high are approximately constant across the chunk,
            # provided n_chunks is large enough.
            sigma_kernel_sq = sigma_low_res_median**2 - sigma_high_res_median**2

            #TD: remove testing
            # If CV is small, values are approximately constant
            cv_low_res = np.std(sigma_low_res_chunk) / np.median(sigma_low_res_chunk)
            cv_high_res = np.std(sigma_high_res_chunk) / np.median(sigma_high_res_chunk)
            # Rule of thumb: CV < 0.05-0.10 means "approximately constant"
            # print(f"CV low res: {cv_low_res}, CV high res: {cv_high_res}")

            # Fractional spread within chunk
            spread_low = (np.max(sigma_low_res_chunk) - np.min(sigma_low_res_chunk)) / np.median(sigma_low_res_chunk)
            spread_high = (np.max(sigma_high_res_chunk) - np.min(sigma_high_res_chunk)) / np.median(sigma_high_res_chunk)
            # print(f"Spread low res: {spread_low}, Spread high res: {spread_high}")
            #

            if sigma_kernel_sq > EPS:
                #TD: remove testing
                # print(f"sigma_kernel_sq: {sigma_kernel_sq}", flush=True)
                n_smoothed += 1
                #
                sigma_pix = np.sqrt(sigma_kernel_sq) / wavelength_step
                temp_blurred = gaussian_filter1d(flux_high_res, sigma=sigma_pix)
                blurred[high_res_chunk_start_idx:high_res_chunk_end_idx] = temp_blurred[high_res_chunk_start_idx:high_res_chunk_end_idx]
            #TD: remove testing
            else:
                # print(f"too small sigma_kernel_sq: {sigma_kernel_sq}")
                # print(f"\tNo blurring applied in chunk {i+1}", flush=True)
                n_ignored += 1
            #
        else:
            raise ValueError("No low_res coverage in this chunk. This should never happen.")
    
    #TD: remove testing
    # print(f"n_smoothed: {n_smoothed}, n_ignored: {n_ignored}")
    #
    return blurred

def gaussian_blur_after_resampling(
    low_res: np.ndarray | float, 
    high_res: np.ndarray | float, 
    lam: np.ndarray,
    flux_high_res: np.ndarray, 
    wavelength_step: float, 
    n_chunks: int = 20
) -> np.ndarray:
    """
    Gaussian blur using resolution (R = λ/FWHM) arrays.
    Degrades high_res spectrum to match low_res spectrum.
    
    Handles the case where high_res <= low_res at some wavelengths
    (no blurring needed there).
    """
    if np.isscalar(low_res):
        low_res = np.full_like(lam, low_res)
    if np.isscalar(high_res):
        high_res = np.full_like(lam, high_res)

    fwhm_low_res = lam / low_res
    fwhm_high_res = lam / high_res

    sigma_low_res = fwhm_low_res / 2.355
    sigma_high_res = fwhm_high_res / 2.355
    
    # Calculate the kernel sigma needed to convolve high_res to low_res
    # If high_res is already lower resolution, no blurring needed (set to 0)
    sigma_diff_sq = sigma_low_res**2 - sigma_high_res**2
    sigma_diff_sq = np.maximum(sigma_diff_sq, 0)  # Clamp negative values to 0
    sigma_kernel_arr = np.sqrt(sigma_diff_sq)
    
    blurred = flux_high_res.copy()
    chunk_size = len(lam) // n_chunks
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else len(lam)
        
        sigma_pix = np.median(sigma_kernel_arr[start:end]) / wavelength_step
        
        if sigma_pix > EPS:
            temp_blurred = gaussian_filter1d(flux_high_res, sigma=sigma_pix)
            #TD: remove testing
            # plt.plot(lam, flux_high_res)
            # plt.plot(lam, temp_blurred)
            # plt.xlim((lam[start], lam[end-1]))
            # plt.show()
            #
            blurred[start:end] = temp_blurred[start:end]
        # else: keep original flux (already copied above)
    
    #TD: remove testing
    # plt.plot(lam, flux_high_res)
    # plt.plot(lam, blurred)
    # plt.show()
    #

    return blurred