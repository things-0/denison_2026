import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import constants as const

def clip_sami_blue_edge(
    unclipped_sami_flux: np.ndarray,
    unclipped_sami_lam: np.ndarray,
    unclipped_sami_err: np.ndarray,
    min_ssd_lam: float,
    good_pixels_15_blue: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Clip the SAMI blue edge to the minimum SDSS wavelength.

    Parameters
    ----------
    unclipped_sami_flux: np.ndarray
        The flux of the blue SAMI spectrum.
    unclipped_sami_lam: np.ndarray
        The wavelength of the blue SAMI spectrum.
    unclipped_sami_err: np.ndarray
        The error on the flux of the blue SAMI spectrum.
    min_ssd_lam: float
        The minimum wavelength of the SDSS spectrum.
    good_pixels_15_blue: np.ndarray | None
        The indices of the good pixels in the blue SAMI spectrum.

    Returns
    -------
    clipped_sami_lam: np.ndarray
        The clipped wavelength array.
    clipped_sami_flux: np.ndarray
        The clipped flux array.
    clipped_sami_err: np.ndarray
        The clipped error array.
    clipped_good_pixels15_blue: np.ndarray | None
        The clipped good pixels mask.
    """
    # Find the index of the first wavelength greater than min_ssd_lam
    clip_idx = np.where(unclipped_sami_lam > min_ssd_lam)[0][0]
    
    # Clip the flux and variance arrays
    clipped_sami_flux = unclipped_sami_flux[clip_idx:]
    clipped_sami_lam = unclipped_sami_lam[clip_idx:]
    clipped_sami_err = unclipped_sami_err[clip_idx:]
    clipped_good_pixels_15_blue = good_pixels_15_blue[clip_idx:] if good_pixels_15_blue is not None else None

    return clipped_sami_lam, clipped_sami_flux, clipped_sami_err, clipped_good_pixels_15_blue


def gaussian_blur_before_resampling(
    low_resolving_power: np.ndarray | float, 
    high_resolving_power: np.ndarray | float, 
    lam_low_res: np.ndarray,            # should always be lam01?
    lam_high_res: np.ndarray,
    flux_high_res: np.ndarray,
    n_chunks: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian blur using resolving power R = λ/FWHM_resolution.
    Blurs high_res spectrum to match low_res spectrum.

    Parameters
    ----------
    low_resolving_power: np.ndarray | float
        The resolving power of the low-resolution spectrum.
    high_resolving_power: np.ndarray | float
        The resolving power of the high-resolution spectrum.
    lam_low_res: np.ndarray
        The wavelength of the low-resolution spectrum.
    lam_high_res: np.ndarray
        The wavelength of the high-resolution spectrum.
    flux_high_res: np.ndarray
        The flux of the high-resolution spectrum.
    n_chunks: int = 100
        The number of chunks to divide the high-resolution spectrum into
        when blurring.

    Returns
    -------
    blurred: np.ndarray
        The blurred high-resolution spectrum.
    new_fwhm_high_res: np.ndarray
        The new FWHM of the high-resolution spectrum, which now (approximately)
        matches that of the low-resolution spectrum.
    """
    # Compute sigma arrays on their native grids
    fwhm_low_res = lam_low_res / low_resolving_power
    fwhm_high_res = lam_high_res / high_resolving_power
    sigma_low_res_arr = fwhm_low_res / const.SIGMA_TO_FWHM
    sigma_high_res_arr = fwhm_high_res / const.SIGMA_TO_FWHM

    #TD: remove testing
    # print(f"Low res sigma {np.mean(sigma_low_res_arr):.2f} - high res sigma {np.mean(sigma_high_res_arr):.2f} = {np.mean(sigma_low_res_arr) - np.mean(sigma_high_res_arr):.2f}")
    #
    if np.mean(sigma_low_res_arr) < np.mean(sigma_high_res_arr) or np.median(sigma_low_res_arr) < np.median(sigma_high_res_arr):
        raise ValueError("ERROR: low-resolution sigma is less than high-resolution sigma")

    new_sigma_high_res = np.full_like(sigma_high_res_arr, np.nan)
    

    #TD: remove testing
    n_smoothed = 0
    n_ignored = 0
    #

    blurred = flux_high_res.copy()
    chunk_size = int(np.ceil(len(lam_high_res) / n_chunks))
    
    # for i in range(n_chunks):
    #     if i == n_chunks:
    #         pass
    i = 0
    while True:
        high_res_chunk_start_idx = i * chunk_size
        high_res_chunk_end_idx = high_res_chunk_start_idx + chunk_size

        if high_res_chunk_start_idx >= len(lam_high_res):
            break
        
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

            if sigma_kernel_sq > const.EPS:
                #TD: remove testing
                # print(f"sigma_kernel_sq: {sigma_kernel_sq}", flush=True)
                n_smoothed += 1
                #
                sigma_pix = np.sqrt(sigma_kernel_sq) / wavelength_step
                temp_blurred = gaussian_filter1d(flux_high_res, sigma=sigma_pix)
                blurred[high_res_chunk_start_idx:high_res_chunk_end_idx] = temp_blurred[high_res_chunk_start_idx:high_res_chunk_end_idx]
                new_sigma_high_res[high_res_chunk_start_idx:high_res_chunk_end_idx] = sigma_low_res_median
            #TD: remove testing
            else:
                # print(f"too small sigma_kernel_sq: {sigma_kernel_sq}")
                # print(f"\tNo blurring applied in chunk {i+1}", flush=True)
                new_sigma_high_res[high_res_chunk_start_idx:high_res_chunk_end_idx] = sigma_high_res_arr[high_res_chunk_start_idx:high_res_chunk_end_idx]
                n_ignored += 1
            #
        else:
            raise ValueError("No low_res coverage in this chunk. This should never happen.")
        i += 1
    
    if np.any(~np.isfinite(new_sigma_high_res)):
        raise ValueError("ERROR: new_sigma_high_res contains non-finite values")
    new_fwhm_high_res = new_sigma_high_res * const.SIGMA_TO_FWHM
    #TD: remove testing
    # print(f"\n\nnew 'high' res - old high res average: {np.mean(new_fwhm_high_res) - np.mean(fwhm_high_res):.2f}")
    # print("(should be greater than 0 (decreasing resolving power increases FWHM values))")
    # print(f"new 'high' res - old low res average: {np.mean(new_fwhm_high_res) - np.mean(fwhm_low_res):.2f}")
    # print("(should be close to 0)\n\n")
    #
    return blurred, new_fwhm_high_res

def gaussian_blur_after_resampling(
    low_resolving_power: np.ndarray | float, 
    high_resolving_power: np.ndarray | float, 
    lam: np.ndarray,
    flux_high_res: np.ndarray, 
    n_chunks: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian blur using resolution (R = λ/FWHM) arrays.
    Degrades high_res spectrum to match low_res spectrum (or
    leaves as is if high_res <= low_res at particular wavelengths).

    Parameters
    ----------
    low_resolving_power: np.ndarray | float
        The resolving power of the low-resolution spectrum.
    high_resolving_power: np.ndarray | float
        The resolving power of the high-resolution spectrum.
    lam: np.ndarray
        The wavelength of the spectrum.
    flux_high_res: np.ndarray
        The flux of the high-resolution spectrum.
    n_chunks: int = 20
        The number of chunks to divide the spectrum into when blurring.
        Each chunk is blurred according to its median sigma (from resolving power).

    Returns
    -------
    blurred: np.ndarray
        The blurred high-resolution spectrum.
    """
    wavelength_step = np.median(np.diff(lam))

    fwhm_low_res = lam / low_resolving_power
    fwhm_high_res = lam / high_resolving_power

    sigma_low_res = fwhm_low_res / const.SIGMA_TO_FWHM
    sigma_high_res = fwhm_high_res / const.SIGMA_TO_FWHM

    new_sigma_high_res = np.full_like(sigma_high_res, np.nan)
    
    # Calculate the kernel sigma needed to convolve high_res to low_res
    # If high_res is already lower resolution, no blurring needed (set to 0)
    sigma_diff_sq = sigma_low_res**2 - sigma_high_res**2
    sigma_diff_sq = np.maximum(sigma_diff_sq, 0)  # Clamp negative values to 0
    sigma_kernel_arr = np.sqrt(sigma_diff_sq)
    
    blurred = flux_high_res.copy()
    chunk_size = int(np.ceil(len(lam) / n_chunks))
    i = 0
    while True:
        start = i * chunk_size
        end = start + chunk_size

        if start >= len(lam):
            break
    # for i in range(n_chunks):
    #     start = i * chunk_size
    #     end = start + chunk_size
        
        sigma_pix = np.median(sigma_kernel_arr[start:end]) / wavelength_step
        low_res_median_sigma = np.median(sigma_low_res[start:end])
        finite_flux_within_chunk_mask = np.isfinite(flux_high_res)[start:end]
        
        if sigma_pix > const.EPS:
            temp_blurred = gaussian_filter1d(flux_high_res, sigma=sigma_pix) # this preserves nans
            blurred[start:end] = temp_blurred[start:end]
            # only change indices within chunk that have finite flux (preserve nans)
            new_sigma_high_res[start:end][finite_flux_within_chunk_mask] = low_res_median_sigma
        else: # keep original flux (already copied above)
            new_sigma_high_res[start:end][finite_flux_within_chunk_mask] = sigma_high_res[start:end][finite_flux_within_chunk_mask]

        i += 1

    # if np.any(~np.isfinite(new_sigma_high_res)):
        # raise ValueError("ERROR: new_sigma_high_res contains non-finite values")
    new_fwhm_high_res = new_sigma_high_res * const.SIGMA_TO_FWHM

    return blurred, new_fwhm_high_res