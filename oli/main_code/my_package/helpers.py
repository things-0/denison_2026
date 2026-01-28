import numpy as np

from .constants import H_ALPHA, C_KM_S, Z_SPEC, SMOOTH_FACTOR

def convert_lam_to_vel(
    lam: np.ndarray | float,
    lam_centre_rest_frame: float = H_ALPHA,
    lam_is_rest_frame: bool = False
) -> np.ndarray | float:
    """
    Convert wavelength (Å) to velocity (km/s). 
    Set `lam_is_rest_frame` to `True` if the `lam`
    float or array to be converted is already in
    the rest frame.
    """
    # v = c * Δλ / λ_cent
    if lam_is_rest_frame:
        return (lam - lam_centre_rest_frame) * C_KM_S / lam_centre_rest_frame
    else:
        return (lam / (1 + Z_SPEC) - lam_centre_rest_frame) * C_KM_S / lam_centre_rest_frame

def convert_vel_to_lam(
    vel: np.ndarray | float,
    lam_centre_rest_frame: float = H_ALPHA,
    return_rest_frame: bool = False
) -> np.ndarray | float:
    """
    Convert velocity (km/s) to wavelength (Å).
    """
    # λ = λ_cent * (1 + v / c)
    if return_rest_frame:
        return lam_centre_rest_frame * (1 + vel / C_KM_S)
    else:
        return (lam_centre_rest_frame * (1 + vel / C_KM_S)) * (1 + Z_SPEC)

def get_lam_bounds(lam: float, width: float, is_rest_frame: bool = True, width_is_vel: bool = False) -> tuple[float, float]:
    if is_rest_frame:
        obs_lam = lam * (1+Z_SPEC)
        rest_lam = lam
    else:
        obs_lam = lam
        rest_lam = lam/(1+Z_SPEC)
    if width_is_vel:
        left = convert_vel_to_lam(-width / 2, lam_centre_rest_frame=rest_lam)
        right = convert_vel_to_lam(width / 2, lam_centre_rest_frame=rest_lam)
    else:
        left = obs_lam - width / 2
        right = obs_lam + width / 2
    return left, right

def get_min_res(
    res_01: np.ndarray,
    res_21: np.ndarray,
    res_22: np.ndarray
) -> np.ndarray:
    # res_min = np.minimum(np.minimum(
    #     res_21, 
    #     res_22
    # ), RES_15_BLUE)

    #TODO: check logic with Scott
    res_min = np.minimum(
        res_21, 
        res_22
    )
    #

    res_min =  res_min / SMOOTH_FACTOR

    return res_min

def bin_data_by_median(x: np.ndarray, y: np.ndarray, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    step_size = np.median(np.diff(x))

    points_per_bin = int(bin_width / step_size)

    n_complete_bins = len(y) // points_per_bin
    n_points_to_keep = n_complete_bins * points_per_bin

    x_trimmed = x[:n_points_to_keep]
    y_trimmed = y[:n_points_to_keep]

    x_2d = x_trimmed.reshape(n_complete_bins, points_per_bin)
    y_2d = y_trimmed.reshape(n_complete_bins, points_per_bin)

    x_binned = np.median(x_2d, axis=1)
    y_binned = np.median(y_2d, axis=1)

    return x_binned, y_binned