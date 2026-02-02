from matplotlib.pylab import plot
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib.colors import Colormap

from .constants import *
from .helpers import (
    get_lam_bounds, convert_lam_to_vel, convert_to_vel_data,
    get_flux_bounds, get_vel_lam_mask
)

def plot_vert_emission_lines(
    ions: dict[str, float] | None, 
    plot_x_bounds: tuple[float, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = 0.5,
    vlines_cmap: Colormap = COLOUR_MAP,
    is_rest_frame: bool = True,
    vel_centre_ang: float | None = None
) -> None:
    plt.xlim(plot_x_bounds)
    if fill_between_bounds is not None:
        plt.axvspan(
            fill_between_bounds[0], fill_between_bounds[1],
            color='lightgrey', alpha=fill_between_opacity,
            label=fill_between_label
        )
    if ions is None:
        return
    for i, (name, lam) in enumerate(ions.items()):
        if is_rest_frame:
            obs_lam = lam * (1+Z_SPEC)
        else:
            obs_lam = lam
        if vel_centre_ang is None:
            x_val = obs_lam
        else:
            x_val = convert_lam_to_vel(obs_lam, lam_centre_rest_frame=vel_centre_ang)
        if plot_x_bounds is None or (plot_x_bounds[0] < obs_lam < plot_x_bounds[1]):
            plt.axvline(
                x_val, linestyle='--', lw=LINEWIDTH,
                color=vlines_cmap(i), label=name
            )


def plot_min_res(
    lam01: np.ndarray,
    lam15_blue: np.ndarray,
    lam15_red: np.ndarray,
    lam21: np.ndarray,
    lam22: np.ndarray,
    res_min: np.ndarray,
    res_01: np.ndarray,
    res_21: np.ndarray,
    res_22: np.ndarray,
    plot_RES_15_RED: bool = False
) -> None:
    """
    Plot the coverage of the resolution of the spectra.
    """

    lam15_blue_min = np.min(lam15_blue)
    lam15_blue_max = np.max(lam15_blue)
    lam15_red_min = np.min(lam15_red)
    lam15_red_max = np.max(lam15_red)

    if plot_RES_15_RED:
        res_plot_bounds = [np.min(res_min), max(
            np.max(res_21), np.max(res_22), 
            np.max(res_01), RES_15_BLUE, RES_15_RED
        )]
    else:
        res_plot_bounds = [np.min(res_min), max(
            np.max(res_21), np.max(res_22), 
            np.max(res_01), RES_15_BLUE
        )]

    plt.axhline(RES_15_BLUE, color='blue', linestyle='--', label="SAMI blue")
    if plot_RES_15_RED:
        plt.axhline(RES_15_RED, color='red', linestyle='--', label="SAMI red")
    plt.plot(lam21, res_21, alpha=0.5, label="2021")
    plt.plot(lam22, res_22, alpha=0.5, label="2022")
    plt.plot(lam01, res_01, alpha=0.5, label="SDSS Average")
    plt.plot(lam01, res_min, color='black', alpha=0.5, lw=4, linestyle='--', label="Minimum")
    plt.fill_betweenx(res_plot_bounds, lam15_blue_min, lam15_blue_max, color='lightblue', alpha=0.5, label="SAMI blue coverage")
    plt.fill_betweenx(res_plot_bounds, lam15_red_min, lam15_red_max, color='red', alpha=0.2, label="SAMI red coverage")
    plt.xlabel(ANG_LABEL)
    plt.ylabel("Resolving Power")
    plt.title("Resolution of SDSS spectra")
    plt.legend()
    plt.show()

def plot_spectra(
    lam01: np.ndarray,
    lam15: np.ndarray | tuple[np.ndarray, np.ndarray],
    lam21: np.ndarray,
    lam22: np.ndarray,
    flux01: np.ndarray,
    flux15: np.ndarray | tuple[np.ndarray, np.ndarray],
    flux21: np.ndarray,
    flux22: np.ndarray,
    plot_errors: bool = False,
    flux01_err: np.ndarray | None = None,
    flux15_err: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    flux21_err: np.ndarray | None = None,
    flux22_err: np.ndarray | None = None,
    title: str | None = None,
    y_axis_label: str = SFD_Y_AX_LABEL,
    x_axis_label: str = ANG_LABEL,
    error_opacity: float = ERR_OPAC,
    ions: dict[str, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = 0.5,
    legend_loc: str | None = "best",
) -> None:
    sami_is_split = True if isinstance(flux15, tuple) else False

    plt.figure(figsize=FIG_SIZE)
    plt.plot(lam01, flux01, color='black', label='2001 (SDSS)', lw = LINEWIDTH)

    if sami_is_split:
        flux15_blue, flux15_red = flux15
        lam15_blue, lam15_red = lam15

        plt.plot(lam15_blue, flux15_blue, color='blue', label='2015 blue arm (SAMI)', lw = LINEWIDTH)
        plt.plot(lam15_red, flux15_red, color='red', label='2015 red arm (SAMI)', lw = LINEWIDTH)
    else:
        plt.plot(lam15, flux15, color='purple', label='2015 (SAMI)', lw = LINEWIDTH)

    plt.plot(lam21, flux21, color='orange', label='2021 (SDSS)', lw = LINEWIDTH)
    plt.plot(lam22, flux22, color='green', label='2022 (SDSS)', lw = LINEWIDTH)

    if plot_errors:
        if flux01_err is not None:
            plt.fill_between(lam01, flux01 - flux01_err, flux01 + flux01_err, color='black', alpha=error_opacity)
        if flux15_err is not None:
            if sami_is_split:
                flux15_blue_err, flux15_red_err = flux15_err
                plt.fill_between(lam15_blue, flux15_blue - flux15_blue_err, flux15_blue + flux15_blue_err, color='blue', alpha=error_opacity)
                plt.fill_between(lam15_red, flux15_red - flux15_red_err, flux15_red + flux15_red_err, color='red', alpha=error_opacity)
            else:
                plt.fill_between(lam15, flux15 - flux15_err, flux15 + flux15_err, color='purple', alpha=error_opacity)
        if flux21_err is not None:
            plt.fill_between(lam21, flux21 - flux21_err, flux21 + flux21_err, color='orange', alpha=error_opacity)
        if flux22_err is not None:
            plt.fill_between(lam22, flux22 - flux22_err, flux22 + flux22_err, color='green', alpha=error_opacity)

    plot_vert_emission_lines(
        ions, plot_x_bounds=x_bounds,
        fill_between_bounds=fill_between_bounds,
        fill_between_label=fill_between_label,
        fill_between_opacity=fill_between_opacity
    )
    
    if y_bounds is not None:
        plt.ylim(y_bounds)
    else:
        min_flux, smallest_range, total_range = get_flux_bounds(
            lam01, lam15, lam21, lam22,
            flux01, flux15, flux21, flux22,
            x_bounds
        )
        if total_range > 5 * smallest_range:
            plt.ylim((min_flux / 1.2, min_flux + 1.2 * smallest_range))

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title is not None:
        plt.title(title)
    if legend_loc is not None:
        plt.legend(loc=legend_loc)
    else:
        plt.legend()
    plt.show()

def plot_polynomial_ratio(
    lambdas: np.ndarray,
    vals: np.ndarray,
    polynom_vals: np.ndarray | None = None,
    binned_lambdas: np.ndarray | None = None,
    binned_vals: np.ndarray | None = None,
    vals_removed: np.ndarray | None = None,
    degree: float = 6,
    bin_width: float = 40,
    bin_by_med: bool = True,
    title: str | None = None,
    plot_selection: bool = False,
) -> None:
    plt.figure(figsize=FIG_SIZE)
    
    if bin_by_med:
        plt.plot(binned_lambdas, binned_vals, color='black', label=f'ratio binned by median (width {bin_width} Å)', lw=4*LINEWIDTH)
        poly_label = f"polynomial fit (degree {degree}) to binned ratio"
    else:
        poly_label = f"polynomial fit (degree {degree})"

    if plot_selection:
        ratio_label = 'Spectral flux density ratio'
        plt.plot(lambdas, vals_removed, color='red', label=f'{ratio_label} (ignored Balmer)', lw = LINEWIDTH)
    else:
        ratio_label = 'actual ratio'
        plt.plot(binned_lambdas, polynom_vals, color='red', label=poly_label, lw=4*LINEWIDTH)

    plt.plot(lambdas, vals, alpha=0.5, color='black', label=ratio_label, lw = LINEWIDTH)
    plt.xlabel(ANG_LABEL)
    plt.ylabel("Ratio")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_adjusted_spectrum(
    lam: np.ndarray,
    baseline_flux: np.ndarray,
    unadjusted_flux: np.ndarray,
    adjusted_flux: np.ndarray,
    year_to_adjust: int,
    baseline_year: int,
    ions: dict[str, float] | None = None,
    lam_bounds: tuple[float] | None = (3800, 8000),
    flux_y_bounds: tuple[float] | None = None,
    title: str | None = None,
)-> None:
    plt.figure(figsize=FIG_SIZE)
    plt.plot(lam, baseline_flux, color='black', label=f'{baseline_year}', lw = LINEWIDTH)
    plt.plot(lam, unadjusted_flux, color='orange', label=f'{year_to_adjust}', lw = LINEWIDTH)
    plt.plot(lam, adjusted_flux, color='red', label=f'{year_to_adjust} (polynomial fit to {baseline_year})', lw = LINEWIDTH)
    plt.xlabel(ANG_LABEL)
    plt.ylabel(SFD_Y_AX_LABEL)

    plot_vert_emission_lines(ions, lam_bounds)
    plt.title(title)
    if flux_y_bounds is not None:
        plt.ylim(flux_y_bounds)
    elif ((
        np.nanmax(adjusted_flux) - np.nanmin(adjusted_flux)) >
        10 * (np.nanmax(unadjusted_flux) - np.nanmin(unadjusted_flux)
    )):
        plt.ylim((0, 1.2 * np.nanmax(unadjusted_flux)))
    plt.legend()
    plt.show()


def plot_diff_spectra(
    lam: np.ndarray,
    # years_to_plot: list[int] = [2015, 2021, 2022],
    diff_15: np.ndarray | None = None,
    diff_21: np.ndarray | None = None,
    diff_22: np.ndarray | None = None,
    diff_15_err: np.ndarray | None = None,
    diff_21_err: np.ndarray | None = None,
    diff_22_err: np.ndarray | None = None,
    ions: dict[str, float] | bool = False,
    vel_plot_width: float | None = VEL_PLOT_WIDTH,
    hlines: dict[str, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = FILL_BETWEEN_OPAC,
    plot_centres: float | list[float] = [H_ALPHA, H_BETA],
    plot_labels: list[str] | None = [r"H-${\alpha}$", r"H-${\beta}$"],
    use_ang_x_axis: bool = False,
    plot_y_bounds: tuple[float, float] | bool = True,
    error_opacity: float = ERR_OPAC,
    colour_map: Colormap = COLOUR_MAP,
) -> None:

    if isinstance(plot_centres, list) and plot_labels is not None and len(plot_centres) != len(plot_labels):
        raise ValueError("plot_centres and plot_labels must have the same length and correspond to each other")
    if not isinstance(plot_centres, list) and isinstance(plot_labels, list) and len(plot_labels) != 1:
        raise ValueError("plot_labels must be a list of length 1 if plot_centres is just a number")

    num_centres = len(plot_centres) if isinstance(plot_centres, list) else 1
    if plot_labels is None:
        plot_labels = [""] * num_centres

    if use_ang_x_axis:
        if isinstance(plot_centres, list):
            raise ValueError("plot_centres must be a single number if use_ang_x_axis is True")

        x_axis_label = ANG_LABEL
        x_15, x_21, x_22 = [lam], [lam], [lam]
        diffs_15 = [diff_15] if diff_15 is not None else None
        diffs_21 = [diff_21] if diff_21 is not None else None
        diffs_22 = [diff_22] if diff_22 is not None else None
        diffs_15_err = [diff_15_err] if diff_15_err is not None else None
        diffs_21_err = [diff_21_err] if diff_21_err is not None else None
        diffs_22_err = [diff_22_err] if diff_22_err is not None else None
    else:
        x_axis_label = VEL_LABEL
        x_15, diffs_15, diffs_15_err = convert_to_vel_data(
            lam, diff_15, diff_15_err, plot_centres, vel_plot_width
        )
        x_21, diffs_21, diffs_21_err = convert_to_vel_data(
            lam, diff_21, diff_21_err, plot_centres, vel_plot_width
        )
        x_22, diffs_22, diffs_22_err = convert_to_vel_data(
            lam, diff_22, diff_22_err, plot_centres, vel_plot_width
        )


    plt.figure(figsize=FIG_SIZE)
    for i in range(num_centres):
        if diffs_15 is not None:
            label_info = plot_labels[i]
            flux = diffs_15[i]
            flux_err = diffs_15_err[i] if diffs_15_err is not None else None
            colour_15 = colour_map(3*i) if num_centres > 1 else 'black'
            plt.plot(x_15[i], flux, alpha=0.7, color=colour_15, label=f'{label_info} 2015 - 2001', lw = LINEWIDTH)
            
            if flux_err is not None:
                plt.fill_between(x_15[i], flux - flux_err, flux + flux_err, color=colour_15, alpha=error_opacity)
        if diffs_21 is not None:
            label_info = plot_labels[i]
            flux = diffs_21[i]
            flux_err = diffs_21_err[i] if diffs_21_err is not None else None
            colour_21 = colour_map(3*i+1) if num_centres > 1 else 'red'
            plt.plot(x_21[i], flux, alpha=0.7, color='red', label=f'{label_info} 2021 - 2001', lw = LINEWIDTH)
            
            if flux_err is not None:
                plt.fill_between(x_21[i], flux - flux_err, flux + flux_err, color=colour_21, alpha=error_opacity)
        if diffs_22 is not None:
            label_info = plot_labels[i]
            flux = diffs_22[i]
            flux_err = diffs_22_err[i] if diffs_22_err is not None else None
            colour_22 = colour_map(3*i+2) if num_centres > 1 else 'blue'
            plt.plot(x_22[i], flux, alpha=0.7, color=colour_22, label=f'{label_info} 2022 - 2001', lw = LINEWIDTH)
            
            if flux_err is not None:
                plt.fill_between(x_22[i], flux - flux_err, flux + flux_err, color=colour_22, alpha=error_opacity)

    there_are_ions_to_plot = True
    if isinstance(ions, bool):
        if ions:
            ions_near_h_alpha = {r"H-${\alpha}$": H_ALPHA, "S[II] (1)": SII_1, "S[II] (2)": SII_2, "N[II] (2)": NII_2}
            ions_near_h_beta = {r"H-${\beta}$": H_BETA, "O[III] (1)": OIII_1, "O[III] (2)": OIII_2}
            if isinstance(plot_centres, list):
                ions = {}
                if H_ALPHA in plot_centres:
                    ions = ions | ions_near_h_alpha
                if H_BETA in plot_centres:
                    ions = ions | ions_near_h_beta
            else:
                if plot_centres == H_ALPHA:
                    ions = ions_near_h_alpha
                elif plot_centres == H_BETA:
                    ions = ions_near_h_beta
                else:
                    ions = None
        else:
            ions = None

    if use_ang_x_axis:        
        x_bounds = get_lam_bounds(plot_centres, vel_plot_width, width_is_vel=True) if vel_plot_width is not None else None
        vel_centre_ang = None
    else:
        vel_centre_ang = plot_centres[0] if isinstance(plot_centres, list) else plot_centres
        x_bounds = (-vel_plot_width / 2, vel_plot_width / 2) if vel_plot_width is not None else None
        

    plot_vert_emission_lines(
        ions, x_bounds, vel_centre_ang=vel_centre_ang,
        fill_between_bounds=fill_between_bounds,
        fill_between_label=fill_between_label,
        fill_between_opacity=fill_between_opacity
    )
    if ions is not None and not use_ang_x_axis and num_centres > 1:
        warn_msg = (
            f"\nEmission lines only plotted with respect to {plot_labels[0]}."
        )
        warnings.warn(warn_msg)

    if hlines is not None:
        for i, (name, val) in enumerate(hlines.items()):
            plt.axhline(val, color=colour_map(i), linestyle='--', lw=LINEWIDTH, label=name)

    if isinstance(plot_y_bounds, tuple):
        plt.ylim(plot_y_bounds[0], plot_y_bounds[1])
    elif plot_y_bounds:
        if not isinstance(plot_centres, list):
            if plot_centres == H_ALPHA:
                plt.ylim(-10, 30)
            elif plot_centres == H_BETA:
                plt.ylim(-10, 20)
        else:
            plt.ylim(-10, 30)

    plt.xlabel(x_axis_label)
    plt.ylabel(SFD_Y_AX_LABEL)
    plt.title(f"Spectral flux density difference from 2001")
    plt.legend()
    plt.show()

def plot_gaussians(
    x: np.ndarray,
    y_data: np.ndarray,
    sep_gaussian_vals: np.ndarray[np.ndarray],
    summed_gaussian_vals: np.ndarray,
    y_data_errs: np.ndarray | None = None,
    summed_gaussian_errs: np.ndarray | None = None,
    colour_map: Colormap = COLOUR_MAP,
    error_opacity: float = ERR_OPAC,
    y_axis_label: str = SFD_Y_AX_LABEL,
    x_axis_label: str = VEL_LABEL,
    title: str | None = None,
    mask_vel_width: float | None = VEL_PLOT_WIDTH,
    mask_lam_centre: float | None = None,
    red_chi_sq: float | None = None
) -> None:
    if mask_vel_width is not None:
        if mask_lam_centre is None:
            raise ValueError("mask_lam_centre must be provided if mask_vel_width is provided")
        mask = get_vel_lam_mask(x, mask_vel_width, mask_lam_centre)
        x = x[mask]
        y_data = y_data[mask]
        y_data_errs = y_data_errs[mask] if y_data_errs is not None else None

    plt.figure(figsize=FIG_SIZE)
    plt.plot(x, y_data, color='black', label='Data', lw = LINEWIDTH)
    plt.plot(x, summed_gaussian_vals, color='red', label='Total Gaussian fit', lw = 2*LINEWIDTH)
    if summed_gaussian_errs is not None:
        plt.fill_between(
            x, summed_gaussian_vals - summed_gaussian_errs,
            summed_gaussian_vals + summed_gaussian_errs,
            color='red', alpha=error_opacity
        )
    if y_data_errs is not None:
        plt.fill_between(
            x, y_data - y_data_errs,
            y_data + y_data_errs,
            color='black', alpha=error_opacity
        )
    for i in range(len(sep_gaussian_vals)):
        plt.plot(
            x, sep_gaussian_vals[i],
            color=colour_map(i), label=f'Gaussian {i+1}',
            linestyle='--', lw = LINEWIDTH
        )
    if red_chi_sq is not None:
        label = r"Reduced $\chi^2$ = "
        label += f"{red_chi_sq:.2f}"
        plt.text(
            0.05, 0.95, label,
            transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top'
        )
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()




def compare_balmer_decrements(
    results: list[list[dict[str, np.ndarray]]],
    num_gaussians_list: list[int],
    num_bins_list: list[int],
    year: int,
    colour_map: Colormap = COLOUR_MAP
) -> float | None:
    bd_mean = None
    for i, num_bins in enumerate(num_bins_list):
        if num_bins == 1:
            one_bin_results = [result[i] for result in results]
            one_bin_bds = [result["bd"] for result in one_bin_results]
            plt.errorbar(
                x=num_gaussians_list,
                y=one_bin_bds,
                yerr=[result["bd_err"] for result in one_bin_results],
                # make data points not connected by lines
                linestyle="None",
                fmt='o', capsize=4
            )
            bd_mean = np.mean(one_bin_bds)
            plt.axhline(
                bd_mean, linestyle="--", color="black",
                lw=LINEWIDTH, label=f"Mean Balmer Decrement ({bd_mean:.2f})"
            )
            plt.xlabel("Number of Gaussians")
            plt.ylabel("Balmer Decrement")
            plt.ylim(0, 10)
            plt.title(f"{year} Balmer Decrement vs Number of Gaussians (no binning)")
            plt.legend()
            plt.show()
            continue
        
        for j, num_gaussians in enumerate(num_gaussians_list):
            plt.errorbar(
                x=results[j][i]["vel_centres"],
                y=results[j][i]["bd"],
                yerr=results[j][i]["bd_err"],
                fmt='o', capsize=4,
                label=f"{num_gaussians} gaussians",
                color=colour_map(j)
            )
        plt.xlabel(VEL_LABEL)
        plt.ylabel("Balmer Decrement")
        plt.ylim(0, 10)
        plt.title(f"{year} Balmer Decrement vs Velocity ({num_bins} bins)")
        plt.legend()
        plt.show()
    return bd_mean




def compare_balmer_decrements_old(
    balmer_decrements_0_bins: np.ndarray,
    balmer_decrements_many_bins: np.ndarray,
    vel_bin_centres_all: np.ndarray,
    year: int,
    num_bins_bounds: tuple[int, int] = (1, 7),
    num_gaussians_bounds: tuple[int, int] = (0, 5),
    colour_map: Colormap = COLOUR_MAP
) -> None:

    num_bins_range = range(num_bins_bounds[0], num_bins_bounds[1] + 1)
    num_gaussians_range = range(num_gaussians_bounds[0], num_gaussians_bounds[1] + 1)
    print(f"num_bins_range: {num_bins_range}")
    print(f"num_gaussians_range: {num_gaussians_range}")

    plt.plot(num_gaussians_range, balmer_decrements_0_bins, color='black', label='0 bins', lw = LINEWIDTH)
    plt.xlabel("Number of Gaussians")
    plt.ylabel("Balmer Decrement")
    plt.title(f"{year} Balmer Decrement vs Number of Gaussians")
    plt.show()

    # for > 0 num_bins, plot the balmer decrement on the y axis vs the velocity on the x axis, 
    # with multiple dashed lines for each number of gaussians
    
    for num_bins in num_bins_range:
        for num_gaussians in num_gaussians_range:
            plt.plot(
                vel_bin_centres_all[num_gaussians],
                balmer_decrements_many_bins[num_gaussians],
                color=colour_map(num_gaussians),
                label=f'{num_gaussians} gaussians',
                linestyle='None',
                lw = LINEWIDTH
            )
        plt.xlabel(VEL_LABEL)
        plt.ylabel("Balmer Decrement")
        plt.title(f"{year} Balmer Decrement vs Velocity ({num_bins} bins)")
        plt.legend()
        plt.show()