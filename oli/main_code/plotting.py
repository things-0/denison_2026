import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap

from .constants import *
from .helpers import get_lam_bounds, convert_lam_to_vel

def plot_vert_emission_lines(
    ions: dict[str, float] | None, 
    plot_x_bounds: tuple[float, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = 0.5,
    vlines_cmap: Colormap | None = plt.cm.tab10,
    is_rest_frame: bool = True,
) -> None:
    plt.xlim(plot_x_bounds)
    if ions is None:
        return
    for i, (name, lam) in enumerate(ions.items()):
        if is_rest_frame:
            obs_lam = lam * (1+Z_SPEC)
        else:
            obs_lam = lam
        if plot_x_bounds is None or (plot_x_bounds[0] < obs_lam < plot_x_bounds[1]):
            plt.axvline(
                obs_lam, linestyle='--', lw=LINEWIDTH,
                color=vlines_cmap(i), label=name
            )
    if fill_between_bounds is not None:
        if fill_between_label is None:
            plt.axvspan(
                fill_between_bounds[0],
                fill_between_bounds[1],
                color='lightgrey', alpha=fill_between_opacity
            )
        else:
            plt.axvspan(
                fill_between_bounds[0],
                fill_between_bounds[1],
                color='lightgrey', alpha=fill_between_opacity,
                label=fill_between_label
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
    plt.xlabel("Wavelength (Å)")
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
    x_axis_label: str = "Wavelength (Å)",
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
    
    max_flux = max(
        np.nanmax(flux01),
        np.nanmax(flux15),
        np.nanmax(flux21),
        np.nanmax(flux22)
    )
    min_flux = min(
        np.nanmin(flux01),
        np.nanmin(flux15),
        np.nanmin(flux21),
        np.nanmin(flux22)
    )
    total_range = max_flux - min_flux
    smallest_range = min(
        np.nanmax(flux01) - np.nanmin(flux01),
        np.nanmax(flux15) - np.nanmin(flux15),
        np.nanmax(flux21) - np.nanmin(flux21),
        np.nanmax(flux22) - np.nanmin(flux22)
    )
    if y_bounds is not None:
        plt.ylim(y_bounds)
    elif total_range > 10 * smallest_range:
        plt.ylim((0, 1.2 * smallest_range))

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
    plt.xlabel("Wavelength (Å)")
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
    plt.xlabel("Wavelength (Å)")
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
    years_to_plot: list[int] = [2015, 2021, 2022],
    diff_15: np.ndarray | None = None,
    diff_21: np.ndarray | None = None,
    diff_22: np.ndarray | None = None,
    diff_15_err: np.ndarray | None = None,
    diff_21_err: np.ndarray | None = None,
    diff_22_err: np.ndarray | None = None,
    ions: dict[str, float] | bool = True,
    plot_centre: float | list[float] = H_ALPHA,
    vel_plot_width: float | None = VEL_PLOT_WIDTH,
    use_ang_x_axis: bool = False,
    plot_y_bounds: tuple[float, float] | bool = True,
    plot_errors: bool = False,
    error_opacity: float = ERR_OPAC,
) -> None:

    if not isinstance(plot_centre, list):
        plot_centre = [plot_centre]

    #TODO: add support for using velocity on the x-axis
    if use_ang_x_axis:
        x_axis_label = "Wavelength (Å)"
        x = lam
    else:
        x_axis_label = "Velocity (km/s)"
        for centre in plot_centre:
            x = convert_lam_to_vel(lam, centre)
    
    #TODO: add support for multiple plot centres




    plt.figure(figsize=FIG_SIZE)
    if 2015 in years_to_plot and diff_15 is not None:
        plt.plot(lam, diff_15, alpha=0.7, color='black', label=f'2015 - 2001', lw = LINEWIDTH)
    if 2021 in years_to_plot and diff_21 is not None:
        plt.plot(lam, diff_21, alpha=0.7, color='red', label=f'2021 - 2001', lw = LINEWIDTH)
    if 2022 in years_to_plot and diff_22 is not None:
        plt.plot(lam, diff_22, alpha=0.7, color='blue', label=f'2022 - 2001', lw = LINEWIDTH)
        
    there_are_ions_to_plot = True
    if isinstance(ions, bool):
        if ions:
            if plot_centre == H_ALPHA:
                ions = {r"H-${\alpha}$": H_ALPHA, "S[II] (1)": SII_1, "S[II] (2)": SII_2, "N[II] (2)": NII_2}
            elif plot_centre == H_BETA:
                ions = {r"H-${\beta}$": H_BETA, "O[III] (1)": OIII_1, "O[III] (2)": OIII_2}
            else:
                there_are_ions_to_plot = False
        else:
            there_are_ions_to_plot = False

    x_bounds = get_lam_bounds(plot_centre, vel_plot_width, width_is_vel=True) if vel_plot_width is not None else None

    if plot_errors:
        if 2015 in years_to_plot and diff_15_err is not None:
            plt.fill_between(lam, diff_15 - diff_15_err, diff_15 + diff_15_err, color='black', alpha=error_opacity)
        if 2021 in years_to_plot and diff_21_err is not None:
            plt.fill_between(lam, diff_21 - diff_21_err, diff_21 + diff_21_err, color='red', alpha=error_opacity)
        if 2022 in years_to_plot and diff_22_err is not None:
            plt.fill_between(lam, diff_22 - diff_22_err, diff_22 + diff_22_err, color='blue', alpha=error_opacity)


    if there_are_ions_to_plot:
        plot_vert_emission_lines(ions, x_bounds)
    elif x_bounds is not None:
        plt.xlim(x_bounds)

    if isinstance(plot_y_bounds, tuple):
        plt.ylim(plot_y_bounds[0], plot_y_bounds[1])
    elif plot_y_bounds:
        if plot_centre == H_ALPHA:
            plt.ylim(-10, 30)
        elif plot_centre == H_BETA:
            plt.ylim(-10, 20)

    plt.xlabel("Wavelength (Å)")
    plt.ylabel(SFD_Y_AX_LABEL)
    plt.title(f"Spectral flux density difference from 2001")
    plt.legend()
    plt.show()