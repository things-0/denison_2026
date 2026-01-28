import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap

from .constants import (
    LINEWIDTH, Z_SPEC, RES_15_BLUE, RES_15_RED,
    FIG_SIZE, SFD_Y_AX_LABEL
)

def plot_vert_emission_lines(
    ions: dict[str, float] | None, 
    plot_x_bounds: tuple[float, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = 0.5,
    vlines_cmap: Colormap | None = plt.cm.tab10,
    is_rest_frame: bool = True,
) -> None:
    if ions is None:
        plt.xlim(plot_x_bounds)
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
    error_opacity: float = 0.5,
    ions: dict[str, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None
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

    plot_vert_emission_lines(ions, plot_x_bounds=x_bounds)
    
    if y_bounds is not None:
        plt.ylim(y_bounds)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()

def plot_polynomial_ratio(
    lambdas: np.ndarray,
    vals: np.ndarray,
    polynom_vals: np.ndarray,
    binned_lambdas: np.ndarray | None = None,
    binned_vals: np.ndarray | None = None,
    vals_removed: np.ndarray | None = None,
    degree: float = 6,
    bin_width: float = 40,
    bin_by_med: bool = True,
    title: str | None = None,
    plot_selection: bool = True,
) -> None:
    plt.figure(figsize=FIG_SIZE)

    if plot_selection:
        ratio_label = 'Spectral flux density ratio'
        plt.plot(lambdas, vals_removed, color='red', label=f'{ratio_label} (ignored Balmer)', lw = LINEWIDTH)
    else:
        ratio_label = 'actual ratio'

    plt.plot(lambdas, vals, alpha=0.5, color='black', label=ratio_label, lw = LINEWIDTH)
    
    if bin_by_med:
        plt.plot(binned_lambdas, binned_vals, color='black', label=f'ratio binned by median (width {bin_width} Å)', lw=4*LINEWIDTH)
        poly_label = f"polynomial fit (degree {degree}) to binned ratio"
    else:
        poly_label = f"polynomial fit (degree {degree})"
    plt.plot(binned_lambdas, polynom_vals, color='red', label=poly_label, lw=4*LINEWIDTH)
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
    lam_bounds: tuple[float] | None = None,
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
