import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from astropy.io import fits
import numpy as np
import warnings
from pathlib import Path

from typing import Any

from . import constants as const
from .helpers import (
    get_lam_bounds, convert_lam_to_vel, convert_to_vel_data,
    get_scaled_y_bounds, get_lam_mask, get_better_y_bounds
)

def my_savefig(
    save_fig_name: str | None,
    fig: plt.Figure | None = None,
    output_dir: Path = const.OUTPUT_DIR
) -> None:
    """
    Save a figure to the output directory.

    Parameters
    ----------
    save_fig_name: str | None
        The name of the figure to save.
    fig: plt.Figure | None
        The figure object to save. If None, the current figure is used.
    """
    my_fig = plt.gcf() if fig is None else fig
    fig_size = my_fig.get_size_inches()
    if tuple(fig_size) not in (const.FIG_SIZE, const.DOUBLE_FIG_SIZE):
        warn_msg = f"figure size {fig_size} is not const.FIG_SIZE {const.FIG_SIZE} or "
        warn_msg += f"const.DOUBLE_FIG_SIZE {const.DOUBLE_FIG_SIZE}"
        warnings.warn(warn_msg)
    # my_fig.tight_layout() # use "constrained" instead when first creating the figure
    if const.SAVE_FIGS and save_fig_name is not None and save_fig_name != "":
        # check if output_dir exists, if not create it
        if not output_dir.exists():
            warn_msg = f"output_dir {output_dir} does not exist. Creating it."
            warnings.warn(warn_msg)
            output_dir.mkdir(parents=True, exist_ok=True)

        prefix_suffix = save_fig_name.split(".")
        if len(prefix_suffix) < 2:
            # no extension provided, add .pdf
            save_fig_name += ".pdf"
        elif len(prefix_suffix) > 2:
            raise ValueError("too many . in save_fig_name")
        elif prefix_suffix[1] != "pdf":
            # extension provided, make it .pdf anyway
            save_fig_name = prefix_suffix[0] + ".pdf"
        while (output_dir / save_fig_name).exists():
            # append _cpy to the filename until it is unique
            warn_msg = f"{output_dir / save_fig_name} already exists. Creating copy"
            warnings.warn(warn_msg)
            save_fig_name = save_fig_name[:-4] + "_cpy.pdf"
        my_fig.savefig(output_dir / save_fig_name)
            

def plot_vert_emission_lines(
    ions: dict[str, float] | None, 
    plot_x_bounds: tuple[float, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = 0.5,
    vlines_cmap: Colormap = const.ALT_COLOUR_MAP,
    lam_centre: float | None = None,
    ax: plt.Axes | None = None
) -> None:
    """
    Plot dashed vertical emission lines on the spectrum.

    Parameters
    ----------
    ions: dict[str, float]
        The ions to plot. Each key is the line label, and each value is the rest wavelength of the
        line. E.g. {"Hα": 6564.61, "Hβ": 4862.68}.
    plot_x_bounds: tuple[float, float] | None
        The x-axis limits of the plot.
    fill_between_bounds: tuple[float, float] | None
        The x-axis limits of the region to fill between with light grey.
    fill_between_label: str | None
        The label of the region to fill between.
    fill_between_opacity: float
        The opacity of the region to fill between.
    vlines_cmap: Colormap
        The colormap to use for the emission lines.
    lam_centre: float | None
        If plotting in velocity space, this is the centre wavelength of the spectrum (0 km/s).
    ax: plt.Axes | None
        The axes to plot on.
    """

    if ax is None:
        ax = plt.gca()
    ax.set_xlim(plot_x_bounds)
    if fill_between_bounds is not None:
        ax.axvspan(
            fill_between_bounds[0], fill_between_bounds[1],
            color='lightgrey', alpha=fill_between_opacity,
            label=fill_between_label
        )
    if ions is None:
        return
    for i, (name, lam) in enumerate(ions.items()):
        if lam_centre is None:
            # if plotting in wavelength space, use the rest wavelength
            x_val = lam
        else:
            # if plotting in velocity space, convert the rest wavelength to velocity
            x_val = convert_lam_to_vel(lam, lam_centre)
        if plot_x_bounds is None or (plot_x_bounds[0] < lam < plot_x_bounds[1]):
            # only plot if it is within the plot x bounds
            ax.axvline(
                x_val, linestyle='--', lw=0.7*const.LINEWIDTH,
                color=vlines_cmap(i), label=name
            )

#TODO: add more comments from here down (including polynomial_fit & ppxf_funcs)

def plot_min_res(
    lam_sdss: np.ndarray,
    lam15_blue: np.ndarray,
    lam15_red: np.ndarray,
    res_min: np.ndarray,
    res01: np.ndarray,
    res21: np.ndarray,
    res22: np.ndarray,
    plot_RES_15_RED: bool = False,
    save_fig_name: str | None = "min_res.pdf"
) -> None:
    """
    Plot the resolving power and coverage of the spectra.

    Parameters
    ----------
    lam_sdss: np.ndarray
        The wavelength of the SDSS spectra.
    lam15_blue: np.ndarray
        The wavelength of the blue arm of the 2015 SAMI spectra.
    lam15_red: np.ndarray
        The wavelength of the red arm of the 2015 SAMI spectra.
    res_min: np.ndarray
        The minimum resolving power of the spectra.
    res01: np.ndarray
        The resolving power of the 2001 SDSS spectra.
    res21: np.ndarray
        The resolving power of the 2021 SDSS spectra.
    res22: np.ndarray
        The resolving power of the 2022 SDSS spectra.
    plot_RES_15_RED: bool
        If True, plot the resolving power of the red arm of the 2015 SAMI spectra.
        Note: this is much larger than all other resolving powers.
    save_fig_name: str | None
        The name of the figure to save.
    """

    lam15_blue_min = np.min(lam15_blue)
    lam15_blue_max = np.max(lam15_blue)
    lam15_red_min = np.min(lam15_red)
    lam15_red_max = np.max(lam15_red)

    if plot_RES_15_RED:
        # y_bounds of the SAMI rectangle coverage
        res_plot_bounds = [np.min(res_min), np.max((
            np.max(res21), np.max(res22), 
            np.max(res01), const.RES_15_BLUE, const.RES_15_RED
        ))]
    else:
        # y_bounds of the SAMI rectangle coverage
        res_plot_bounds = [np.min(res_min), np.max((
            np.max(res21), np.max(res22), 
            np.max(res01), const.RES_15_BLUE
        ))]
    # lam_plot_range = np.max(lam_sdss) - lam15_blue_min
    # axhline_end = (lam15_blue_max - lam15_blue_min) / lam_plot_range

    plt.figure(figsize=const.FIG_SIZE, layout=const.FIG_LAYOUT)
    plt.axhline(
        const.RES_15_BLUE, # xmax=axhline_end,
        color='blue', linestyle='--', label="SAMI blue"
    )
    if plot_RES_15_RED:
        plt.axhline(const.RES_15_RED, color='red', linestyle='--', label="SAMI red")
    plt.plot(lam_sdss, res01, alpha=0.5, label="2001")
    plt.plot(lam_sdss, res21, alpha=0.5, label="2021")
    plt.plot(lam_sdss, res22, alpha=0.5, label="2022")
    plt.plot(lam_sdss, res_min, color='black', alpha=0.5, lw=4, linestyle='--', label="Minimum")
    plt.fill_betweenx(res_plot_bounds, lam15_blue_min, lam15_blue_max, color='lightblue', alpha=0.5, label="SAMI blue coverage")
    plt.fill_betweenx(res_plot_bounds, lam15_red_min, lam15_red_max, color='red', alpha=0.2, label="SAMI red coverage")
    plt.xlabel(const.REST_ANG_LABEL)
    plt.ylabel("Resolving Power")
    if const.PLOT_TITLES:
        plt.title("Resolution of SDSS spectra")
    plt.legend(loc="upper left", fontsize=const.LEGEND_SCALE_FACTOR * const.TEXT_SIZE)
    my_savefig(save_fig_name)
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
    y_axis_label: str = const.SFD_Y_AX_LABEL,
    x_axis_label: str = const.REST_ANG_LABEL,
    error_opacity: float = const.ERR_OPAC,
    ions: dict[str, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = 0.5,
    y_offset: float = 0.0,
    figsize: tuple[float, float] = const.FIG_SIZE,
    legend_loc: str | None = "best",
    save_fig_name: str | None = ""
) -> None:
    """
    Plots the spectra from 2001 to 2022.

    Parameters
    ----------
    lam01: np.ndarray
        The wavelength of the 2001 SDSS spectra.
    lam15: np.ndarray | tuple[np.ndarray, np.ndarray]
        The wavelength of the 2015 SAMI spectra, or a tuple of the
        wavelength arrays of the blue and red arms, respectively.
    lam21: np.ndarray
        The wavelength of the 2021 SDSS spectra.
    lam22: np.ndarray
        The wavelength of the 2022 SDSS spectra.
    flux01: np.ndarray
        The flux of the 2001 SDSS spectra.
    flux15: np.ndarray | tuple[np.ndarray, np.ndarray]
        The flux of the 2015 SAMI spectra, or a tuple of the
        flux arrays of the blue and red arms, respectively.
    flux21: np.ndarray
        The flux of the 2021 SDSS spectra.
    flux22: np.ndarray
        The flux of the 2022 SDSS spectra.
    plot_errors: bool
        If True, plot the flux errors as ± shaded regions.
    flux01_err: np.ndarray | None
        The flux errors of the 2001 SDSS spectra.
    flux15_err: np.ndarray | tuple[np.ndarray, np.ndarray] | None
        The flux errors of the 2015 SAMI spectra, or a tuple of the
        flux error arrays of the blue and red arms, respectively.
    flux21_err: np.ndarray | None
        The flux errors of the 2021 SDSS spectra.
    flux22_err: np.ndarray | None
        The flux errors of the 2022 SDSS spectra.
    title: str | None
        The title of the plot.
    y_axis_label: str
        The label of the y-axis.
    x_axis_label: str
        The label of the x-axis.
    error_opacity: float
        The opacity of the flux error regions.
    ions: dict[str, float] | None
        The ions to plot. Each key is the line label, and each value is the
        rest wavelength of the line. E.g. {"Hα": 6564.61, "Hβ": 4862.68}.
    x_bounds: tuple[float, float] | None
        The x-axis limits of the plot.
    y_bounds: tuple[float, float] | None
        The y-axis limits of the plot. These will be adjusted to sensibly fit the
        data if this is not already the case. See :func:`get_better_y_bounds`.
    fill_between_bounds: tuple[float, float] | None
        The x-axis limits of the region to fill between with light grey.
    fill_between_label: str | None
        The label of the region to fill between.
    fill_between_opacity: float
        The opacity of the region to fill between.
    y_offset: float
        When plotting each spectrum, the flux of the entire spectrum will be increased
        by a multiple of this value to separate them. Set to 0 to plot the actual
        values of each spectrum.
    figsize: tuple[float, float]
        The size of the figure (x, y) in inches.
    legend_loc: str | None
        The location of the legend.
    save_fig_name: str | None
        The name of the figure to save.
    """
    
    # define offset fluxes
    flux01_os = flux01
    flux21_os = flux21 + 2 * y_offset
    flux22_os = flux22 + 3 * y_offset

    sami_is_split = True if isinstance(flux15, tuple) else False

    plt.figure(figsize=figsize, layout=const.FIG_LAYOUT)
    plt.plot(lam01, flux01_os, color=const.COL_01, label='2001 (SDSS)', lw = const.LINEWIDTH)

    if sami_is_split:
        flux15_blue, flux15_red = flux15
        lam15_blue, lam15_red = lam15

        # define offset fluxes
        flux15_blue_os, flux15_red_os = flux15_blue + y_offset, flux15_red + y_offset
        flux15_os = (flux15_blue_os, flux15_red_os)

        plt.plot(lam15_blue, flux15_blue_os, color='blue', label='2015 blue arm (SAMI)', lw = const.LINEWIDTH)
        plt.plot(lam15_red, flux15_red_os, color='red', label='2015 red arm (SAMI)', lw = const.LINEWIDTH)
    else:
        # define offset flux
        flux15_os = flux15 + y_offset

        plt.plot(lam15, flux15_os, color=const.COL_15, label='2015 (SAMI)', lw = const.LINEWIDTH)

    plt.plot(lam21, flux21_os, color=const.COL_21, label='2021 (SDSS)', lw = const.LINEWIDTH)
    plt.plot(lam22, flux22_os, color=const.COL_22, label='2022 (SDSS)', lw = const.LINEWIDTH)

    if plot_errors:
        if flux01_err is not None:
            plt.fill_between(lam01, flux01_os - flux01_err, flux01_os + flux01_err, color=const.COL_01, alpha=error_opacity)
        if flux15_err is not None:
            if sami_is_split:
                flux15_blue_err, flux15_red_err = flux15_err
                plt.fill_between(lam15_blue, flux15_blue_os - flux15_blue_err, flux15_blue_os + flux15_blue_err, color='blue', alpha=error_opacity)
                plt.fill_between(lam15_red, flux15_red_os - flux15_red_err, flux15_red_os + flux15_red_err, color='red', alpha=error_opacity)
            else:
                plt.fill_between(lam15, flux15_os - flux15_err, flux15_os + flux15_err, color=const.COL_15, alpha=error_opacity)
        if flux21_err is not None:
            plt.fill_between(lam21, flux21_os - flux21_err, flux21_os + flux21_err, color=const.COL_21, alpha=error_opacity)
        if flux22_err is not None:
            plt.fill_between(lam22, flux22_os - flux22_err, flux22_os + flux22_err, color=const.COL_22, alpha=error_opacity)

    plot_vert_emission_lines(
        ions, plot_x_bounds=x_bounds,
        fill_between_bounds=fill_between_bounds,
        fill_between_label=fill_between_label,
        fill_between_opacity=fill_between_opacity
    )

    better_y_bounds = get_better_y_bounds(
        y_bounds = y_bounds,
        lams = [lam01, lam15, lam21, lam22],
        fluxes = [flux01_os, flux15_os, flux21_os, flux22_os]
    )

    plt.ylim(bottom=better_y_bounds[0], top=better_y_bounds[1])

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title is not None and const.PLOT_TITLES:
        plt.title(title)
    # scale legend font size according to figure size
    if figsize == const.DOUBLE_FIG_SIZE:
        leg_font_size = const.LEGEND_SCALE_FACTOR * const.DOUBLE_TEXT_SIZE
    elif figsize == const.FIG_SIZE:
        leg_font_size = const.LEGEND_SCALE_FACTOR * const.TEXT_SIZE
    else:
        # use default matplotlib legend font size
        leg_font_size = None
    if legend_loc is not None:
        plt.legend(loc=legend_loc, fontsize=leg_font_size)
    else:
        plt.legend(fontsize=leg_font_size)
    my_savefig(save_fig_name)
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
    save_fig_name: str | None = ""
) -> None:
    """
    Plot the flux ratio of a particular spectrum and the spectrum (2015)
    used for recalibration.

    Parameters
    ----------
    lambdas: np.ndarray
        The wavelengths of the spectra.
    vals: np.ndarray
        The actual ratio of flux between the two epochs (excluding values to
        be ignored when fitting the polynomial - i.e. Balmer lines. These are
        plotted separately as `vals_removed`).
    polynom_vals: np.ndarray | None
        The values of the polynomial fit to the flux ratio.
    binned_lambdas: np.ndarray | None
        The wavelengths of the binned flux ratio.
    binned_vals: np.ndarray | None
        The binned flux ratio. See :func:`helpers.bin_data_by_median`.
    vals_removed: np.ndarray | None
        The actual ratio of flux between the two epochs in wavelength regions
        where the Balmer lines are ignored for the polynomial fit.
    degree: float
        The degree of the polynomial fit.
    bin_width: float
        The width of the `binned_vals` bins (angstroms).
    bin_by_med: bool
        If True, the polynomial was fit to the binned flux ratio, else to the
        the actual flux ratio.
    title: str | None
        The title of the plot.
    plot_selection: bool
        If True, plots `vals_removed` but not `polynom_vals`. Else, `polynom_vals`
        is plotted and `vals_removed` is not.
    save_fig_name: str | None
        The name of the figure to save.
    """
    plt.figure(figsize=const.FIG_SIZE, layout=const.FIG_LAYOUT)
    
    if bin_by_med:
        plt.plot(binned_lambdas, binned_vals, color='black', label=f'ratio binned by median (width {bin_width} Å)', lw=2*const.LINEWIDTH)
        poly_label = f"polynomial fit (degree {degree}) to binned ratio"
    else:
        poly_label = f"polynomial fit (degree {degree})"

    if plot_selection:
        ratio_label = 'Spectral flux density ratio'
        # plot the balmer vals_removed
        plt.plot(lambdas, vals_removed, color='red', label=f'{ratio_label} (ignored Balmer)', lw = const.LINEWIDTH)
    else:
        ratio_label = 'actual ratio'
        # plot the polynomial fit (only when vals_removed aren't plotted)
        plt.plot(binned_lambdas, polynom_vals, color='red', label=poly_label, lw=2*const.LINEWIDTH)

    # plot the actual ratio
    plt.plot(lambdas, vals, alpha=0.4, color='black', label=ratio_label, lw = 0.5*const.LINEWIDTH)
    plt.xlabel(const.REST_ANG_LABEL)
    plt.ylabel("Ratio")
    if const.PLOT_TITLES:
        plt.title(title)
    plt.legend(fontsize=const.LEGEND_SCALE_FACTOR * const.TEXT_SIZE)
    my_savefig(save_fig_name)
    plt.show()

def plot_adjusted_spectrum(
    lam: np.ndarray,
    adjusted_lam: np.ndarray,
    baseline_flux: np.ndarray,
    unadjusted_flux: np.ndarray,
    adjusted_flux: np.ndarray,
    year_to_adjust: int,
    baseline_year: int,
    ions: dict[str, float] | None = None,
    lam_bounds: tuple[float] | None = None, # const.TOTAL_LAM_BOUNDS, # (3800, 8000),
    flux_y_bounds: tuple[float] | None = None,
    title: str | None = None,
    save_fig_name: str | None = ""
)-> None:
    """
    Plot the adjusted spectrum against the unadjusted and baseline (2015) spectra.

    Parameters
    ----------
    lam: np.ndarray
        The wavelengths of the spectra.
    adjusted_lam: np.ndarray
        The wavelengths of the adjusted spectrum. This may be the same as `lam`,
        or it may be trimmed if the unadjusted spectrum wavelength range extended
        beyond the baseline spectrum wavelength range and no extrapolation was
        performed.
    baseline_flux: np.ndarray
        The flux of the baseline (2015) spectrum.
    unadjusted_flux: np.ndarray
        The flux of the unadjusted spectrum.
    adjusted_flux: np.ndarray
        The flux of the spectrum to adjust after applying the polynomial fit.
    year_to_adjust: int
        The year of the spectrum to adjust.
    baseline_year: int
        The year of the baseline (2015) spectrum.
    ions: dict[str, float] | None
        The ions to plot. Each key is the line label, and each value is the
        rest wavelength of the line. E.g. {"Hα": 6564.61, "Hβ": 4862.68}.
    lam_bounds: tuple[float] | None
        The wavelength bounds of the plot.
    flux_y_bounds: tuple[float] | None
        The flux bounds of the plot. These will be adjusted to sensibly fit the
        data if this is not already the case. See :func:`get_better_y_bounds`.
    title: str | None
        The title of the plot.
    save_fig_name: str | None
        The name of the figure to save.
    """
    plt.figure(figsize=const.FIG_SIZE, layout=const.FIG_LAYOUT)
    plt.plot(lam, baseline_flux, color='black', label=f'{baseline_year}', lw = 0.7*const.LINEWIDTH)
    plt.plot(lam, unadjusted_flux, color='orange', label=f'{year_to_adjust}', lw = 0.7*const.LINEWIDTH)
    plt.plot(adjusted_lam, adjusted_flux, color='red', label=f'{year_to_adjust} (polynomial fit to {baseline_year})', lw = 0.7*const.LINEWIDTH)
    plt.xlabel(const.REST_ANG_LABEL)
    plt.ylabel(const.SFD_Y_AX_LABEL)

    plot_vert_emission_lines(ions, lam_bounds)
    if const.PLOT_TITLES:
        plt.title(title)

    better_y_bounds = get_better_y_bounds(
        y_bounds = flux_y_bounds,
        lams = [lam, lam, lam],
        fluxes = [baseline_flux, unadjusted_flux, adjusted_flux]
    )

    plt.ylim(bottom=better_y_bounds[0], top=better_y_bounds[1])

    plt.legend(fontsize=const.LEGEND_SCALE_FACTOR * const.TEXT_SIZE)
    my_savefig(save_fig_name)
    plt.show()


def plot_diff_spectra_one_fig(
    fig: plt.Figure,
    lam: np.ndarray,
    diff_15: np.ndarray | None = None,
    diff_21: np.ndarray | None = None,
    diff_22: np.ndarray | None = None,
    diff_15_err: np.ndarray | None = None,
    diff_21_err: np.ndarray | None = None,
    diff_22_err: np.ndarray | None = None,
    ions: dict[str, float] | bool = False,
    vel_plot_width: float | None = const.VEL_PLOT_WIDTH,
    hlines: dict[str, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = const.FILL_BETWEEN_OPAC,
    plot_centres: float | list[float] = [const.H_ALPHA, const.H_BETA],
    plot_labels: list[str] | None = [r"H${\alpha}$", r"H${\beta}$"],
    use_ang_x_axis: bool = False,
    plot_y_bounds: tuple[float, float] | bool = True,
    scale_axes: bool = False,
    y_top_scale_factor: float = 1.25,
    error_opacity: float = const.ERR_OPAC,
    colour_map: Colormap = const.COLOUR_MAP,
    n_ticks_x: int | None = None
) -> tuple[str, str | None, str]:
    """
    Plots some or all of the 2015, 2021, and 2022 difference (from 2001) spectra
    onto a single (sub)figure.

    Parameters
    ----------
    fig: plt.Figure
        The figure to plot the spectra onto.
    lam: np.ndarray
        The wavelengths of the spectra.
    diff_15: np.ndarray | None
        The flux density of the 2015 - 2001 difference spectrum.
    diff_21: np.ndarray | None
        The flux density of the 2021 - 2001 difference spectrum.
    diff_22: np.ndarray | None
        The flux density of the 2022 - 2001 difference spectrum.
    diff_15_err: np.ndarray | None
        The error of the flux density of the 2015 - 2001 difference spectrum.
    diff_21_err: np.ndarray | None
        The error of the flux density of the 2021 - 2001 difference spectrum.
    diff_22_err: np.ndarray | None
        The error of the flux density of the 2022 - 2001 difference spectrum.
    ions: dict[str, float] | bool
        The ions to plot. Each key is the line label, and each value is the
        rest wavelength of the line. E.g. {"Hα": 6564.61, "Hβ": 4862.68}.
        Default values according to `plot_centres` will be used if True.
    vel_plot_width: float | None
        The width of the plot in velocity units.
    hlines: dict[str, float] | None
        The horizontal lines to plot. Each key is the line label, and each value is the
        value of the line. E.g. {"1 σ": 20.0, "2 σ": 25.0}.
    fill_between_bounds: tuple[float, float] | None
        The bounds of the region to fill between with light grey.
    fill_between_label: str | None
        The label of the region to fill between.
    fill_between_opacity: float
        The opacity of the region to fill between.
    plot_centres: float | list[float]
        If a single number, this is the centre wavelength (or 0 km/s velocity) of the plot.
        If a list, different sections of the spectrum will be overplotted on the same
        velocity scale each with their own centre wavelength set to 0 km/s.
    plot_labels: list[str] | None
        Dictates the prefix of the epoch legend labels, and, if `scale_axes` is True,
        the y-axis labels of the plot.
    use_ang_x_axis: bool
        If True, the x-axis is in angstroms, else it is in km/s.
    plot_y_bounds: tuple[float, float] | bool
        If a tuple, the y-axis bounds of the plot. If True, the y-axis bounds are
        set according to the `plot_centre` (if it is a single number), else,
        :class:`matplotlib.pyplot` sets the bounds automatically.
    scale_axes: bool
        If True and `plot_centres` is a list of length 2, the y-axis is scaled to the
        same bounds for all spectra such that their maxima and 0 fluxes are aligned.
        See :func:`get_scaled_y_bounds` for more details.
    y_top_scale_factor: float
        The factor that the max of y2 will be scaled by to create the top of the second axis.
    error_opacity: float
        The opacity of the error regions.
    colour_map: Colormap
        The colour map to use for the separate epochs and plot centres.
    n_ticks_x: int | None
        The number of major ticks on the x-axis. If None, :class:`matplotlib.pyplot` will set the
        ticks automatically.
    """

    if isinstance(plot_centres, list) and plot_labels is not None and len(plot_centres) != len(plot_labels):
        raise ValueError("plot_centres and plot_labels must have the same length and correspond to each other")
    if not isinstance(plot_centres, list) and isinstance(plot_labels, list) and len(plot_labels) != 1:
        raise ValueError("plot_labels must be a list of length 1 if plot_centres is just a number")

    num_centres = len(plot_centres) if isinstance(plot_centres, list) else 1
    if plot_labels is None:
        # add nothing to the y axis or legend labels 
        plot_labels = [""] * num_centres

    if use_ang_x_axis:
        if isinstance(plot_centres, list):
            raise ValueError("plot_centres must be a single number if use_ang_x_axis is True")


        x_axis_label = const.REST_ANG_LABEL
        x_15, x_21, x_22 = [lam], [lam], [lam]
        # not overplotting different velocity centres on top of each other
        diffs_15 = [diff_15] if diff_15 is not None else None
        diffs_21 = [diff_21] if diff_21 is not None else None
        diffs_22 = [diff_22] if diff_22 is not None else None
        diffs_15_err = [diff_15_err] if diff_15_err is not None else None
        diffs_21_err = [diff_21_err] if diff_21_err is not None else None
        diffs_22_err = [diff_22_err] if diff_22_err is not None else None
    else:
        if vel_plot_width is None:
            raise ValueError("use_ang_x_axis must be True if vel_plot_width is None")
        x_axis_label = const.VEL_LABEL
        # each diffs_xx is a list of difference spectra for a single epoch, each with a different plot centre
        # (or None if diff_xx is None)
        x_15, diffs_15, diffs_15_err = convert_to_vel_data(
            lam, diff_15, diff_15_err, plot_centres, vel_plot_width
        )
        x_21, diffs_21, diffs_21_err = convert_to_vel_data(
            lam, diff_21, diff_21_err, plot_centres, vel_plot_width
        )
        x_22, diffs_22, diffs_22_err = convert_to_vel_data(
            lam, diff_22, diff_22_err, plot_centres, vel_plot_width
        )


    all_diffs = {
        "2015": (x_15, diffs_15, diffs_15_err),
        "2021": (x_21, diffs_21, diffs_21_err),
        "2022": (x_22, diffs_22, diffs_22_err)
    }
    # only keep epochs with not None difference spectra
    all_diffs_not_none = {k: v for k, v in all_diffs.items() if v[1] is not None}
    if scale_axes:
        if not isinstance(plot_y_bounds, bool) or plot_y_bounds == True:
            raise ValueError("plot_y_bounds must be False if using scale_axes")
        if len(all_diffs_not_none) != 1:
            raise ValueError("One of diffs_15, diffs_21, diffs_22 must be not None if using scale_axes")
        year, (_, diffs_xx, _) = next(iter(all_diffs_not_none.items()))
        if num_centres == 2:
            # diffs_xx has 2 difference spectra
            ax1 = fig.add_subplot()
            ax2 = ax1.twinx()
            axes = [ax1, ax2]
            # scale bounds between both plots in this (sub)figure
            y_bounds_1, y_bounds_2 = get_scaled_y_bounds(
                y1=diffs_xx[0],
                y2=diffs_xx[1],
                y_top_scale_factor=y_top_scale_factor
            )
            ax1.set_ylim(y_bounds_1)
            ax2.set_ylim(y_bounds_2)

        else:
            raise NotImplementedError("scale_axes is only supported for 2 centres")
    else:
        # get axis for the current figure (store as list for iterations below)
        axes = [fig.add_subplot()]
    for i in range(num_centres):
        ax = axes[i] if scale_axes else axes[0] # only one axis if not scaling axes
        label_info = plot_labels[i] # prefix of the epoch legend labels to denote plot centres. e.g. "Hα" or "Hβ"
        # set the number of major ticks on the x-axis
        if n_ticks_x is not None:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=n_ticks_x))
        for j, (year, (x_xx, diffs_xx, diffs_xx_err)) in enumerate(
            all_diffs_not_none.items()
        ):
            if diffs_xx is None:
                continue
            # get the current flux, wavelength, and error for the i-th plot centre of the xx epoch
            cur_flux = diffs_xx[i]
            cur_lam = x_xx[i]
            cur_flux_err = diffs_xx_err[i] if diffs_xx_err is not None else None
            
            if num_centres == 2:
                # use red or blue, depending on which wavelength is longer (larger)
                colour = const.COL_RED if plot_centres[i] > plot_centres[(i+1)%2] else const.COL_BLUE
            elif num_centres > 2:
                colour = colour_map(len(all_diffs_not_none)*i+j)
            else:
                colour = const.EPOCHS_INFO[year]["colour"]
                
            ax.plot(cur_lam, cur_flux, alpha=0.7, color=colour, label=f'{label_info} {year} - 2001', lw = const.LINEWIDTH)
            
            if cur_flux_err is not None:
                ax.fill_between(cur_lam, cur_flux - cur_flux_err, cur_flux + cur_flux_err, color=colour, alpha=error_opacity)

    if isinstance(ions, bool):
        if ions:
            # set ions automatically
            ions_near_h_alpha = {
                const.HA_LATEX: const.H_ALPHA,
                const.SII_BLUE_LATEX: const.SII_BLUE,
                const.SII_RED_LATEX: const.SII_RED,
                const.NII_RED_LATEX: const.NII_RED
            }
            ions_near_h_beta = {
                const.HB_LATEX: const.H_BETA,
                const.OIII_BLUE_LATEX: const.OIII_BLUE,
                const.OIII_RED_LATEX: const.OIII_RED
            }
            if isinstance(plot_centres, list):
                # plot any lines near the plot centres (in velocity space so these lines will be confusing - warning given below)
                ions = {}
                if const.H_ALPHA in plot_centres:
                    ions = ions | ions_near_h_alpha
                if const.H_BETA in plot_centres:
                    ions = ions | ions_near_h_beta
            else:
                # plot any lines near the plot centre
                if plot_centres == const.H_ALPHA:
                    ions = ions_near_h_alpha
                elif plot_centres == const.H_BETA:
                    ions = ions_near_h_beta
                else:
                    ions = None
        else:
            # don't set ions automatically
            ions = None
    # else leave ions as a dict

    if use_ang_x_axis:   
        # already verified that plot_centres is not a list if use_ang_x_axis is True     
        x_bounds = get_lam_bounds(plot_centres, vel_plot_width, width_is_vel=True) if vel_plot_width is not None else None
        # not plotting in velocity space, so plot_vert_emission_lines does not need a lam_centre to convert to velocity
        lam_centre = None
    else:
        # arbitrarilychoose first plot centre (warn below)
        lam_centre = plot_centres[0] if isinstance(plot_centres, list) else plot_centres
        x_bounds = (-vel_plot_width / 2, vel_plot_width / 2) if vel_plot_width is not None else None
        

    plot_vert_emission_lines(
        ions, x_bounds, lam_centre=lam_centre,
        fill_between_bounds=fill_between_bounds,
        fill_between_label=fill_between_label,
        fill_between_opacity=fill_between_opacity
    )
    if ions is not None and not use_ang_x_axis and num_centres > 1:
        warn_msg = (
            f"Emission lines only plotted with respect to {plot_labels[0]}."
        )
        warnings.warn(warn_msg)

    # plot horizontal lines
    if hlines is not None:
        for i, (name, val) in enumerate(hlines.items()):
            plt.axhline(val, color=colour_map(i), linestyle='--', lw=const.LINEWIDTH, label=name)

    if isinstance(plot_y_bounds, tuple):
        # use user-provided y-axis bounds
        plt.ylim(plot_y_bounds[0], plot_y_bounds[1])
    elif plot_y_bounds and not scale_axes:
        # set y-axis bounds automatically based on the plot centre
        # (don't adjust ylim if scale_axes is True as this is already done above)
        if not isinstance(plot_centres, list):
            if plot_centres == const.H_ALPHA:
                plt.ylim(-5, 30)
            elif plot_centres == const.H_BETA:
                # typically not as much flux around Hβ
                plt.ylim(-6, 9)
        else:
            # if using multiple plot centres, (-5, 30) is a good default to cover most data
            plt.ylim(-5, 30)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5, lw=const.LINEWIDTH)
    title = "" if len(all_diffs_not_none) != 1 else f"{year} "
    title += "Spectral flux density difference from 2001"
    for i, ax in enumerate(axes):
        if scale_axes:
            # prefix of y-axis label to denote plot centres. e.g. "Hα" or "Hβ"
            y_axis_label = f"{plot_labels[i]}   {const.SFD_Y_AX_LABEL}"
            ax.set_ylabel(y_axis_label)
            # set to None so that plot_diff_spectra_all adds a y-axis label for each subplot
            # and doesn't set a sup y axis label
            y_axis_label = None
        else:
            # don't set a y-axis label for each subplot
            # let plot_diff_spectra_all set a sup y axis label
            y_axis_label = const.SFD_Y_AX_LABEL
        # axes is length 2 if scale_axes is True, otherwise length 1
        # split the legend between the two axes if scale_axes is True to clearly show which axis is 
        # which, since the different legends will contain the plot labels info (e.g. "Hα" or "Hβ")
        loc = "upper left" if i == 0 else "upper right"
        ax.legend(loc=loc, fontsize=const.LEGEND_SCALE_FACTOR * const.TEXT_SIZE)
    return x_axis_label, y_axis_label, title

def plot_diff_spectra_all(
    lam: np.ndarray,
    diff_15: np.ndarray | None = None,
    diff_21: np.ndarray | None = None,
    diff_22: np.ndarray | None = None,
    diff_15_err: np.ndarray | None = None,
    diff_21_err: np.ndarray | None = None,
    diff_22_err: np.ndarray | None = None,
    ions_list: list[dict[str, float] | bool] = [
        {const.HB_LATEX: const.H_BETA, const.OIII_RED_LATEX: const.OIII_RED},
        {const.HA_LATEX: const.H_ALPHA}
    ],
    vel_plot_width: float | None = const.VEL_PLOT_WIDTH,
    hlines_list: list[dict[str, float] | None] = [None, None],
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = const.FILL_BETWEEN_OPAC,
    plot_centres_list: list[list[float] | float] = [const.H_BETA, const.H_ALPHA],
    plot_labels_list: list[list[str] | None] = [None, None], # [[r"H${\beta}$"], [r"H${\alpha}$"]],
    use_ang_x_axis: bool = False,
    plot_y_bounds_list: list[tuple[float, float] | bool] = [False, False],
    scale_axes_list: list[bool] = [False, False],
    scale_all_axes: bool = True,
    n_ticks_x_list: list[int | None] = [6, 6],
    y_top_scale_factor: float = 1.25,
    error_opacity: float = const.ERR_OPAC,
    colour_map: Colormap = const.COLOUR_MAP,
    save_fig_name: str | None = "",
    figsize: tuple[float, float] = const.FIG_SIZE
) -> None:
    """
    Plots some or all of the 2015, 2021, and 2022 difference (from 2001) spectra. Creates
    subplots for each zoomed in wavelength region (for each element of `*_list` arguments).

    Parameters
    ----------
    lam: np.ndarray
        The wavelengths of the spectra.
    diff_15: np.ndarray | None
        The flux density of the 2015 - 2001 difference spectrum.
    diff_21: np.ndarray | None
        The flux density of the 2021 - 2001 difference spectrum.
    diff_22: np.ndarray | None
        The flux density of the 2022 - 2001 difference spectrum.
    diff_15_err: np.ndarray | None
        The error of the flux density of the 2015 - 2001 difference spectrum.
    diff_21_err: np.ndarray | None
        The error of the flux density of the 2021 - 2001 difference spectrum.
    diff_22_err: np.ndarray | None
        The error of the flux density of the 2022 - 2001 difference spectrum.
    ions_list: list[dict[str, float] | bool]
        For each subplot, this list constains a dictionary of the ions to plot, or a boolean
        indicating whether to plot the default ions for the given plot centre.
    vel_plot_width: float | None
        The width of the velocity plots.
    hlines_list: list[dict[str, float] | None]
        The horizontal lines to plot for each subplot.
    fill_between_bounds: tuple[float, float] | None
        The bounds of the region to fill between.
    fill_between_label: str | None
        The label of the region to fill between.
    fill_between_opacity: float
        The opacity of the region to fill between.
    plot_centres_list: list[list[float] | float]
        For each subplot, the list or float in the outer list describes the centre wavelength
        of the plot. If a single number, this is the centre wavelength (or 0 km/s velocity) of
        the plot. If a list, different sections of the spectrum will be overplotted on the same
        velocity scale each with their own centre wavelength set to 0 km/s.
    plot_labels_list: list[list[str] | None]
        For each subplot, the elements of the outer list dictate the prefix of the epoch legend
        labels, and, if `scale_axes` is True, the y-axis labels of the plot. 
    use_ang_x_axis: bool
        If True, the x axis will be in angstroms.
    plot_y_bounds_list: list[tuple[float, float] | bool]
        For each subplot, the tuple of (y_min, y_max), or a bool: if True the y-axis bounds are
        set according to the corresponding `plot_centre` in `plot_centres_list` (if it is a single
        number), else, :class:`matplotlib.pyplot` sets the bounds automatically.
    scale_axes_list: list[bool]
        For each subplot, if True and the corresponding `plot_centres` (within `plot_centres_list`)
        is a list of length 2, the y-axis is scaled to the same bounds for all spectra in this plot
        such that their maxima and 0 fluxes are aligned. See :func:`get_scaled_y_bounds` for more
        details.
    scale_all_axes: bool
        If True, the spectra with the largest flux in each plot are used to scale the y-axis bounds
        such that the maxima and 0 fluxes of these maximal spectra are aligned. See
        :func:`get_scaled_y_bounds` for more details. Note: this is different to each argument in
        `scale_axes_list`, which scales the y-axis bounds within a single subplot for all spectra.
    n_ticks_x_list: list[int | None]
        The number of major ticks on the x-axis for each subplot. If None, :class:`matplotlib.pyplot`
        will set the ticks automatically.
    y_top_scale_factor: float
        The factor that the max of y2 will be scaled by to create the top of the second axis.
    error_opacity: float
        The opacity of the error regions.
    colour_map: Colormap
        The colour map to use for the separate epochs and plot centres.
    save_fig_name: str | None
        The name of the file to save the plot to.
    figsize: tuple[float, float]
        The size of the figure in inches (x, y).
    """
    num_plots_options = [
        len(plot_centres_list), len(plot_labels_list),
        len(ions_list), len(n_ticks_x_list), len(scale_axes_list),
        len(plot_y_bounds_list), len(hlines_list)
    ]
    if not np.all(np.array(num_plots_options) == num_plots_options[0]):
        raise ValueError(
            f"num_plots must be the same for all lists\n"
            f"plot_centres_list: {len(plot_centres_list)}\n"
            f"plot_labels_list: {len(plot_labels_list)}\n"
            f"ions_list: {len(ions_list)}\n"
            f"n_ticks_x_list: {len(n_ticks_x_list)}\n"
            f"scale_axes_list: {len(scale_axes_list)}\n"
            f"plot_y_bounds_list: {len(plot_y_bounds_list)}\n"
            f"hlines_list: {len(hlines_list)}"
        )
    num_plots = num_plots_options[0]

    fig = plt.figure(figsize=figsize, layout=const.FIG_LAYOUT)

    if scale_all_axes:
        if num_plots != 2:
            raise NotImplementedError("scale_all_axes is only supported for 2 plots")
        if not np.all(np.array(scale_axes_list) == False):
            raise ValueError("scale_axes_list must be False if scale_all_axes is True")
        if not np.all(np.array(plot_y_bounds_list) == False):
            raise ValueError("plot_y_bounds_list must be False if scale_all_axes is True")
        for plot_centres in plot_centres_list:
            if isinstance(plot_centres, list):
                raise ValueError("plot_centres_list must be a list of single numbers if scale_all_axes is True")

        # each diffs_xx is a list of difference spectra for a single epoch, each with a different plot centre
        # (or None if diff_xx is None)
        x_15, diffs_15, diffs_15_err = convert_to_vel_data(
            lam, diff_15, diff_15_err, plot_centres_list, vel_plot_width
        )
        x_21, diffs_21, diffs_21_err = convert_to_vel_data(
            lam, diff_21, diff_21_err, plot_centres_list, vel_plot_width
        )
        x_22, diffs_22, diffs_22_err = convert_to_vel_data(
            lam, diff_22, diff_22_err, plot_centres_list, vel_plot_width
        )

        all_diffs = {
            "2015": diffs_15,
            "2021": diffs_21,
            "2022": diffs_22
        }
        # only keep epochs with not None difference spectra
        all_diffs_not_none = {k: v for k, v in all_diffs.items() if v is not None}

        # find the epoch with the largest flux in the first (left) plot and the epoch with the largest flux in the second (right) plot
        # (gauranteed only 2 plots above)

        max_left_diffs_year = None
        max_left_diffs_val = -np.inf
        max_right_diffs_year = None
        max_right_diffs_val = -np.inf

        for year, diffs in all_diffs_not_none.items():
            left_diffs = diffs[0]
            right_diffs = diffs[1]
            if max_left_diffs_year is None or np.max(left_diffs) > max_left_diffs_val:
                max_left_diffs_year = year
                max_left_diffs_val = np.max(left_diffs)
            if max_right_diffs_year is None or np.max(right_diffs) > max_right_diffs_val:
                max_right_diffs_year = year
                max_right_diffs_val = np.max(right_diffs)
        
        # pick the difference spectra to use when scaling the y-axis bounds (can only choose 2 (functional limitation of get_scaled_y_bounds))
        left_diffs_to_scale = all_diffs_not_none[max_left_diffs_year][0]
        right_diffs_to_scale = all_diffs_not_none[max_right_diffs_year][1]

        y_bounds_1, y_bounds_2 = get_scaled_y_bounds(
            y1=left_diffs_to_scale,
            y2=right_diffs_to_scale,
            y_top_scale_factor=y_top_scale_factor
        )

        plot_y_bounds_list = [y_bounds_1, y_bounds_2]


    subfigs = fig.subfigures(nrows=1, ncols=num_plots)
    if not isinstance(subfigs, np.ndarray):
        # make iterable
        subfigs = np.array([subfigs])

    x_axis_labels = np.zeros(num_plots, dtype=object)
    y_axis_labels = np.zeros(num_plots, dtype=object)
    titles = np.zeros(num_plots, dtype=object)
    for i in range(num_plots):
        x_axis_label, y_axis_label, title = plot_diff_spectra_one_fig(
            fig=subfigs[i],
            lam=lam,
            diff_15=diff_15,
            diff_21=diff_21,
            diff_22=diff_22,
            diff_15_err=diff_15_err,
            diff_21_err=diff_21_err,
            diff_22_err=diff_22_err,
            ions=ions_list[i],
            vel_plot_width=vel_plot_width,
            hlines=hlines_list[i],
            fill_between_bounds=fill_between_bounds,
            fill_between_label=fill_between_label,
            fill_between_opacity=fill_between_opacity,
            plot_centres=plot_centres_list[i],
            plot_labels=plot_labels_list[i],
            use_ang_x_axis=use_ang_x_axis,
            plot_y_bounds=plot_y_bounds_list[i],
            scale_axes=scale_axes_list[i],
            y_top_scale_factor=y_top_scale_factor,
            error_opacity=error_opacity,
            colour_map=colour_map,
            n_ticks_x=n_ticks_x_list[i]
        )
        x_axis_labels[i] = x_axis_label
        y_axis_labels[i] = y_axis_label
        titles[i] = title
    if not np.all(x_axis_labels == x_axis_labels[0]):
        raise ValueError(f"conflicting x_axis_labels: {x_axis_labels}")
    if not np.all(y_axis_labels == y_axis_labels[0]):
        raise ValueError(f"conflicting y_axis_labels: {y_axis_labels}")
    if not np.all(titles == titles[0]):
        raise ValueError(f"conflicting titles: {titles}")
    if const.PLOT_TITLES:
        fig.suptitle(titles[0]) # pick the first one since they are all the same
    fig.supxlabel(x_axis_labels[0]) # pick the first one since they are all the same
    if y_axis_labels[0] is not None:
        fig.supylabel(y_axis_labels[0]) # pick the first one since they are all the same
    my_savefig(save_fig_name, fig=fig)
    fig.show()

#TODO include 2x3 of gaussian fit plots or 1x2 of just total gaussian fit
# (not all components) and relate colours to main diff spec plot
def plot_gaussians( 
    x: np.ndarray,
    y_data: np.ndarray,
    sep_gaussian_vals: np.ndarray[np.ndarray],
    summed_gaussian_vals: np.ndarray,
    y_data_errs: np.ndarray | None = None,
    summed_gaussian_errs: np.ndarray | None = None,
    colour_map: Colormap = const.COLOUR_MAP,
    error_opacity: float = const.ERR_OPAC,
    y_axis_label: str = const.SFD_Y_AX_LABEL,
    x_axis_label: str = const.VEL_LABEL,
    title: str | None = None,
    mask_vel_width: float | None = const.VEL_PLOT_WIDTH,
    mask_lam_centre: float | None = None,
    red_chi_sq: float | None = None,
    save_fig_name: str | None = ""
) -> None:
    """
    Plots the Gaussian fit to the data, showing all n Gaussians and
    the total summed fit.

    Parameters
    ----------
    x: np.ndarray
        x values (e.g. wavelength array).
    y_data: np.ndarray
        y values (e.g. flux).
    sep_gaussian_vals: np.ndarray[np.ndarray]
        The values of the individual Gaussians (shape `(num_gaussians, len(x))`).
    summed_gaussian_vals: np.ndarray
        The values of the total summed Gaussian fit.
    y_data_errs: np.ndarray | None
        The errors on the y values.
    summed_gaussian_errs: np.ndarray | None
        The errors on the total summed Gaussian fit.
    colour_map: Colormap
        The colour map to use for the n different Gaussians.
    error_opacity: float
        The opacity of the error regions.
    y_axis_label: str
        The label for the y axis.
    x_axis_label: str
        The label for the x axis.
    title: str | None
        The title of the plot.
    mask_vel_width: float | None
        The width of the plot in velocity units. This will trim the data if
        not None.
    mask_lam_centre: float | None
        The centre wavelength of the plot if trimming the data.
    red_chi_sq: float | None
        The reduced chi squared of the fit.
    save_fig_name: str | None
        The name of the file to save the plot to.
    """
    if mask_vel_width is not None:
        if mask_lam_centre is None:
            raise ValueError("mask_lam_centre must be provided if mask_vel_width is provided")
        mask = get_lam_mask(x, mask_vel_width, mask_lam_centre)
        x = x[mask]
        y_data = y_data[mask]
        y_data_errs = y_data_errs[mask] if y_data_errs is not None else None

    plt.figure(figsize=const.FIG_SIZE, layout=const.FIG_LAYOUT)
    # plot actual flux
    plt.plot(x, y_data, color='black', label='Data', lw = const.LINEWIDTH)
    # plot total Gaussian fit
    plt.plot(x, summed_gaussian_vals, color='red', label='Total Gaussian fit', lw = 2*const.LINEWIDTH)
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
            linestyle='--', lw = const.LINEWIDTH
        )
    if red_chi_sq is not None:
        label = r"Reduced $\chi^2$ = "
        label += f"{red_chi_sq:.2f}"
        plt.text(
            x=0.05, y=0.95,
            s=label, # text string
            transform=plt.gca().transAxes, # uses fraction of axes width and height rather than actual x, y values as coordinates
            fontsize=const.TEXT_SIZE, verticalalignment='top'
        )
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title is not None and const.PLOT_TITLES:
        plt.title(title)
    plt.legend(loc="upper right", fontsize=const.LEGEND_SCALE_FACTOR * const.TEXT_SIZE)
    my_savefig(save_fig_name)
    plt.show()


def plot_ppxf_comp(
    infile_name: str = "ppxf_components",
    infile_suffix: str = "",
    infile_path: Path = const.PPXF_DATA_DIR,
    fit_is_normalised: bool = True,
    plot_individual_gas_components: bool = False,
    plot_all_gas_components: bool = True,
) -> None:
    """
    Plots the components of a pPXF fit and the host subtracted spectrum.

    Parameters
    ----------
    infile_name: str
        The name of the file to read the data from: "<infile_name>_<infile_suffix>.fits".
    infile_suffix: str
        The suffix of the infile name: "<infile_name>_<infile_suffix>.fits".
    infile_path: Path
        The path to the directory containing the infile.
    fit_is_normalised: bool
        If True, flux will be re-multiplied by the median flux of the galaxy.
    plot_individual_gas_components: bool
        If True, the individual gas components are plotted on separate figures.
    plot_all_gas_components: bool
        If True, all gas components are plotted on a single figure.
    """
    if infile_suffix != "":
        infile_suffix = "_" + infile_suffix
    full_infile_path = infile_path / (infile_name + infile_suffix + ".fits")
    fits.info(full_infile_path)
    with fits.open(full_infile_path) as hdul:
        try:
            medflux = hdul['GALAXY'].header.get('MEDFLUX')
        except KeyError:
            warnings.warn("MEDFLUX not found in GALAXY header. Using 1 instead.")
            medflux = 1
        scale_factor = medflux if fit_is_normalised else 1

        max_narrow_component = hdul['GAS_ALL'].header.get('MAX_NL_COMP')
        if not isinstance(max_narrow_component, int):
            raise ValueError("valid MAX_NL_COMP not found in GAS_ALL header")

        lam = hdul['WAVELENGTH'].data
        galaxy = hdul['GALAXY'].data * scale_factor     # re multiply by median flux (if necessary)
        stellar = hdul['STELLAR'].data * scale_factor
        bestfit = hdul['BESTFIT'].data * scale_factor
        goodpixels = hdul['GOODPIXELS'].data

        print(goodpixels)

        # get gas components:
        gas_components = {}
        for hdu in hdul:
            name = hdu.name
            if name.startswith('GAS_COMP_'):
                k = int(name.split('_')[-1])
                gas_components[k] = hdu.data * scale_factor

    
    gas_components = dict(sorted(gas_components.items()))

    if plot_individual_gas_components:
        for k, spec in gas_components.items():
            plt.figure(figsize=const.FIG_SIZE, layout=const.FIG_LAYOUT)
            plt.plot(lam, spec, 'k')
            plt.title(f'Gas component {k}')
            plt.xlabel(r'Wavelength [$\AA$]')
            plt.ylabel('Flux')
            # plt.tight_layout()
            plt.show()

    if plot_all_gas_components:
        plt.figure(figsize=const.FIG_SIZE, layout=const.FIG_LAYOUT)
        for i, (k, spec) in enumerate(gas_components.items()):
            plt.plot(lam, spec, 'k', color=const.ALT_COLOUR_MAP(i), label=f'Gas component {k}')
            plt.xlabel(r'Wavelength [$\AA$]')
            plt.ylabel('Flux')
            # plt.tight_layout()
        plt.title(f"All Gas components ({infile_suffix.strip('_')})")
        plt.legend(fontsize=const.LEGEND_SCALE_FACTOR * const.TEXT_SIZE)
        plt.show()
        
    # get the narrow IDs:
    narrow_ids = [k for k in gas_components if k <= max_narrow_component]

    narrow_gas = np.zeros_like(galaxy)
    for k in narrow_ids:
        narrow_gas += gas_components[k]
    
    galaxy_sub_host = galaxy - stellar - narrow_gas

    # plot spectrum with host removed:
    fig1 = plt.figure(figsize=const.FIG_SIZE, layout="constrained")

    ax1 = fig1.add_subplot(2,1,1)
    ax1.plot(lam, galaxy, 'k', label="galaxy")
    ax1.plot(lam, bestfit, 'm', label="bestfit")
    ax1.plot(lam, stellar, 'r', label="stellar")
    ax1.set(xlabel=r'Wavelength [$\AA$]',ylabel='Flux',title=f"full spectrum ({infile_suffix.strip('_')})")
    ymax = galaxy[lam > 3700].max()
    ymin = galaxy[lam > 3700].min()
    yrange = ymax-ymin
    ymin=0.0
    ax1.set(ylim=[ymin-0.05*yrange,ymax+0.05*yrange]) #, xlim=[6450, 6800])
    ax1.legend(fontsize=const.LEGEND_SCALE_FACTOR * const.TEXT_SIZE)
    
    
    ax2 = fig1.add_subplot(2,1,2)
    ax2.plot(lam, galaxy_sub_host, 'k')
    ax2.set(xlabel=r'Wavelength [$\AA$]',ylabel='Flux',title=f'host subtracted')
    # ax2.axvline(const.OIII_STRONG, linestyle="--")

    ymax = galaxy_sub_host[lam > 3700].max()
    ymin = galaxy_sub_host[lam > 3700].min()
    yrange = ymax-ymin
    ax2.set(ylim=[ymin-0.05*yrange,ymax+0.05*yrange])
    plt.show()



def compare_balmer_decrements( #TODO: overplot diff spectra (remove make_new_fig arg)
    results: list[list[dict[str, np.ndarray]]],
    num_gaussians_list: list[int],
    num_bins_list: list[int],
    year: int,
    colour_map: Colormap = const.COLOUR_MAP,
    ylim: tuple[int, int] = (0, 10),
    save_fig_name: str | None = "",
    **diff_kwargs: Any
) -> float | None:
    """
    Compares the balmer decrements for different numbers of Gaussians and velocity bins.
    NOTE: This function is not yet implemented. See GitHub history for older versions.

    Parameters
    ----------
    results: list[list[dict[str, np.ndarray]]]
        The results of the balmer decrement calculations. See
        :func:`integrate.get_bd_comparison_info` for more details.
    num_gaussians_list: list[int]
        The number of Gaussians used for each calculation (inclusive).
    num_bins_list: list[int]
        The number of bins used for each calculation (inclusive).
    year: int
        The year of the data.
    colour_map: Colormap
        The colour map to use for the plots.
    ylim: tuple[int, int]
        The y-axis bounds of the plots.
    save_fig_name: str | None
        The name of the figure to save.
    **diff_kwargs: Any
        Additional keyword arguments to pass to :func:`plot_diff_spectra`.
    """
    pass