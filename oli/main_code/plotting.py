import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from matplotlib.figure import Figure
from matplotlib.colors import Colormap
import numpy as np
import warnings
import os
from typing import Any

from . import constants as const
from .helpers import (
    get_lam_bounds, convert_lam_to_vel, convert_to_vel_data,
    get_flux_bounds, get_scaled_y_bounds, get_vel_lam_mask, convert_vel_to_lam
)

def my_savefig(save_fig_name: str | None, fig: plt.Figure | None = None) -> None:
    if const.SAVE_FIGS and save_fig_name is not None and save_fig_name != "":
        prefix_suffix = save_fig_name.split(".")
        if len(prefix_suffix) < 2:
            save_fig_name += ".pdf"
        elif len(prefix_suffix) > 2:
            raise ValueError("too many . in save_fig_name")
        elif prefix_suffix[1] != "pdf":
            save_fig_name = prefix_suffix[0] + ".pdf"
        while os.path.exists(const.FIG_OUTPUT_DIR + save_fig_name):
            warn_msg = f"{const.FIG_OUTPUT_DIR + save_fig_name} already exists. Creating copy"
            warnings.warn(warn_msg)
            save_fig_name = save_fig_name[:-4] + "_cpy.pdf"
        if fig is None:
            plt.savefig(const.FIG_OUTPUT_DIR + save_fig_name)
        else:
            fig.savefig(const.FIG_OUTPUT_DIR + save_fig_name)
            

def plot_vert_emission_lines(
    ions: dict[str, float] | None, 
    plot_x_bounds: tuple[float, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = 0.5,
    vlines_cmap: Colormap = const.COLOUR_MAP,
    is_rest_frame: bool = True,
    vel_centre_ang: float | None = None,
    ax: plt.Axes | None = None
) -> None:
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
        if is_rest_frame:
            obs_lam = lam * (1+const.Z_SPEC)
        else:
            obs_lam = lam
        if vel_centre_ang is None:
            x_val = obs_lam
        else:
            x_val = convert_lam_to_vel(obs_lam, lam_centre_rest_frame=vel_centre_ang)
        if plot_x_bounds is None or (plot_x_bounds[0] < obs_lam < plot_x_bounds[1]):
            ax.axvline(
                x_val, linestyle='--', lw=const.LINEWIDTH,
                color=vlines_cmap(i), label=name
            )


def plot_min_res(
    lam_sdss: np.ndarray,
    lam15_blue: np.ndarray,
    lam15_red: np.ndarray,
    res_min: np.ndarray,
    res_01: np.ndarray,
    res_21: np.ndarray,
    res_22: np.ndarray,
    plot_RES_15_RED: bool = False,
    save_fig_name: str | None = "min_res.pdf"
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
            np.max(res_01), const.RES_15_BLUE, const.RES_15_RED
        )]
    else:
        res_plot_bounds = [np.min(res_min), max(
            np.max(res_21), np.max(res_22), 
            np.max(res_01), const.RES_15_BLUE
        )]
    lam_plot_range = np.max(lam_sdss) - lam15_blue_min
    axhline_end = (lam15_blue_max - lam15_blue_min) / lam_plot_range

    plt.figure(figsize=const.FIG_SIZE)
    plt.axhline(
        const.RES_15_BLUE, # xmax=axhline_end,
        color='blue', linestyle='--', label="SAMI blue"
    )
    if plot_RES_15_RED:
        plt.axhline(const.RES_15_RED, color='red', linestyle='--', label="SAMI red")
    plt.plot(lam_sdss, res_21, alpha=0.5, label="2021")
    plt.plot(lam_sdss, res_22, alpha=0.5, label="2022")
    plt.plot(lam_sdss, res_01, alpha=0.5, label="SDSS Average")
    plt.plot(lam_sdss, res_min, color='black', alpha=0.5, lw=4, linestyle='--', label="Minimum")
    plt.fill_betweenx(res_plot_bounds, lam15_blue_min, lam15_blue_max, color='lightblue', alpha=0.5, label="SAMI blue coverage")
    plt.fill_betweenx(res_plot_bounds, lam15_red_min, lam15_red_max, color='red', alpha=0.2, label="SAMI red coverage")
    plt.xlabel(const.REST_ANG_LABEL)
    plt.ylabel("Resolving Power")
    if const.PLOT_TITLES:
        plt.title("Resolution of SDSS spectra")
    plt.legend(loc="upper left")
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
    x_axis_label: str = const.ANG_LABEL,
    error_opacity: float = const.ERR_OPAC,
    ions: dict[str, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    fill_between_bounds: tuple[float, float] | None = None,
    fill_between_label: str | None = None,
    fill_between_opacity: float = 0.5,
    legend_loc: str | None = "best",
    save_fig_name: str | None = ""
) -> None:
    sami_is_split = True if isinstance(flux15, tuple) else False

    plt.figure(figsize=const.FIG_SIZE)
    plt.plot(lam01, flux01, color='black', label='2001 (SDSS)', lw = const.LINEWIDTH)

    if sami_is_split:
        flux15_blue, flux15_red = flux15
        lam15_blue, lam15_red = lam15

        plt.plot(lam15_blue, flux15_blue, color='blue', label='2015 blue arm (SAMI)', lw = const.LINEWIDTH)
        plt.plot(lam15_red, flux15_red, color='red', label='2015 red arm (SAMI)', lw = const.LINEWIDTH)
    else:
        plt.plot(lam15, flux15, color='purple', label='2015 (SAMI)', lw = const.LINEWIDTH)

    plt.plot(lam21, flux21, color='orange', label='2021 (SDSS)', lw = const.LINEWIDTH)
    plt.plot(lam22, flux22, color='green', label='2022 (SDSS)', lw = const.LINEWIDTH)

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
    if title is not None and const.PLOT_TITLES:
        plt.title(title)
    if legend_loc is not None:
        plt.legend(loc=legend_loc)
    else:
        plt.legend()
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
    plt.figure(figsize=const.FIG_SIZE)
    
    if bin_by_med:
        plt.plot(binned_lambdas, binned_vals, color='black', label=f'ratio binned by median (width {bin_width} Å)', lw=2*const.LINEWIDTH)
        poly_label = f"polynomial fit (degree {degree}) to binned ratio"
    else:
        poly_label = f"polynomial fit (degree {degree})"

    if plot_selection:
        ratio_label = 'Spectral flux density ratio'
        plt.plot(lambdas, vals_removed, color='red', label=f'{ratio_label} (ignored Balmer)', lw = const.LINEWIDTH)
    else:
        ratio_label = 'actual ratio'
        plt.plot(binned_lambdas, polynom_vals, color='red', label=poly_label, lw=2*const.LINEWIDTH)

    plt.plot(lambdas, vals, alpha=0.4, color='black', label=ratio_label, lw = 0.5*const.LINEWIDTH)
    plt.xlabel(const.ANG_LABEL)
    plt.ylabel("Ratio")
    if const.PLOT_TITLES:
        plt.title(title)
    plt.legend()
    my_savefig(save_fig_name)
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
    save_fig_name: str | None = ""
)-> None:
    plt.figure(figsize=const.FIG_SIZE)
    plt.plot(lam, baseline_flux, color='black', label=f'{baseline_year}', lw = const.LINEWIDTH)
    plt.plot(lam, unadjusted_flux, color='orange', label=f'{year_to_adjust}', lw = const.LINEWIDTH)
    plt.plot(lam, adjusted_flux, color='red', label=f'{year_to_adjust} (polynomial fit to {baseline_year})', lw = const.LINEWIDTH)
    plt.xlabel(const.ANG_LABEL)
    plt.ylabel(const.SFD_Y_AX_LABEL)

    plot_vert_emission_lines(ions, lam_bounds)
    if const.PLOT_TITLES:
        plt.title(title)
    if flux_y_bounds is not None:
        plt.ylim(flux_y_bounds)
    elif ((
        np.nanmax(adjusted_flux) - np.nanmin(adjusted_flux)) >
        10 * (np.nanmax(unadjusted_flux) - np.nanmin(unadjusted_flux)
    )):
        plt.ylim((0, 1.2 * np.nanmax(unadjusted_flux)))
    plt.legend()
    my_savefig(save_fig_name)
    plt.show()

def _plot_diff_spectra_on_axis(
    *,
    ax: plt.Axes,
    lam: np.ndarray,
    diff_15: np.ndarray | None,
    diff_21: np.ndarray | None,
    diff_22: np.ndarray | None,
    diff_15_err: np.ndarray | None,
    diff_21_err: np.ndarray | None,
    diff_22_err: np.ndarray | None,
    ions: dict[str, float] | bool,
    vel_plot_width: float | None,
    hlines: dict[str, float] | None,
    fill_between_bounds: tuple[float, float] | None,
    fill_between_label: str | None,
    fill_between_opacity: float,
    plot_centres: float | list[float],
    plot_labels: list[str] | None,
    use_ang_x_axis: bool,
    plot_y_bounds: tuple[float, float] | bool,
    scale_axes: bool,
    y_top_scale_factor: float,
    error_opacity: float,
    colour_map: Colormap,
    apply_xlabel: bool = True,
    apply_ylabel: bool = True,
    apply_title: bool = True,
    apply_legend: bool = True,
    vel_units: str = "km/s"
) -> None:
    plt.sca(ax)

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

        x_axis_label = const.ANG_LABEL
        x_15, x_21, x_22 = [lam], [lam], [lam]
        diffs_15 = [diff_15] if diff_15 is not None else None
        diffs_21 = [diff_21] if diff_21 is not None else None
        diffs_22 = [diff_22] if diff_22 is not None else None
        diffs_15_err = [diff_15_err] if diff_15_err is not None else None
        diffs_21_err = [diff_21_err] if diff_21_err is not None else None
        diffs_22_err = [diff_22_err] if diff_22_err is not None else None
    else:
        if vel_plot_width is None:
            raise ValueError("use_ang_x_axis must be True if vel_plot_width is None")
        if vel_units == "km/s":
            x_axis_label = const.VEL_LABEL
        elif vel_units == "Mm/s":
            x_axis_label = const.MMS_VEL_LABEL
        else:
            raise NotImplementedError("vel_units must be km/s or Mm/s")
        x_15, diffs_15, diffs_15_err = convert_to_vel_data(
            lam, diff_15, diff_15_err, plot_centres, vel_plot_width, vel_units
        )
        x_21, diffs_21, diffs_21_err = convert_to_vel_data(
            lam, diff_21, diff_21_err, plot_centres, vel_plot_width, vel_units
        )
        x_22, diffs_22, diffs_22_err = convert_to_vel_data(
            lam, diff_22, diff_22_err, plot_centres, vel_plot_width, vel_units
        )

    all_diffs = []
    all_years = []
    if diffs_15 is not None:
        all_diffs.append(diffs_15)
        all_years.append(2015)
    if diffs_21 is not None:
        all_diffs.append(diffs_21)
        all_years.append(2021)
    if diffs_22 is not None:
        all_diffs.append(diffs_22)
        all_years.append(2022)

    if scale_axes:
        if not isinstance(plot_y_bounds, bool) or plot_y_bounds == True:
            raise ValueError("plot_y_bounds must be False if using scale_axes")
        if len(all_diffs) != 1:
            raise ValueError("One of diffs_15, diffs_21, diffs_22 must be not None if using scale_axes")
        if num_centres == 2:
            ax2 = ax.twinx()
            axes = [ax, ax2]
            y_bounds_1, y_bounds_2 = get_scaled_y_bounds(
                y1=all_diffs[0][0],
                y2=all_diffs[0][1],
                y_top_scale_factor=y_top_scale_factor
            )
            ax.set_ylim(y_bounds_1)
            ax2.set_ylim(y_bounds_2)
            if apply_xlabel:
                ax.set_xlabel(x_axis_label)
        else:
            raise ValueError("scale_axes is only supported for 2 centres")
    else:
        axes = [ax]
        if apply_xlabel:
            ax.set_xlabel(x_axis_label)

    for i in range(num_centres):
        ax_i = axes[i] if scale_axes else axes[0]
        if diffs_15 is not None:
            label_info = plot_labels[i]
            flux = diffs_15[i]
            flux_err = diffs_15_err[i] if diffs_15_err is not None else None
            colour_15 = colour_map(3*i) if num_centres > 1 else 'black'
            ax_i.plot(x_15[i], flux, alpha=0.7, color=colour_15, label=f'{label_info} 2015 - 2001', lw=const.LINEWIDTH)
            if flux_err is not None:
                ax_i.fill_between(x_15[i], flux - flux_err, flux + flux_err, color=colour_15, alpha=error_opacity)
        if diffs_21 is not None:
            label_info = plot_labels[i]
            flux = diffs_21[i]
            flux_err = diffs_21_err[i] if diffs_21_err is not None else None
            colour_21 = colour_map(3*i+1) if num_centres > 1 else 'red'
            ax_i.plot(x_21[i], flux, alpha=0.7, color='red', label=f'{label_info} 2021 - 2001', lw=const.LINEWIDTH)
            if flux_err is not None:
                ax_i.fill_between(x_21[i], flux - flux_err, flux + flux_err, color=colour_21, alpha=error_opacity)
        if diffs_22 is not None:
            label_info = plot_labels[i]
            flux = diffs_22[i]
            flux_err = diffs_22_err[i] if diffs_22_err is not None else None
            colour_22 = colour_map(3*i+2) if num_centres > 1 else 'blue'
            ax_i.plot(x_22[i], flux, alpha=0.7, color=colour_22, label=f'{label_info} 2022 - 2001', lw=const.LINEWIDTH)
            if flux_err is not None:
                ax_i.fill_between(x_22[i], flux - flux_err, flux + flux_err, color=colour_22, alpha=error_opacity)

    # preserve user-provided ions dict unless ions is a bool flag
    local_ions = ions
    if isinstance(ions, bool):
        if ions:
            ions_near_h_alpha = {r"H${\alpha}$": const.H_ALPHA, "S[II] (blue)": const.SII_BLUE, "S[II] (red)": const.SII_RED, "N[II] (strong)": const.NII_STRONG}
            ions_near_h_beta = {r"H${\beta}$": const.H_BETA, "O[III] (weak)": const.OIII_WEAK, "O[III] (strong)": const.OIII_STRONG}
            if isinstance(plot_centres, list):
                local_ions = {}
                if const.H_ALPHA in plot_centres:
                    local_ions = local_ions | ions_near_h_alpha
                if const.H_BETA in plot_centres:
                    local_ions = local_ions | ions_near_h_beta
            else:
                if plot_centres == const.H_ALPHA:
                    local_ions = ions_near_h_alpha
                elif plot_centres == const.H_BETA:
                    local_ions = ions_near_h_beta
                else:
                    local_ions = None
        else:
            local_ions = None

    if use_ang_x_axis:
        x_bounds = get_lam_bounds(plot_centres, vel_plot_width, width_is_vel=True) if vel_plot_width is not None else None
        vel_centre_ang = None
    else:
        vel_centre_ang = plot_centres[0] if isinstance(plot_centres, list) else plot_centres
        x_bounds = (-vel_plot_width / 2, vel_plot_width / 2) if vel_plot_width is not None else None

    plot_vert_emission_lines(
        local_ions, x_bounds, vel_centre_ang=vel_centre_ang,
        fill_between_bounds=fill_between_bounds,
        fill_between_label=fill_between_label,
        fill_between_opacity=fill_between_opacity
    )
    if local_ions is not None and not use_ang_x_axis and num_centres > 1:
        warn_msg = (f"Emission lines only plotted with respect to {plot_labels[0]}.")
        warnings.warn(warn_msg)

    if hlines is not None:
        for i, (name, val) in enumerate(hlines.items()):
            ax.axhline(val, color=colour_map(i), linestyle='--', lw=const.LINEWIDTH, label=name)

    if isinstance(plot_y_bounds, tuple):
        ax.set_ylim(plot_y_bounds[0], plot_y_bounds[1])
    elif plot_y_bounds:
        if not isinstance(plot_centres, list):
            if plot_centres == const.H_ALPHA:
                ax.set_ylim(-10, 30)
            elif plot_centres == const.H_BETA:
                ax.set_ylim(-7, 9)
        elif not scale_axes:
            ax.set_ylim(-10, 30)

    ax.axhline(0, color='black', linestyle='--', alpha=0.5, lw=const.LINEWIDTH)

    # Ensure each subplot uses its own x-range (centered on its own line)
    if use_ang_x_axis:
        ax.set_xlim(float(np.min(lam)), float(np.max(lam)))
    else:
        if vel_plot_width is not None:
            half = 0.5 * vel_plot_width
            ax.set_xlim(-half, half)

    title = "" if len(all_years) != 1 else f"{all_years[0]} "
    title += "Spectral flux density difference from 2001"
    if const.PLOT_TITLES and apply_title:
        ax.set_title(title)

    if apply_ylabel:
        ax.set_ylabel(const.SFD_Y_AX_LABEL)

    if apply_legend:
        ax.legend(loc="upper left")


def ai_plot_diff_spectra(
    lam: np.ndarray,
    # years_to_plot: list[int] = [2015, 2021, 2022],
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
    make_new_fig: bool = True,
    y_top_scale_factor: float = 1.25,
    error_opacity: float = const.ERR_OPAC,
    colour_map: Colormap = const.COLOUR_MAP,
    save_fig_name: str | None = "",
    side_by_side_plot_centres: list[float] | None = None,
    vel_units: str = "km/s"
) -> None:

    if side_by_side_plot_centres is not None:
        if not isinstance(side_by_side_plot_centres, list) or len(side_by_side_plot_centres) != 2:
            raise ValueError("side_by_side_plot_centres must be a list of exactly two centres")
        if plot_labels is not None and len(plot_labels) != 2:
            raise ValueError("plot_labels must be length 2 when using side_by_side_plot_centres")
        if plot_labels is None:
            plot_labels = ["", ""]

        fig, axes = plt.subplots(
            1, 2, sharey=True, figsize=const.FIG_SIZE,
            gridspec_kw={"wspace": 0.08}
        )

        for i, centre in enumerate(side_by_side_plot_centres):
            _plot_diff_spectra_on_axis(
                ax=axes[i],
                lam=lam,
                diff_15=diff_15,
                diff_21=diff_21,
                diff_22=diff_22,
                diff_15_err=diff_15_err,
                diff_21_err=diff_21_err,
                diff_22_err=diff_22_err,
                ions=ions,
                vel_plot_width=vel_plot_width,
                hlines=hlines,
                fill_between_bounds=fill_between_bounds,
                fill_between_label=fill_between_label,
                fill_between_opacity=fill_between_opacity,
                plot_centres=centre,
                plot_labels=[plot_labels[i]],
                use_ang_x_axis=use_ang_x_axis,
                plot_y_bounds=plot_y_bounds,
                scale_axes=scale_axes,
                y_top_scale_factor=y_top_scale_factor,
                error_opacity=error_opacity,
                colour_map=colour_map,
                apply_xlabel=True,
                apply_ylabel=(i == 0),
                apply_title=False,
                apply_legend=True,
                vel_units=vel_units
            )

        if use_ang_x_axis:
            x_axis_label = const.ANG_LABEL
        elif vel_units == "km/s":
            x_axis_label = const.VEL_LABEL
        elif vel_units == "Mm/s":
            x_axis_label = const.MMS_VEL_LABEL
        else:
            raise NotImplementedError("vel_units must be km/s or Mm/s")

        fig.supxlabel(x_axis_label)

        title = ""
        years = []
        if diff_15 is not None:
            years.append(2015)
        if diff_21 is not None:
            years.append(2021)
        if diff_22 is not None:
            years.append(2022)
        if len(years) == 1:
            title = f"{years[0]} "
        title += "Spectral flux density difference from 2001"
        if const.PLOT_TITLES:
            fig.suptitle(title)

        my_savefig(save_fig_name)
        plt.show()
        return

    if make_new_fig:
        plt.figure(figsize=const.FIG_SIZE)

    _plot_diff_spectra_on_axis(
        ax=plt.gca(),
        lam=lam,
        diff_15=diff_15,
        diff_21=diff_21,
        diff_22=diff_22,
        diff_15_err=diff_15_err,
        diff_21_err=diff_21_err,
        diff_22_err=diff_22_err,
        ions=ions,
        vel_plot_width=vel_plot_width,
        hlines=hlines,
        fill_between_bounds=fill_between_bounds,
        fill_between_label=fill_between_label,
        fill_between_opacity=fill_between_opacity,
        plot_centres=plot_centres,
        plot_labels=plot_labels,
        use_ang_x_axis=use_ang_x_axis,
        plot_y_bounds=plot_y_bounds,
        scale_axes=scale_axes,
        y_top_scale_factor=y_top_scale_factor,
        error_opacity=error_opacity,
        colour_map=colour_map,
        apply_xlabel=True,
        apply_ylabel=True,
        apply_title=True,
        apply_legend=True
    )

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
) -> tuple[str, str, str]:

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

        x_axis_label = const.ANG_LABEL
        x_15, x_21, x_22 = [lam], [lam], [lam]
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
        x_15, diffs_15, diffs_15_err = convert_to_vel_data(
            lam, diff_15, diff_15_err, plot_centres, vel_plot_width
        )
        x_21, diffs_21, diffs_21_err = convert_to_vel_data(
            lam, diff_21, diff_21_err, plot_centres, vel_plot_width
        )
        x_22, diffs_22, diffs_22_err = convert_to_vel_data(
            lam, diff_22, diff_22_err, plot_centres, vel_plot_width
        )


    all_diffs = []
    all_years = []
    if diffs_15 is not None:
        all_diffs.append(diffs_15)
        all_years.append(2015)
    if diffs_21 is not None:
        all_diffs.append(diffs_21)
        all_years.append(2021)
    if diffs_22 is not None:
        all_diffs.append(diffs_22)
        all_years.append(2022)
    if scale_axes: #TODO: finish setting up plot parameters
        if not isinstance(plot_y_bounds, bool) or plot_y_bounds == True:
            raise ValueError("plot_y_bounds must be False if using scale_axes")
        if len(all_diffs) != 1:
            raise ValueError("One of diffs_15, diffs_21, diffs_22 must be not None if using scale_axes")
        if num_centres == 2:
            ax1 = fig.add_subplot()
            # ax1 = fig.subplots()
            ax2 = ax1.twinx()
            axes = [ax1, ax2]
            y_bounds_1, y_bounds_2 = get_scaled_y_bounds(
                y1=all_diffs[0][0],
                y2=all_diffs[0][1],
                y_top_scale_factor=y_top_scale_factor
            )
            ax1.set_ylim(y_bounds_1)
            ax2.set_ylim(y_bounds_2)
            # ax1.set_xlabel(x_axis_label)

        else:
            raise ValueError("scale_axes is only supported for 2 centres")
    else:
        # plt.xlabel(x_axis_label)
        axes = [fig.add_subplot()]
    for i in range(num_centres):
        ax = axes[i] if scale_axes else axes[0]
        if n_ticks_x is not None:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=n_ticks_x))
        if diffs_15 is not None:
            label_info = plot_labels[i]
            flux = diffs_15[i]
            flux_err = diffs_15_err[i] if diffs_15_err is not None else None
            colour_15 = colour_map(3*i) if num_centres > 1 else 'black'
            ax.plot(x_15[i], flux, alpha=0.7, color=colour_15, label=f'{label_info} 2015 - 2001', lw = const.LINEWIDTH)
            
            if flux_err is not None:
                ax.fill_between(x_15[i], flux - flux_err, flux + flux_err, color=colour_15, alpha=error_opacity)
        if diffs_21 is not None:
            label_info = plot_labels[i]
            flux = diffs_21[i]
            flux_err = diffs_21_err[i] if diffs_21_err is not None else None
            colour_21 = colour_map(3*i+1) if num_centres > 1 else 'red'
            ax.plot(x_21[i], flux, alpha=0.7, color='red', label=f'{label_info} 2021 - 2001', lw = const.LINEWIDTH)
            
            if flux_err is not None:
                ax.fill_between(x_21[i], flux - flux_err, flux + flux_err, color=colour_21, alpha=error_opacity)
        if diffs_22 is not None:
            label_info = plot_labels[i]
            flux = diffs_22[i]
            flux_err = diffs_22_err[i] if diffs_22_err is not None else None
            colour_22 = colour_map(3*i+2) if num_centres > 1 else 'blue'
            ax.plot(x_22[i], flux, alpha=0.7, color=colour_22, label=f'{label_info} 2022 - 2001', lw = const.LINEWIDTH)
            
            if flux_err is not None:
                ax.fill_between(x_22[i], flux - flux_err, flux + flux_err, color=colour_22, alpha=error_opacity)

    if isinstance(ions, bool):
        if ions:
            ions_near_h_alpha = {r"H${\alpha}$": const.H_ALPHA, "S[II] (blue)": const.SII_BLUE, "S[II] (red)": const.SII_RED, "N[II] (strong)": const.NII_STRONG}
            ions_near_h_beta = {r"H${\beta}$": const.H_BETA, "O[III] (weak)": const.OIII_WEAK, "O[III] (strong)": const.OIII_STRONG}
            if isinstance(plot_centres, list):
                ions = {}
                if const.H_ALPHA in plot_centres:
                    ions = ions | ions_near_h_alpha
                if const.H_BETA in plot_centres:
                    ions = ions | ions_near_h_beta
            else:
                if plot_centres == const.H_ALPHA:
                    ions = ions_near_h_alpha
                elif plot_centres == const.H_BETA:
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
            f"Emission lines only plotted with respect to {plot_labels[0]}."
        )
        warnings.warn(warn_msg)

    if hlines is not None:
        for i, (name, val) in enumerate(hlines.items()):
            plt.axhline(val, color=colour_map(i), linestyle='--', lw=const.LINEWIDTH, label=name)

    if isinstance(plot_y_bounds, tuple):
        plt.ylim(plot_y_bounds[0], plot_y_bounds[1])
    elif plot_y_bounds:
        if not isinstance(plot_centres, list):
            if plot_centres == const.H_ALPHA:
                plt.ylim(-5, 30)
            elif plot_centres == const.H_BETA:
                plt.ylim(-6, 9)
        elif not scale_axes:
            plt.ylim(-5, 30)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5, lw=const.LINEWIDTH)
    title = "" if len(all_years) != 1 else f"{all_years[0]} "
    title += "Spectral flux density difference from 2001"
    for i, ax in enumerate(axes):
        y_axis_label = f"{plot_labels[i]}   {const.SFD_Y_AX_LABEL}" if scale_axes else const.SFD_Y_AX_LABEL
        # ax.set_ylabel(y_axis_label)
        loc = "upper left" if i == 0 else "upper right"
        # loc = "best" if i == 0 else "upper right"
        ax.legend(loc=loc)

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
        {r"H${\beta}$": const.H_BETA, "O[III] (strong)": const.OIII_STRONG},
        {r"H${\alpha}$": const.H_ALPHA}
    ],
    vel_plot_width: float | None = const.VEL_PLOT_WIDTH,
    hlines: dict[str, float] | None = None,
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
    num_plots_options = [
        len(plot_centres_list), len(plot_labels_list),
        len(ions_list), len(n_ticks_x_list), len(scale_axes_list),
        len(plot_y_bounds_list)
    ]
    if not np.all(np.array(num_plots_options) == num_plots_options[0]):
        raise ValueError(
            f"num_plots must be the same for all lists\n"
            f"plot_centres_list: {len(plot_centres_list)}\n"
            f"plot_labels_list: {len(plot_labels_list)}\n"
            f"ions_list: {len(ions_list)}\n"
            f"n_ticks_x_list: {len(n_ticks_x_list)}\n"
            f"scale_axes_list: {len(scale_axes_list)}\n"
            f"plot_y_bounds_list: {len(plot_y_bounds_list)}"
        )
    num_plots = num_plots_options[0]

    fig = plt.figure(figsize=figsize, layout='constrained')

    if scale_all_axes:
        if num_plots != 2:
            raise ValueError("scale_all_axes is only supported for 2 plots")
        if not np.all(np.array(scale_axes_list) == False):
            raise ValueError("scale_axes_list must be False if scale_all_axes is True")
        if not np.all(np.array(plot_y_bounds_list) == False):
            raise ValueError("plot_y_bounds_list must be False if scale_all_axes is True")
        for plot_centres in plot_centres_list:
            if isinstance(plot_centres, list):
                raise ValueError("plot_centres_list must be a list of single numbers if scale_all_axes is True")

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
            "15": diffs_15,
            "21": diffs_21,
            "22": diffs_22
        }

        all_diffs_not_none = {k: v for k, v in all_diffs.items() if v is not None}

        max_left_diffs_year = None
        max_left_diffs_val = None
        max_right_diffs_year = None
        max_right_diffs_val = None

        for year, diffs in all_diffs_not_none.items():
            left_diffs = diffs[0]
            right_diffs = diffs[1]
            if max_left_diffs_year is None or np.max(left_diffs) > max_left_diffs_val:
                max_left_diffs_year = year
                max_left_diffs_val = np.max(left_diffs)
            if max_right_diffs_year is None or np.max(right_diffs) > max_right_diffs_val:
                max_right_diffs_year = year
                max_right_diffs_val = np.max(right_diffs)
        
        left_diffs_to_scale = all_diffs_not_none[max_left_diffs_year][0]
        right_diffs_to_scale = all_diffs_not_none[max_right_diffs_year][1]

        # print(f"left_diffs_to_scale: {left_diffs_to_scale}")
        # print(f"right_diffs_to_scale: {right_diffs_to_scale}")

        y_bounds_1, y_bounds_2 = get_scaled_y_bounds(
            y1=left_diffs_to_scale,
            y2=right_diffs_to_scale,
            y_top_scale_factor=y_top_scale_factor
        )

        plot_y_bounds_list = [y_bounds_1, y_bounds_2]


    subfigs = fig.subfigures(1, num_plots) 
    if not isinstance(subfigs, np.ndarray):
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
            hlines=hlines,
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
        fig.suptitle(titles[0])
    fig.supxlabel(x_axis_labels[0])
    fig.supylabel(y_axis_labels[0])
    my_savefig(save_fig_name, fig=fig)
    fig.show()


def old_plot_diff_spectra(
    lam: np.ndarray,
    # years_to_plot: list[int] = [2015, 2021, 2022],
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
    make_new_fig: bool = True,
    y_top_scale_factor: float = 1.25,
    error_opacity: float = const.ERR_OPAC,
    colour_map: Colormap = const.COLOUR_MAP,
    save_fig_name: str | None = ""
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

        x_axis_label = const.ANG_LABEL
        x_15, x_21, x_22 = [lam], [lam], [lam]
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
        x_15, diffs_15, diffs_15_err = convert_to_vel_data(
            lam, diff_15, diff_15_err, plot_centres, vel_plot_width
        )
        x_21, diffs_21, diffs_21_err = convert_to_vel_data(
            lam, diff_21, diff_21_err, plot_centres, vel_plot_width
        )
        x_22, diffs_22, diffs_22_err = convert_to_vel_data(
            lam, diff_22, diff_22_err, plot_centres, vel_plot_width
        )


    all_diffs = []
    all_years = []
    if diffs_15 is not None:
        all_diffs.append(diffs_15)
        all_years.append(2015)
    if diffs_21 is not None:
        all_diffs.append(diffs_21)
        all_years.append(2021)
    if diffs_22 is not None:
        all_diffs.append(diffs_22)
        all_years.append(2022)
    if scale_axes: #TODO: finish setting up plot parameters
        if not isinstance(plot_y_bounds, bool) or plot_y_bounds == True:
            raise ValueError("plot_y_bounds must be False if using scale_axes")
        if len(all_diffs) != 1:
            raise ValueError("One of diffs_15, diffs_21, diffs_22 must be not None if using scale_axes")
        if num_centres == 2:
            fig, ax1 = plt.subplots(figsize=const.FIG_SIZE)
            ax2 = ax1.twinx()
            axes = [ax1, ax2]
            y_bounds_1, y_bounds_2 =  get_scaled_y_bounds(
                y1=all_diffs[0][0],
                y2=all_diffs[0][1],
                y_top_scale_factor=y_top_scale_factor
            )
            ax1.set_ylim(y_bounds_1)
            ax2.set_ylim(y_bounds_2)
            ax1.set_xlabel(x_axis_label)

        else:
            raise ValueError("scale_axes is only supported for 2 centres")
    else:
        if make_new_fig:
            plt.figure(figsize=const.FIG_SIZE)
        plt.xlabel(x_axis_label)
        axes = [plt.gca()]
    for i in range(num_centres):
        ax = axes[i] if scale_axes else axes[0]
        if diffs_15 is not None:
            label_info = plot_labels[i]
            flux = diffs_15[i]
            flux_err = diffs_15_err[i] if diffs_15_err is not None else None
            colour_15 = colour_map(3*i) if num_centres > 1 else 'black'
            ax.plot(x_15[i], flux, alpha=0.7, color=colour_15, label=f'{label_info} 2015 - 2001', lw = const.LINEWIDTH)
            
            if flux_err is not None:
                ax.fill_between(x_15[i], flux - flux_err, flux + flux_err, color=colour_15, alpha=error_opacity)
        if diffs_21 is not None:
            label_info = plot_labels[i]
            flux = diffs_21[i]
            flux_err = diffs_21_err[i] if diffs_21_err is not None else None
            colour_21 = colour_map(3*i+1) if num_centres > 1 else 'red'
            ax.plot(x_21[i], flux, alpha=0.7, color='red', label=f'{label_info} 2021 - 2001', lw = const.LINEWIDTH)
            
            if flux_err is not None:
                ax.fill_between(x_21[i], flux - flux_err, flux + flux_err, color=colour_21, alpha=error_opacity)
        if diffs_22 is not None:
            label_info = plot_labels[i]
            flux = diffs_22[i]
            flux_err = diffs_22_err[i] if diffs_22_err is not None else None
            colour_22 = colour_map(3*i+2) if num_centres > 1 else 'blue'
            ax.plot(x_22[i], flux, alpha=0.7, color=colour_22, label=f'{label_info} 2022 - 2001', lw = const.LINEWIDTH)
            
            if flux_err is not None:
                ax.fill_between(x_22[i], flux - flux_err, flux + flux_err, color=colour_22, alpha=error_opacity)

    if isinstance(ions, bool):
        if ions:
            ions_near_h_alpha = {r"H${\alpha}$": const.H_ALPHA, "S[II] (blue)": const.SII_BLUE, "S[II] (red)": const.SII_RED, "N[II] (strong)": const.NII_STRONG}
            ions_near_h_beta = {r"H${\beta}$": const.H_BETA, "O[III] (weak)": const.OIII_WEAK, "O[III] (strong)": const.OIII_STRONG}
            if isinstance(plot_centres, list):
                ions = {}
                if const.H_ALPHA in plot_centres:
                    ions = ions | ions_near_h_alpha
                if const.H_BETA in plot_centres:
                    ions = ions | ions_near_h_beta
            else:
                if plot_centres == const.H_ALPHA:
                    ions = ions_near_h_alpha
                elif plot_centres == const.H_BETA:
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
            f"Emission lines only plotted with respect to {plot_labels[0]}."
        )
        warnings.warn(warn_msg)

    if hlines is not None:
        for i, (name, val) in enumerate(hlines.items()):
            plt.axhline(val, color=colour_map(i), linestyle='--', lw=const.LINEWIDTH, label=name)

    if isinstance(plot_y_bounds, tuple):
        plt.ylim(plot_y_bounds[0], plot_y_bounds[1])
    elif plot_y_bounds:
        if not isinstance(plot_centres, list):
            if plot_centres == const.H_ALPHA:
                plt.ylim(-10, 30)
            elif plot_centres == const.H_BETA:
                plt.ylim(-7, 9)
        elif not scale_axes:
            plt.ylim(-10, 30)

    plt.axhline(0, color='black', linestyle='--', alpha=0.5, lw=const.LINEWIDTH)
    title = "" if len(all_years) != 1 else f"{all_years[0]} "
    title += "Spectral flux density difference from 2001"
    if const.PLOT_TITLES:
        plt.title(title)
    for i, ax in enumerate(axes):
        label = f"{plot_labels[i]}   {const.SFD_Y_AX_LABEL}" if scale_axes else const.SFD_Y_AX_LABEL
        ax.set_ylabel(label)
        loc = "upper left" if i == 0 else "upper right"
        # loc = "best" if i == 0 else "upper right"
        ax.legend(loc=loc)
    my_savefig(save_fig_name)
    plt.show()

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
    if mask_vel_width is not None:
        if mask_lam_centre is None:
            raise ValueError("mask_lam_centre must be provided if mask_vel_width is provided")
        mask = get_vel_lam_mask(x, mask_vel_width, mask_lam_centre)
        x = x[mask]
        y_data = y_data[mask]
        y_data_errs = y_data_errs[mask] if y_data_errs is not None else None

    plt.figure(figsize=const.FIG_SIZE)
    plt.plot(x, y_data, color='black', label='Data', lw = const.LINEWIDTH)
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
            0.05, 0.95, label,
            transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top'
        )
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title is not None and const.PLOT_TITLES:
        plt.title(title)
    plt.legend(loc="upper right")
    my_savefig(save_fig_name)
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
    pass

# def compare_balmer_decrements(
#     results: list[list[dict[str, np.ndarray]]],
#     num_gaussians_list: list[int],
#     num_bins_list: list[int],
#     year: int,
#     colour_map: Colormap = const.COLOUR_MAP,
#     ylim: tuple[int, int] = (0, 10),
#     save_fig_name: str | None = "",
#     **diff_kwargs: Any
# ) -> float | None:
#     bd_mean = None
#     for i, num_bins in enumerate(num_bins_list):
#         if diff_kwargs == {}:
#             plt.figure(figsize=const.FIG_SIZE)
#             bd_ax = plt.gca()
#         else:
#             fig, bd_ax = plt.subplots(figsize=const.FIG_SIZE)
#             diff_ax = bd_ax.twinx()
#         if num_bins == 1:
#             one_bin_results = [result[i] for result in results]
#             one_bin_bds = [result["bd"] for result in one_bin_results]
#             bd_ax.errorbar(
#                 x=num_gaussians_list,
#                 y=one_bin_bds,
#                 yerr=[result["bd_err"] for result in one_bin_results],
#                 # make data points not connected by lines
#                 linestyle="None",
#                 fmt='o', capsize=4
#             )
#             bd_mean = np.mean(one_bin_bds)
#             bd_ax.axhline(
#                 bd_mean, color="purple",
#                 lw=2*const.LINEWIDTH, label=f"Mean Balmer Decrement ({bd_mean:.2f})"
#             )
#             bd_ax.set_xlabel("Number of Gaussians")
#             bd_ax.set_ylabel("Balmer Decrement")
#             bd_ax.set_ylim(ylim)
#             if const.PLOT_TITLES:
#                 plt.title(f"{year} Balmer Decrement vs Number of Gaussians (no binning)")
#             bd_ax.legend(loc="upper left")
#             if diff_kwargs == {}:
#                 my_savefig(save_fig_name)
#                 plt.show()
#             elif ("make_new_fig", False) in diff_kwargs.items():
#                 plt.sca(diff_ax)
#                 plot_diff_spectra(**diff_kwargs)
#             else:
#                 print(diff_kwargs.items())
#                 raise ValueError("make_new_fig must be set to false if plotting diff spectra")
#             continue
        
#         for j, num_gaussians in enumerate(num_gaussians_list):
#             plt.errorbar(
#                 x=results[j][i]["vel_centres"],
#                 y=results[j][i]["bd"],
#                 yerr=results[j][i]["bd_err"],
#                 fmt='o', capsize=4,
#                 label=f"{num_gaussians} gaussians",
#                 color=colour_map(j)
#             )
#         plt.xlabel(const.VEL_LABEL)
#         plt.ylabel("Balmer Decrement")
#         plt.ylim(ylim)
#         if const.PLOT_TITLES:
#             plt.title(f"{year} Balmer Decrement vs Velocity ({num_bins} bins)")
#         plt.legend(loc="upper left")
#         if diff_kwargs == {}:
#             my_savefig(save_fig_name)
#             plt.show()
#         elif ("make_new_fig", False) in diff_kwargs.items():
#             plt.sca(diff_ax)
#             plot_diff_spectra(**diff_kwargs)
#         else:
#             print(diff_kwargs.items())
#             raise ValueError("make_new_fig must be set to false if plotting diff spectra")
#     return bd_mean




# def compare_balmer_decrements_older(
#     balmer_decrements_0_bins: np.ndarray,
#     balmer_decrements_many_bins: np.ndarray,
#     vel_bin_centres_all: np.ndarray,
#     year: int,
#     num_bins_bounds: tuple[int, int] = (1, 7),
#     num_gaussians_bounds: tuple[int, int] = (0, 5),
#     colour_map: Colormap = const.COLOUR_MAP,
#     save_fig_name: str | None = ""
# ) -> None:

#     num_bins_range = range(num_bins_bounds[0], num_bins_bounds[1] + 1)
#     num_gaussians_range = range(num_gaussians_bounds[0], num_gaussians_bounds[1] + 1)
#     print(f"num_bins_range: {num_bins_range}")
#     print(f"num_gaussians_range: {num_gaussians_range}")

#     plt.plot(num_gaussians_range, balmer_decrements_0_bins, color='black', label='0 bins', lw = const.LINEWIDTH)
#     plt.xlabel("Number of Gaussians")
#     plt.ylabel("Balmer Decrement")
#     if const.PLOT_TITLES:
#         plt.title(f"{year} Balmer Decrement vs Number of Gaussians")
#     my_savefig(save_fig_name)
#     plt.show()

#     # for > 0 num_bins, plot the balmer decrement on the y axis vs the velocity on the x axis, 
#     # with multiple dashed lines for each number of gaussians
    
#     for num_bins in num_bins_range:
#         for num_gaussians in num_gaussians_range:
#             plt.plot(
#                 vel_bin_centres_all[num_gaussians],
#                 balmer_decrements_many_bins[num_gaussians],
#                 color=colour_map(num_gaussians),
#                 label=f'{num_gaussians} gaussians',
#                 linestyle='None',
#                 lw = const.LINEWIDTH
#             )
#         plt.xlabel(const.VEL_LABEL)
#         plt.ylabel("Balmer Decrement")
#         if const.PLOT_TITLES:
#             plt.title(f"{year} Balmer Decrement vs Velocity ({num_bins} bins)")
#         plt.legend()
#         my_savefig(save_fig_name)
#         plt.show()