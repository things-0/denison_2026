def adjust_calibration(
    year_to_change: str = "2022",
    calibration_year: str = "2015",
    lambdas_to_ignore_width: int = 200,
    plot_ratio_selection: bool = True, poly_degree: float = 6,
    bin_by_med: bool = True, bin_width: float = 50,
    plot_poly_ratio: bool = True, plot_adjusted: bool = True,
    adjusted_plot_lam_bounds: tuple[float] | None = None,
    adjusted_plot_flux_bounds: tuple[float] | None = None,
    vlines: dict[str, float] | None = None,
    blur_before_resampling: bool = True,
) -> tuple[np.poly1d, np.ndarray]:
    possible_years = ["2001", "2015", "2021", "2022"]
    if year_to_change not in possible_years or calibration_year not in possible_years:
        raise ValueError(f"year should be in {possible_years}")

    lam, (data01, data15, data21, data22) = get_data(
        plot_resampled_and_blurred=False, 
        blur_before_resampling=blur_before_resampling
    )
    flux01, var01 = data01
    flux15, var15 = data15
    flux21, var21 = data21
    flux22, var22 = data22

    flux_map = {
        "2022": flux22,
        "2021": flux21,
        "2015": flux15,
        "2001": flux01
    }

    flux = flux_map.get(year_to_change)
    cal_flux = flux_map.get(calibration_year)

    non_constant_mult = flux / cal_flux
    balmer_mask = np.zeros(lam.shape, dtype=bool)

    lambdas_to_ignore = [
        get_lam_bounds(H_ALPHA, lambdas_to_ignore_width),
        get_lam_bounds(H_BETA, lambdas_to_ignore_width)
    ]

    for start, end in lambdas_to_ignore:
        current_range_mask = (lam >= start) & (lam <= end)
        balmer_mask = balmer_mask | current_range_mask

    removed = np.copy(non_constant_mult)
    removed[~balmer_mask] = np.nan

    non_constant_mult[balmer_mask] = np.nan

    if plot_ratio_selection:
        plt.figure(figsize=(12,5))
        plt.plot(lam, non_constant_mult, color='black', label=f'{year_to_change} to {calibration_year}', lw = LINEWIDTH)
        plt.plot(lam, removed, color='red', label=f'{year_to_change} to {calibration_year} (ignored Balmer)', lw = LINEWIDTH)
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Ratio")
        plt.title(f"Flux ratio of {year_to_change} to {calibration_year}")
        plt.legend()
        plt.show()
    
    polynom, _ = get_polynom_fit(
        lambdas=lam, vals=non_constant_mult,
        degree=poly_degree, bin_by_med=bin_by_med,
        bin_width=bin_width, plot_result=plot_poly_ratio,
        title = f"Spectrum ratio of {year_to_change} to {calibration_year}"
    )

    adjusted_flux = flux / polynom(lam)

    polynom, _ = get_polynom_fit(
        lambdas=lam, vals=non_constant_mult,
        degree=poly_degree, bin_by_med=bin_by_med,
        bin_width=bin_width, plot_result=plot_poly_ratio,
        title = f"Spectrum ratio of {year_to_change} to {calibration_year}"
    )

    adjusted_flux = flux / polynom(lam)

    if plot_adjusted:
        plt.figure(figsize=(12,5))
        plt.plot(lam, cal_flux, color='black', label=calibration_year, lw = LINEWIDTH)
        plt.plot(lam, flux, color='orange', label=year_to_change, lw = LINEWIDTH)
        plt.plot(lam, adjusted_flux, color='red', label=f'{year_to_change} (polynomial fit to {calibration_year})', lw = LINEWIDTH)
        if adjusted_plot_lam_bounds is not None:
            plt.xlim(adjusted_plot_lam_bounds)
        if adjusted_plot_flux_bounds is not None:
            plt.ylim(adjusted_plot_flux_bounds)
        elif (np.nanmax(adjusted_flux) - np.nanmin(adjusted_flux)) > 10 * (np.nanmax(flux) - np.nanmin(flux)):
            plt.ylim((0, 1.2 * np.nanmax(flux)))
        plt.xlabel("Wavelength (Å)")
        plt.ylabel("Flux (10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)")
        if blur_before_resampling:
            title = f"Spectra {calibration_year}, {year_to_change} (blurred then resampled to {calibration_year} grid and with polynomial fitting)"
        else:
            title = f"Spectra {calibration_year}, {year_to_change} (resampled to {calibration_year} grid then blurred and with polynomial fitting)"
        plt.title(title)
        if vlines is not None:
            cmap = plt.cm.tab10  # or 'Set1', 'Dark2', etc.
            for i, (name, emission_lam) in enumerate(vlines.items()):
                if adjusted_plot_lam_bounds is None or (adjusted_plot_lam_bounds[0] < emission_lam < adjusted_plot_lam_bounds[1]):
                    plt.axvline(
                        emission_lam, linestyle='--', lw=LINEWIDTH,
                        color=cmap(i), label=name
                    )
            # for name, emission_lam in vlines.items():
            #     plt.axvline(emission_lam, linestyle='--', lw=LINEWIDTH, label=name)
        plt.legend()
        plt.show()
    
    return polynom, adjusted_flux, adjusted_flux_err