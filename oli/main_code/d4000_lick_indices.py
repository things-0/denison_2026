import numpy as np
import scipy as sp

from .constants import *

#Conversion from FWHM to Gaussian sigma
FWHM_sigma=1/SIGMA_TO_FWHM

##A Gaussian function
#x=independenat variable
#a=amplitude
#b=phase shift (left or right)
#c=sigma
#d=zero-point offset (up or down)
def gaussian(x,a,b,c,d):
    return a*np.exp(-(x-b)**2/(2*c**2))+d

## Convert fluxes per Angstrom to fluxes per pixel
#wave=Wavelength vector
#spec=spectrum in units of flux per angstrom
def flux_per_pix(wave,spec):
    d_lam=[]
    for i in range(1,len(wave)):
        d_lam.append(wave[i]-wave[i-1])
    d_lam.append(d_lam[-1])
    d_lam=np.array(d_lam)
    return spec*d_lam

#resolution at which to calculate indices. 
#These values should be sigmas in Angstroms
lick_res={'D4000': 11.0/FWHM_sigma,   #This may not be the right resolution. Was 16 A FWHM in Balogh et al. 1999, but smoothing may not have been applied in subsequent papers. Ask Nic Scott!
          'Hdelta_A': 10.0/FWHM_sigma,
          'Ca4227': 9.0/FWHM_sigma,
          'G4300': 9.5/FWHM_sigma,
          'Fe4383': 9.0/FWHM_sigma,
          'Fe5270': 8.4/FWHM_sigma,
          'Hbeta':8.4/FWHM_sigma,
          'Mg_b': 8.4/FWHM_sigma,
          'CN_1': 10.0/FWHM_sigma,
          'CN_2': 10.0/FWHM_sigma        
}

#Index passbands in order of increasing wavelength
lick_window={'D4000_1':[3850.0,3950.0],  #Index invented by Balogh et al. 1999
             'D4000_2':[4000.0,4100.0],  #Index invented by Balogh et al. 1999
             'Mg_b_1':[4041.60,4079.75],  #Worthey & Ottaviani 1997
             'Mg_b_2':[4083.50,4122.25],
             'Mg_b_3':[4128.50,4161.00],
             'Ca4227_1':[4211.000,4219.750], #Lick system
             'Ca4227_2':[4222.250,4234.750],
             'Ca4227_3':[4241.000,4251.000],
             'G4300_1':[4266.375,4282.625], #Lick system
             'G4300_2':[4281.375,4316.375],
             'G4300_3':[4318.875,4335.125],
             'Fe4383_1':[4359.125,4370.375],
             'Fe4383_2':[4369.125,4420.375],
             'Fe4383_3':[4442.875,4455.375], # maybe this
             'Fe5270_1':[5233.150,5248.150],
             'Fe5270_2':[5245.650,5285.650],
             'Fe5270_3':[5285.650,5318.150],
             'Hbeta_1':[4827.875,4847.875],
             'Hbeta_2':[4847.875,4876.625],
             'Hbeta_3':[4876.625,4891.625],
             'Mg_b_1':[5142.625,5161.375], # this
             'Mg_b_2':[5160.125,5192.625],
             'Mg_b_3':[5191.375,5206.375],
             'CN_1_1':[4080.125,4117.625],
             'CN_1_2':[4142.125,4177.125],
             'CN_1_3':[4244.125,4284.125],
             'CN_2_1':[4083.875,4096.375],
             'CN_2_2':[4142.125,4177.125],
             'CN_2_3':[4244.125,4284.125]
}

## This function will blur the spectrum to the appropriate resolution.
#wavelength=wavelength vector
#spec=spectrum
#res_in=Native resolution in Angstroms. Sigma, not FWHM.
#res_out=Desired resolution of the spectrum
#Disp_cor=dispersion correction for high velocity dispersion spectra. Default=0
def blur_spectrum(wavelength,spec,res_in,res_out,disp_cor=0):
    res_ker=np.sqrt(res_out**2 - res_in**2 - disp_cor**2)
    kernel=(1.0/np.sqrt(2*np.pi*res_ker**2))*gaussian(wavelength,1.0,np.median(wavelength),res_ker,0)
    spec_smooth=sp.signal.convolve(np.nan_to_num(spec),kernel,mode='same')
    transfer=np.ones(spec.shape)
    badpix=np.where(np.isnan(spec)==True)
    transfer[badpix]=0
    transfer_smooth=sp.signal.convolve(transfer,kernel,mode='same')
    return spec_smooth/transfer_smooth

def flam_fnu(wave,flam):
    return (flam*wave**2)/(C_ANG_S)

##D4000 index
#wave=wavelength scale
#spec=input spectrum
#z=redshift of the spectrum
#res=native resolution of the spectrum
def D4000_w_errs(wave,spec,var,z,res=None):
    '''
    A calculation of Dn4000 that gives an error estimate too 
    '''
    ind_res=lick_res['D4000']
    if res!=None:
        spec_smooth=blur_spectrum(wave,spec,res,ind_res)
    else:
        spec_smooth=spec
    spec_smooth=flam_fnu(wave,spec_smooth)
    spec_smooth=flux_per_pix(wave,spec_smooth)
    # Adam's version
    #ind1=array_if(spec_smooth,(wave>=lick_window['D4000_1'][0]*(1+z)) & (wave<=lick_window['D4000_1'][1]*(1+z)))
    #ind2=array_if(spec_smooth,(wave>=lick_window['D4000_2'][0]*(1+z)) & (wave<=lick_window['D4000_2'][1]*(1+z)))
    #d1=np.nanmean(ind1)
    #d2=np.nanmean(ind2)
    # should do the same, but getting indices:
    idx1 = np.where((wave>=lick_window['D4000_1'][0]*(1+z)) & (wave<=lick_window['D4000_1'][1]*(1+z)))
    idx2 = np.where((wave>=lick_window['D4000_2'][0]*(1+z)) & (wave<=lick_window['D4000_2'][1]*(1+z)))
    d1 = np.nanmean(spec_smooth[idx1])
    d2 = np.nanmean(spec_smooth[idx2])
    
    d4000=d2/d1
    #Compute errors. Need variance spectrum first...
    ##
    var=var*((wave**2)/(C_ANG_S))**2
    var=flux_per_pix(wave,var) #conversion from wavelength to frequency
    var=flux_per_pix(wave,var) #I have to do this twice becasue you multiply variance by square of number
    #var_ind1=array_if(var,(wave>=lick_window['D4000_1'][0]*(1+z)) & (wave<=lick_window['D4000_1'][1]*(1+z)))
    #var_ind2=array_if(var,(wave>=lick_window['D4000_2'][0]*(1+z)) & (wave<=lick_window['D4000_2'][1]*(1+z)))
    #[var_ind1]=reject_invalid([var_ind1])
    #[var_ind2]=reject_invalid([var_ind2])
    #v1=np.sum(var_ind1)/(len(var_ind1)**2)
    #v2=np.sum(var_ind2)/(len(var_ind2)**2)
    
    vidx1=np.where((wave>=lick_window['D4000_1'][0]*(1+z)) & (wave<=lick_window['D4000_1'][1]*(1+z)))
    vidx2=np.where((wave>=lick_window['D4000_2'][0]*(1+z)) & (wave<=lick_window['D4000_2'][1]*(1+z)))

    #[var_ind1]=reject_invalid([var_ind1])
    #[var_ind2]=reject_invalid([var_ind2])
    # need to check th scaling and nan handling...
    v1=np.nansum(var[vidx1])/(len(var[vidx1])**2)
    v2=np.nansum(var[vidx2])/(len(var[vidx2])**2)
    s1=np.sqrt(v1)
    s2=np.sqrt(v2)
    d4000_err=d4000*np.sqrt((s1/d1)**2 + (s2/d2)**2)
    return d4000,d4000_err

#idx_name=name of index - e.g. Mg_b or Hdelta_A. must have 3 indices in lick_window
#wave=wavelength scale
#spec=input spectrum
#z=redshift of the spectrum
#res=resolution in Angstroms. Sigma, not FWHM.
def idx_w_errs(
    idx_name: str,
    wave: np.ndarray,
    spec: np.ndarray,
    var: np.ndarray,
    z: float,
    res: float | None = None
) -> tuple[float, float]:
    ind_res=lick_res[idx_name]
    if res!=None:
        spec_smooth=blur_spectrum(wave,spec,res,ind_res)
    else:
        spec_smooth=spec
    spec_smooth=flux_per_pix(wave,spec_smooth)
    #ind1=array_if(spec_smooth,(wave>=lick_window['Mg_b_1'][0]*(1+z)) & (wave<=lick_window['Mg_b_1'][1]*(1+z)))
    #ind2=array_if(spec_smooth,(wave>=lick_window['Mg_b_2'][0]*(1+z)) & (wave<=lick_window['Mg_b_2'][1]*(1+z)))
    #ind3=array_if(spec_smooth,(wave>=lick_window['Mg_b_3'][0]*(1+z)) & (wave<=lick_window['Mg_b_3'][1]*(1+z)))
    #hda1=np.nanmean(ind1)
    #hda2=np.nanmean(ind2)
    #hda3=np.nanmean(ind3)
    idx1=np.where((wave>=lick_window[idx_name + '_1'][0]*(1+z)) & (wave<=lick_window[idx_name + '_1'][1]*(1+z)))
    idx2=np.where((wave>=lick_window[idx_name + '_2'][0]*(1+z)) & (wave<=lick_window[idx_name + '_2'][1]*(1+z)))
    idx3=np.where((wave>=lick_window[idx_name + '_3'][0]*(1+z)) & (wave<=lick_window[idx_name + '_3'][1]*(1+z)))
    named_lick1=np.nanmean(spec_smooth[idx1])
    named_lick2=np.nanmean(spec_smooth[idx2])
    named_lick3=np.nanmean(spec_smooth[idx3])
    named_lick_sideband=np.nanmean([named_lick1,named_lick3])
    named_index=named_lick2
    lam1=lick_window[idx_name + '_2'][0]*(1+z)
    lam2=lick_window[idx_name + '_2'][1]*(1+z)
    named_lick=(lam2-lam1)*(1-named_index/named_lick_sideband)
    ##Now calculate the error
    var=flux_per_pix(wave,var)
    var=flux_per_pix(wave,var) #I have to do this twice becasue you multiply variance by square of number
    #var_ind1=array_if(var,(wave>=lick_window['Mg_b_1'][0]*(1+z)) & (wave<=lick_window['Mg_b_1'][1]*(1+z)))
    #var_ind2=array_if(var,(wave>=lick_window['Mg_b_2'][0]*(1+z)) & (wave<=lick_window['Mg_b_2'][1]*(1+z)))
    #var_ind3=array_if(var,(wave>=lick_window['Mg_b_3'][0]*(1+z)) & (wave<=lick_window['Mg_b_3'][1]*(1+z)))
    #[var_ind1]=reject_invalid([var_ind1])
    #[var_ind2]=reject_invalid([var_ind2])
    #[var_ind3]=reject_invalid([var_ind3])
    #v1=np.sum(var_ind1)/(len(var_ind1)**2)
    #v2=np.sum(var_ind2)/(len(var_ind2)**2)
    #v3=np.sum(var_ind3)/(len(var_ind3)**2)
    vidx1=np.where((wave>=lick_window[idx_name + '_1'][0]*(1+z)) & (wave<=lick_window[idx_name + '_1'][1]*(1+z)))
    vidx2=np.where((wave>=lick_window[idx_name + '_2'][0]*(1+z)) & (wave<=lick_window[idx_name + '_2'][1]*(1+z)))
    vidx3=np.where((wave>=lick_window[idx_name + '_3'][0]*(1+z)) & (wave<=lick_window[idx_name + '_3'][1]*(1+z)))
    #[var_ind1]=reject_invalid([var_ind1])
    #[var_ind2]=reject_invalid([var_ind2])
    #[var_ind3]=reject_invalid([var_ind3])
    v1=np.nansum(var[vidx1])/(len(var[idx1])**2)
    v2=np.nansum(var[vidx2])/(len(var[idx2])**2)
    v3=np.nansum(var[vidx3])/(len(var[idx3])**2)
    v_sb=(v1 + v3) / 4.0
    s_sb=np.sqrt(v_sb)
    s_i=np.sqrt(v2)
    named_lick_err=np.abs(named_lick)*(lam2-lam1)*np.sqrt((s_sb/named_lick_sideband)**2 + (s_i/named_index)**2)
    return named_lick, named_lick_err

def get_results(
    lambdas: list[np.ndarray],
    fluxes: list[np.ndarray],
    variances: list[np.ndarray],
    years: list[str] = ["2001", "2015", "2021", "2022"],
    idx_names: list[str] = ["D4000", "Hbeta", "Mg_b", "Fe4383"],
    z: float = Z_SPEC,
    print_results: bool = True
) -> ...:

    results = np.zeros((len(years), len(idx_names)), dtype=object)

    for i, year in enumerate(years):
        for j, idx_name in enumerate(idx_names):
            if idx_name == "D4000":
                res, res_err = D4000_w_errs(lambdas[i], fluxes[i], variances[i], z)
                str_repr = f"D4000 for {year}: {res:.3f} ± {res_err:.3f}"
                results[i, j] = {"value": res, "error": res_err, "str_repr": str_repr}
            else:
                res, res_err = idx_w_errs(idx_name, lambdas[i], fluxes[i], variances[i], z)
                str_repr = f"{idx_name} for {year}: {res:.3f} ± {res_err:.3f}"
                results[i, j] = {"value": res, "error": res_err, "str_repr": str_repr}

    if print_results:
        for j, idx_name in enumerate(idx_names):
            for i, year in enumerate(years):
                print(results[i, j]["str_repr"])
            print()

    return results

    # blue_lambdas = [lam01, lam15_blue, lam21, lam22]
    # blue_fluxes = [flux01, flux15_blue, flux21, flux22]
    # blue_variances = [var01, var15_blue, var21, var22]
    # # blue_fluxes = [flux01, flux15_blue_shifted, flux21_shifted, flux22_shifted]
    # # blue_variances = [var01, var15_blue_shifted, var21_shifted, var22_shifted]

    # #TODO: make dict and plot instead of print
    # idx_names = ["Hbeta", "Mg_b", "Fe4383"]

    # for i, year in enumerate([2001, 2015, 2021, 2022]):
    #     D4000_idx, D4000_idx_err = D4000_w_errs(blue_lambdas[i], blue_fluxes[i], blue_variances[i], z)
    #     print(f"\nD4000 index for year {year}: {D4000_idx} with error {D4000_idx_err}")
    #     for idx_name in idx_names:
    #         idx, idx_err = idx_w_errs(idx_name, blue_lambdas[i], blue_fluxes[i], blue_variances[i], z)
    #         print(f"{idx_name} index for year {year}: {idx} with error {idx_err}")