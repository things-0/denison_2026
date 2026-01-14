import numpy as np
import pylab as py
import scipy
import scipy.signal
import scipy.stats as stats
import scipy.interpolate as interp
#from astropy.io import fits
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from matplotlib import rc
import glob
import os
import smc_plot # probably don't need to use
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

################################################################################
# short script to generate a new FITS binray table from D4000 results, but
# only include 1 row per cube that contains the ifuprobe info for each file.
#
def get_ifuprobes(infile):
    
# open and read FITS file:
    hdulist = pf.open(infile)
    tbdata = hdulist[1].data
    print(hdulist[1].columns)

    catid = tbdata['CATID']
    cubenames = tbdata['Cubename']
    # note that the v0.9.1 cubes do not have IFUPROBE as a keyword in the header, so can't
    # test for that, unless we match against the newer data first.
    ifuprobe = tbdata['IFUPROBE']
    n = np.size(catid)

    cubename_c = np.empty(n,dtype='S64')
    ifuprobe_c = np.empty(n,dtype='S10')
    
    #loop through array and only output when you get to a new cube:
    prevcube = ' '
    nc = 0
    for i in range(n):
        if (cubenames[i] != prevcube):
            print(i,cubenames[i],ifuprobe[i])
            prevcube = cubenames[i]
            cubename_c[nc] = cubenames[i]
            ifuprobe_c[nc] = ifuprobe[i]
            nc = nc +1
            
    print(nc)
    col1 = pf.Column(name='Cubename', format='64A', array=cubename_c[0:nc])
    col2 = pf.Column(name='IFUPROBE', format='10A', array=ifuprobe_c[0:nc])
 
     # next we make the column definitions:
    cols = pf.ColDefs([col1, col2])

    #Now, create a new binary table HDU object:
    tbhdu = pf.BinTableHDU.from_columns(cols)

    # finally write the table HDU to a file:
    outfile = 'cube_probes.fits'
    tbhdu.writeto(outfile,clobber=True)
        


    

################################################################################
# get rms of array, including sigma clipping:

def clipped_mean_rms(a, nsig,verbose=False,niter_max=10):

    # get size of array:
    n = np.size(a)
    
    # get the mean:
    mean = np.nansum(a)/n
    
    # get rms:
    rms = np.sqrt(np.nansum(((a-mean)**2)/n))

    if (verbose):
        print(mean,rms,n)
        
    nprev = n
    for i in range(niter_max):

        a_clipped = a[np.where(abs(a-mean) < nsig*rms)]
        n_clipped = np.size(a_clipped)

        mean = np.nansum(a_clipped)/n_clipped
        rms = np.sqrt(np.nansum(((a_clipped-mean)**2)/n_clipped))
        if (verbose):
            print(i,mean,rms,n_clipped,nprev)
            
        if (n_clipped == nprev):
            break
        else:
            nprev = n_clipped
        
    return mean,rms,n_clipped

    

#######################################################################
# function to plot D4000/Hdelta etc reading from fits binary table
#
def plot_d4000_hdelta(infile,dopdf=True,snlim=5.0,col='r',compfile=None,edgeonly=False,probes=[1,2,3,4,5,6,7,8,9,10,11,12,13],label='',lstyle='solid'):


    py.rc('text', usetex=True)
    py.rc('font', family='sans-serif')
    py.rcParams.update({'font.size': 12})
    py.rcParams.update({'lines.linewidth': 1})
    
# open and read FITS file:
    hdulist = pf.open(infile)
    tbdata = hdulist[1].data
    print(hdulist[1].columns)

    catid = tbdata['CATID']
    cubenames = tbdata['Cubename']
    # note that the v0.9.1 cubes do not have IFUPROBE as a keyword in the header, so can't
    # test for that, unless we match against the newer data first.
    ifuprobe = tbdata['IFUPROBE']
    sn = tbdata['SN']
    d4000 = tbdata['D4000n']
    d4000_err = tbdata['D4000n_err']
    hdelta = tbdata['Hdelta']
    hdelta_err = tbdata['Hdelta_err']
    n = np.size(catid)

    goodprobe = np.zeros(n,dtype=bool)
    npr = np.size(probes)
    
    # set the goodprobe flag based on whether the probe number is in the probes[] list:
    for i in range(n):
        for j in range(npr):
            if (ifuprobe[i] == str(probes[j])):
                goodprobe[i] = True
                break
    

    # if there is a comparison file, read this in, so we can only select objects
    # that are also in the comparison:
    if (compfile != None):
        hdulist_c = pf.open(compfile)
        tbdata_c = hdulist_c[1].data
        catid_c = tbdata_c['CATID']
        ifuprobe_c = tbdata_c['IFUPROBE']
        # check for good objects:
        
        idx = np.where((np.in1d(catid,catid_c)) & (sn>snlim) & (goodprobe) & (~np.isnan(d4000)) & (~np.isnan(hdelta)))
        
    else:
    
        idx = np.where((sn>snlim) & (goodprobe) & (~np.isnan(d4000)) & (~np.isnan(hdelta)))
    
    py.figure(1)
    py.plot(d4000[idx],hdelta[idx],',',color=col)
    #py.errorbar(d4000[0:n],hdelta[0:n],xerr=d4000_err[0:n],yerr=hdelta_err[0:n],fmt='o',color='k')
    py.xlim(xmin=0.7,xmax=2.5)
    py.ylim(ymin=-6.0,ymax=12.0)
    py.xlabel('D$_n$(4000)')
    py.ylabel('H$\delta_A$')

    py.figure(2)
    py.subplot(2,2,1)
    py.hist(d4000[idx],bins=40,range=(0.5,2.5),color=col,histtype='step',label='D4000',normed=1)
    py.subplot(2,2,2)
    py.hist(hdelta[idx],bins=30,range=(-5.0,10.0),color=col,histtype='step',label='Hdelta',normed=1)
    py.subplot(2,2,3)
    py.hist(sn[idx],bins=50,range=(0.0,50.0),color=col,histtype='step',label='S/N',normed=1)

    # look at distribution in narrow D4000 bins to measure the scatter.
    nbin=34
    d4000_range = [0.8,2.5]
    hdelta_med, edges2, bc = stats.binned_statistic(d4000[idx],hdelta[idx], np.nanmedian, bins=nbin,range=d4000_range)
    d4000_med, edges2, bc = stats.binned_statistic(d4000[idx],d4000[idx], np.nanmedian, bins=nbin,range=d4000_range)
    hdelta_rms, edges2, bc = stats.binned_statistic(d4000[idx],hdelta[idx], np.nanstd, bins=nbin,range=d4000_range)
    
    print(hdelta_med)
    print(d4000_med)
    py.figure(1)
    py.plot(d4000_med,hdelta_med,'o',color='k')
    py.plot(d4000_med,hdelta_med,'-',color='k')
    py.plot(d4000_med,hdelta_med-hdelta_rms,'--',color='k')
    py.plot(d4000_med,hdelta_med+hdelta_rms,'--',color='k')

    # get mean RMS:
    medrms_hdelta = np.nanmean(hdelta_rms)
    print('mean rms:',medrms_hdelta)

    print(d4000_med)
    print(hdelta_med)

    good = np.where(~np.isnan(d4000_med))
    
    # fit a spline to the median values:
    spline_1d = interp.splrep(d4000_med[good],hdelta_med[good])
    splmodel = interp.splev(d4000_med,spline_1d)
    py.plot(d4000_med[good],splmodel[good],'-',color=col)

    # subtract spline fit from th Hdelta:
    hdelta_sub = np.copy(hdelta)
    hdelta_sub = hdelta_sub - interp.splev(d4000,spline_1d)

    # plot spline subtracted version:
    py.figure(3)
    py.plot(d4000[idx],hdelta_sub[idx],',',color=col)
    py.xlim(xmin=0.7,xmax=2.5)
    py.ylim(ymin=-20.0,ymax=20.0)

    hdelta_sub_med, edges2, bc = stats.binned_statistic(d4000[idx],hdelta_sub[idx], np.nanmedian, bins=nbin,range=d4000_range)
    d4000_sub_med, edges2, bc = stats.binned_statistic(d4000[idx],d4000[idx], np.nanmedian, bins=nbin,range=d4000_range)

    hdelta_sub_rms, edges2, bc = stats.binned_statistic(d4000[idx],hdelta_sub[idx], np.nanstd, bins=nbin,range=d4000_range)
    py.plot(d4000_sub_med,hdelta_sub_med,'o',color='k')
    py.plot(d4000_sub_med,hdelta_sub_med,'-',color='k')
    py.plot(d4000_sub_med,hdelta_sub_med-hdelta_sub_rms,'--',color='k')
    py.plot(d4000_sub_med,hdelta_sub_med+hdelta_sub_rms,'--',color='k')
    
    (mean,rms,nc) = clipped_mean_rms(hdelta_sub[idx],5.0,verbose=True)

    
    print('global rms: ',rms)

    if (dopdf):
        pdf = PdfPages('d4000_hdelta.pdf')

    
    # call code to print contours and points:
    fig = plt.figure(4)
    ax = fig.add_subplot(111) 
    smc_plot.cont_points(d4000[idx],hdelta[idx],ax,bins=(40,40),range=[[0.7,2.5], [-6.0,12.0]],color=col,lstyle=lstyle)
    py.xlim(xmin=0.7,xmax=2.5)
    py.ylim(ymin=-6.0,ymax=12.0)
    py.xlabel('D$_n$(4000)')
    py.ylabel('H$\delta_A$')

    #adding legend by hand for the paper plots (DR2 paper):
    line1 = mlines.Line2D([], [], color='k', linestyle = 'dashed', label='IFUs 2-12')
    line2 = mlines.Line2D([], [], color='r', linestyle = 'solid', label='IFUs 1,13')
    py.legend(handles=[line1,line2])

    py.text(0.05, 0.9,label, horizontalalignment='left',verticalalignment='center',transform=ax.transAxes)
    
    if (dopdf):
        py.savefig(pdf, format='pdf')        
        pdf.close()


    

########################################################################
# function to read in cubes and measure various stellar population
# indices.
#
def get_stel_index_many(inlist,catfile,doplot=True,doplotall=False,snlim=5.0,col='k',verbose=False):


    rc('text', usetex=True)
    #rc('lines',linewidth=2)
    #rc('axes', linewidth=2)
    #rc('font', family='sanserif')
    #rc('font', weight='bold')
    rc('font', size=14)
    # get the input list:
    files = glob.glob(inlist)

    # get the cat file info:
    print("reading "+catfile)
    cat = pf.open(catfile)
    primary_header=cat['PRIMARY'].header
    tbdata = cat[1].data  # assume the first extension is a table

    # get required table cols:
    catid = tbdata['CATID']
    mstar = tbdata['MSTAR']
    zspec= tbdata['Z_SPEC']
    ncat = np.size(catid)
    
    # set up plotting:
    if (doplotall):
        py.figure(1)

    nfmax = np.size(files)
    d4000 = np.zeros(nfmax*50*50)
    d4000_err = np.zeros(nfmax*50*50)
    hdelta = np.zeros(nfmax*50*50)
    hdelta_err = np.zeros(nfmax*50*50)
    sn = np.zeros(nfmax*50*50)
    name = np.zeros(nfmax*50*50,dtype=int)
    probe = np.empty(nfmax*50*50,dtype='S10')
    ixx = np.zeros(nfmax*50*50,dtype=int)  
    iyy = np.zeros(nfmax*50*50,dtype=int)
    basename = np.empty(nfmax*50*50,dtype='S64')
    dirname = np.empty(nfmax*50*50,dtype='S128')

    
    n=0
    nf = 0
    for cubefile in files:

        nf = nf + 1
        print('reading ',cubefile,'  file ',nf,' of ',nfmax)

        header = pf.getheader(cubefile)
        cubedata = pf.getdata(cubefile)
        cubevar = pf.getdata(cubefile,extname='VARIANCE')

        zs,ys,xs = np.shape(cubedata)
        #print('cube size:',zs,ys,xs)

        objname = header['NAME']

        # get the IFu probe number, but we only have this for later versions
        # of the cubes:
        try:
            ifuprobe_tmp = header['IFUPROBE']
            # We are getting this as a string...
            ifuprobe = ifuprobe_tmp
        except KeyError:
            ifuprobe = -1
            
        crval3=header['CRVAL3']
        cdelt3=header['CDELT3']
        crpix3=header['CRPIX3']
        naxis3=header['NAXIS3']
        x=np.arange(naxis3)+1
        L0=crval3-crpix3*cdelt3 #Lc-pix*dL        
        lam=L0+x*cdelt3

        # need to remove redshift, so get it from cat file list:
        z=0.0
        for i in range(ncat):
            #print(objname,catid[i])
            if (int(objname) == int(catid[i])):
                if (verbose):
                    print('match',objname,zspec[i])
                z = zspec[i]
                
        ###########
        
        # loop over each spaxel:
        for ix in range(xs):
            for iy in range(ys):
        
                spec = cubedata[:,iy,ix]
                var = cubevar[:,iy,ix]

                # do simple tests on spectra and check that the data exists and is good:
                med_signal = np.nanmedian(spec)
                med_noise = np.sqrt(np.nanmedian(var))
                med_sn = med_signal/med_noise
                if ((np.isnan(med_signal)) | (np.isnan(med_noise))):
                    continue

                # cut for S/N:
                if (med_sn < snlim):
                    continue

                sn[n] = med_sn
                name[n] = int(objname)
                probe[n] = ifuprobe
                ixx[n] = ix
                iyy[n] = iy

                basename[n]=os.path.basename(cubefile)
                dirname[n]=os.path.dirname(cubefile)
                
                # SAMI spectra are in units of:
                # BUNIT   = '10**(-16) erg /s /cm**2 /angstrom /pixel' / Units
                # test code (actually use Adam's version below):
                # d4000n = d4000n_index(spec,lam)

                # run Adam's version:
                d4000[n],d4000_err[n]=D4000_w_errs(lam,spec,var,z=z,res=None)

                # also get the H_delta from adam's code:
                hdelta[n],hdelta_err[n] = Hdelta_A_w_errs(lam,spec,var,z=z,res=None)

                if (verbose):
                    print(d4000[n],d4000_err[n],hdelta[n],hdelta_err[n])

                # plot individual spectra if required:
                if (doplotall):
                    py.figure(1)
                    py.clf()
                    py.plot(lam,spec)
                    # mark D4000 index bands:
                    py.axvspan(lick_window['D4000_1'][0]*(1+z),lick_window['D4000_1'][1]*(1+z),alpha=0.2,color='red')
                    py.axvspan(lick_window['D4000_2'][0]*(1+z),lick_window['D4000_2'][1]*(1+z),alpha=0.2,color='red')

                    # mark Hdelta bands:
                    py.axvspan(lick_window['Hdelta_A_1'][0]*(1+z),lick_window['Hdelta_A_1'][1]*(1+z),alpha=0.2,color='blue')
                    py.axvspan(lick_window['Hdelta_A_2'][0]*(1+z),lick_window['Hdelta_A_2'][1]*(1+z),alpha=0.2,color='green')
                    py.axvspan(lick_window['Hdelta_A_3'][0]*(1+z),lick_window['Hdelta_A_3'][1]*(1+z),alpha=0.2,color='blue')

                    py.xlim(xmin=3700.0,xmax=4500.0)
                    py.xlabel('Wavelength \AA')
                    py.ylabel('flux')

                    py.draw()
                    py.show()

                    print('Continue?')
                    #yntest = input()
                    yntest = raw_input()

                n=n+1

    # set up plotting:
    if (doplot):
       py.figure(2)
       py.plot(d4000[0:n],hdelta[0:n],'.',color=col)
       #py.errorbar(d4000[0:n],hdelta[0:n],xerr=d4000_err[0:n],yerr=hdelta_err[0:n],fmt='o',color='k')
       py.xlim(xmin=0.7,xmax=2.5)
       py.ylim(ymin=-6.0,ymax=12.0)
       py.xlabel('D$_n$(4000)')
       py.ylabel('H$\delta_A$')

       py.figure(3)
       py.subplot(1,2,1)
       py.hist(d4000[0:n],bins=40,range=(0.5,2.5),histtype='step',color=col,label='D4000')
       py.subplot(1,2,2)
       py.hist(hdelta[0:n],bins=30,range=(-5.0,10.0),histtype='step',color=col,label='Hdelta')

    # finally write out measurements to a FITS binary table
    #
    # we need to start by making the columns:
    col1 = pf.Column(name='CATID', format='I', array=name[0:n])
    col2 = pf.Column(name='Cubename', format='64A', array=basename[0:n])
    col3 = pf.Column(name='Path', format='128A', array=dirname[0:n])
    # set IFU probe to be a string, as it may contain more than one number:
    col4 = pf.Column(name='IFUPROBE', format='10A', array=probe[0:n])
    #col2 = pf.Column(name='IFUPROBE', format='I', array=probe[0:n])
    col5 = pf.Column(name='IX', format='I', array=ixx[0:n])
    col6 = pf.Column(name='IY', format='I', array=iyy[0:n])
    col7 = pf.Column(name='SN', format='D', array=sn[0:n])
    col8 = pf.Column(name='D4000n', format='D', array=d4000[0:n],unit=' ')
    col9 = pf.Column(name='D4000n_err', format='D', array=d4000_err[0:n],unit=' ')
    col10 = pf.Column(name='Hdelta', format='D', array=hdelta[0:n],unit=' ')
    col11 = pf.Column(name='Hdelta_err', format='D', array=hdelta_err[0:n],unit=' ')

    # next we make the column definitions:
    cols = pf.ColDefs([col1, col2, col3, col4, col5, col6,col7,col8,col9,col10,col11])

    #Now, create a new binary table HDU object:
    tbhdu = pf.BinTableHDU.from_columns(cols)

    # finally write the table HDU to a file:
    outfile = 'd4000_hdelta.fits'
    tbhdu.writeto(outfile,clobber=True)
        


##A Gaussian function
#x=independenat variable
#a=amplitude
#b=phase shift (left or right)
#c=sigma
#d=zero-point offset (up or down)

def gaussian(x,a,b,c,d):
    return a*np.exp(-(x-b)**2/(2*c**2))+d



##Return values of an array where a condition is true.
#For multiple conditions use "(condition1) & (condition2)"
#array=input array to choose values from
#condition must be defined with respect to numpy arrays, not lists.
def array_if(array,condition=None):
    array1=np.array(array)
    if condition==None:
        return array1
    ret=array1[np.where(condition)]
    return ret

##Return values in several arrays where a condition is true.
#array_list=list of arrays to extract values from
#contition=condition governing which values are extracted.

def multi_array_if(array_list,condition):
    ret=[]
    for arrays in array_list:
        ret.append(array_if(arrays,condition))
    return ret



##Reject invalid numbers across multiple arrays:

def reject_invalid(variables,bad_flag=None):
    '''
    This function takes in a list of variables stored in numpy arrays and returns these arrays where there are no nans.
    variables=[variable1,variable2,variable3...]
    bad_flag=a value that denotes bad data e.g. -999
    '''
    if type(variables)!=list:
        print("please input a list of numpy arrays")
        return
    good=np.ones(variables[0].shape)
    for var in variables:
        bad=np.where((np.isnan(var)==True) | (np.isinf(var)==True) | (var==bad_flag))
        good[bad]=0
    var_out=[]
    for var in variables:
        var_out.append(var[good==1])
    return var_out






#Speed of light in a vacuum in km/s
c=299792.458



## This function will blur the spectrum to the appropriate resolution.
#wavelength=wavelength vector
#spec=spectrum
#res_in=Native resolution in Angstroms. Sigma, not FWHM.
#res_out=Desired resolution of the spectrum
#Disp_cor=dispersion correction for high velocity dispersion spectra. Default=0
def blur_spectrum(wavelength,spec,res_in,res_out,disp_cor=0):
    res_ker=np.sqrt(res_out**2 - res_in**2 - disp_cor**2)
    kernel=(1.0/np.sqrt(2*np.pi*res_ker**2))*gaussian(wavelength,1.0,np.median(wavelength),res_ker,0)
    spec_smooth=scipy.signal.convolve(np.nan_to_num(spec),kernel,mode='same')
    transfer=np.ones(spec.shape)
    badpix=np.where(np.isnan(spec)==True)
    transfer[badpix]=0
    transfer_smooth=scipy.signal.convolve(transfer,kernel,mode='same')
    return spec_smooth/transfer_smooth

#Conversion from FWHM to Gaussian sigma
FWHM_sigma=2.0*np.sqrt(2.0*np.log(2.0))

#Spectral resolutions of the SAMI red and blue arms 
SAMI_res_B=1.15
SAMI_res_R=0.61
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
             'Hdelta_A_1':[4041.60,4079.75],  #Worthey & Ottaviani 1997
             'Hdelta_A_2':[4083.50,4122.25],
             'Hdelta_A_3':[4128.50,4161.00],
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




##D4000 index
#wave=wavelength scale
#spec=input spectrum
#z=redshift of the spectrum
#res=native resolution of the spectrum
def D4000(wave,spec,z,res=None):
    ind_res=lick_res['D4000']
    if res!=None:
        spec_smooth=blur_spectrum(wave,spec,res,ind_res)
    else:
        spec_smooth=spec
    spec_smooth=flam_fnu(wave,spec_smooth)
    spec_smooth=flux_per_pix(wave,spec_smooth)
    ind1=array_if(spec_smooth,(wave>=lick_window['D4000_1'][0]*(1+z)) & (wave<=lick_window['D4000_1'][1]*(1+z)))
    ind2=array_if(spec_smooth,(wave>=lick_window['D4000_2'][0]*(1+z)) & (wave<=lick_window['D4000_2'][1]*(1+z)))
    d1=np.nanmean(ind1)
    d2=np.nanmean(ind2)
    d4000=d2/d1
    #Compute errors. Need variance spectrum first...
    return d4000



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
    var=var*((wave**2)/(c*10**13))**2
    var=flux_per_pix(wave,var)
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




##Hdelta_A index
#wave=wavelength scale
#spec=spectrum
#z=redshift of the spectrum
#res=native resolution of the spectrum
def Hdelta_A(wave,spec,z,res=None):
    ind_res=lick_res['Hdelta_A']
    if res!=None:
        spec_smooth=blur_spectrum(wave,spec,res,ind_res)
    else:
        spec_smooth=spec
    spec_smooth=flux_per_pix(wave,spec_smooth)
    ind1=array_if(spec_smooth,(wave>=lick_window['Hdelta_A_1'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_1'][1]*(1+z)))
    ind2=array_if(spec_smooth,(wave>=lick_window['Hdelta_A_2'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_2'][1]*(1+z)))
    ind3=array_if(spec_smooth,(wave>=lick_window['Hdelta_A_3'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_3'][1]*(1+z)))
    hda1=np.nanmean(ind1)
    hda2=np.nanmean(ind2)
    hda3=np.nanmean(ind3)
    hda_sideband=np.nanmean([hda1,hda3])
    hda_index=hda2
    lam1=lick_window['Hdelta_A_2'][0]*(1+z)
    lam2=lick_window['Hdelta_A_2'][1]*(1+z)
    hda=(lam2-lam1)*(1-hda_index/hda_sideband)
    return hda





def Hdelta_A_w_errs(wave,spec,var,z,res=None):
    ind_res=lick_res['Hdelta_A']
    if res!=None:
        spec_smooth=blur_spectrum(wave,spec,res,ind_res)
    else:
        spec_smooth=spec
    spec_smooth=flux_per_pix(wave,spec_smooth)
    #ind1=array_if(spec_smooth,(wave>=lick_window['Hdelta_A_1'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_1'][1]*(1+z)))
    #ind2=array_if(spec_smooth,(wave>=lick_window['Hdelta_A_2'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_2'][1]*(1+z)))
    #ind3=array_if(spec_smooth,(wave>=lick_window['Hdelta_A_3'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_3'][1]*(1+z)))
    #hda1=np.nanmean(ind1)
    #hda2=np.nanmean(ind2)
    #hda3=np.nanmean(ind3)
    idx1=np.where((wave>=lick_window['Hdelta_A_1'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_1'][1]*(1+z)))
    idx2=np.where((wave>=lick_window['Hdelta_A_2'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_2'][1]*(1+z)))
    idx3=np.where((wave>=lick_window['Hdelta_A_3'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_3'][1]*(1+z)))
    hda1=np.nanmean(spec_smooth[idx1])
    hda2=np.nanmean(spec_smooth[idx2])
    hda3=np.nanmean(spec_smooth[idx3])
    hda_sideband=np.nanmean([hda1,hda3])
    hda_index=hda2
    lam1=lick_window['Hdelta_A_2'][0]*(1+z)
    lam2=lick_window['Hdelta_A_2'][1]*(1+z)
    hda=(lam2-lam1)*(1-hda_index/hda_sideband)
    ##Now calculate the error
    var=flux_per_pix(wave,var)
    var=flux_per_pix(wave,var) #I have to do this twice becasue you multiply variance by square of number
    #var_ind1=array_if(var,(wave>=lick_window['Hdelta_A_1'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_1'][1]*(1+z)))
    #var_ind2=array_if(var,(wave>=lick_window['Hdelta_A_2'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_2'][1]*(1+z)))
    #var_ind3=array_if(var,(wave>=lick_window['Hdelta_A_3'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_3'][1]*(1+z)))
    #[var_ind1]=reject_invalid([var_ind1])
    #[var_ind2]=reject_invalid([var_ind2])
    #[var_ind3]=reject_invalid([var_ind3])
    #v1=np.sum(var_ind1)/(len(var_ind1)**2)
    #v2=np.sum(var_ind2)/(len(var_ind2)**2)
    #v3=np.sum(var_ind3)/(len(var_ind3)**2)
    vidx1=np.where((wave>=lick_window['Hdelta_A_1'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_1'][1]*(1+z)))
    vidx2=np.where((wave>=lick_window['Hdelta_A_2'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_2'][1]*(1+z)))
    vidx3=np.where((wave>=lick_window['Hdelta_A_3'][0]*(1+z)) & (wave<=lick_window['Hdelta_A_3'][1]*(1+z)))
    #[var_ind1]=reject_invalid([var_ind1])
    #[var_ind2]=reject_invalid([var_ind2])
    #[var_ind3]=reject_invalid([var_ind3])
    v1=np.nansum(var[vidx1])/(len(var[idx1])**2)
    v2=np.nansum(var[vidx2])/(len(var[idx2])**2)
    v3=np.nansum(var[vidx3])/(len(var[idx3])**2)
    v_sb=(v1 + v3) / 4.0
    s_sb=np.sqrt(v_sb)
    s_i=np.sqrt(v2)
    hda_err=np.abs(hda)*(lam2-lam1)*np.sqrt((s_sb/hda_sideband)**2 + (s_i/hda_index)**2)
    return hda, hda_err

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




##Plot spectrum and index passbands
#wave=wavelength
#spec=spectrum to plot
#z=redshift
#index=String identifier of the index e.g. 'Hdelta' or 'D4000'
#res=spectral resolution, default is the SAMI Blue arm resolution
#buff=buffer zone in Angstroms on either side of the index region to plot
def plot_index(wave,spec,z,index,res=SAMI_res_B,buff=100,plot_smoothed=False,**kwargs):
    for lres in lick_res.keys():
        if index in lres:
            ind_res=lick_res[lres]
            ind_name=lres
        else:
            ind_name=index
            ind_res=None
    bandct=0
    for names in lick_window.keys():
        if index in names:
            bandct=bandct+1
    wavemin=min(lick_window[ind_name+'_1'])*(1+z)-buff
    wavemax=max(lick_window[ind_name+'_'+str(bandct)])*(1+z)+buff
    if ind_res!=None:
        spec_smooth=blur_spectrum(wave,spec,res,ind_res)
    else:
        spec_smooth=spec
    wave_win,spec_win,spec_smooth_win=multi_array_if([wave,spec,spec_smooth],(wave<wavemax) & (wave>wavemin))
    plt.plot(wave_win,spec_win,c='b')
    if plot_smoothed==True:
        plt.plot(wave_win,spec_smooth_win,c='r')
    cols=['b','r','b']
    for i in range(1,bandct+1):
        ran=lick_window[ind_name+'_'+str(i)]
        plt.axvspan(ran[0]*(1+z),ran[1]*(1+z),color=cols[i-1],alpha=0.4)
    #plt.show()
    return



def flam_fnu(wave,flam):
    return (flam*wave**2)/(c*10**13)



