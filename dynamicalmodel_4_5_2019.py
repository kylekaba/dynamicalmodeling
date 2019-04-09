#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:15:29 2019

@author: kylekabasares
"""

#################### MAIN MODELING CODE ROUTINE #############################
def residual(freeparams,x_range,y_range,x_0,y_0,a,q,f_range,FluxMap,r_int=None,vc_st=None,data=None,eps=None):
# x_range, y_range are the lengths of the convolution region box in pixels
# x_0, y_0 is the center of the fitting region ellipse and the central pixel of the convoution region
    
    mbh = freeparams['mbh']
    MtoL = freeparams['MtoL']
    xc = freeparams['xc']
    yc = freeparams['yc']
    vsys = freeparams['vsys']
    z = freeparams['z']
    theta = freeparams['theta']
    incl = freeparams['incl']
    F_0 = freeparams['F_0']
    F_1 = freeparams['F_1']
    r_s = freeparams['r_s']
    sigma_0 = freeparams['sigma_0']
    
    ### Import the FIXED Parameters from the ALMA Parameter File 
    #f_init = hdul[0].header['CRVAL3']
    FILE = open("3258testparam_4.txt","r")
    parameters = defaultdict(str)
    for line in FILE:
        paramval = line.strip().split('=')
        parameters[paramval[0].strip()] = paramval[1].strip()
    
    f_spacing = float(parameters['freq_del'])
    z_i = int(parameters['nu_i'])
    z_f = int(parameters['nu_f'])
    
    # Disk inclination angle and rotation angle (PA)
    # in degrees, where i=0 is face-on and i=90 deg.
    # is edge-on and th=0 is north (up) and th=90 deg. is
    # east (left) of the receding disk major axis.
    # Afterwards, transform these angles to radians and
    # from the inclination angle compute the
    # disk minor/major axis ratio, qv
          
    incl=incl/180.*np.pi
    theta=(270.-theta)/180.*np.pi
    qv=np.cos(incl)

    # Construct matrices that define the native x/y positions
    # of an array with (ndx,ndy) dimensions. The native x/y positions
    # are shifted by the (xc,yc) disk centers at this stage.
    
    ndx = int(parameters['ndx'])
    ndy = int(parameters['ndy'])
    ndz = int(len(f_range))
    
    ssf = int(parameters['sub'])
    rebin = int(parameters['rebin'])
    
    xva=np.linspace(1,ssf*ndx,ssf*ndx)/ssf-xc
    yva=(np.linspace(1,1,ssf*ndy)/ssf)*np.linspace(1,ssf*ndy,ssf*ndy)-yc
    xva, yva = np.meshgrid(xva,yva)    
    
    # Transform the native x/y positions into
    # physical (disk) x/y positions using the
    # disk inclination and rotation angles.
    # Construct array that maps observed pixel
    # to a physical disk radius
    xv=(xva*np.cos(theta)-yva*np.sin(theta))
    yv=(yva*np.cos(theta)+xva*np.sin(theta))/qv
    # Radius at a projected location (x',y') on the sky plane
    rv=np.sqrt((xv)**2+(yv)**2)
    
    # Radius map
    plt.figure(1)
    plt.imshow(rv,origin='lower')
    plt.title('Radius (pixels)')
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    cb = plt.colorbar()
    cb.set_label('Radius ',fontsize = 16)
    
    # Compute the fraction of the maximum line-of-sight (LOS)
    # velocity (along the disk major axis) at each observed
    # pixel position
    los_frac=(xva*np.cos(theta)-yva*np.sin(theta))/rv

    # Convert the radius array from pixels to parsecs (pc)
    # using a 5.91 pc/pixel scaling for NGC 3258
    scale = float(parameters['scale'])
    rv_pc=rv*scale
    
    if r_int is None and vc_st is None:
        vc2ml = 50*np.arctan(rv_pc/150)
    else:
        vc2ml = np.interp(rv_pc,r_int,vc_st)
        
    # Example stellar contribution to the gravitational potential,
    # specifically the circular velocity profile
    #vcstellar=50.*np.arctan(rv_pc/150.)
    #vc2ml = np.interp(rv_pc,r_int,vcst_int)
    vcstellar = np.sqrt(MtoL*vc2ml)
    
    # Determine Total and LOS circular velocity using the stellar mass-to-light
    # ratio, ml, and Newton's constant G
    vctotal=np.sqrt(MtoL*vc2ml +(G*mbh/rv_pc))
    vlostotal=vctotal*los_frac*np.sin(incl)
    # Subtract the systemic velocity of the galaxy to compute observed velocity 
    vobs=vsys-vlostotal
    sigmaturb = sigma_0*np.ones((ndx*ssf,ndy*ssf)) # test value in km/s
    
    # Plotting total circular velocity at each spatial position on the grid
    plt.figure(2) 
    plt.imshow(vctotal,cmap='viridis',origin='lower')
    plt.xlabel('X Disk')
    plt.ylabel('Y Disk')
    cb = plt.colorbar()
    cb.set_label('$V_{Circular}$ (km/s)',fontsize = 16) 
    
    # Plotting the LOS velocity 
    plt.figure(3)
    plt.imshow(vlostotal,cmap='viridis',origin='lower')
    plt.xlabel('X Disk')
    plt.ylabel('Y Disk')
    cb = plt.colorbar()
    cb.set_label('$V_{LOS}$ (km/s)',fontsize = 16)   
    
    # Determine the corresponding frequency centroid and frequency width at each spatial position
    # f1 and f2 define the starting and ending frequency channels that define the line profile.
    cs = float(parameters['cs'])
    f_0 = float(parameters['restfreq'])/1e9
    f_obs = (f_0/(1+z))*(1-(vlostotal/cs))
    df_obs = (f_0/(1+z))*(sigmaturb/cs)
    
    # Create FITS (Flexible Image Transport System) files 
    # from scratch to save the centroid frequencies and frequency widths
    hdu=fits.PrimaryHDU(f_obs)
    hdu.writeto('3258fobs_array.fits',overwrite=True)
    
    hdu=fits.PrimaryHDU(df_obs)
    hdu.writeto('3258_dfobs_array.fits',overwrite=True)
    
    # Create a velocity range based on the optical definition of radial velocity 
    # c(f-f_0)/f
    v_range = cs*(f_range-f_0)/(f_range)
    
    # Model the PSF as an elliptical gaussian with a mean = 0 and standard deviation proportional to the 
    # FWHM of the synthesized beam 
    # Set the center of the Gaussian at (ndx/2,ndy/2)
    
    wf = int(parameters['wf'])
    PSF_sigma_x = (float(parameters['beammaj'])/2.3548)/float(parameters['ra_del_arcsec'])
    PSF_sigma_y = (float(parameters['beammin'])/2.3548)/float(parameters['dec_del_arcsec'])
    
    resbeamPA = float(parameters['resbeamPA'])
    # In degrees
    PSF_PA = (90.+resbeamPA)*(np.pi/180)
    
    elliptical_gaussian = Gaussian2D(1,int(ndx/2),int(ndy/2),PSF_sigma_x*wf,PSF_sigma_y*wf,theta=((np.pi/2)-PSF_PA))
    x, y = np.mgrid[0:int(parameters['ndx']), 0:int(parameters['ndy'])]
    PSF = np.array(elliptical_gaussian(x,y))

    # Normalize the PSF by dividing by the sum of the PSF 
    PSF_norm = PSF/np.sum(PSF)
    # Crop the inner region of the PSF to have a 51x51 pixel image
    # Set the fitting region to be from [(ndx/2)-26:(ndx/2)+25,(ndy/2)-26:(ndy/2)+25]
    
    PSF_sub = PSF_norm[int((ndx/2)-26):int((ndx/2)+25),int((ndy/2)-26):int((ndy/2)+25)]
    
    # Testing J. Cohn's PSF code 
    
    PSF_sub = make_beam(51,0.044,1,0,0,0.479199,0.401434,-76)
    # Create line profiles from the f_centroid and f_width arrays
    # Sigma is the dispersion, which is assumed to be flat for now. 
    # Centroid velocity is the systemic velocity of the galaxy
    
    # Sampling the line by just evaluating the Gaussian with given centroid frequency and line width 
    
    glineprof = np.swapaxes(np.swapaxes(np.array([(1/(np.sqrt(2*np.pi*df_obs**2)))*np.exp(-0.5*((i-f_obs)**2)/(df_obs**2)) for i in f_range]),2,0),0,1)
    #hdu=fits.PrimaryHDU(np.swapaxes(glineprof,0,2))
    #hdu.writeto('3258Gaussian_LineProfileEvaluated.fits',overwrite=True)
    
    # Test glineprof by not including normalization factor 
    # Change name to glineflux to run it in the rest of the code
    glineflux = np.swapaxes(np.swapaxes(np.array([np.exp(-0.5*((i-f_obs)**2)/(df_obs**2)) for i in f_range]),2,0),0,1)

    delta_f = f_spacing/1e9
    
    # Create an integrated gaussian line profile following the methodology of B. Boizelle
    # The integrated line profile will be weighted by the deconvolved flux map in the following step
    #glineflux = np.swapaxes(np.swapaxes(-0.5*np.array([-scipy.erf((i-(delta_f/2)-f_obs)/(np.sqrt(2)*df_obs))+scipy.erf(((i+(delta_f/2)-f_obs)/(np.sqrt(2)*df_obs))) for i in f_range]),2,0),0,1)
    
    # Write the Gaussian Line Profile to a FITS file
    hdu=fits.PrimaryHDU(np.swapaxes(glineflux,0,2))
    hdu.writeto('3258GLineFluxTest_BeforeFluxMap.fits',overwrite=True)
    
    # Upscale the flux map by (ssf x ssf) 
    # Deconvolve the flux map with the Richardson-Lucy algorithm 
    # It is important that the flux map has been made to be strictly-non negative before the deconvolution
    
    FluxMap = restoration.richardson_lucy(FluxMap,PSF_sub,iterations=1)
    FluxMap = upsample(FluxMap,ssf)
    
    # Save the deconvovled flux map as a FITS file for future use
    hdu=fits.PrimaryHDU(FluxMap)
    hdu.writeto('3258DeconvolvedFluxMap.fits',overwrite=True)
    
    # Weight the line profile by multiplying each slice by the flux map.
    for i in range(int(ssf*ndx)):
        for j in range(int(ssf*ndy)):
            glineflux[i,j,:] = F_0*FluxMap[i,j]*glineflux[i,j,:]
    
    glineflux[glineflux < np.max(glineflux)*1e-6] = 0
    
    # Write the Gaussian Line Profile to a FITS file
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(glineflux,0,2),0,1))
    hdu.writeto('3258GLineFluxTest.fits',overwrite=True)
    
    #Display the PSF 
    plt.figure(4)
    plt.imshow(PSF_sub,interpolation='none',origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    cb.set_label('PSF',fontsize = 16) 
    
    # Display the flux map
    plt.figure(5)
    plt.imshow(FluxMap,origin='lower')
    plt.xlabel('x ')
    plt.ylabel('y ')
    cb = plt.colorbar()
    cb.set_label('Flux Map',fontsize = 16) 
        
    # Create FITS (Flexible Image Transport System) files 
    # from scratch to save the radial position and LOS velocity arrays.
    hdu=fits.PrimaryHDU(rv_pc)
    hdu.writeto('3258rv_array.fits',overwrite=True)
    hdu=fits.PrimaryHDU(los_frac)
    hdu.writeto('3258losfrac_array.fits',overwrite=True)
    hdu=fits.PrimaryHDU(vlostotal)
    hdu.writeto('3258vlos_array.fits',overwrite=True)
    hdu=fits.PrimaryHDU(vctotal)
    hdu.writeto('3258vctot.fits',overwrite=True)
    hdu=fits.PrimaryHDU(PSF)
    hdu.writeto('3258PSFModel.fits',overwrite=True)
    
    # Re-bin the integrated gaussian line profile to the scale of the original ALMA data for convolution efficiency
    # Pre-allocate the array first
    rebinned_glineflux = np.zeros((ndx,ndy,ndz))
    for i in range(ndz):
        rebinned_glineflux[:,:,i] = block_reduce(glineflux[:,:,i],ssf,np.sum)

    #Convolve the PSF with the integrated Gaussian line profile 
    convolvetest = np.zeros((ndx,ndy,ndz))
    
    convolvetest_sub = np.zeros((int(ndx/(rebin)),int(ndy/(rebin)),ndz))
    
    # Define the convolution box region by determining the central pixel and then adding +/- x_range/2 on both sides 
    # of the central pixel 
    xlo = int(np.rint(x_0) - np.rint(x_range/2))
    xhi = int(np.rint(x_0) + np.rint(x_range/2))
    
    ylo = int(np.rint(y_0) - np.rint(y_range/2))
    yhi = int(np.rint(y_0) + np.rint(y_range/2))
    
    for i in list(range(z_i,z_f)): 
        convolvetest[xlo:xhi,ylo:yhi,i] = convolve(rebinned_glineflux[xlo:xhi,ylo:yhi,i],PSF_sub,boundary='extend',normalize_kernel=True) 
        convolvetest[convolvetest < 1e-6*np.max(convolvetest)] = 0 
        
        
    # Re-Bin the Model by averaging over ssf x ssf blocks in the spatial domain.
    for i in range(ndz):
        convolvetest_sub[:,:,i] = block_reduce(convolvetest[:,:,i],rebin,func=np.mean)
      
    model = convolvetest_sub
    
    # Create moment maps based on a center of mass calculation
    # Moment 0 is the total flux
    # Moment 1 is the mean line of sight velocity
    # Moment 2 is the mean line of sight velocity dispersion 
    
    Moment0 = np.sum(model,2)*np.abs(f_spacing)
    Moment1_FullSize = np.zeros((np.size(model,0),np.size(model,1),np.size(model,2)))
    Moment2_FullSize = np.zeros((np.size(model,0),np.size(model,1),np.size(model,2)))

    for i in range(ndz):
        Moment1_FullSize[:,:,i] = v_range[i]*model[:,:,i]
    
    Moment1 = np.sum(Moment1_FullSize,2)
    
    plt.figure(6)
    plt.imshow(Moment0,origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    cb.set_label('Total Flux',fontsize = 16)  
                  
    plt.figure(7)
    plt.imshow(Moment1,origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    cb.set_label('$V_{LOS}$',fontsize = 16)
    
    # Save the Synthetic Data Cube Before Flattening 
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(model,0,2),1,2))
    hdu.writeto('3258Convolve_Testnew.fits',overwrite=True)
    
    # Define the fitting ellipse that determines where the model optimizations occur
    x0, y0 = x_0,y_0
    Gamma = int(parameters['GammaEllipse'])
    semimaj = a
    semimin = q*a
    Gamma=(90.+Gamma)/180.*np.pi
    e = Ellipse2D(amplitude=1., x_0=x0, y_0=y0, a=semimaj, b=semimin,
    theta=Gamma)
    y, x = np.mgrid[0:int(parameters['ndx']), 0:int(parameters['ndy'])]

    # Select the regions of the ellipse we want to fit
    fitting_ellipse = np.array(e(x,y))
    region = np.where(fitting_ellipse == 1)
    x_reg = region[0]
    y_reg = region[1]
    
    # Sub-sampling factor:
    m = int(parameters['rebin'])
    ellipse_reduce = np.zeros((int(len(x)/m),int(len(y)/m)))
    ellipse_reduce = block_reduce(fitting_ellipse,m,np.mean)
    
    plt.figure(8)              
    plt.imshow(fitting_ellipse,origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.title('Elliptical Fitting Region')
    cb = plt.colorbar()

    reducedregion = np.where(ellipse_reduce != 0)
    x_reduced = reducedregion[0]
    y_reduced = reducedregion[1]
    
    fit_region = np.ones((int(len(x)/m),int(len(y)/m),ndz))
    fit_cube = np.zeros((int(len(x)/m),int(len(y)/m),ndz))
    for i in range(ndz):
        if i < z_i:
            fit_cube[:,:,i] = 0
        elif i > z_f:
            fit_cube[:,:,i] = 0
        else: 
            fit_cube[:,:,i] = ellipse_reduce*fit_region[:,:,i]
    
    fittingregion = np.where(fit_cube == 1)
    
    plt.figure(9)              
    plt.imshow(model[:,:,45],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Model Frequency Slice 45')
    
    plt.figure(10)              
    plt.imshow(data_rebin[:,:,45],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Data Frequency Slice 45')
    plt.show()
    
    plt.figure(11)              
    plt.imshow(model[:,:,55],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Model Frequency Slice 55')
    
    plt.figure(12)              
    plt.imshow(data_rebin[:,:,55],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Data Frequency Slice 55')
    plt.show()
    
    plt.figure(13)              
    plt.imshow(model[:,:,65],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Model Frequency Slice 65')
    
    plt.figure(14)              
    plt.imshow(data_rebin[:,:,65],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Data Frequency Slice 65')
    plt.show()

    # Identify the fitting regions in the data, model, and noise cube and flatten them to a 1D array
    # This must be done because LMFIT only accepts 1D arrays to perform chi-squared minimization. 
    model_vector =  model[fittingregion].flatten('C')
    if data is None:
        return model
    
    data_vector = data[fittingregion].flatten('C')
    
    if eps is None:
        return data_vector-model_vector
    
    eps_vector = eps[fittingregion].flatten('C')
    
    return (data_vector-model_vector)/eps_vector