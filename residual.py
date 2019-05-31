#################### MAIN MODELING CODE ROUTINE #############################
def residual(freeparams,gridparams,f_range,FluxMap,r_int=None,vc_st=None,data=None,eps=None,SaveFITS=False):
# x_range, y_range are the lengths of the convolution region box in pixels
# x_0, y_0 is the center of the fitting region ellipse and the central pixel of the convolution region
## Clear the previous output
    clear_output()
    ### INPUT GRID (FIXED) PARAMETERS FROM TEXT PARAMETER FILE
    GRIDPARAMS = open(gridparams,"r")
    gridparameters = defaultdict(str)
    for line in GRIDPARAMS:
        gridparamval = line.strip().split('=')
        gridparameters[gridparamval[0].strip()] = gridparamval[1].strip()
           
#     Generate a beam spread function
#     grid_size: size of grid (must be odd!)
#     res: resolution of the grid (arcsec/pixel)
#     amp: amplitude of the 2d gaussian
#     x0: mean of x axis of 2d gaussian
#     y0: mean of y axis of 2d gaussian
#     x_std: FWHM of beam in x (to use for standard deviation of Gaussian) (in arcsec)
#     y_std: FWHM of beam in y (to use for standard deviation of Gaussian) (in arcsec)
#     rot: rotation angle in radians

    gridsize = int(gridparameters['gridsize'])
    res = float(gridparameters['res'])
    amp = float(gridparameters['amp'])
    x0 = float(gridparameters['x0'])
    y0 = float(gridparameters['y0'])
    x_std = float(gridparameters['x_std'])
    y_std = float(gridparameters['y_std'])
    rot = float(gridparameters['rot'])
    a = float(gridparameters['a'])
    q = float(gridparameters['q'])
    xc_ellipse = float(gridparameters['xc_ellipse'])
    yc_ellipse = float(gridparameters['yc_ellipse'])
    
    ### FREE PARAMETERS OF THE MODEL
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
    sigma_0 = freeparams['sigma_0']
    sigma_1 = freeparams['sigma_1']
    mu = freeparams['mu']
    r_0 = freeparams['r_0']
    
    print('The black hole mass in solar masses is', float(mbh))
    print('The central x pixel is',float(xc))
    print('The central y pixel is',float(yc))
    print('The position angle is at',float(theta))
    print('The inclination angle is at',float(incl))
    print('The redshift is',float(z))
    print('The Mass to Light ratio is',float(MtoL))
    print('The flux multiplier constant is',float(F_0))
    print('The constant velocity dispersion term is',float(sigma_0))
    print('The systemic velocity is',float(vsys))
    print('The amplitude of the velocity dispersion Gaussian is',float(sigma_1))
    print('The radius offset of the velocity dispersion Gaussian is',float(r_0))
    print('The standard deviation of the velocity dispersion Gaussian is',float(mu))
    
    ### Import the FIXED Parameters from the ALMA Parameter File 
    #f_init = hdul[0].header['CRVAL3']
    
    # Start a timer 
    start = time.time()
    
    FILE = open("3258testparam_4.txt","r")
    parameters = defaultdict(str)
    for line in FILE:
        paramval = line.strip().split('=')
        parameters[paramval[0].strip()] = paramval[1].strip()
    
    f_spacing = float(parameters['freq_del'])
    xi = int(parameters['xi'])
    xf = int(parameters['xf'])
    yi = int(parameters['yi'])
    yf = int(parameters['yf'])
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
    
    ssf = int(parameters['sub'])
    rebin = int(parameters['rebin'])
    
    if data is None:
        ndx = int(parameters['ndx'])
        ndy = int(parameters['ndy'])
        ndz = int(len(f_range))
    else:
        hdul = fits.open(data)
        hdul.info()
        data = hdul[0].data
        data = data[0,:,:,:]
        data = np.swapaxes(np.swapaxes(data,0,2),0,1)
        # Truncate the data to work on a smaller grid 
        data = data[yi:yf,xi:xf,:]
        plt.imshow(data[:,:,55],origin='lower')
        plt.show()
        ndx = np.size(data,1)
        ndy = np.size(data,0)
        ndz = np.size(data,2)
        # Pre-Allocate Data Rebin Array
        data_rebin = np.zeros((int(ndy/rebin),int(ndx/rebin),ndz))
        for i in range(ndz):
            data_rebin[:,:,i] = block_reduce(data[:,:,i],rebin,np.mean)
        print('ndx,ndy,and ndz are',(ndx,ndy,ndz))
        

    # SAVE THE RE-BINNED DATA FOR FUTURE USE
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(data_rebin,0,2),1,2))
    hdu.writeto('3258RebinnedData.fits',overwrite=True)
    
    # Now, we must shift the xc and yc coordinates by the xi,xf, yi, and yf shift.
    xc = xc - xi
    yc = yc - yi
    
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
    plt.show()
    
    # Compute the fraction of the maximum line-of-sight (LOS)
    # velocity (along the disk major axis) at each observed
    # pixel position
    los_frac=(xva*np.cos(theta)-yva*np.sin(theta))/rv

    # Convert the radius array from pixels to parsecs (pc)
    scale = float(parameters['scale'])
    rv_pc=rv*scale
    
    if r_int is None and vc_st is None:
        vc2ml = 50*np.arctan(rv_pc/150)
    else:
        vc2ml = np.interp(rv_pc,r_int,vc_st)
        
    # Stellar contribution to the gravitational potential,
    # specifically the circular velocity profile
    vcstellar = np.sqrt(MtoL*vc2ml)
    
    # Determine Total and LOS circular velocity using the stellar mass-to-light
    # ratio, MtoL, and Newton's constant G
    vctotal=np.sqrt(MtoL*vc2ml +(G*mbh/rv_pc))
    vlostotal=vctotal*los_frac*np.sin(incl)
    
    end_grid = time.time()
    print('The model grid construction time in seconds is',end_grid-start)
    # Subtract the systemic velocity of the galaxy to compute observed velocity 

    # Turbulent velocity disperison profile, composed of a constant and Gaussian expression
    sigmaturb = sigma_0*np.ones((ndx*ssf,ndy*ssf)) + sigma_1*np.exp(-(rv_pc-r_0)**2/(2*mu**2))
    
    # Plotting total circular velocity at each spatial position on the grid
    plt.figure(2) 
    plt.imshow(vctotal,cmap='viridis',origin='lower')
    plt.xlabel('X Disk')
    plt.ylabel('Y Disk')
    cb = plt.colorbar()
    cb.set_label('$V_{Circular}$ (km/s)',fontsize = 16) 
    plt.show()
    
    # Plotting the LOS velocity 
    plt.figure(3)
    plt.imshow(vlostotal,cmap='viridis',origin='lower')
    plt.xlabel('X Disk')
    plt.ylabel('Y Disk')
    cb = plt.colorbar()
    cb.set_label('$V_{LOS}$ (km/s)',fontsize = 16)  
    plt.show()
    
    # Determine the corresponding frequency centroid and frequency width at each spatial position
    cs = float(parameters['cs'])
    f_0 = float(parameters['restfreq'])/1e9
    f_obs = (f_0/(1+z))*(1-(vlostotal/cs))
    df_obs = (f_0/(1+z))*(sigmaturb/cs)
    
    # Create a velocity range based on the optical definition of radial velocity 
    # c(f-f_0)/f
    v_range = cs*(f_range-f_0)/(f_range)
    
    # Model the PSF as an elliptical gaussian with a mean = 0 and standard deviation proportional to the 
    # FWHM of the synthesized beam 
  
    # PSF generated by J. Cohn's make_beam Python function
    PSF_sub = make_beam(gridsize,res,amp,x0,y0,x_std,y_std,rot)
    
    
    # Create line profiles from the f_centroid and f_width arrays
    # Sigma is the dispersion, which is assumed to be flat for now. 
    # Centroid velocity is the systemic velocity of the galaxy
    
    delta_f = f_spacing/1e9
    
    # Create an integrated gaussian line profile following the methodology of B. Boizelle
    # The integrated line profile will be weighted by the deconvolved flux map in the following step
    glineflux = np.swapaxes(np.swapaxes(-0.5*np.array([-scipy.special.erf((i-(delta_f/2)-f_obs)/(np.sqrt(2)*df_obs))+scipy.special.erf(((i+(delta_f/2)-f_obs)/(np.sqrt(2)*df_obs))) for i in f_range]),2,0),0,1)
    glineflux[glineflux < 1e-5*np.max(glineflux)] = 0
    end_lineprofile = time.time()
    print('The time to construct a model line profile in seconds is',end_lineprofile-start)
    
    
    # Save the unweighted integrated Gaussian line profile 
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(glineflux,0,2),1,2))
    hdu.writeto('GLINEFLUX_UNWEIGHTED.fits',overwrite=True)
    
    # Upscale the flux map by (ssf x ssf) 
    # Deconvolve the flux map with the Richardson-Lucy algorithm 
    # It is important that the flux map has been made to be strictly-non negative before the deconvolution
    
    FluxMap = restoration.richardson_lucy(FluxMap,PSF_sub,iterations=10)
    FluxMap = upsample(FluxMap[yi:yf,xi:xf],ssf)

    # Save the Flux map as a FITS file and plot it to visualize
    hdu=fits.PrimaryHDU(FluxMap)
    hdu.writeto('FluxMap_BeforeWeight.fits',overwrite=True)
    plt.imshow(FluxMap,origin='lower')
    plt.title('Flux Map (Jy/Beam) (Post RL Deconvolution)')
    cb = plt.colorbar()
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    plt.show()
    
    # Weight the line profile by multiply each slice by the flux map
    for i in range(ndz):
        glineflux[:,:,i] = F_0*np.multiply(glineflux[:,:,i],FluxMap[:,:])
    
    # Set any values less than the max of the 
    glineflux[glineflux < np.max(glineflux)*1e-5] = 0
    
    # Plot a collapsed integrated Gaussian line profile to visualize if it was applied correctly.
    plt.imshow(np.sum(glineflux,2),origin='lower')
    cb=plt.colorbar()
    plt.title('Collapsed Normalized Line Profile')
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    plt.show()
    ZZZ = np.sum(glineflux,2)
    hdu=fits.PrimaryHDU(ZZZ)
    hdu.writeto('CollapsedNormalizedLineProfile.fits',overwrite=True)
    
    # Save the now weighted integrated Gaussian line profile as a FITS file
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(glineflux,0,2),1,2))
    hdu.writeto('3258ConvolveTest_FluxWeighted.fits',overwrite=True)
    
    glineflux[glineflux < np.max(glineflux)*1e-6] = 0
    print('The shape of the model line profile array is',glineflux.shape)
      
    #Display the PSF 
    plt.figure(4)
    plt.imshow(PSF_sub,interpolation='none',origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    cb.set_label('PSF',fontsize = 16) 
    plt.show()
    
    # Display the flux map
    plt.figure(5)
    plt.imshow(FluxMap,origin='lower')
    plt.xlabel('x ')
    plt.ylabel('y ')
    cb = plt.colorbar()
    cb.set_label('Flux Map',fontsize = 16)
    plt.show()
        
    # Re-bin the integrated gaussian line profile to the scale of the original ALMA data for convolution efficiency
    # Pre-allocate the array first
    rebinned_glineflux = np.zeros((ndx,ndy,ndz))
    for i in range(ndz):
        rebinned_glineflux[:,:,i] = block_reduce(glineflux[:,:,i],ssf,np.sum)
        rebinned_glineflux[rebinned_glineflux < 1e-6*np.max(rebinned_glineflux)]

    # Convolve the PSF with the integrated Gaussian line profile 
    # First pre-allocate arrays to be filled at both the original scale and at the block-averaged scale.
    convolvetest = np.zeros((ndx,ndy,ndz))
    convolvetest_sub = np.zeros((int(ndx/(rebin)),int(ndy/(rebin)),ndz))
    
    # Define the fitting ellipse that determines where the model optimizations occur
    Gamma = int(parameters['GammaEllipse'])
    semimaj = a
    semimin = q*a
    Gamma=(90.+Gamma)/180.*np.pi
    e = Ellipse2D(amplitude=1., x_0=(xc_ellipse-xi), y_0=(yc_ellipse-yi), a=semimaj, b=semimin,
    theta=Gamma)
    y, x = np.mgrid[0:ndx,0:ndy]

    # Select the regions of the ellipse we want to fit
    # Create a fitting-cube that will contain the regions where the fits will occur
    fitting_ellipse = np.array(e(x,y))
    
    # Plot the elliptical region on the scale of the ALMA data 
    plt.figure(6)              
    plt.imshow(fitting_ellipse,origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.title('Elliptical Fitting Region')
    cb = plt.colorbar()
    plt.show()
    
    ### FIND THE HIGHEST AND LOWEST X AND Y VALUES TO USE FOR THE CONVOLUTION BOX
    ### STORE THE X AND Y POINTS OF THE ELLIPSE IN ARRAYS
    good_ellipse_pixels = np.where(fitting_ellipse == 1)
    y_ellipse = good_ellipse_pixels[0]
    x_ellipse = good_ellipse_pixels[1]
    
    ### FIND THE MAX AND MIN VALUES OF BOTH THE X AND Y ARRAYS 
    x_ellipse_max = np.max(x_ellipse)
    x_ellipse_min = np.min(x_ellipse)
    y_ellipse_max = np.max(x_ellipse)
    y_ellipse_min = np.min(y_ellipse)
    
    ### ADD A BUFFER OF ABOUT 2*FWHM ON BOTH ENDS
    ### USE THESE VALUES TO PROPERLY CHOOSE THE CORRECT VALUES OF THE CUBE TO CONVOLVE
    box_xlo = int(np.rint(x_ellipse_min-(2*PSF_sigma_x)))
    box_xhi = int(np.rint(x_ellipse_max+(2*PSF_sigma_x)))
    box_ylo = int(np.rint(y_ellipse_min-(2*PSF_sigma_y)))
    box_yhi = int(np.rint(y_ellipse_max+(2*PSF_sigma_y)))
    
    ### GENERATE THE CONVOLUTION BOX'S X WIDTH AND Y WIDTH VALUES
    box_x_width = np.abs(x_ellipse_max-x_ellipse_min) + 4*PSF_sigma_x
    box_y_width = np.abs(y_ellipse_max-y_ellipse_min) + 4*PSF_sigma_y
    

    print('The convolution box width in the x-direction in pixels is',box_x_width)
    print('The convolution box width in the y-direction in pixels is',box_y_width)
    
    ### THE REPLACEMENT CONVOLUTION METHOD IS TO CREATE A RECTANGULAR REGION THAT ENCAPUSLATES THE FITTING ELLIPSE
    box_region = Box2D(amplitude=1.,x_0=(xc_ellipse-xi),y_0=(yc_ellipse-yi),x_width=box_x_width,y_width=box_y_width)
    convolve_box = np.array(box_region(x,y))
        
    ### PLOT THE CONVOLUTION BOX ON THE SCALE OF THE ALMA DATA
    plt.figure(7)              
    plt.imshow(convolve_box,origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.title('Convolution Box Region')
    cb = plt.colorbar()
    plt.show()
    
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(rebinned_glineflux[box_ylo:box_yhi,box_xlo:box_xhi,:],0,2),1,2))
    hdu.writeto('3258ConvolveTest_SmallRegion.fits',overwrite=True)
    
    # Switch the x and y coordinates as they are flipped in the sense that 
    # Python interprets axis 0 as the rows (y) and axis 1 as the columns (x)
    for i in list(range(z_i-1,z_f-1)): 
        convolvetest[box_ylo:box_yhi,box_xlo:box_xhi,i] = convolve(rebinned_glineflux[box_ylo:box_yhi,box_xlo:box_xhi,i],PSF_sub,boundary='extend',normalize_kernel=True) 
        convolvetest[convolvetest < 1e-6*np.max(convolvetest)] = 0 
    
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(convolvetest,0,2),1,2))
    hdu.writeto('3258ModelCube_ALMASize.fits',overwrite=True)
    end_convolve = time.time()
    print('The time to perform the convolution in seconds has taken',end_convolve-start)
    # Re-Bin the Model by averaging over ssf x ssf blocks in the spatial domain.
    for i in range(ndz):
        convolvetest_sub[:,:,i] = block_reduce(convolvetest[:,:,i],rebin,func=np.mean)
        
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(convolvetest_sub,0,2),1,2))
    hdu.writeto('3258Convolve_Testnew.fits',overwrite=True)
    model = convolvetest_sub
    
    # Sub-sampling factor
    # Create a smaller fitting ellipse that will be used to perform fit on the sub-sampled scale
    m = int(parameters['rebin'])
    fitting_ellipse_small = np.array(block_reduce(fitting_ellipse,m,np.mean))
    plt.imshow(fitting_ellipse_small,origin='lower')
    plt.title('Smaller Elliptical Region for Chi-Squared Fits')
    plt.show()
    # Create a smaller fit cube
    fit_cube_small = np.ones((int(ndx/m),int(ndy/m),ndz))
    for i in range(ndz):
        if i < (z_i-1):
            fit_cube_small[:,:,i] = 0
        elif i > z_f-1:
            fit_cube_small[:,:,i] = 0
        else: 
            fit_cube_small[:,:,i] = fitting_ellipse_small*fit_cube_small[:,:,i]
            
    ### SAVE THE FIT CUBE REGION
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(fit_cube_small,0,2),1,2))
    hdu.writeto('3258FITCUBE.fits',overwrite=True)

    # Save the Fitting Region Cube (Down-sampled version)
    #hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(fit_cube_small,0,1),0,2))
    #hdu.writeto('Small_Fit_Cube.fits',overwrite=True)
    
    # Select the Fit region to be the regions that only contain a value of 1.
    fittingregion = np.where(fit_cube_small > 0)
    print('The amount of points in the fitting region is',np.size(fittingregion))
    
    x_fit = fittingregion[1]
    y_fit = fittingregion[0]
    
    # Create a residual cube on the scale of the rebinned ALMA data
    residual_cube = model-data_rebin
    
    ### PRINT OUT SLICES OF BOTH THE MODEL AND DATA CUBES FOR A QUICK, QUALITATIVE INSPECTION
    plt.figure(8)              
    plt.imshow(model[:,:,45],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Model Frequency Slice 45')
    plt.show()
    
    plt.figure(9)              
    plt.imshow(data_rebin[:,:,45],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Data Frequency Slice 45')
    plt.show()

    plt.figure(10)              
    plt.imshow(model[:,:,55],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Model Frequency Slice 55')
    plt.show()
    
    plt.figure(11)              
    plt.imshow(data_rebin[:,:,55],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Data Frequency Slice 55')
    plt.show()
 
    plt.figure(12)              
    plt.imshow(model[:,:,65],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Model Frequency Slice 65')
    plt.show()
    
    plt.figure(13)              
    plt.imshow(data_rebin[:,:,65],origin='lower')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    cb = plt.colorbar()
    plt.title('Data Frequency Slice 65')
    plt.show()
    
    # Save the Synthetic Data Cube Before Flattening 
    hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(convolvetest_sub,0,2),1,2))
    hdu.writeto('3258Convolve_Testnew.fits',overwrite=True)
    
    # Identify the fitting regions in the data, model, and noise cube and flatten them to a 1D array
    # This must be done because LMFIT only accepts 1D arrays to perform chi-squared minimization.  
    model_vector =  model[fittingregion].flatten('C')
    print('The number of data points in the chi-squared fit is',np.size(model_vector))
    end_chi = time.time()
    print('The time it has taken to calculate chi in seconds is',end_chi-start)
    
    if data is None:
        return model
    
    data_vector = data_rebin[fittingregion].flatten('C')
    
    if eps is None:
        return data_vector-model_vector
    
    eps_vector = eps[fittingregion].flatten('C')
    
    chi = (data_vector-model_vector)/(eps_vector)
    chi = chi.flatten('C')
    print('The reduced chi-squared value is',np.sum(chi**2)/(np.size(chi)-np.size(freeparams)-2))
    
     # Save all the FITS files with the usage of boolean argument
    if SaveFITS is True:
    # Create FITS (Flexible Image Transport System) files 
    # from scratch to save the centroid frequencies and frequency widths
        hdu=fits.PrimaryHDU(f_obs)
        hdu.writeto('3258fobs_array.fits',overwrite=True)
    
        hdu=fits.PrimaryHDU(df_obs)
        hdu.writeto('3258_dfobs_array.fits',overwrite=True)
    
    # Save the deconvovled flux map as a FITS file for future use
        hdu=fits.PrimaryHDU(FluxMap)
        hdu.writeto('3258DeconvolvedFluxMap.fits',overwrite=True)
    
    # Write the Gaussian Line Profile to a FITS file
        hdu=fits.PrimaryHDU(np.swapaxes(np.swapaxes(glineflux,0,2),0,1))
        hdu.writeto('3258GLineFluxTest.fits',overwrite=True)
    
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
        
    # Save the residual cube to a FITS file.
        hdu = fits.PrimaryHDU(np.swapaxes(residual_cube,0,2))
        hdu.writeto('Residual_Cube.fits',overwrite=True)
     
    return chi