# data_cube is the input ALMA data cube, typically the .pbcor file
# xlo, xhi, ylo, yhi are the initial and final points of the rectangular box in the spatial region that will be 
# used to calculate the noise in the flux, which will be used for the signal-to-noise (SNR) map
def CreateFluxMap(data_cube,xlo,xhi,ylo,yhi,zlo,zhi,threshold):
    
    # Open and extract the data from the input data cube 
    hdu = fits.open(data_cube)
    hdu.info()
    data = hdu[0].data

    # Pick out the 3 dimensions of the data cube and swap axes to have the axes ordered 
    # as x,y,z
    data = data[0,:,:,:]
    # At this point, the order of the axes is: (z,x,y). 
    # We have to interchange z with y to get (y,x,z), and then switch x and y to get it in regular (x,y,z) -> (RA,dec,freq)
    data = np.swapaxes(np.swapaxes(data,0,2),0,1)
    
    # Dimensions of each axis in the data cube 
    dx = int(np.size(data,0))
    dy = int(np.size(data,1))
    dz = int(np.size(data,2))
    
    # Extract the beam size from the header values
    # Import the beam major axis and minor axis, which is given in degrees
    bmaj = hdu[0].header['BMAJ']*3600
    bmin = hdu[0].header['BMIN']*3600
    
    # Convert the FWHMs into Gaussian dispersions
    sigma_x = bmaj/2.3548
    sigma_y = bmin/2.3548
    
    print('The major and minor axes of the beam FWHM in arcseconds are',(bmaj,bmin))
    # Import arcsecond to pixel ratio
    print('The area of the beam in square arcseconds is',bmaj*bmin*np.pi/(4*np.log(2)))
    arcsec2pix = (hdu[0].header['CDELT2']*3600)
    beamarea = bmaj*bmin*np.pi/(4*np.log(2))
    
    # The amount of pixels per beam is given by the ratio of the beam area to the ratio of arcseconds per pixel
    print('The amount of pixels per beam is', beamarea/arcsec2pix**2)
    pix2arcsec = 1/(arcsec2pix)
    pix2beam = beamarea/arcsec2pix**2
    
    # Create a frequency axis 
    f_step = float(hdu[0].header['CDELT3'])  # frequency step in the data cube
    restfreq = hdu[0].header['RESTFRQ']/1e9
    f_init = hdu[0].header['CRVAL3']
    f_spacing = hdu[0].header['CDELT3']
    f_final  = (f_init+(dz*f_spacing))
    f_range = np.arange(f_init,f_final,f_spacing)/1e9
    
    # Divdie by 1e9 to get this in GHz
    f_spacing = f_spacing/1e9
    
    # Create the iniital flux map by summing over the frequency direction, using only the channels which contain emission
    BadFluxMap = np.sum(data[:,:,zlo:zhi],2)
    
    # Determine the noise level in the flux map
    noise = np.std(BadFluxMap[xlo:xhi,ylo:yhi])
    
    # Create a signal-to-noise map by dividing "FluxMap" by "noise"
    SNRMap = BadFluxMap/noise

    # Determine the "good pixels" to be the ones above the input "threshold"
    goodpixels = np.where(SNRMap >= threshold)
    mask = np.zeros((dx,dy))
    mask[goodpixels] = 1
    
    # Create a new flux map that collapses the data cube once more, but only including the spatial pixels that
    # are considered "good pixels" and the frequency channels that only contain emission.
    FluxMap_Cube = np.zeros((dx,dy,dz))
    
    # Properly integrate the flux in the channel by multiplying the frequency width
    for i in range(zlo-1,zhi-1):
        FluxMap_Cube[:,:,i] = mask*data[:,:,i]*np.abs(f_spacing)
        
    FluxMap = np.sum(FluxMap_Cube,2)/np.abs(f_spacing)
    
    
    # Set the values that are equal to 0 to 1e-9 to enable deconvolution with the Richardson-Lucy algorithm
    # as it requires the values be non-negative
    # Once the deconvoution is over, set the values back to 0. 
    FluxMap[FluxMap == 0] = 1e-9
    
    plt.figure(1)
    plt.imshow(FluxMap,origin='lower')
    plt.title('Flux Map (Jy/beam)')
    cb=plt.colorbar()
    
    plt.figure(2)
    plt.imshow(SNRMap,origin='lower')
    plt.title('Signal to Noise Map')
    cb=plt.colorbar()
    plt.show()
    
    plt.figure(3)
    plt.imshow(mask,origin='lower')
    plt.title('Masked Region')
    cb=plt.colorbar()
    
    return FluxMap,SNRMap,mask