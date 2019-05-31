# Creating a New ALMA Noise Modeling Code With Only the .pbcor and Primary beam cube 

# data = the .pbcor data cube, typically seen as an image.image.fits file.
# PB = the primary beam data cube, typically seen as an image.flux.fits file.
# (x0,y0) is the spatial position of the center of a rectangular box that will be used to get a measure
# of the standard deviation of the noise in a given data slice.
# x_range and y_range are the width of the x and y directions of the box.

# For example, for NGC 1380:
# Define the rectangular region that we will determine the noise from in the non-pb.cor image.
#  x:320-420, y:400:600
# The given values would be:
#x0, y0 = 370,500
#x_range = 100
#y_range = 100

def modelnoise(pb_data,PB,xc,yc,x_range,y_range,s):
    # Define the region we wish to take the noise measurement on the full-sized, flat-noise data cube. 
    # It is best if this region is at least (50x50 pixels)
    # This region will then be downsampled by the factor is s
    dx,dy,dz = np.size(pb_data,0),np.size(pb_data,1),np.size(pb_data,2)
    
    box = Box2D(amplitude=1.,x_0=xc,y_0=yc,x_width=x_range,y_width=y_range)
    y,x = np.mgrid[0:dx,0:dy]
    fitting_box = box(x,y)
    region = np.where(fitting_box != 0)
    x_reg = region[0]
    y_reg = region[1]
    
    # Sub-sampling factor
    box_reduce = np.zeros((len(x_reg),len(y_reg)))
    box_reduce = block_reduce(fitting_box,s,np.mean)
   
    reducedregion = np.where(box_reduce != 0)
    x_reduced = reducedregion[0]
    y_reduced = reducedregion[1]
    
    # Create the flat uniform data cube by multiplying each slice of the .pbcor cube with each slice of the
    # primary beam cube
    flatcube = np.zeros((dx,dy,dz))   
    for i in range(dz):
        flatcube[:,:,i] = pb_data[:,:,i]*PB[:,:,i]
          
    # Take the Flat-Free, Full-Resolution Data Cube and Block Average it by the given factor s.
    rebinned_data = np.zeros((int(dx/s),int(dy/s),int(dz)))
    rebinned_pbcor_data = np.zeros((int(dx/s),int(dy/s),int(dz)))
    
    for i in range(dz):
        rebinned_data[:,:,i] = block_reduce(flatcube[:,:,i],s,np.mean)
        rebinned_pbcor_data[:,:,i] = block_reduce(pb_data[:,:,i],s,np.mean)
    
    # Pre-Allocate Noise Slices and Cubes
    # noise_slice is a noise vector that contains the RMS for the i-th channel in the cube
    # Flat_noise_cube is a cube initially populated with 1s that will all be multiplied by noise_slice, which 
    # generates a slice full of the value of noise_slice[i] for spatial pixel in a given frequency slice.
    # model_noise_cube will then be Flat_noise_cube[:,:,i]/PB_downsample[:,:,i]
    # model noise cube pbcor is taking the noise measurement from the pbcor data cube itself directly
    noise_slice = np.zeros(dz)
    noise_slice_pbcor = np.zeros(dz)
    flat_noise_cube = np.ones((int(dx/s),int(dy/s),int(dz)))
    model_noise_cube_pbcor = np.ones((int(dx/s),int(dy/s),int(dz)))
    model_noise_cube = np.zeros((int(dx/s),int(dy/s),int(dz)))

    
    #Down sample the Primary Beam Cube to Match the Resolution of the Down-sampled Data
    PB_downsample = np.zeros((int(dx/s),int(dy/s),int(dz)))
    for i in range(dz):
        PB_downsample[:,:,i] = block_reduce(PB[:,:,i],s,np.mean)
    
    for i in range(dz):
        noise_slice[i] = np.std(rebinned_data[x_reduced,y_reduced,i])
        noise_slice_pbcor[i] = np.std(rebinned_pbcor_data[x_reduced,y_reduced,i])
        model_noise_cube_pbcor[:,:,i] = noise_slice_pbcor[i]*model_noise_cube_pbcor[:,:,i]
        flat_noise_cube[:,:,i] = noise_slice[i]*flat_noise_cube[:,:,i]
        model_noise_cube[:,:,i] = flat_noise_cube[:,:,i]/PB_downsample[:,:,i]  
    
    noise_3258test_PBCOR = fits.PrimaryHDU(np.swapaxes(model_noise_cube_pbcor,0,2))
    noise_3258test_PBCOR.writeto('NGC3258_Noise_Test_PBCOR_UNIFORM_ONSLICE.fits',overwrite=True)
        
    return model_noise_cube