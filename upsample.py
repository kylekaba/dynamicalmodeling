# Upsampling function used to upscale the initial surface brightness map to the correct size 
# Given a subsampling factor, ssf, and an input flux map, the function works as follows:
# If the dimensions of the input flux map is (n x n), the dimensions of the new flux map is (ssf*n x ssf*n)
# Also, the surface brightness in the original flux map is now divided by a factor of ssf^2, so each (s x s) pixel block corresponding to an original ALMA pixel 
# Is weighted equally with surface brightness S/(ssf^2).

def upsample(A,ssf):
    B = np.repeat(A,ssf,axis=0)
    C = np.repeat(B,ssf,axis=1)
    C = C/(ssf**2)
    return C