import numpy as np
from scipy.interpolate import interp1d
from astropy.constants import c  # the speed of light

def auto_bootstrap(func, args, nbootstrap=50):
    results = [func(*args) for _ in range(nbootstrap)]
    results = np.array(results)
    mean = np.nanmean(results, axis=0)
    std = np.nanstd(results, axis=0, ddof=1)
    return mean,std

def get_N_subsamples_len_M(sample, N, M, norepeats=False, suppress_warning=False, seed=None):
    """
    Returns (N,M,*) array containing N subsamples, each of length M.
    We require M < L, where (L,*) is the shape of `sample`.
    If norepeats=True, then we require N*M < L."""
    assert(is_arraylike(sample))
    maxM = len(sample)
    assert(M <= maxM)
    maxN = np.math.factorial(maxM)//(np.math.factorial(M)*np.math.factorial(maxM-M))
    if N is None:
        N = int(maxM // M)
    assert(N <= maxN)
    
    if norepeats:
        if N*M > maxM:
            msg = "Warning: Cannot make %d subsamples without repeats\n" %N
            N = int(maxM/float(M))
            msg += "Making %d subsamples instead" %N
            if not suppress_warning:
                print(msg)
        
        sample = np.asarray(sample)
        newshape = (N,M,*sample.shape[1:])
        newsize = N*M
        if not seed is None: np.random.seed(seed)
        np.random.shuffle(sample)
        if not seed is None: np.random.seed()
        return sample[:newsize].reshape(newshape)
    
    if isinstance(sample, np.ndarray):
        return get_N_subsamples_len_M_numpy(sample, N, M, maxN, maxM, seed)
    else:
        return get_N_subsamples_len_M_list(sample, N, M, maxN, maxM, seed)

def get_N_subsamples_len_M_list(sample, N, M, maxN, maxM, seed=None):
    i = 0
    subsamples = []
    while i < N:
        if not seed is None: np.random.seed(seed)
        subsample = set(np.random.choice(maxM, M, replace=False))
        if not seed is None: np.random.seed()
        if not subsample in subsamples:
            subset = []
            for index in subsample:
                subset.append(sample[index])
            subsamples.append(subset)
            i += 1
    
    return subsamples

def get_N_subsamples_len_M_numpy(sample, N, M, maxN, maxM, seed=None):
    i = 0
    subsamples = np.zeros((N,maxM), dtype=bool)
    while i < N:
        if not seed is None: np.random.seed(seed)
        subsample = np.random.choice(maxM, M, replace=False)
        if not seed is None: np.random.seed()
        subsample = np.bincount(subsample, minlength=maxM).astype(bool)
        if np.any(np.all(subsample[None,:] == subsamples[:i,:], axis=1)):
            continue
        else:
            subsamples[i,:] = subsample
            i += 1
    
    subset = np.tile(sample,(N,*[1]*len(sample.shape)))[subsamples].reshape((N,M,*sample.shape[1:]))
    return subset

def is_arraylike(a):
    return isinstance(a, (tuple, list, np.ndarray))

def angular_separation(ra1, dec1, ra2=0, dec2=0):
    """All angles (ra1, dec1, ra2=0, dec2=0) must be given in radians"""
    cos_theta = np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)
    return np.arccos(cos_theta)

def angle_lim_to_dist(angle, mean_redshift, cosmo, deg=False):
    if deg:
        angle *= np.pi/180.
    angle = min([angle, np.pi/2.])
    z = cosmo.comoving_distance(mean_redshift).value * cosmo.h
    dist = z * 2*np.tan(angle/2.)
    return dist

def redshift_lim_to_dist(delta_redshift, mean_redshift, cosmo):
    redshifts = np.asarray([mean_redshift - delta_redshift/2., mean_redshift + delta_redshift/2.])
    distances = cosmo.comoving_distance(redshifts).value.astype(np.float32) * cosmo.h
    dist = distances[1] - distances[0]
    return dist

def get_random_center(Lbox, fieldshape, numcenters=None, seed=None, pad=None):
    #origin_shift = np.asarray(origin_shift)
    Lbox = np.asarray(Lbox)
    fieldshape = np.asarray(fieldshape)
    if numcenters is None:
        shape = 3
    else:
        shape = (numcenters,3)
    if pad is None:
        pad = np.array([0.,0.,0.])
    else:
        pad = np.asarray(pad)
    
    assert(np.all(2.*pad + fieldshape <= Lbox))
    if not seed is None: np.random.seed(seed)
    xyz_min = np.random.random(shape)*(Lbox - fieldshape - 2.*pad) #- origin_shift
    if not seed is None: np.random.seed()
    center = xyz_min + fieldshape/2. + pad
    return center

def get_random_gridded_center(Lbox, fieldshape, numcenters=None, fieldpad=None, replace=False, seed=None):
    Lbox = np.asarray(Lbox); fieldshape = np.asarray(fieldshape)
    if numcenters is None:
        numcenters = 1
        one_dim = True
        
    centers = grid_centers(Lbox, fieldshape, fieldpad)
    if not replace and numcenters > len(centers):
        raise ValueError("numcenters=%d, but there are only %d possible grid centers" %(numcenters, len(centers)))
    if isinstance(numcenters, str) and numcenters.lower().startswith("all"):
        numcenters = len(centers)

    if not seed is None: np.random.seed(seed)
    which = np.random.choice(np.arange(len(centers)), numcenters, replace=replace)
    if not seed is None: np.random.seed()
    centers = centers[which]
    if one_dim:
        return centers[0]
    else:
        return centers

def grid_centers(Lbox, fieldshape, fieldpad=None):
    if fieldpad is None: fieldpad = np.array([0.,0.,0.])
    nbd = (Lbox / fieldshape).astype(int)
    xcen, ycen, zcen = [np.linspace(0,Lbox[i],nbd[i]+1)[1:] for i in range(3)]
    centers = np.asarray(np.meshgrid(xcen,ycen,zcen)).reshape((3,xcen.size*ycen.size*zcen.size)).T
    centers -= np.asarray([xcen[0],ycen[0],zcen[0]])[np.newaxis,:]/2.
    return centers
    

def sample_fraction(sample, fraction, seed=None):
    assert(0 <= fraction <= 1)
    if hasattr(sample, "__len__"):
        N = len(sample)
        return_indices = False
    else:
        N = sample
        return_indices = True
    Nfrac = int(N*fraction)
    if not seed is None: np.random.seed(seed)
    indices = np.random.choice(N, Nfrac, replace=False)
    if not seed is None: np.random.seed()
    if return_indices:
        return indices
    else:
        return sample[indices]

def kwargs2attributes(obj, kwargs):
    class_name = str(obj.__class__).split("'")[1].split(".")[1]
    if not set(kwargs).issubset(obj.__dict__):
        valid_keys = obj.__dict__.keys()
        bad_keys = set(kwargs) - valid_keys
        raise KeyError("Invalid keys %s in %s creation. Valid keys are: %s"
                                        %(str(bad_keys), class_name, str(valid_keys)))
    
    for key in set(kwargs.keys()):
        if kwargs[key] is None:
            del kwargs[key]
    
    obj.__dict__.update(kwargs)

def rdz2xyz(rdz, cosmo):
    """If cosmo=None then z is assumed to already be distance, not redshift."""
    if cosmo is None:
        dist = rdz[:,2]
    else:
        dist = (cosmo.comoving_distance(rdz[:,2]) * cosmo.h).value.astype(np.float32)
    z = (dist * np.cos(rdz[:,1]) * np.cos(rdz[:,0])).astype(np.float32)
    x = (dist * np.cos(rdz[:,1]) * np.sin(rdz[:,0])).astype(np.float32)
    y = (dist * np.sin(rdz[:,1])).astype(np.float32)
    return np.vstack([x,y,z]).T # in Mpc/h


def distance2redshift(dist, vr, cosmo, zprec=1e-3, h_scaled=True):
    redshift_hard_max = 10.
    if len(dist) == 0:
        return np.array([])
    c_km_s = c.to('km/s').value
    rmin, rmax = dist.min(), dist.max()
    
    # First construct a low resolution grid to check the range of redshifts
    # needed for the high resolution grid
    yy = np.arange(0, redshift_hard_max+.005, 0.1)
    xx = cosmo.comoving_distance(yy).value
    if h_scaled:
        xx *= cosmo.h
    imin,imax = None,None
    for i in range(len(yy)):
        if xx[i] >= rmin:
            imin = max((0, i-1))
            break
    for i in range(len(yy)):
        if xx[-1-i] <= rmax:
            imax = -i
            break
    if imax == 0:
        dmin, dmax = cosmo.comoving_distance([0., redshift_hard_max]).value * cosmo.h
        msg = "You can't observe galaxies at that distance. min/max distance from input array = (%.1f,%.1f) but 0 < z < %.1f is required." %(min(dist), max(dist), redshift_hard_max)
        msg += "This corresponds to a requirement of %.1f < distance < %.1f" %(dmin,dmax)
        msg += "If you want to observe galaxies further away, change the value of `redshift_hard_max`"
        raise ValueError(msg)

    zmin, zmax = yy[imin], yy[imax]

    # compute cosmological redshift and add contribution from peculiar velocity
    num_points = int((zmax-zmin)/zprec) + 1
    yy = np.linspace(zmin, zmax, num_points, dtype=np.float32)
    xx = cosmo.comoving_distance(yy).value.astype(np.float32)
    if h_scaled:
        xx *= cosmo.h
    f = interp1d(xx, yy, kind='cubic')
    z_cos = f(dist).astype(np.float32)
    redshift = z_cos+(vr/c_km_s)*(1.0+z_cos)
    return redshift

def ra_dec_z(xyz, vel=None, cosmo=None, zprec=1e-3):
    """
    Convert position array `xyz` and velocity array `vel` (optional), each of shape (N,3), into ra/dec/redshift array, using the (ra,dec) convention of :math:`\\hat{z} \\rightarrow  ({\\rm ra}=0,{\\rm dec}=0)`, :math:`\\hat{x} \\rightarrow ({\\rm ra}=\\frac{\\pi}{2},{\\rm dec}=0)`, :math:`\\hat{y} \\rightarrow ({\\rm ra}=0,{\\rm dec}=\\frac{\\pi}{2})`
    """
    if not cosmo is None:
    # remove h scaling from position so we can use the cosmo object
        xyz = xyz/cosmo.h

    # comoving distance from observer (at origin)
    r = np.sqrt(xyz[:, 0]**2+xyz[:, 1]**2+xyz[:, 2]**2)
    r_cyl_eq = np.sqrt(xyz[:, 2]**2 + xyz[:, 0]**2)

    # radial velocity from observer (at origin)
    if vel is None:
        vr = np.zeros(xyz.shape[0])
    else:
        r_safe = r.copy()
        r_safe[r_safe==0] = 1.
        vr = np.sum(xyz*vel, axis=1)/r_safe

    if cosmo is None:
        redshift = r
    else:
        redshift = distance2redshift(r, vr, cosmo, zprec, h_scaled=False)

    # calculate spherical coordinates
    # theta = np.arccos(xyz[:, 2]/r) # <--- causes large roundoff error near (x,y) = (0,0)
    # theta = np.arctan2(r_cyl, xyz[:, 2])
    # phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    
    # make ra,dec = 0,0 in the z direction
    dec = np.arctan2(xyz[:, 1], r_cyl_eq)
    ra = np.arctan2(xyz[:, 0], xyz[:, 2])

    # convert spherical coordinates into ra,dec
    # ra = phi
    # dec = np.pi/2.0 - theta

    return np.vstack([ra, dec, redshift]).T.astype(np.float32)

def update_table(table, coldict):
    for key in coldict:
        table[key] = coldict[key]

def xyz_array(struc_array, keys=['x', 'y', 'z'], attributes=False):
    if attributes:
        struc_array = struc_array.__dict__

    x = struc_array[keys[0]]
    y = struc_array[keys[1]]
    z = struc_array[keys[2]]
    return np.vstack([x, y, z]).T

def logN(data, rands, njackknife=None, volume_factor=1):
    if njackknife is None:
        jackknife_factor = 1
    else:
        n = np.product(njackknife)
        jackknife_factor = n/float(n-1)
    return np.log(volume_factor*jackknife_factor*len(data))

def ln_density(data, rands, volume, njackknife=None):
    if njackknife is None:
        jackknife_factor = 1
    else:
        n = np.product(njackknife)
        jackknife_factor = n/float(n-1)
    return np.log(jackknife_factor * len(data) / float(volume))

# def logN(data, rands, njackknife=None, volume_factor=1):
#         return logN_box(data, njackknife, volume_factor)
# logN.__doc__ = "**Only use this function if `rands` argument is required. Else use logN_box**\n" + logN_box.__doc__

def rand_rdz(N, ralim, declim, zlim, seed=None):
    """
Returns an array of shape (N,3) with columns ra,dec,z (z is treated as distance) within the specified limits such that the selected points are chosen randomly over a uniform distribution
    """
    N = int(N)
    if not seed is None: np.random.seed(seed)
    ans = np.random.random((N,3))
    if not seed is None: np.random.seed()
    ans[:,0] = (ralim[1] - ralim[0]) * ans[:,0] + ralim[0]
    ans[:,1] = np.arcsin( (np.sin(declim[1]) - np.sin(declim[0])) * ans[:,1] + np.sin(declim[0]) )
    ans[:,2] = ( (zlim[1]**3 - zlim[0]**3) * ans[:,2] + zlim[0]**3 )**(1./3.)
    
    return ans

def volume_rdz(ralim, declim, zlim, cosmo=None):
    """
    Returns the volume of a chunk of a sphere of given limits in ra, dec, and z
    z is interpreted as distance unless cosmo is given
    """
    if not cosmo is None:
        zlim = cosmo.comoving_distance(zlim).value * cosmo.h
    return 2./3. * (ralim[1]-ralim[0]) * np.sin((declim[1]-declim[0])/2.) * (zlim[1]**3-zlim[0]**3)

def rdz_distance(rdz, rdz_prime, cosmo=None):
    """
    rdz is an array of coordinates of shape (N,3) while rdz_prime is a single point of comparison of shape (3,)
    z is interpreted as radial distance unless cosmo is given
    
    Returns the distance of each rdz coordinate from rdz_prime
    """
    if not cosmo is None:
        rdz[:,2] = (cosmo.comoving_distance(rdz[:,2]).value * cosmo.h).astype(np.float32)
    r,d,z = rdz.T
    rp,dp,zp = rdz_prime
    return np.sqrt( r**2 + rp**2 - 2.*r*rp*(np.cos(d)*np.cos(dp)*np.cos(r-rp) + np.sin(d)*np.sin(dp)) )

def xyz_distance(xyz, xyz_prime):
    x,y,z = xyz.T
    xp,yp,zp = xyz_prime
    return np.sqrt((x-xp)**2 + (y-yp)**2 + (z-zp)**2)

def unit_vector(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return (x,y,z)

def make_npoly(radius, n):
    import spherical_geometry.polygon as spoly
    phi1 = 2.*np.pi/float(n)
    points = [unit_vector(radius, phi1*i) for i in range(n)]
    inside = (0.,0.,1.)
    return spoly.SingleSphericalPolygon(points, inside)