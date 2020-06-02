import sys
import os
import shutil
import collections
import warnings
from contextlib import contextmanager
import wget
import numpy as np
import scipy.special as spec
from scipy.interpolate import interp1d
from astropy.constants import c  # the speed of light
from astropy import cosmology
from astropy import units
import halotools.utils as ht_utils

from . import mocksurvey as ms


@contextmanager
def temp_seed(seed):
    if seed is None:
        # Do NOT alter random state
        do_alter = False
    elif isinstance(seed, int):
        # New state with specified seed
        do_alter = True
    else:
        # New state with random seed
        assert isinstance(seed, str) and seed.lower() == "none", \
            f"seed={seed} not understood. Must be int, None, or 'none'"
        do_alter = True
        seed = None

    state = np.random.get_state()
    if do_alter:
        np.random.seed(seed)
    try:
        yield
    finally:
        if do_alter:
            # noinspection PyTypeChecker
            np.random.set_state(state)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def explicit_path(path: str,
                  assert_dir=False, assert_file=False) -> bool:
    ans = path.startswith((os.path.sep,
                           os.path.curdir,
                           os.path.pardir))
    if ans and assert_dir and not os.path.isdir(path):
        raise NotADirectoryError(f"Explicit path {path} must "
                                 f"already exist as a directory.")
    if ans and assert_file and not os.path.isfile(path):
        raise FileNotFoundError(f"Explicit path {path} must "
                                f"already exist as a file.")
    return ans


def selector_from_meta(meta: dict):
    selector = ms.LightConeSelector(
        meta["z_low"], meta["z_high"],
        meta["x_arcmin"] * meta["y_arcmin"] / 3600,
        fieldshape="square", realspace=True)
    for key in meta.keys():
        if key.startswith("selector_"):
            selector &= eval(
                meta[key].replace(" km / (Mpc s),", ",")
                .replace(" K,", ",")
                .replace("LightConeSelector", "ms.LightConeSelector")
                .replace("FlatLambdaCDM", "cosmology.FlatLambdaCDM"))
    return selector


def make_struc_array(dtypes, name_val_dict):
    names, vals = list(name_val_dict.keys()), list(name_val_dict.values())

    if vals:
        lengths = [len(val) for val in vals]
        length = lengths[0]
        assert np.all([l == length for l in lengths]), \
            f"Arrays must be same length: names={names}, lengths={lengths}"
    else:
        length = 0

    dtype = [(n, d) for n, d in zip(names, dtypes)]
    ans = np.zeros(length, dtype=dtype)
    for n, v in zip(names, vals):
        ans[n] = v

    return ans


def lightcone_array(id=None, upid=None, x=None, y=None, z=None, obs_sm=None,
                    obs_sfr=None, redshift=None, ra=None, dec=None, **kwargs):
    dtypes = []
    array = dict(id=id, upid=upid, x=x, y=y, z=z, obs_sm=obs_sm,
                 obs_sfr=obs_sfr, redshift=redshift, ra=ra, dec=dec, **kwargs)

    for name in list(array.keys()):
        if array[name] is None:
            del array[name]
        elif name in ("id", "upid"):
            dtypes.append("<i8")
        else:
            dtypes.append("<f4")

    return make_struc_array(dtypes, array)


def apply_over_window(func, a, window, axis=-1, edge_case=None, **kwargs):
    """
    `func` must be a numpy-friendly function which accepts
    an array as a positional argument and utilizes
    an `axis` keyword argument
    
    This function is just a wrapper for rolling_window,
    and is essentially implemented by the following code:
    
    >>> def apply_over_window(func, a, window, **kwargs):
    >>>     return func(rolling_window(a, window), axis=-1, **kwargs)
    
    See rolling_window docstring for more info
    """
    return func(rolling_window(a, window, axis=axis, edge_case=edge_case),
                axis=-1, **kwargs)


def rolling_window(a, window, axis=-1, edge_case=None):
    """
    Append a new axis of length `window` on `a` over which to roll over
    the specified `axis` of `a`. The new window axis is always the LAST
    axis in the returned array.
    
    WARNING: Memory usage for this function is O(a.size * window)
    
    Parameters
    ----------
    
    a : array-like
        Input array over which to add a new axis containing windows
    
    window : int
        Number of elements in each window
    
    axis : int (default = -1)
        The specified axis with which windows are drawn from
    
    edge_case : str (default = "replace")
        We need to choose a scheme of how to deal with the ``window - 1``
        windows which contain entries beyond the edge of our specified axis.
        The following options are supported:
        
        edge_case = None | "replace"
            Windows containing entries beyond the edges will still exist, but
            those entries will be replaced with the edge entry. The nth axis
            element will be positioned in the ``window//2``th element of the 
            nth window

        edge_case = "wrap"
            Windows containing entries beyond the edges will still exist, and
            those entries will wrap around the axis. The nth axis element will
            be positioned in the ``window//2``th element of the nth window

        edge_case = "contract"
            Windows containing entries beyond the edges will be removed 
            entirely (e.g., if ``a.shape = (10, 10, 10)``, ``window = 4``,
            and ``axis = 1`` then the output array will have shape 
            ``(10, 7, 10, 4)`` because the specified axis will be reduced by
            a length of ``window-1``). The nth axis element will be positioned
            in the 0th element of the nth window.
    """
    # Input sanitization
    a = np.asarray(a)
    window = int(window)
    axis = int(axis)
    ndim = len(a.shape)
    axis = axis + ndim if axis < 0 else axis
    assert -1 < axis < ndim, "Invalid value for `axis`"
    assert 1 < window < a.shape[axis], "Invalid value for `window`"
    assert edge_case in [None, "replace", "contract", "wrap"], \
        "Invalid value for `edge_case`"

    # Convenience function 'onaxes' maps smaller-dimensional arrays 
    # along the desired axes of dimension of the output array
    def onaxes(*axes):
        return tuple(slice(None) if i in axes else None for i in range(ndim + 1))

    # Repeat the input array `window` times, adding a new axis at the end
    rep = np.repeat(a[..., None], window, axis=-1)

    # Create `window`-lengthed index arrays that increase by one 
    # for each window (i.e., rolling indices)
    ind = np.repeat(np.arange(a.shape[axis])[:, None], window, axis=-1
                    )[onaxes(axis, ndim)]
    ind += np.arange(window)[onaxes(ndim)]

    # Handle the edge cases
    if (edge_case is None) or (edge_case == "replace"):
        ind -= window // 2
        ind[ind < 0] = 0
        ind[ind >= a.shape[axis]] = a.shape[axis] - 1
    elif edge_case == "wrap":
        ind -= window // 2
        ind %= a.shape[axis]
    elif edge_case == "contract":
        ind = ind[tuple(slice(1 - window) if i == axis else slice(None)
                        for i in range(ndim + 1))]

    # Select the output array using our array of rolling indices `ind`
    selection = tuple(ind if i == axis else np.arange(rep.shape[i])[onaxes(i)]
                      for i in range(ndim + 1))
    return rep[selection]


def unbiased_std_factor(n):
    """
    Returns 1/c4(n)
    """
    if is_arraylike(n):
        wh = n < 343
        ans = np.ones(np.shape(n)) * (4. * n - 3.) / (4. * n - 4.)
        n = n[wh]
        ans[wh] = spec.gamma((n - 1) / 2) / np.sqrt(2 / (n - 1)) / spec.gamma(n / 2)

        return ans
    else:
        ans = spec.gamma((n - 1) / 2) / np.sqrt(2 / (n - 1)) / spec.gamma(n / 2)
        return ans if n < 343 else (4. * n - 3.) / (4. * n - 4.)


def auto_bootstrap(func, args, nbootstrap=50):
    results = [func(*args) for _ in range(nbootstrap)]
    results = np.array(results)
    mean = np.nanmean(results, axis=0)
    std = np.nanstd(results, axis=0, ddof=1)
    return mean, std


def get_N_subsamples_len_M(sample, N, M, norepeats=False, suppress_warning=False, seed=None):
    """
    Returns (N,M,*) array containing N subsamples, each of length M.
    We require M < L, where (L,*) is the shape of `sample`.
    If norepeats=True, then we require N*M < L."""
    with temp_seed(seed):
        assert (is_arraylike(sample))
        maxM = len(sample)
        assert (M <= maxM)
        maxN = np.math.factorial(maxM) // (np.math.factorial(M) * np.math.factorial(maxM - M))
        if N is None:
            N = int(maxM // M)
        assert (N <= maxN)

        if norepeats:
            if N * M > maxM:
                msg = "Warning: Cannot make %d subsamples without repeats\n" % N
                N = int(maxM / float(M))
                msg += "Making %d subsamples instead" % N
                if not suppress_warning:
                    print(msg)

            sample = np.asarray(sample)
            newshape = (N, M, *sample.shape[1:])
            newsize = N * M
            np.random.shuffle(sample)
            return sample[:newsize].reshape(newshape)

        if isinstance(sample, np.ndarray):
            return get_N_subsamples_len_M_numpy(sample, N, M, maxM, seed)
        else:
            return get_N_subsamples_len_M_list(sample, N, M, maxM, seed)


def get_N_subsamples_len_M_list(sample, N, M, maxM, seed=None):
    with temp_seed(seed):
        i = 0
        subsamples = []
        while i < N:
            subsample = set(np.random.choice(maxM, M, replace=False))
            if not subsample in subsamples:
                subset = []
                for index in subsample:
                    subset.append(sample[index])
                subsamples.append(subset)
                i += 1

        return subsamples


def get_N_subsamples_len_M_numpy(sample, N, M, maxM, seed=None):
    with temp_seed(seed):
        i = 0
        subsamples = np.zeros((N, maxM), dtype=bool)
        while i < N:
            subsample = np.random.choice(maxM, M, replace=False)
            subsample = np.bincount(subsample, minlength=maxM).astype(bool)
            if np.any(np.all(subsample[None, :] == subsamples[:i, :], axis=1)):
                continue
            else:
                subsamples[i, :] = subsample
                i += 1

        subset = np.tile(sample, (N, *[1] * len(sample.shape)))[subsamples].reshape((N, M, *sample.shape[1:]))
        return subset


def is_arraylike(a):
    # return isinstance(a, (tuple, list, np.ndarray))
    return isinstance(a, collections.abc.Container
                      ) and not isinstance(a, str)


def angular_separation(ra1, dec1, ra2=0, dec2=0):
    """All angles (ra1, dec1, ra2=0, dec2=0) must be given in radians"""
    cos_theta = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.arccos(cos_theta)


def angle_lim_to_dist(angle, mean_redshift, cosmo, deg=False):
    if deg:
        angle *= np.pi / 180.
    angle = min([angle, np.pi / 2.])
    z = comoving_disth(mean_redshift, cosmo)
    dist = z * 2 * np.tan(angle / 2.)
    return dist


def redshift_lim_to_dist(delta_redshift, mean_redshift, cosmo):
    redshifts = np.asarray([mean_redshift - delta_redshift / 2., mean_redshift + delta_redshift / 2.])
    distances = comoving_disth(redshifts, cosmo)
    dist = distances[1] - distances[0]
    return dist


def get_random_center(Lbox, fieldshape, numcenters=None, seed=None, pad=30):
    with temp_seed(seed):
        # origin_shift = np.asarray(origin_shift)
        Lbox = np.asarray(Lbox)
        fieldshape = np.asarray(fieldshape)
        if numcenters is None:
            shape = 3
        else:
            shape = (numcenters, 3)
        if pad is None:
            pad = np.array([0., 0., 0.])
        else:
            pad = np.asarray(pad)

        assert (np.all(2. * pad + fieldshape <= Lbox))
        xyz_min = np.random.random(shape) * (Lbox - fieldshape - 2. * pad)  # - origin_shift
        center = xyz_min + fieldshape / 2. + pad
        return center


def get_random_gridded_center(Lbox, fieldshape, numcenters=None, pad=None, replace=False, seed=None):
    with temp_seed(seed):
        Lbox = np.asarray(Lbox)
        fieldshape = np.asarray(fieldshape)
        if numcenters is None:
            numcenters = 1
            one_dim = True
        else:
            one_dim = False

        centers = grid_centers(Lbox, fieldshape, pad)
        if not replace and numcenters > len(centers):
            raise ValueError("numcenters=%d, but there are only %d possible grid centers" % (numcenters, len(centers)))
        if isinstance(numcenters, str) and numcenters.lower().startswith("all"):
            numcenters = len(centers)

        which = np.random.choice(np.arange(len(centers)), numcenters, replace=replace)
        centers = centers[which]
        if one_dim:
            return centers[0]
        else:
            return centers


def grid_centers(Lbox, fieldshape, pad=None):
    if pad is None: pad = np.array([0., 0., 0.])
    nbd = (Lbox / (np.asarray(fieldshape) + 2 * pad)).astype(int)
    xcen, ycen, zcen = [np.linspace(0, Lbox[i], nbd[i] + 1)[1:] for i in range(3)]
    centers = np.asarray(np.meshgrid(xcen, ycen, zcen)).reshape((3, xcen.size * ycen.size * zcen.size)).T
    centers -= np.asarray([xcen[0], ycen[0], zcen[0]])[np.newaxis, :] / 2.
    return centers


def sample_fraction(sample, fraction, seed=None):
    with temp_seed(seed):
        assert (0 <= fraction <= 1)
        if hasattr(sample, "__len__"):
            N = len(sample)
            return_indices = False
        else:
            N = sample
            return_indices = True
        Nfrac = int(N * fraction)
        indices = np.random.choice(N, Nfrac, replace=False)
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
                       % (str(bad_keys), class_name, str(valid_keys)))

    for key in set(kwargs.keys()):
        if kwargs[key] is None:
            del kwargs[key]

    obj.__dict__.update(kwargs)


def rdz2xyz(rdz, cosmo, use_um_convention=False):
    """If cosmo=None then z is assumed to already be distance, not redshift."""
    if cosmo is None:
        dist = rdz[:, 2]
    else:
        dist = comoving_disth(rdz[:, 2], cosmo)

    z = (dist * np.cos(rdz[:, 1]) * np.cos(rdz[:, 0])).astype(np.float32)
    x = (dist * np.cos(rdz[:, 1]) * np.sin(rdz[:, 0])).astype(np.float32)
    y = (dist * np.sin(rdz[:, 1])).astype(np.float32)
    xyz = np.vstack([x, y, z]).T
    if use_um_convention:
        xyz = xyz_convention_ms2um(xyz)
    return xyz  # in Mpc/h


def xyz_convention_ms2um(xyz):
    # Convert an xyz array from mocksurvey convention (left-handed; z=los)
    # to UniverseMachine convention (right-handed; x=los)
    xyz = xyz[:, [2, 0, 1]]
    xyz[:, 1] *= -1
    return xyz


def comoving_disth(redshifts, cosmo):
    z = np.asarray(redshifts)
    dist = cosmo.comoving_distance(z).value * cosmo.h
    return (dist.astype(z.dtype)
            if is_arraylike(dist) else dist)


def distance2redshift(dist, vr, cosmo, zprec=1e-3, h_scaled=True):
    if len(dist) == 0:
        return np.array([])

    c_km_s = c.to('km/s').value
    dist_units = units.Mpc / cosmo.h if h_scaled else units.Mpc
    zmin, zmax = [cosmology.z_at_value(cosmo.comoving_distance,
                                       f(dist) * dist_units) for f in [min, max]]
    zmin = (zmin + 1) * .99 - 1
    zmax = (zmax + 1) * 1.01 - 1

    # compute cosmological redshift + doppler shift
    num_points = int((zmax - zmin) / zprec) + 1
    yy = np.linspace(zmin, zmax, num_points, dtype=np.float32)
    xx = cosmo.comoving_distance(yy).value.astype(np.float32)
    if h_scaled:
        xx *= cosmo.h
    f = interp1d(xx, yy, kind='cubic')
    z_cos = f(dist).astype(np.float32)

    # Add velocity distortion
    if vr is None:
        redshift = z_cos
    else:
        redshift = z_cos + (vr / c_km_s) * (1.0 + z_cos)
    return redshift


def ra_dec_z(xyz, vel=None, cosmo=None, zprec=1e-3):
    """
    Convert position array `xyz` and velocity array `vel` (optional), each of shape (N,3), into ra/dec/redshift array, using the (ra,dec) convention of :math:`\\hat{z} \\rightarrow  ({\\rm ra}=0,{\\rm dec}=0)`, :math:`\\hat{x} \\rightarrow ({\\rm ra}=\\frac{\\pi}{2},{\\rm dec}=0)`, :math:`\\hat{y} \\rightarrow ({\\rm ra}=0,{\\rm dec}=\\frac{\\pi}{2})`
    """
    if not cosmo is None:
        # remove h scaling from position so we can use the cosmo object
        xyz = xyz / cosmo.h

    # comoving distance from observer (at origin)
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
    r_cyl_eq = np.sqrt(xyz[:, 2] ** 2 + xyz[:, 0] ** 2)

    # radial velocity from observer (at origin)
    if vel is None:
        vr = np.zeros(xyz.shape[0])
    else:
        r_safe = r.copy()
        r_safe[r_safe == 0] = 1.
        vr = np.sum(xyz * vel, axis=1) / r_safe

    if cosmo is None:
        redshift = r
    else:
        redshift = distance2redshift(r, vr, cosmo, zprec, h_scaled=False)

    # calculate spherical coordinates
    # theta = np.arccos(xyz[:, 2]/r) # <--- causes large round off error near (x,y) = (0,0)
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


def xyz_array(struc_array, keys=None, attributes=False):
    if keys is None:
        keys = ['x', 'y', 'z']
    if attributes:
        struc_array = struc_array.__dict__

    return np.vstack([struc_array[key] for key in keys]).T


def logN(data, rands, njackknife=None, volume_factor=1):
    del rands
    if njackknife is None:
        jackknife_factor = 1
    else:
        n = np.product(njackknife)
        jackknife_factor = n / float(n - 1)
    return np.log(volume_factor * jackknife_factor * len(data))


def logn(data, rands, volume, njackknife=None):
    del rands
    if njackknife is None:
        jackknife_factor = 1
    else:
        n = np.product(njackknife)
        jackknife_factor = n / float(n - 1)
    return np.log(jackknife_factor * len(data) / float(volume))


# def logN(data, rands, njackknife=None, volume_factor=1):
#         return logN_box(data, njackknife, volume_factor)
# logN.__doc__ = "**Only use this function if `rands` argument is required. Else use logN_box**\n" + logN_box.__doc__

def rand_rdz(N, ralim, declim, zlim, seed=None):
    """
Returns an array of shape (N,3) with columns ra,dec,z
(z is treated as distance) within the specified limits
such that the selected points are chosen randomly over a uniform distribution
    """
    with temp_seed(seed):
        N = int(N)
        ans = np.random.random((N, 3))
        ans[:, 0] = (ralim[1] - ralim[0]) * ans[:, 0] + ralim[0]
        ans[:, 1] = np.arcsin((np.sin(declim[1]) - np.sin(declim[0])) * ans[:, 1] + np.sin(declim[0]))
        ans[:, 2] = ((zlim[1] ** 3 - zlim[0] ** 3) * ans[:, 2] + zlim[0] ** 3) ** (1. / 3.)

        return ans


def volume(sqdeg, zlim, cosmo=None):
    """
    Returns the comoving volume of a chunk of a sphere of given limits in
    ra, dec, and z. z is interpreted as distance unless cosmo is given
    """
    if cosmo is not None:
        zlim = comoving_disth(zlim, cosmo)
    omega = sqdeg * (np.pi/180.) ** 2
    return omega/3. * (zlim[1] ** 3 - zlim[0] ** 3)


def volume_rdz(ralim, declim, zlim, cosmo=None):
    """
    Returns the comoving volume of a chunk of a sphere of given limits in
    ra, dec, and z. z is interpreted as distance unless cosmo is given
    """
    if cosmo is not None:
        zlim = comoving_disth(zlim, cosmo)
    return 2. / 3. * (ralim[1] - ralim[0]) * np.sin((declim[1] - declim[0]) / 2.) \
        * (zlim[1] ** 3 - zlim[0] ** 3)


def rdz_distance(rdz, rdz_prime, cosmo=None):
    """
    rdz is an array of coordinates of shape (N,3) while rdz_prime is a single point of comparison of shape (3,)
    z is interpreted as radial distance unless cosmo is given
    
    Returns the distance of each rdz coordinate from rdz_prime
    """
    if not cosmo is None:
        rdz[:, 2] = comoving_disth(rdz[:, 2], cosmo)
    r, d, z = rdz.T
    rp, dp, zp = rdz_prime
    return np.sqrt(r ** 2 + rp ** 2 - 2. * r * rp * (np.cos(d) * np.cos(dp) * np.cos(r - rp) + np.sin(d) * np.sin(dp)))


def xyz_distance(xyz, xyz_prime):
    x, y, z = xyz.T
    xp, yp, zp = xyz_prime
    return np.sqrt((x - xp) ** 2 + (y - yp) ** 2 + (z - zp) ** 2)


def unit_vector(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return (x, y, z)


def make_npoly(radius, n):
    import spherical_geometry.polygon as spoly
    phi1 = 2. * np.pi / float(n)
    points = [unit_vector(radius, phi1 * i) for i in np.arange(n) + .5]
    inside = (0., 0., 1.)
    return spoly.SingleSphericalPolygon(points, inside)


def vpmax2pimax(vpmax, z, cosmo):
    """
    Converts peculiar velocity (km/s) to comoving distance (Mpc/h) due to redshift distortion

    pimax = dChi/dz * z_vp
    ======================
    where z_vp = vpmax/c * (1 + z_cosmo)
    and dChi/dz = c/(H0*E(z))
    """
    H0_on_h = 100  # km/s/Mpc (use in place of H0 for h-scaling)
    pimax = vpmax / H0_on_h * (1 + z) / cosmo.efunc(z)
    return pimax


def factor_velocity(v, halo_v, halo_vel_factor=None, gal_vel_factor=None, inplace=False):
    if not inplace:
        v = v.copy()
    if not halo_vel_factor is None:
        new_halo_v = halo_vel_factor * halo_v
        v += new_halo_v - halo_v
    else:
        new_halo_v = halo_v

    if gal_vel_factor is not None:
        v -= new_halo_v
        v *= gal_vel_factor
        v += new_halo_v

    return v


def reduce_dim(arr):
    arr = np.asarray(arr)
    shape = arr.shape
    if len(shape) == 0:
        return arr.tolist()

    s = tuple(0 if shape[i] == 1 else slice(None) for i in range(len(shape)))
    return arr[s]


def logggnfw(x, x0, y0, m1, m2, alpha):
    """
    LOGarithmic Generalized^2 NFW profile
    ====================================================================
    log_{base}((base**(x-x0))**m1 * (1 + base**(x-x0)**(m2-m1))) + const
    ====================================================================
    Every parameter is allowed to range from -inf to +inf
    m1 is the slope far left of x0 and m2 is the slope far right
    of x0, with a smooth transition. The smaller alpha, the
    smoother the transition.

    Warning: Large magnitudes of m1 and m2 and large positive
    values of alpha may prevent the exact solution from being
    solved, in which case, approximations will be made automatically

    Parameters
    ----------
    x : float | array of floatsp at Pitt
        This is not a parameter, this is the abscissa

    x0 : float
        The characteristic position where the slope changes

    y0 : float
        The value of f(x0)

    m1 : float
        The slope at x << x0

    m2 : float
        The slope as x >> x0

    alpha : float
        The sharpness of the slope transition from m1 to m2

    Returns
    -------
    y : float | array of floats
        >>> base = 1. + np.exp(alpha)
        >>> const = (m1 - m2) / np.log2(base) + y0
        >>> np.log((base**(x-x0))**m1 * (1 + base**(x-x0)**(m2-m1))
        >>>        ) / np.log(base) + const

        where ``base = 1 + exp(alpha)``
        and   ``const = `y0 + (m1 - m2) / np.log2(base)`

    """
    x = np.asarray(x)
    scalar = len(x.shape) == 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ans = np.asarray(logggnfw_exact(x, x0, y0, m1, m2, alpha))

    fixthese = ~ np.isfinite(ans)
    x = x[fixthese]
    ans[fixthese] = logggnfw_approx(x, x0, y0, m1, m2, alpha)

    return ans.reshape(()).tolist() if scalar else ans


def logggnfw_exact(x, x0, y0, m1, m2, alpha):
    """
    exact form, inspired by gNFW potential
    OverFlow warning is easily raised by somewhat
    large values of m1, m2, and base
    """
    base = 1. + np.exp(alpha)
    x = x - x0
    return np.log((base ** x) ** m1 *
                  (1 + base ** x) ** (m2 - m1)
                  ) / np.log(base) + y0 + (m1 - m2) / np.log2(base)


def logggnfw_approx(x, x0, y0, m1, m2, alpha):
    """
    Depending on value of x, either Taylor expand around x0 or
    use the linear when sufficiently far from x0.
    It does a pretty good job of being continuous, but may
    drift far from the exact form at intermediate x-x0 values
    """
    base = 1. + np.exp(alpha)
    x = np.asarray(x)
    scalar = len(x.shape) == 0

    # Convergence value chosen to make this
    # function as continuous as possible
    blackmagic = 4.2 / np.log2(base)
    is_line1 = x - x0 < -blackmagic
    is_line2 = x - x0 > blackmagic
    is_curve = ~ (is_line1 | is_line2)

    ans = np.zeros_like(x)
    const = (m1 - m2) / np.log2(base) + y0
    ans[is_line1] = m1 * (x[is_line1] - x0) + const
    ans[is_line2] = m2 * (x[is_line2] - x0) + const

    x = x[is_curve]
    ans[is_curve] = logggnfw_taylor(x, x0, y0, m1, m2, base)

    return ans.reshape(()).tolist() if scalar else ans


def logggnfw_taylor(x, x0, y0, m1, m2, base):
    x = x - x0
    return (y0 + 0.5 * (m1 + m2) * x +
            np.log(base) / 8. * (m2 - m1) * x ** 2 +
            np.log(base) ** 3 / 192. * (m1 - m2) * x ** 4 +
            np.log(base) ** 5 / 2880. * (m2 - m1) * x ** 6 +
            17. * np.log(base) ** 7 / 645120. * (m1 - m2) * x ** 8)


def fuzzy_digitize_improved(x, centroids, **args):
    """
    Added functionality to halotools.utils.fuzzy_digitize
    allowing data to be above and below the highest and
    lowest centroid, respectively. In these cases, the data
    are assigned the the highest and lowest centroid with 100%
    probability, respectively. See original docstring below.

    """
    # Add a new centroid above and below the lowest and highest data values
    vals = np.concatenate([np.ravel(x), centroids])
    # noinspection PyArgumentList
    centroids = np.concatenate([[-2 * abs(vals.min()) - 1], centroids,
                                [2 * abs(vals.max()) + 1]])

    # Use fuzzy_digitize normally and reindex bins

    # TODO: Make exception for AssertionError where there aren't any bins with > nwin points
    centroid_indices = ht_utils.fuzzy_digitize(x, centroids, **args)
    centroid_indices[centroid_indices != 0] -= 1
    centroid_indices[centroid_indices == len(centroids) - 2] -= 1

    return centroid_indices


fuzzy_digitize_improved.__doc__ += ht_utils.fuzzy_digitize.__doc__


def correction_for_empty_bins(original_centroids, original_indices):
    """
    You may want to use this function after using
    halotools.utils.fuzzy_digitize (or the modified version of it)
    In case there were centroids with fewer than `min_count` data
    in the bin, this function will remove those centroids and
    shift the centroid_indices down so that no numbers are skipped.

    Arguments
    ---------
    original_centroids : 1d array of floats/ints
        The centroids you passed to fuzzy_digitize

    original_indices : array of floats/ints
        The array returned to you by fuzzy_digitize

    Returns
    -------
    new_centroids : 1d array of floats/ints
        similar to `original_centroids`, but removing any points
        corresponding to not a single index in `original_indices`

    new_indices : array of floats/ints
        array with same shape as `original_indices`, but indices
        are shifted to correspond to `new_centroids`
    """
    bin_is_empty = ~np.isin(np.arange(len(original_centroids)),
                            np.unique(original_indices))

    find_inds = (original_indices[..., None] ==
                 np.arange(len(original_centroids)))
    correction = np.sum(find_inds * np.cumsum(bin_is_empty), axis=-1)

    new_centroids: np.ndarray
    new_indices: np.ndarray
    new_centroids = np.asarray(original_centroids)[~bin_is_empty]
    new_indices = original_indices - correction
    return new_centroids, new_indices


def choose_close_index(value, values, tolerance=0.05):
    """
    Return the index of `values` containing the specified `value`
    within some `tolerance`. If there is not exactly one index within
    the tolerance, a ValueError is raised.

    Set ``tolerance="none"`` to return the closest index to your specified
    `value`, regardless of how far off it is.
    """
    if isinstance(tolerance, str):
        # Return the closest index, regardless of how far away the value is
        if value == np.inf:
            return np.argmax(values)
        elif value == -np.inf:
            return np.argmin(values)
        else:
            return np.argmin(np.abs(value - np.asarray(values)))
    else:
        # Demand exactly one value within the tolerance
        # else raise ValueError
        wh = np.where(np.isclose(values, value,
                                 rtol=0, atol=tolerance))[0]
        if len(wh) < 1:
            raise ValueError(f"No values matching {value}. Try "
                             f"increasing tolerance from {tolerance}. Available "
                             f"values: {values}")
        if len(wh) > 1:
            raise ValueError("Multiple matching redshifts:"
                             f"{values[wh]}")
        return wh[0]


def wget_download(file_url, outfile, overwrite=False):
    if not overwrite and os.path.isfile(outfile):
        return
    print("wget " + file_url)
    actual = wget.download(file_url, out=outfile)
    print()
    if overwrite:
        shutil.move(actual, outfile)