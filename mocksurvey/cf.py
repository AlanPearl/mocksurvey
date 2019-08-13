try:
    corrfunc_works = True
    import Corrfunc.theory
    from Corrfunc.utils import convert_rp_pi_counts_to_wp#, convert_3d_counts_to_cf
except ImportError:
    corrfunc_works = False
    import halotools.mock_observables as mockobs
import numpy as np
import warnings
from . import hf

class PairCounts:
    def __init__(self, Ndata, Nrand, DD, DR, RR, n_rpbins, pimax=None):
        """Class used to store information about the pair counts DD,
        DR, and RR needed for calculating correlation functions"""
        self.Ndata = Ndata
        self.Nrand = Nrand
        self.DD = np.asarray(DD)
        self.DR = np.asarray(DR)
        self.RR = np.asarray(RR)
        self.n_rpbins = n_rpbins
        self.pimax = pimax

    def __add__(self, other):
        sums = (self.Ndata + other.Ndata,
                self.Nrand + other.Nrand,
                self.DD + other.DD,
                self.DR + other.DR,
                self.RR + other.RR)
        return PairCounts(*sums, self.n_rpbins, self.pimax)

    def __repr__(self):
        msg = "\tPairCounts\n\t==========\n"
        msg += "\tNdata = %d\n\tNrand = %d\n" %(self.Ndata,self.Nrand)
        msg += "DD = " + str(self.DD) + "\n"
        msg += "DR = " + str(self.DR) + "\n"
        msg += "RR = " + str(self.RR)
        return msg

# Count pairs in 3D r bins
# ========================
def paircount_r(data, rands, rbins, nthreads=1, pair_counter_func="DD",
                kwargs=None, pc_kwargs=None, precomputed=(None, None, None)):
    if kwargs is None:
        kwargs = {}
    if pc_kwargs is None:
        pc_kwargs = {}
    x,y,z = data.T
    xr,yr,zr = rands.T
    
    if callable(pair_counter_func):
        pass
    elif pair_counter_func.lower() == "dd":
        pair_counter_func = Corrfunc.theory.DD
    else:
        raise ValueError("pair_counter_func must be callable")
    
    DD_counts, DR_counts, RR_counts = precomputed
    if len(data)<2:
        DD_counts = [np.nan]
    if len(data)<2 or len(rands)<2:
        DR_counts = [np.nan]
    if len(rands)<2:
        RR_counts = [np.nan]

    if DD_counts is None:
        DD_counts = pair_counter_func(autocorr=True, nthreads=nthreads, **kwargs,
                binfile=rbins, X1=x, Y1=y, Z1=z, periodic=False)["npairs"]
    if DR_counts is None:
        DR_counts = pair_counter_func(autocorr=False, nthreads=nthreads, **kwargs,
                binfile=rbins, X1=x, Y1=y, Z1=z, X2=xr, Y2=yr, Z2=zr, periodic=False)["npairs"]
    if RR_counts is None:
        RR_counts = pair_counter_func(autocorr=True, nthreads=nthreads, **kwargs,
                binfile=rbins, X1=xr, Y1=yr, Z1=zr, periodic=False)["npairs"]
    
    args = [data.shape[0], rands.shape[0]]
    args += [DD_counts, DR_counts, RR_counts]
    args += [len(rbins)-1]
    return PairCounts(*args, **pc_kwargs)

# Count pairs in rp and pi bins
# =============================
def paircount_rp_pi(data, rands, rpbins, pimax=50.0, nthreads=1, precomputed=(None,None,None)):
    answer = paircount_r(data, rands, rpbins, nthreads, Corrfunc.theory.DDrppi,
                 {"pimax":pimax}, {"pimax":pimax}, precomputed)

    return answer

def counts_to_wp(pc):
    # Only estimator available: Landy & Szalay (1993)
    return convert_rp_pi_counts_to_wp(pc.Ndata, pc.Ndata, pc.Nrand, pc.Nrand,
                    pc.DD, pc.DR, pc.DR, pc.RR, pc.n_rpbins, pc.pimax)

def counts_to_xi(pc):
    """
    Returns xi(r) if given Paircounts object has no pimax value
    Else, returns xi(rp, pi)"""
    # Use the Landy & Szalay (1993) estimator
    factor = pc.Nrand / float(pc.Ndata)
    factor2 = factor**2
    xi = (factor2*pc.DD - 2.*factor*pc.DR + pc.RR)/pc.RR
    if not pc.pimax is None:
        Nrp = pc.n_rpbins
        Npi = len(xi)//Nrp
        assert(len(xi)%Nrp == 0)
        xi = np.reshape(xi, (Nrp, Npi))
    return xi

def RRrppi_periodic(N, boxsize, rpbins, pibins):
    drp2 = np.diff(rpbins**2)
    dpi = np.diff(pibins)
    return N**2 / boxsize**3 * 2*np.pi * drp2[:,None] * dpi[None,:]

# Returns the bias as a function of rp (rpbins must be given)
# ===========================================================
def bias_rp(data, rands, rpbins, boxsize=None, wp_dms=None, pimax=50., suppress_warning=False):
    rpbinses = np.asarray(rpbins)
    if len(rpbinses.shape) == 1:
        rpbins = rpbinses
        rpbinses = []
        for i in range(len(rpbins)-1):
            rpbinses.append([rpbins[i],rpbins[i+1]])
    if not hf.is_arraylike(wp_dms):
        wp_dms = [wp_dms] * len(rpbinses)

    biases = []
    for rpbins,wp_dm in zip(rpbinses, wp_dms):
        rpbins = np.asarray(rpbins)
        if wp_dm is None:
            rpcens = np.sqrt(rpbins[:-1]*rpbins[1:])
            wp_dm = wp_darkmatter(rpcens)
            if not suppress_warning:
                print(f"Using default dark matter wp(rp={rpcens}) = {wp_dm}")
        wp_gal = wp_rp(data, rands, rpbins, pimax=pimax, boxsize=boxsize)[0]
        bias = np.sqrt(wp_gal/wp_dm)
        biases.append(bias)
    
    return hf.reduce_dim(biases)


def wp_darkmatter(rp):
    """best fit power-law for MDR1 z=1 (pimax=50)"""
    r0, alpha = (41.437187675742656, -0.832326251664125)
    return (rp/r0)**alpha


# Returns the 2D correlation function xi(rp, pi) using Corrfunc
# =============================================================
def xi_rp_pi(data, rands, rpbins, pibins, boxsize=None, nthreads=1, estimator='Landy-Szalay'):
    # Corrfunc implementation requires evenly spaced pibins, with first bin starting at pi=0
    pimax = pibins[-1]
    n_pibins = len(pibins) - 1
    n_rpbins = len(rpbins) - 1

    pi_factor = n_pibins / pimax
    pimax *= pi_factor
    array_factor = np.array([[1.], [1.], [pi_factor]])

    x,y,z = data.T * array_factor
    
    if not corrfunc_works:
        return mockobs.rp_pi_tpcf(data, rpbins, pibins, randoms=rands, 
                                  num_threads=nthreads, estimator=estimator)

    if rands is None:
        if boxsize is None:
            raise ValueError("`boxsize` cannot be None if `rands` is None")
        if np.any((data > boxsize) | (data < 0)):
            data = data%boxsize
        
        with warnings.catch_warnings():
          with hf.suppress_stdout():
            warnings.simplefilter("ignore")
            
            DD = Corrfunc.theory.DDrppi(autocorr=True, nthreads=nthreads, pimax=pimax, binfile=rpbins, X1=x, Y1=y, Z1=z)
            DD = np.reshape(DD['npairs'], (n_rpbins, n_pibins))
            RR = RRrppi_periodic(len(data), boxsize, rpbins, pibins)
        return DD/RR - 1.
    else:
        xr,yr,zr = rands.T * array_factor
        
        with warnings.catch_warnings():
          with hf.suppress_stdout():
            warnings.simplefilter("ignore")
            
            DD = Corrfunc.theory.DDrppi(autocorr=True, nthreads=nthreads, pimax=pimax, binfile=rpbins, X1=x, Y1=y, Z1=z, periodic=False)
            DD = np.reshape(DD['npairs'], (n_rpbins, n_pibins))
        
            DR = Corrfunc.theory.DDrppi(autocorr=False, nthreads=nthreads, pimax=pimax, binfile=rpbins, X1=x, Y1=y, Z1=z,
                                                                                        X2=xr, Y2=yr, Z2=zr, periodic=False)
            DR = np.reshape(DR['npairs'], (n_rpbins, n_pibins))
        
            RR = Corrfunc.theory.DDrppi(autocorr=True, nthreads=nthreads, pimax=pimax, binfile=rpbins, X1=xr, Y1=yr, Z1=zr, periodic=False)
            RR = np.reshape(RR['npairs'], (n_rpbins, n_pibins))
    
        factor = len(rands) / float(len(data))
        factor2 = factor**2
        if estimator.lower() == 'landy-szalay':
            return (factor2*DD - 2.*factor*DR + RR)/RR # Landy & Szalay (1993) Estimator
        elif estimator.lower() == 'natural':
            return factor**2 * DD/RR - 1. # Natural Estimator
        else:
            raise KeyError("Estimator must be `Natural` or `Landy-Szalay`")

# Returns the 3D correlation function xi(r) using Corrfunc
# ========================================================
def xi_r(data, rands, rbins, boxsize=None, nthreads=1, estimator='Landy-Szalay'):
    if rands is None:
        # Periodic boundary conditions
        if boxsize is None:
            raise ValueError("`boxsize` cannot be None if `rands` is None")
        if hf.is_arraylike(boxsize):
            raise ValueError(f"`boxsize` must be a scalar, not {boxsize.__class__}")
        if rbins[-1]*3 > boxsize:
            raise ValueError(f"cube side length must be at least 3x the largest r bin, but"
                             "`boxsize={boxsize}` and `3*rbins[-1]={3*rbins[-1]}`")
        if np.any((data > boxsize) | (data < 0)):
            data = data%boxsize
        
        with warnings.catch_warnings():
          with hf.suppress_stdout():
            warnings.simplefilter("ignore")
            return Corrfunc.theory.xi(boxsize, nthreads, rbins, *data.T)["xi"]
    x,y,z = data.T
    xr,yr,zr = rands.T
    if len(data)==0 or len(rands)==0:
        return np.nan
    
    if not corrfunc_works:
        return mockobs.tpcf(data, rbins, randoms=rands, estimator=estimator, num_threads=nthreads)
    
    with warnings.catch_warnings():
      with hf.suppress_stdout():
        warnings.simplefilter("ignore")
        
        DD_counts = Corrfunc.theory.DD(autocorr=True, nthreads=nthreads, binfile=rbins, X1=x, Y1=y, Z1=z, periodic=False)
        DD_counts = DD_counts['npairs']
    
        DR_counts = Corrfunc.theory.DD(autocorr=False, nthreads=nthreads, binfile=rbins, X1=x, Y1=y, Z1=z, X2=xr, Y2=yr, Z2=zr, periodic=False)
        DR_counts = DR_counts['npairs']
    
        RR_counts = Corrfunc.theory.DD(autocorr=True, nthreads=nthreads, binfile=rbins, X1=xr, Y1=yr, Z1=zr, periodic=False)
        RR_counts = RR_counts['npairs']

    factor = len(rands) / float(len(data))
    factor2 = factor**2

    if estimator.lower() == 'natural':
        return factor2 * DD_counts/RR_counts - 1
    elif estimator.lower() == 'landy-szalay':
        return (factor2*DD_counts - 2*factor*DR_counts + RR_counts) / RR_counts
    else:
        raise KeyError("Estimator must be `Natural` or `Landy-Szalay`")

def wp_rp(data, rands, rpbins, pimax=50., boxsize=None, nthreads=1,
          use_halotools_version=False):
    """
    wp_rp(data, rands, rpbins, pimax, boxsize=None, nthreads=1)
    
    Return the projected correlation function :math:`w_p(r_p)` to measure the clustering of `data`, compared to uniformly distributed `rands`. If `rands` is ``None``, then `data` must be selected from a cube of side length `boxsize`. Wrapper for Corrfunc. Only Landy & Szalay estimator available.
    
    Parameters
    ----------
    data : array_like, with shape (N,3)
        Array containing columns x, y, z of data to be measured. Units should be
        
    rands : array_like, with shape (Nrand,3)
        Array containing columns x, y, z of random data uniformly selected in the same space as `data`.
    
    rpbins : array_like, with shape (Nbins+1,)
        Array containing the edges of bins of separation ``rp = sqrt(dx^2 + dy^2)``. Must be increasing in value.
    
    pimax : float (default = 50.)
        Line-of-sight (z-direction) separation to integrate the correlation function over.
    
    boxsize : float (default = None)
        Side length of the cube which `data` is selected in. Only used if `rands` is set to ``None``.
    
    nthreads : int (default = 2)
        Number of CPU cores to be used for multiprocessing.
    
    Returns
    -------
    wp : ndarray, with shape (Nbins,)
        Projected two-point correlation function (units of distance) evaluated within each specified bin enclosed by `rpbins`.
    """
    if rands is None:
        # Periodic boundary conditions
        if boxsize is None:
            raise ValueError("`boxsize` cannot be None if `rands` is None")
        if hf.is_arraylike(boxsize):
            raise ValueError(f"`boxsize` must be a scalar, not {boxsize.__class__}")
        if rpbins[-1]*3 > boxsize:
            raise ValueError(f"cube side length must be at least 3x the largest rp bin, but"
                             "`boxsize={boxsize}` and `3*rpbins[-1]={3*rpbins[-1]}`")
        if np.any((data > boxsize) | (data < 0)):
            data = data%boxsize
        
        if use_halotools_version or not corrfunc_works:
            import halotools.mock_observables as mockobs
            if rands is None:
                return mockobs.wp(data, rpbins, pimax, period=boxsize,
                        estimator="Landy-Szalay", num_threads=nthreads)
            else:
                return mockobs.wp(data, rpbins, pimax, randoms=rands,
                        estimator="Landy-Szalay", num_threads=nthreads)
        
        with warnings.catch_warnings():
          with hf.suppress_stdout():
            warnings.simplefilter("ignore")
            return Corrfunc.theory.wp(boxsize, pimax, nthreads,
                                      rpbins, *data.T)["wp"]
    n_rpbins = len(rpbins) - 1

    x,y,z = data.T
    xr,yr,zr = rands.T

    N = len(data)
    Nran = len(rands)
    if N==0 or Nran==0:
        return np.array([np.nan]*n_rpbins)
    
    if not corrfunc_works:
        import halotools.mock_observables as mockobs
        return mockobs.wp(data, rpbins, pimax, randoms=rands, 
                          estimator="Landy-Szalay", num_threads=nthreads)
    
    with warnings.catch_warnings():
      with hf.suppress_stdout():
        warnings.simplefilter("ignore")
        
        DD = Corrfunc.theory.DDrppi(autocorr=True, nthreads=nthreads, pimax=pimax, binfile=rpbins, X1=x, Y1=y, Z1=z, periodic=False)
        DD = DD['npairs']
    
        DR = Corrfunc.theory.DDrppi(autocorr=False, nthreads=nthreads, pimax=pimax, binfile=rpbins, X1=x, Y1=y, Z1=z, X2=xr, Y2=yr, Z2=zr, periodic=False)
        DR = DR['npairs']
    
        RR = Corrfunc.theory.DDrppi(autocorr=True, nthreads=nthreads, pimax=pimax, binfile=rpbins, X1=xr, Y1=yr, Z1=zr, periodic=False)
        RR = RR['npairs']

    wp = convert_rp_pi_counts_to_wp(N, N, Nran, Nran, DD, DR, DR, RR, n_rpbins, pimax)
    return wp

# Calculate any of the above three correlation functions, estimating errors via the block jackknife/bootstrap method
# ==================================================================================================================
def block_jackknife(data, rands, centers, fieldshape, nbins=(2,2,1), data_to_bin=None, rands_to_bin=None, func="xi_r", args=None, kwargs=None, mean_answer=None, rdz_distance=False, debugging_plots=False):
    """
    Given a function which returns a statistic over an array of rbins,
    compute the statistic and its uncertainty.
    The function provided MUST take data and rands as its first two
    arguments, each being ndarrays of shape (N,3)
    ___
    Returns:
    - statistic [rp]
    - covariance matrix [rp_i, rp_j]
    """
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = []
    if data_to_bin is None:
        assert(rands_to_bin is None)
        data_to_bin = data
        rands_to_bin = rands
    if callable(func):
        pass
    elif func.lower() == 'xi_r':
        func = xi_r
    elif func.lower() == 'wp_rp':
        func = wp_rp
    else:
        raise KeyError("Argument func=%s not valid. Must be one of: %s" %(func, '{<callable>, "xi_r", "wp_rp"}'))
    
    N = np.product(nbins)
    if hf.is_arraylike(centers[0]):
        N *= len(centers)
    
    centers = np.asarray(centers)
    fieldshape = np.asarray(fieldshape)
    
    ind_d, ind_r = _assign_block_indices(data_to_bin, rands_to_bin, centers, fieldshape, nbins, rdz_distance)
    
    answer_l = []
    for l in range(N):
        ind_d_sample = np.where(ind_d != l)[0]
        data_sample = data[ind_d_sample]
        if not rands is None:
            ind_r_sample = np.where(ind_r != l)[0]
            rands_sample = rands[ind_r_sample]
        else:
            rands_sample = None
        
        if debugging_plots:
            import matplotlib.pyplot as plt
            if not rands is None: plt.scatter(rands_to_bin[ind_r_sample][:,0], rands_to_bin[ind_r_sample][:,1], s=.1)
            plt.scatter(data_to_bin[ind_d_sample][:,0], data_to_bin[ind_d_sample][:,1], s=.5)
            plt.show()
        
        ans = func(data_sample, rands_sample, *args, **kwargs)
        answer_l.append(np.atleast_1d(ans))
        if np.any(np.isnan(answer_l[-1])):
            print("block_jackknife: NAN encountered", flush=True)
            answer_l.pop()
            N -= 1
    
    # mean_answer/mean_jacks [rp]
    jackknife_mean = np.mean(answer_l, axis=0)
    if isinstance(mean_answer, dict):
        mean_answer = func(data, rands, *args, **{**kwargs, **mean_answer})
    elif isinstance(mean_answer, str):
        mean_answer = func(data, rands, *args, **kwargs)
    else:
        mean_answer = jackknife_mean
    
    # Correction factor to account for the fact that jackknife mean
    # may be systematically biased from the "true" mean answer
    jackknife_bias = mean_answer[None,:]*mean_answer[:,None] / (jackknife_mean[None,:]*jackknife_mean[:,None])
    
    # answer_l [l, rp]
    answer_l = np.asarray(answer_l)
    
    # covar [rp_i, rp_j]
    covar = (N-1)/N * jackknife_bias * np.sum( (answer_l[:,:,None] - jackknife_mean[None,:,None]) * (answer_l[:,None,:] - jackknife_mean[None,None,:]), axis=0)
    
    
    
    
    return mean_answer, covar


def block_bootstrap(data, rands, data_to_bin=None, rands_to_bin=None, func='xi_r', args=None, kwargs=None, nbootstrap=10, bins=50., plot_blocks=False, alpha=.5, Lbox=400., return_better_answer=False):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if data_to_bin is None:
        assert(rands_to_bin is None)
        data_to_bin = data
        rands_to_bin = rands
    if type(func) != str:
        pass
    elif func.lower() == 'xi_r':
        func = xi_r
    elif func.lower() == 'xi_rp_pi':
        func = xi_rp_pi
    elif func.lower() == 'wp_rp':
        func = wp_rp
    else:
        raise KeyError("Argument func=%s not valid. Must be in %s" %(func, '{`xi_r`, xi_rp_pi`, `wp_rp`}'))

    # Set up bins to define blocks within the data
    bins, nx, ny, nz = _setupblockbins(rands_to_bin, bins)
    Nblock = nx * ny * nz

    # Assign each index a number corresponding to which block it has been assigned
    ind_d, ind_r = _assignblocks(data_to_bin, rands_to_bin, bins)

    assert (nbootstrap >= 0)

    if nbootstrap > 0:
        answer, err, err_err = _blockbootstrap_subsample(data, rands, Nblock, ind_d, ind_r, nbootstrap, func, args, kwargs, plot_blocks, alpha)
        if return_better_answer:
            better_answer = func(data, rands, *args, **kwargs)
            return np.array([better_answer, err, err_err])
        else:
            return np.array([answer, err, err_err])
    else:
        return func(data, rands, *args, **kwargs)

# Helper functions for jackknife / bootstrapping
# ==============================================
def _blockbootstrap_subsample(data, rands, Nblock, ind_d, ind_r, nbootstrap, func, args, kwargs, plot_blocks, alpha, seed=None):
    for i in range(nbootstrap):
        # Choose blocks in resample
        if not seed is None: np.random.seed(seed)
        blocks_resample = np.random.choice(np.arange(Nblock), Nblock, replace=True)
        if not seed is None: np.random.seed()

        # Determine indices of data within chosen blocks, including repeats
        if len(blocks_resample)*max([len(ind_d),len(ind_r)]) < 1e7:
            ind_d_resample = np.where(blocks_resample[np.newaxis,:] == ind_d[:,np.newaxis])[0]
            ind_r_resample = np.where(blocks_resample[np.newaxis,:] == ind_r[:,np.newaxis])[0]
        else:
            # If Ndata*Nbootstrap array takes too much memory, then just do one row at a time
            ind_d_resample = []
            ind_r_resample = []
            for block in blocks_resample:
                ind_d_resample += np.where(ind_d == block)[0].tolist()
                ind_r_resample += np.where(ind_r == block)[0].tolist()
            ind_d_resample = np.asarray(ind_d_resample)
            ind_r_resample = np.asarray(ind_r_resample)

        # Select resampled dataNone
        data_resample = data[ind_d_resample,:]
        rands_resample = rands[ind_r_resample,:]

        if plot_blocks and plot_blocks != 'final_plot_only':
            import matplotlib.pyplot as plt
            if plot_blocks == 'noslice':
                slice_d = np.full(data_resample.shape[0], True)
                slice_r = np.full(rands_resample.shape[0], True)
            else:
                slice_d = (-10 < data_resample[:,1]) & (data_resample[:,1] < 10)
                slice_r = (-10 < rands_resample[:,1]) & (rands_resample[:,1] < 10)
            noise_d = np.random.random(data_resample.shape)*.2
            noise_r = np.random.random(rands_resample.shape)*.2
            xd = data_resample[:,0][slice_d] + noise_d[:,0][slice_d]
            yd = data_resample[:,2][slice_d] + noise_d[:,2][slice_d]
            xr = rands_resample[:,0][slice_r] + noise_r[:,0][slice_r]
            yr = rands_resample[:,2][slice_r] + noise_r[:,2][slice_r]
            plt.plot(xd, yd, 'r.', alpha=alpha)
            plt.plot(xr, yr, 'g.', alpha=alpha)
            plt.show()

        # Compute correlation function using resampled data
        xi_resample = np.asarray(func(data_resample, rands_resample, *args, **kwargs))
        if i==0:
            results = np.ones((nbootstrap, *xi_resample.shape), dtype=xi_resample.dtype)
        results[i] = xi_resample

    if len(results.shape) != 2:
        assert(len(results.shape) == 1)
        results = np.reshape(results, (results.shape[0], 1))
    # Return the mean and spread of the statistic computed
    xi = []; xi_err = []; xi_err_err = []
    for result in results.T:
        assert(len(result.shape) == 1)
        N_success = np.sum(~np.isnan(result))
        #print("Mean, std, std_err of:", result)
        result = result.copy()[result==result]
        # print('Number of unusable columns:', len(np.where(result!=result)[0]))
        if len(result) >= 1:
            xi_i = np.nanmean(result)
        else:
            xi_i = np.nan
        if len(result) >= 2:
            xi_err_i = np.nanstd(result, ddof=1)
            xi_err_err_i = xi_err_i / np.sqrt( 2. * (N_success - 1) )
        else:
            xi_err_i = xi_err_err_i = np.nan

        xi.append(xi_i)
        xi_err.append(xi_err_i)
        xi_err_err.append(xi_err_err_i)
    xi = np.asarray(xi)
    xi_err = np.asarray(xi_err)
    xi_err_err = np.asarray(xi_err_err)
    if plot_blocks:
        import matplotlib.pyplot as plt
        x = np.arange(len(xi))
        for result in results:
            plt.plot(x, result, 'g-', alpha=alpha)
        plt.errorbar(x, xi, yerr=xi_err)
        plt.show()
    return xi, xi_err, xi_err_err

def _setupblockbins(rands, bins):
    if len(rands) < 1:
        raise ValueError('`rands` cannot be length zero')
    if hf.is_arraylike(bins):
        assert(len(bins) == 3)
        if hf.is_arraylike(bins[0]):
            # bins = [x_edges, y_edges, z_edges]
            nx, ny, nz = len(bins[0])-1, len(bins[1])-1, len(bins[2])-1
            bins = (*bins, nx, ny, nz)
            return bins, nx, ny, nz
        else:
            # bins = [xblocklength, yblocklength, zblocklength]
            # cut out some data near the sides so that all bins are the exact length
            xblocklength, yblocklength, zblocklength = bins
            x_range = rands[:,0].min(), rands[:,0].max()
            y_range = rands[:,1].min(), rands[:,1].max()
            z_range = rands[:,2].min(), rands[:,2].max()

            xbins = np.arange(x_range[0], x_range[1], xblocklength)
            ybins = np.arange(y_range[0], y_range[1], yblocklength)
            zbins = np.arange(z_range[0], z_range[1], zblocklength)

            xbins += (x_range[-1] - xbins[-1])/2.
            ybins += (y_range[-1] - ybins[-1])/2.
            zbins += (z_range[-1] - zbins[-1])/2.

            nx, ny, nz = len(xbins)-1, len(ybins)-1, len(zbins)-1

            return (xbins, ybins, zbins, nx, ny, nz), nx, ny, nz

    else:
        # Make block bin lengths close to but less than `bins`
        # in each dimension, to allow all data to be placed in a block
        xblocklength = yblocklength = zblocklength = bins
        xlim = rands[:,0].min(), rands[:,0].max()
        ylim = rands[:,1].min(), rands[:,1].max()
        zlim = rands[:,2].min(), rands[:,2].max()

        nx = np.ceil((xlim[1] - xlim[0])/xblocklength).astype(int)
        ny = np.ceil((ylim[1] - ylim[0])/yblocklength).astype(int)
        nz = np.ceil((zlim[1] - zlim[0])/zblocklength).astype(int)

        xbins = np.linspace(*xlim, nx+1); xbins[0] -= 1.; xbins[-1] += 1.
        ybins = np.linspace(*ylim, ny+1); ybins[0] -= 1.; ybins[-1] += 1.
        zbins = np.linspace(*zlim, nz+1); zbins[0] -= 1.; zbins[-1] += 1.

        bins = (xbins, ybins, zbins, nx, ny, nz)

    return bins, nx, ny, nz

def _assignblocks(data, rands, bins_info):
    if data is None or rands is None or len(data) == 0 or len(rands) == 0:
        raise ValueError('data_to_bin: %s \nrands_to_bin: %s' %(str(data),str(rands)))
    xbins, ybins, zbins, nx, ny, nz = bins_info
    xind_d = np.digitize(data[:,0], xbins) - 1; xind_r = np.digitize(rands[:,0], xbins) - 1
    yind_d = np.digitize(data[:,1], ybins) - 1; yind_r = np.digitize(rands[:,1], ybins) - 1
    zind_d = np.digitize(data[:,2], zbins) - 1; zind_r = np.digitize(rands[:,2], zbins) - 1

    ind_d = xind_d * ny * nz   +   yind_d * nz   +   zind_d
    ind_r = xind_r * ny * nz   +   yind_r * nz   +   zind_r

    ind_d[(0 > xind_d) | (xind_d >= nx) | (0 > yind_d) | (yind_d >= ny) | (0 > zind_d) | (zind_d >= nz)] = -1
    ind_r[(0 > xind_r) | (xind_r >= nx) | (0 > yind_r) | (yind_r >= ny) | (0 > zind_r) | (zind_r >= nz)] = -1

    return ind_d, ind_r

def _assign_block_indices(data, rands, centers, fieldshape, nbins, rdz_distance=False):
    if data is None or len(data) == 0 or (rands is not None and len(rands) == 0):
        raise ValueError("data_to_bin: %s \nrands_to_bin: %s" %(str(data), str(rands)))
    
    # Make sure centers.shape = (numfields,3)
    centers = np.atleast_2d(centers); assert(centers.shape[-1] == 3); assert(len(centers.shape) < 3)
    
    # Index every data point and random point according to their block (jackknife region)
    ind_data_rands = ()
    for dat in data,rands:
        if dat is None:
            ind_data_rands += None,
            continue
        dist_to_center = []
        # Find the center which the point is closest to
        for center in centers:
            if rdz_distance:
                dist = hf.rdz_distance(dat, center, rdz_distance)
            else:
                dist = hf.xyz_distance(dat, center)
            
            dist_to_center.append(dist)
        
        dist_to_center = np.asarray(dist_to_center)
        closest_center = np.argmin(dist_to_center, axis=0)
        
        # Assign bins around each center
        ind = -np.ones(len(dat), dtype=np.int32)
        for i,center in enumerate(centers):
            closest_ind = np.where(closest_center==i)
            dat_i = dat[closest_ind]
            lower,upper = center - fieldshape/2., center + fieldshape/2.
            # Fix the z selection so that center is at the center in CARTESIAN space so this isn't necessary
            # if nbins[2] < 3:
            #     lower[2] -= 100.
            #     upper[2] += 100.
            
            nx,ny,nz = nbins
            xbins,ybins,zbins = [np.linspace(lower[i], upper[i], nbins[i]+1) for i in range(3)]
            xbins[0] = -np.inf; xbins[-1] = np.inf
            ybins[0] = -np.inf; ybins[-1] = np.inf
            zbins[0] = -np.inf; zbins[-1] = np.inf
            
            xind = np.digitize(dat_i[:,0], xbins) - 1
            yind = np.digitize(dat_i[:,1], ybins) - 1
            zind = np.digitize(dat_i[:,2], zbins) - 1
            ind_i = xind * ny * nz   +   yind * nz   +   zind
            ind_i += i*nx*ny*nz
            
            ind[closest_ind] = ind_i
            
        
        ind_data_rands += ind,
    
    return ind_data_rands[0], ind_data_rands[1]
