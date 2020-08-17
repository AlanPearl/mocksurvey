import numpy as np

import nbodykit.lab as nbk

from .. import mocksurvey as ms


class ThreePointCalculator:
    r_edges = np.geomspace(30, 180, 61)
    maxpole = 10
    cosmo = ms.bplcosmo

    def __init__(self, rands, numrand=None, r_edges=None, maxpole=None, cosmo=None):
        self.rands = rands
        if r_edges is not None:
            self.r_edges = r_edges
        if maxpole is not None:
            self.maxpole = maxpole
        if cosmo is not None:
            self.cosmo = cosmo

        self.r_cens = np.sqrt(self.r_edges[:-1] * self.r_edges[1:])
        self.volume_weights = np.diff(self.r_edges**3)

        r = self._rdzw_array(self.rands, choice=numrand)
        self.numrand = len(r) if numrand is None else numrand
        self.rrr = self._count_triplets(r, append_rands=False)

        if not np.all(self.rrr):
            import warnings
            warnings.warn("Bins exist with zero randoms",
                          category=RuntimeWarning)

    def __call__(self, data, numrand=None):
        """
        Compute the 3PCF (Slepian & Eisenstein 2015) of given data

        Parameters
        ----------
        data : struc_array
            Array containing "ra" [deg], "dec" [deg], and "redshift"
        numrand : int (default=None)
            If supplied, select a subsample of randoms of this
            size to compute (D-R)(D-R)(D-R)

        Returns
        -------
        xi : np.ndarray
            Multipole expansion of the three-point correlation function
            with shape (M,N,N) where M is the number of poles and N is
            the number of bins in r1 (and r2)
        """
        nnn = self.count_triplets(data, numrand=numrand)
        rrr = self.rrr * (len(data) / self.numrand)**3
        return triplet_counts_to_3pcf(nnn, rrr)

    def count_triplets(self, data, numrand=None):
        return self._count_triplets(data, numrand=numrand)

    def poles(self):
        return list(range(self.maxpole + 1))

    def get_numrand(self, numrand=None):
        return self.numrand if numrand is None else numrand

    def _count_triplets(self, data, append_rands=True, numrand=None):
        tcmb0 = max(self.cosmo.Tcmb0.value, 0.066)  # lowest Tcmb0 that works
        # sigma8=0.823 <--- giving this allows
        cosmo = nbk.cosmology.Cosmology.from_astropy(
            self.cosmo, T0_cmb=tcmb0, n_s=0.96)  # <-- Bolshoi-Planck n_s

        data = self._rdzw_array(data)
        if append_rands:
            weight = -len(data) / self.get_numrand(numrand)
            rands = self._rdzw_array(self.rands, weight=weight, choice=numrand)
            data = np.concatenate([data, rands])
        result = nbk.SurveyData3PCF(nbk.ArrayCatalog(data),
                                    poles=self.poles(),
                                    edges=self.r_edges,
                                    cosmo=cosmo, ra="ra", dec="dec",
                                    redshift="redshift", weight="weight")
        # noinspection PyTypeChecker
        # Indexed by (pole,  r1, r2)
        return np.array([result.poles[f"corr_{i}"]
                         for i in self.poles()])

    @staticmethod
    def _rdzw_array(array, weight=1.0, choice=None):
        if choice is not None:
            array = np.random.choice(array, choice, replace=False)

        names = ["ra", "dec", "redshift", "weight"]
        vals = [array[name]
                for name in names[:-1]] + [np.full(len(array), weight)]
        return ms.util.make_struc_array(names, vals)


def slepian_matrix_eq7(rrr):
    """
    Returns the A matrix from Equation 7 of Slepian+ 2017

    Parameters
    ----------
    rrr : np.ndarray
        Multipole expansion of RRR triplet counts of shape (M,N,N)
        where M is the number of poles and N is the number of r1 bins.

    Returns
    -------
    matrix : np.ndarray
        A matrix of shape (N,N,M,M) which transforms zeta_j -> nnn_j/rrr_0.
    """
    from sympy.physics.wigner import wigner_3j

    shape = np.shape(rrr)
    assert len(shape) == 3
    assert shape[1] == shape[2]

    poles = list(range(shape[0]))
    f = rrr / rrr[:1]

    # M(r1,r2,k,l) matrix from Equation 6
    matrix = np.sum([[[float(wigner_3j(ell, lp, k, 0, 0, 0)**2) * f[lp]
                       for ell in poles]
                      for k in poles]
                     for lp in poles[1:]], axis=0)
    matrix = np.moveaxis(matrix, [0, 1], [-2, -1])
    matrix *= np.array([2*k + 1 for k in poles]
                       )[None, None, :, None]

    # A(r1,r2,k,l) matrix from Equation 7
    matrix += np.identity(len(poles))[None, None, :, :]
    return matrix


def triplet_counts_to_3pcf(nnn, rrr):
    """
    Follows Equations 2-7 of Slepian+ 2017 to solve for the 3PCF.

    Prior to using this function, you should renormalize RRR by a
    factor of (N_R/N_D)^3 where N_R is the number of randoms used
    to calculate RRR and N_D is the number of data used to
    calculate NNN. The number of randoms used for NNN doesn't matter
    at this point, since they should have already been downweighted.

    Parameters
    ----------
    nnn : np.ndarray
        Multipole expansion of NNN=(D-R)(D-R)(D-R) triplet counts.
        Must have shape of (M,N,N) where M is the number of poles
        and N is the number of r1 (and r2) bins. Poles must be in
        order l = 0, 1, 2, ..., M-1.
    rrr : np.ndarray
        Multipole expansion of RRR triplet counts. Must be same
        shape and order as nnn.

    Returns
    -------
    xi : np.ndarray
        Multipole expansion of three-point correlation function.
        Takes the same shape as nnn and rrr.
    """
    assert np.shape(nnn) == np.shape(rrr)

    matrix = np.linalg.inv(slepian_matrix_eq7(rrr))
    return np.einsum("abcd,dab->cab", matrix, nnn / rrr[:1])


def reobtain_triplet_counts(zeta, rrr):
    """
    Apply Eq. 7 of Slepian+ 2017 to reobtain the multipole expansion
    of NNN from RRR and the 3PCF (zeta). Prior to using this function,
    you should renormalize RRR by a factor of (N_R/N_D)^3
    """
    assert np.shape(zeta) == np.shape(rrr)

    matrix = slepian_matrix_eq7(rrr)
    return np.einsum("abcd,dab->cab", matrix, zeta * rrr[:1])


def three_point_compression(cf, r_edges, min_side=30.0, r2max=False,
                            fractional=False, weighted=True):
    """
    Integrates over r2 to get a volume-weighted average.

    Integration bounds for r2 (r2_low and r2_high) will depend
    on the r1 bin, and the values r2max and fractional.
    There are five different compression schemes:

    - For r2max=False & fractional=False (Slepian et al. 2017):
    r2_low = min_side & r2_high = r1_low - min_side

    - For r2max=False & fractional=True:
    r2_low = min_side * r1_low & r2_high = (1 - min_side) * r1_low

    - For r2max=<float> & fractional=False:
    r2_low = r1_high + min_side & r2_high = r2max

    - For r2max=<float> & fractional="lower"
    r2_low = r1_high * (1 + min_side) & r2_high = r2max

    - For r2max=<float> & fractional=True:
    r2_low = r1_high * (1 + min_side) & r2_high = r1_high * r2max

    Parameters
    ----------
    cf : np.ndarray (of shape (N, N))
        The uncompressed three-point correlation function, binned
        symmetrically into r1 and r2
    r_edges : np.ndarray (of shape (N+1,))
        Bin edges for either r1 or r2 (must be the same)
    min_side : float (default = 30)
        Either the minimum triangle side-length, or the minimum
        side-length divided by r1 if fractional = True or "lower"
    r2max : Union[bool, float] (default = False)
        If False, r2 is bound to be less than r1. Otherwise, this
        must be a float to specify the upper bound of r2.
    fractional : Union[bool, str] (default = False)
        See the effect of this parameter on the compression schemes
    weighted : bool (default = True)
        If true, the compression is weighted by volume

    Returns
    -------
    compressed_cf : np.ndarray (of shape (N,))
        Three-point correlation function as a function of r1 only
    """
    weights = np.diff(r_edges**3) if weighted else np.ones_like(r_edges[:-1])

    # Select min_side < r2 < r1 - min_side [Mpc/h]
    if r2max:
        lows = [(1 + min_side) * edge if fractional
                else min_side + edge for edge in r_edges[1:]]
        highs = [r2max * edge if fractional
                 else r2max for edge in r_edges[1:]]
    else:
        lows = [min_side * edge for edge in r_edges[:-1]] if fractional \
            else [min_side] * len(r_edges[:-1])
        highs = [x - y for x, y in zip(r_edges[:-1], lows)]

    low_inds = [_closest(low, r_edges) for low in lows]
    high_inds = [_closest(high, r_edges) for high in highs]

    nans = [high <= low for high, low in zip(high_inds, low_inds)]
    slices = [slice(low, high) for low, high in zip(low_inds, high_inds)]
    ans = [np.nan if nan else np.average(cf[i, s], weights=weights[s])
           for i, (s, nan) in enumerate(zip(slices, nans))]
    return np.array(ans)


def _closest(val, vals):
    """
    Returns index of `vals` containing element which is
    closest to `val`. The lowest index, in case of tie.
    """
    vals = np.asarray(vals)
    return np.argmin(np.abs(vals - val))
