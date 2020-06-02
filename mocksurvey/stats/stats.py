import numpy as np
import halotools as ht

from .. import mocksurvey as ms
from . import cf

def wprp_tot_sf_q(data, rands, rpbins=None, ssfr_cut=1e-11, los=0,
                  shuffle_sfr=False):
    # Argument parsing
    xyz = ms.util.xyz_array(data)
    los = (los + 1) % 3
    if rpbins is None:
        rpbins = np.logspace(-1, 1.5, 11)

    # Find star-forming vs. quiescents
    is_sf = data["obs_sfr"] / data["obs_sm"] > ssfr_cut
    if shuffle_sfr:
        np.random.shuffle(is_sf)

    # Change convention so that z is along line-of-sight
    if not los == 0:
        xyz = np.roll(xyz, -los, axis=1)
        rands = np.roll(rands, -los, axis=1)
    # Calculate wp(rp)
    wp_tot = cf.wp_rp(xyz, rands, rpbins=rpbins)
    wp_sf = cf.wp_rp(xyz[is_sf], rands, rpbins=rpbins)
    wp_q = cf.wp_rp(xyz[~is_sf], rands, rpbins=rpbins)

    return rpbins, wp_tot, wp_sf, wp_q

def qpgf(catalog, rands, search_rad=4.0, search_min=0.3, pimax=None,
         bin_edges=None, primary_mass_lims=None, mass_ratio_lims=None,
         ssfr_cut=1e-11, period=None, primary_search_rad=0.5,
         primary_pimax=None, precomputed_primary_selection=None,
         cosmo=None):
    if pimax is None:
        cosmo = ms.bplcosmo if cosmo is None else cosmo
        pimax = ms.util.vpmax2pimax(1000., catalog["redshift"], cosmo)
    pos = ms.util.xyz_array(catalog)
    mass = catalog["obs_sm"]
    sfr = catalog["obs_sfr"]
    return qpgf_vs_cia(pos, mass, sfr, search_rad, search_min, pimax,
            bin_edges=bin_edges, primary_mass_lims=primary_mass_lims,
            mass_ratio_lims=mass_ratio_lims, ssfr_cut=ssfr_cut,
            period=period, primary_search_rad=primary_search_rad,
            primary_pimax=primary_pimax,
            precomputed_primary_selection=precomputed_primary_selection)

def find_primaries(sample1, sample2, search_rad, pimax, m1, m2, period=None):
    n_larger = _nic(sample1, sample2, search_rad, pimax, m1, m2,
                   inclusive_bounds=[False, True],
                   mass_ratio_lims=[1, np.inf], period=period)
    return n_larger == 0

def qpgf_vs_cia(pos, mass, sfr, search_rad, search_min, pimax,
                bin_edges=None, primary_mass_lims=None,
                mass_ratio_lims=None, ssfr_cut=1e-11,
                period=None, primary_search_rad=0.5,
                primary_pimax=None, precomputed_primary_selection=None):
    """
    Quenched Primary Galaxy Fraction
    vs.
    Counts in Annuli
    ================================
    For each galaxy in `sample1`, this function counts the number of "neighbors"
    in `sample2` within an annulus centered around the galaxy of inner/outer
    radii `search_min` and `search_rad` respectively (xy-plane), and within a
    line-of-sight distance 2*`pimax` (z-axis). This process is done for both
    star-forming galaxies (ssfr >= `ssfr_cut`) and quiescents, and the galaxies
    are binned (according to `bin_edges`) by neighbor counts.

    By default (from Behroozi et al. 2019):
    Primary galaxies are any galaxy whose mass is greater than
    any other within R_p < 0.5 Mpc and 1000 km/s in redshift. They
    are also seleted by 10 < log(M_star/M_sun) < 10.5

    Neighbors are only counted if their mass is between 30% and 100%
    that of the primary. Neighbors are selected by an annulus of
    0.3 Mpc < R_p < 4 Mpc and 1000 km/s in redshift.
    """
    # Default parameters
    primary_pimax = pimax if primary_pimax is None else primary_pimax
    bin_edges = (np.array([-0.5, *np.logspace(-0.375, 2.125, 11)[1:]]
                          ) if bin_edges is None else bin_edges)
    pm_min, pm_max = ([10 ** 10, 10 ** 10.5] if primary_mass_lims
                                   is None else primary_mass_lims)
    mr_lims = ([0.3, 1.0] if mass_ratio_lims
                is None else mass_ratio_lims)

    # First perform the mass selection
    is_neighbor = (mr_lims[0] * pm_min <= mass) & (
                    mass <= mr_lims[1] * pm_max)
    is_primary = (pm_min <= mass) & (mass <= pm_max)

    # Find primaries using given isolation criteria
    if precomputed_primary_selection is None:
        psr = primary_search_rad[is_primary] if ms.util.is_arraylike(
            primary_search_rad) else primary_search_rad
        pp = primary_pimax[is_primary] if ms.util.is_arraylike(
            primary_pimax) else primary_pimax

        is_primary[is_primary] &= find_primaries(
            pos[is_primary], pos[mass > pm_min], psr, pp,
            mass[is_primary], mass[mass > pm_min], period=period)
    else:
        is_primary &= precomputed_primary_selection

    # Divide primaries into star-forming and quenched with sSFR cut
    is_b_prim = np.zeros_like(is_primary)
    is_b_prim[is_primary] = sfr[is_primary] / mass[is_primary] >= ssfr_cut
    is_r_prim = is_primary & (~is_b_prim)

    # Perform neighbor/primary selection on position arrays
    xyz, sm = pos[is_neighbor], mass[is_neighbor]
    xyz_b, sm_b = pos[is_b_prim], mass[is_b_prim]
    xyz_r, sm_r = pos[is_r_prim], mass[is_r_prim]

    # Perform primary selection on cylinder size arrays, if needed
    search_rad_b = search_rad[is_b_prim] if ms.util.is_arraylike(
        search_rad) else search_rad
    search_rad_r = search_rad[is_r_prim] if ms.util.is_arraylike(
        search_rad) else search_rad
    search_min_b = search_min[is_b_prim] if ms.util.is_arraylike(
        search_min) else search_min
    search_min_r = search_min[is_r_prim] if ms.util.is_arraylike(
        search_min) else search_min
    pimax_b = pimax[is_b_prim] if ms.util.is_arraylike(
        pimax) else pimax
    pimax_r = pimax[is_r_prim] if ms.util.is_arraylike(
        pimax) else pimax

    # Counts in the full cylinder
    num_b = _nic(xyz_b, xyz, search_rad_b, pimax_b,
                sm_b, sm, mr_lims, period=period)
    num_r = _nic(xyz_r, xyz, search_rad_r, pimax_r,
                sm_r, sm, mr_lims, period=period)

    # Subtract counts from inner cylinder (so only counts in annulus are included)
    num_b -= _nic(xyz_b, xyz, search_min_b, pimax_b,
                 sm_b, sm, mr_lims, period=period)
    num_r -= _nic(xyz_r, xyz, search_min_r, pimax_r,
                 sm_r, sm, mr_lims, period=period)

    hist_b = np.histogram(num_b, bins=bin_edges)[0]
    hist_r = np.histogram(num_r, bins=bin_edges)[0]
    # Quenched fraction in each bin
    return bin_edges, hist_r / (hist_b + hist_r)

def _nic(sample1, sample2, search_rad, pimax, m1, m2, mass_ratio_lims, period=None, inclusive_bounds=None):
    """
    Neighbors in Cylinders
    ======================
    For each galaxy in `sample1`, this function returns the number of galaxies
    in `sample2` within a cylinder centered around the galaxy of radius
    `search_rad` (xy-plane) and height 2*`pimax` (z-axis).

    Note:
    Neighbors are defined as having a mass ratio between 0.3 and 1.0
    in Behroozi et al. (2019). The annulus used to select them is
    between 0.3 Mpc < R_p < 4 Mpc and 1000 km/s in redshift.

    Note when calculating quenched primary galaxy fractions, they
    define a primary galaxy as any galaxy whose mass is greater than
    any other within R_p < 0.5 Mpc and 1000 km/s in redshift
    """
    if inclusive_bounds is None:
        inclusive_bounds = [False, False]

    return ht.mock_observables.counts_in_cylinders(
        sample1, sample2, search_rad, pimax, period,
        condition="mass_frac", condition_args=
        (m1, m2, mass_ratio_lims, *inclusive_bounds))

class Observable:
    def __init__(self, funcs, names=None, args=None, kwargs=None):
        N = len(funcs)
        self.funcs = funcs
        self.names = range(N) if names is None else names
        self.funcdic = dict(zip(self.names,self.funcs))
        if isinstance(args, dict):
            self.argsdic = args.copy()
        else:
            self.argsdic = dict(zip(self.names,[()]*N)) if args is None else dict(zip(names,args))
        if isinstance(kwargs, dict):
            self.kwargsdic = kwargs.copy()
        else:
            self.kwargsdic = dict(zip(self.names,[{}]*N)) if kwargs is None else dict(zip(names,kwargs))
        self.indexdic = {}
        self.lendic = {}

        self.mean = None
        self.mean_jack = None
        self.covar_jack = None
        self.mean_real = None
        self.covar_real = None
        self.mean_rand = None
        self.covar_rand = None

    def get_jackknife(self, name=None):
        return self.get_data(name, method="jackknife")

    def get_realization(self, name=None):
        return self.get_data(name, method="realization")

    def get_random_realization(self, name=None):
        return self.get_data(name, method="random_realization")

    def get_data(self, name=None, method=None):
        accepted = ["jackknife", "realization", "random_realization"]
        if not method is None and not (method in accepted):
            raise ValueError(f"`method` must be one of {accepted}")
        if method is None:
            mean, covar = self.mean, None
        else:
            mean = self.__dict__["mean_"+method[:4]]
            covar = self.__dict__["covar_"+method[:4]]
        if mean is None:
            if method is None:
                raise ValueError("Must calculate the observables first.\n"
                        "Use <Observable object>.obs_func(data,rands)")
            else:
                raise ValueError("Must run this method first.\n"
                    f"Use <Observable object>.{method}(data,rands,...)")


        if name is None:
            return mean if (method is None) else (mean, covar)
        else:
            index0 = self.indexdic[name]
            index1 = index0 + self.lendic[name]
            s = slice(index0, index1)
            return mean[s] if (method is None) else (mean[s], covar[s,s])

    def jackknife(self, data, rands, centers, fieldshape, nbins=(2,2,1), data_to_bin=None, rands_to_bin=None, **kwargs):

        self.mean_jack, self.covar_jack = cf.block_jackknife(data, rands, centers, fieldshape, nbins,
                                                                   data_to_bin, rands_to_bin, self.obs_func, [], {"store": False}, **kwargs)
        return self.mean_jack, self.covar_jack

    def realization(self, rands, field, nrealization=25, **get_data_kw):
        data = field.get_data(**get_data_kw)
        samples = [self.obs_func(data, rands, store=False)]
        if len(samples[0]) >= nrealization:
            print("`nrealization` should probably be greater than the number of observables", flush=True)

        for i in range(nrealization-1):
            field.simbox.populate_mock()
            data = type(field)(**field._kwargs_).get_data(**get_data_kw)
            samples.append(self.obs_func(data, rands, store=False))

        samples = np.array(samples)
        self.mean_real = np.mean(samples, axis=0)
        self.covar_real = np.cov(samples, rowvar=False)
        return self.mean_real, self.covar_real

    def random_realization(self, data, field, nrealization=25, **get_rands_kw):
        rands = field.get_rands(**get_rands_kw)
        samples = [self.obs_func(data, rands, store=False)]
        if len(samples[0]) >= nrealization:
            print("`nrealization` should probably be greater than the number of observables", flush=True)

        for i in range(nrealization-1):
            field.make_rands()
            data = field.get_rands(**get_rands_kw)
            samples.append(self.obs_func(data, rands, store=False))

        samples = np.array(samples)
        self.mean_real = np.mean(samples, axis=0)
        self.covar_real = np.cov(samples, rowvar=False)
        return self.mean_rand, self.covar_rand

    def obs_func(self, data, rands=None, store=True, param_dict=None):
        if param_dict is None:
            param_dict = {}
        supported_params = {"icc"}
        if not set(param_dict.keys()).issubset(supported_params):
            raise ValueError(f"param_dict={param_dict} contains illegal keys."
                             f"\nAllowed keys must be in: {supported_params}")
        answers = []
        i = 0
        for name in self.names:
            func = self.funcdic[name]
            args = self.argsdic[name]
            kwargs = self.kwargsdic[name]

            ans = np.atleast_1d(func(data,rands,*args,**kwargs))
            # Integral constraint constant
            # ============================
            if "icc" in param_dict and name.lower().startswith("wp"):
                ans -= param_dict["icc"]

            answers.append(ans)

            l = len(ans)
            if not name in self.indexdic:
                self.indexdic[name] = i
                self.lendic[name] = l
            i += l

        answer = np.concatenate(answers)
        if store:
            self.mean = answer
        return answer
