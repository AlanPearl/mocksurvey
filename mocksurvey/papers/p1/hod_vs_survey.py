import functools

import numpy as np
# noinspection PyUnresolvedReferences
import halotools.mock_observables as htmo
# noinspection PyUnresolvedReferences
from scipy import stats
# noinspection PyUnresolvedReferences
import emcee

# noinspection PyUnresolvedReferences
import mocksurvey as ms
from . import config


class LnProbHOD:
    def __init__(self, lnlike, lnprior, model,
                 nreals=1):
        self.lnlike = lnlike
        self.model = model
        self.lnprior = lnprior
        self.nreals = nreals

    def __call__(self, params):
        sigma, alpha, fsat = params
        prior = self.lnprior(sigma, alpha, fsat)
        if not np.isfinite(prior):
            return prior

        # Calculate wp(rp) given HOD parameters
        wp = calc_wp_from_hod(self.model, alpha=alpha, sigma=sigma,
                              reals=self.nreals)

        # Compare model to observation to calculate likelihood
        return self.lnlike(wp) + prior


def load_wpdata(filename, set_all_means_same=True):
    # path, pattern = os.path.split(os.path.realpath(filename))
    # files = [os.path.join(path, x) for x in os.listdir(path) if x == pattern]
    # data = np.array([np.stack(np.load(fn, allow_pickle=True)) for fn in files])
    data = np.load(filename, allow_pickle=True)
    # if not len(files):
    #     raise IOError(f"Could not find any files starting with {pattern} in {path}")

    wp = ms.stats.datadict_array_get(data, "wp")
    # shape = (5, 25+, 6)
    # indexed by (grid_point, realization, rp_scale)
    imax = wp.shape[0]
    cov_grid = np.empty(imax, dtype=object)
    mean_grid = cov_grid.copy()
    for i in range(imax):
        cov_grid[i] = np.cov(wp[i], rowvar=False, ddof=1)
        mean_grid[i] = np.mean(wp[i], axis=0)
    cov_grid = np.array(cov_grid.tolist())
    mean_grid = np.array(mean_grid.tolist())

    if set_all_means_same:
        var_grid = cov_grid[:,
                            np.arange(cov_grid.shape[1]),
                            np.arange(cov_grid.shape[2])]
        mean_of_means = np.average(mean_grid, weights=1/var_grid, axis=0)
        mean_grid[:, :] = mean_of_means[None, :]
    return mean_grid, cov_grid


def mcmc_from_wpdata(model, mean_wp, cov_wp, backend_fn, name, newrun=True,
                     nreals=1, nwalkers=20, niter=500):
    hod = model["hod"]

    lnlike = stats.multivariate_normal(mean=mean_wp, cov=cov_wp).logpdf
    lnprob = LnProbHOD(lnlike, config.lnprior, model, nreals=nreals)

    guess = [hod.param_dict[p] for p in ["sigma", "alpha", "fsat"]]
    ndim = len(guess)

    # Initialize the sampler
    if newrun:
        initial = np.array(guess)[None, :] + np.random.randn(
            nwalkers, ndim) * 0.1
    else:
        initial = None

    backend = emcee.backends.HDFBackend(backend_fn, name=name)
    if newrun:
        backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, backend=backend)

    sampler.run_mcmc(initial, niter, progress=True)

    return sampler


def calc_wp_from_hod(model, sigma=None, alpha=None, fsat=None, logM0=None,
                     reals=1, take_mean=True):
    hodsolver, rp_edges, boxsize, redshift = [
        model[s] for s in ["hod", "rp_edges", "boxsize", "redshift"]]
    halocat = hodsolver.halocat

    params = dict(sigma=sigma, alpha=alpha, fsat=fsat, logM0=logM0)
    hod_params = hodsolver.get_hod_params(**params)

    hod = ms.diffhod.HODZheng07(
        halocat, **hod_params)
    ans = []
    for _ in range(reals):
        data = hod.populate_mock()
        pos = htmo.return_xyz_formatted_array(
            *ms.util.xyz_array(data).T, boxsize, ms.bplcosmo, redshift,
            velocity=data["vz"], velocity_distortion_dimension="z")

        ans.append(ms.stats.cf.wp_rp(
            pos, None, rp_edges, boxsize=boxsize))
    return np.mean(ans, axis=0) if take_mean else ans


def runmcmc(gridname, niter=1000, backend_fn="mcmc.h5", newrun=True,
            which_runs=None, wpdata_file="wpreal_grids.npy"):
    # Load data we already computed
    mean_grid, cov_grid = load_wpdata(filename=wpdata_file)
    params = config.ModelConfig(gridname)

    rp_edges = params.rp_edges
    boxsize = params.boxsize
    redshift = params.redshift
    threshold = params.threshold

    halocat, closest_redshift = ms.UMConfig().load(redshift)
    hod = ms.diffhod.ConservativeHOD(halocat, threshold=threshold)
    print(f"Closest snapshot to z={redshift} is z={closest_redshift}")
    print({"ndensity [h/Mpc]^3": hod.num_gals / boxsize**3,
           **hod.param_dict, **hod.get_hod_params()})

    model = dict(hod=hod, redshift=redshift,
                 boxsize=boxsize, rp_edges=rp_edges)

    if which_runs is None:
        imax = mean_grid.shape[0]  # = 5
        which_runs = list(range(imax))
    num_runs = len(which_runs)
    run_num = 0
    for i in which_runs:
        run_num += 1
        print(f"Beginning run {run_num}/{num_runs}.", flush=True)
        mcmc_from_wpdata(model, mean_grid[i], cov_grid[i], backend_fn,
                         f"{gridname}.{i}", niter=niter, newrun=newrun)


def wpreals(gridname, mockname, save=True, nrand=int(5e5), newrun=True,
            nreal=None):
    # Function that we apply to each lightcone realization
    # ====================================================
    def calc_wp_from_lightcone(lightcone, _loader):
        pos = ms.util.xyz_array(lightcone, ["ra", "dec", "redshift"])
        rands = _loader.selector.make_rands(nrand, rdz=True)
        pos[:, 2] = ms.util.comoving_disth(pos[:, 2], cosmo=ms.bplcosmo)
        rands[:, 2] = ms.util.comoving_disth(rands[:, 2], cosmo=ms.bplcosmo)
        wp = ms.stats.cf.wp_rp(pos, rands, rp_edges, is_celestial_data=True)
        return ms.stats.DataDict({"wp": wp, "N": len(lightcone)})

    @functools.lru_cache
    def wp_reals_from_survey_params(completeness, sqdeg):
        print(f"Completeness = {completeness}, sqdeg = {sqdeg}", flush=True)
        assert 0 <= completeness <= 1, f"Completeness must be between 0 and 1"
        assert max_sqdeg >= sqdeg, f"Area is too big. Must be <= {max_sqdeg}"

        selector2 = ms.LightConeSelector(zlim[0], zlim[1], sqdeg=sqdeg,
                                         sample_fraction=completeness)
        wp_reals = loader.apply(calc_wp_from_lightcone, progress=True,
                                secondary_selector=selector2)
        return np.array(wp_reals)

    # Survey and wp(rp) parameters
    # ============================
    params = config.SurveyParamGrid(gridname)
    rp_edges, zlim, threshold = params.rp_edges, params.zlim, params.threshold
    selector = ms.LightConeSelector(zlim[0], zlim[1],
                                    min_dict=dict(obs_sm=threshold))

    # Initialize the realization loader
    # =================================
    loader = ms.RealizationLoader(mockname, selector=selector, nreal=nreal)
    max_sqdeg = ms.util.selector_from_meta(loader.meta[0]).sqdeg
    loader.load_all()

    # Calculate wp realizations over the following survey parameter grids
    # ===================================================================
    completeness_grid = params.completeness_grid
    sqdeg_grid = params.sqdeg_grid

    wp_grid = np.array([
        wp_reals_from_survey_params(c, s)
        for c, s in zip(completeness_grid, sqdeg_grid)
    ])

    # Save wp realization grids to file "wpreal_grids.npy"
    # ====================================================
    fn = save if isinstance(save, str) else f"wpreal_{gridname}.npy"
    save = bool(save)
    if not newrun:
        saved_grid = np.load(fn, allow_pickle=True)
        wp_grid = np.concatenate([saved_grid, wp_grid], axis=1)
    if save:
        np.save(fn, wp_grid)
