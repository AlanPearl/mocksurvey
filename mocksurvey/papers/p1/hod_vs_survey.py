import os
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
        sigma, alpha = params
        prior = self.lnprior(sigma, alpha)
        if not np.isfinite(prior):
            return prior

        # Calculate wp(rp) given HOD parameters
        wp = calc_wp_from_hod(self.model, alpha=alpha, sigma_logM=sigma,
                              reals=self.nreals)

        # Compare model to observation to calculate likelihood
        return self.lnlike(wp) + prior


def load_wpdata(file_pattern="wpreal_grids"):
    path, pattern = os.path.split(os.path.realpath(file_pattern))
    files = [os.path.join(path, x) for x in os.listdir(path) if x.startswith(pattern)]
    data = np.array([np.stack(np.load(fn, allow_pickle=True)) for fn in files])
    if not len(files):
        raise IOError(f"Could not find any files starting with {pattern} in {path}")

    wp = ms.stats.datadict_array_get(data, "wp")
    # shape = (num_files, 5, 25+, 6)
    # indexed by (grid, grid_point, realization, rp_scale
    imax, jmax = wp.shape[:2]
    cov_grid = np.empty(wp.shape[:2], dtype=object)
    mean_grid = cov_grid.copy()
    for i in range(imax):
        for j in range(jmax):
            cov_grid[i, j] = np.cov(wp[i, j], rowvar=False)
            mean_grid[i, j] = np.mean(wp[i, j], axis=0)
    cov_grid = np.array(cov_grid.tolist())
    mean_grid = np.array(mean_grid.tolist())
    return mean_grid, cov_grid


def mcmc_from_wpdata(model, mean_wp, cov_wp, backend_fn, name, newrun=True,
                     nreals=1, nwalkers=20, niter=500):
    cenhod = model["cenhod"]

    lnlike = stats.multivariate_normal(mean=mean_wp, cov=cov_wp).logpdf
    lnprob = LnProbHOD(lnlike, config.lnprior, model, nreals=nreals)

    guess = [cenhod.param_dict[p] for p in ["sigma_logM", "alpha"]]
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


# Streamline the calculation of wp(rp), varying any HOD parameter
# (except logMmin and logM1 which are set by number density)
def get_hod_params(cenhod, sathod, sigma_logM=None, alpha=None, logM0=None):
    assert cenhod.param_dict == sathod.param_dict
    halocat = cenhod.hod.halocat

    if sigma_logM is None:
        sigma_logM = cenhod.param_dict["sigma_logM"]
    if alpha is None:
        alpha = sathod.param_dict["alpha"]
    if logM0 is None:
        logM0 = sathod.param_dict["logM0"]
    logMmin = ms.diffhod.ConservativeHODZheng07Cen(
        halocat, cenhod.num_gals, sigma_logM=sigma_logM
    ).solve_logMmin()
    logM1 = ms.diffhod.ConservativeHODZheng07Sat(
        halocat, sathod.num_gals, alpha=alpha, logM0=logM0
    ).solve_logM1()
    return logMmin, logM1, sigma_logM, alpha, logM0


def calc_wp_from_hod(model, sigma_logM=None, alpha=None, logM0=None,
                     reals=5, take_mean=True):
    cenhod, sathod, rp_edges, boxsize, redshift = [
        model[s] for s in ["cenhod", "sathod", "rp_edges", "boxsize", "redshift"]]
    halocat = cenhod.hod.halocat

    logMmin, logM1, sigma_logM, alpha, logM0 = get_hod_params(
        cenhod, sathod, sigma_logM, alpha, logM0)
    hod = ms.diffhod.HODZheng07(
        halocat, logMmin=logMmin, sigma_logM=sigma_logM,
        alpha=alpha, logM0=logM0, logM1=logM1)
    ans = []
    for _ in range(reals):
        data = hod.populate_mock()
        pos = htmo.return_xyz_formatted_array(
            *ms.util.xyz_array(data).T, boxsize, ms.bplcosmo, redshift,
            velocity=data["vz"], velocity_distortion_dimension="z")

        ans.append(ms.stats.cf.wp_rp(
            pos, None, rp_edges, boxsize=boxsize))
    return np.mean(ans, axis=0) if take_mean else ans


def runmcmc(niter=1000, backend_fn="mcmc.h5", newrun=True, which_runs=None,
            wpdata_file_pattern="wpreal_grids.npy"):
    # Load data we already computed
    mean_grid, cov_grid = load_wpdata(file_pattern=wpdata_file_pattern)

    rp_edges = np.geomspace(1, 27, 7)
    boxsize = 250.0
    redshift = 1.0
    threshold = 10 ** 10.6
    cenhod, sathod = ms.diffhod.measure_hod(
        ms.UMConfig().load(redshift)[0], threshold=threshold)

    model = dict(cenhod=cenhod, sathod=sathod, redshift=redshift,
                 boxsize=boxsize, rp_edges=rp_edges)

    if which_runs is None:
        imax, jmax = mean_grid.shape[:2]  # = (4, 5)
        which_runs = [(i, j) for i in range(imax) for j in range(jmax)]
    num_runs = len(which_runs)
    run_num = 0
    for i, j in which_runs:
        run_num += 1
        print(f"Beginning run {run_num}/{num_runs}.", flush=True)
        mcmc_from_wpdata(model, mean_grid[i, j], cov_grid[i, j], backend_fn,
                         f"grid{i}-{j}", niter=niter, newrun=newrun)


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
