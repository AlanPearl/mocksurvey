import numpy as np
import halotools.mock_observables as htmo
from scipy import stats
import emcee
import tqdm

import mocksurvey as ms
from . import config


class LnProbHOD:
    def __init__(self, lnlike, lnprior, model,
                 nreals=1):
        self.hod = model["hod"]
        self.lnlike = lnlike
        self.model = model
        self.lnprior = lnprior
        self.nreals = nreals

        self._nanwp = np.array([np.nan] * 6)

    def __call__(self, params):
        # sigma, alpha, fsat = params
        param_dict = dict(zip(["sigma", "alpha", "fsat"], params))
        prior = self.lnprior(**param_dict)
        if not np.isfinite(prior):
            return prior, self._nanwp, -99

        # Calculate wp(rp) given HOD parameters
        wp, n = calc_wp_and_n_from_hod(self.model, **param_dict)

        logmmin = self.model["hod"].cenhod.param_dict["logMmin"]
        logm1 = self.model["hod"].sathod.param_dict["logM1"]
        numcen = self.model["hod"].num_cens
        numsat = self.model["hod"].num_sats
        # Compare model to observation to calculate likelihood
        return self.lnlike(wp) + prior, wp, n, logmmin, logm1, numcen, numsat


def load_wpdata(filename, get_random_real=False, set_all_means_same=True):
    data = np.load(filename, allow_pickle=True)
    if get_random_real and isinstance(get_random_real, bool):
        random_real = np.random.randint(data.shape[1])
    elif get_random_real:
        random_real = get_random_real

    wp = ms.stats.datadict_array_get(data, "wp")
    # shape = (5, 25+, 6)
    # indexed by (grid_point, realization, rp_scale)
    imax = wp.shape[0]
    cov_grid = np.empty(imax, dtype=object)
    mean_grid = cov_grid.copy()
    for i in range(imax):
        cov_grid[i] = np.cov(wp[i], rowvar=False, ddof=1)
        if get_random_real:
            mean_grid[i] = wp[i][random_real]
        else:
            mean_grid[i] = np.mean(wp[i], axis=0)
    cov_grid = np.array(cov_grid.tolist())
    mean_grid = np.array(mean_grid.tolist())

    if set_all_means_same and not get_random_real:
        var_grid = cov_grid[:,
                            np.arange(cov_grid.shape[1]),
                            np.arange(cov_grid.shape[2])]
        mean_of_means = np.average(mean_grid, weights=1/var_grid, axis=0)
        mean_grid[:, :] = mean_of_means[None, :]
    return mean_grid, cov_grid


def mcmc_from_wpdata(model, mean_wp, cov_wp, backend_fn, name, newrun=True,
                     nwalkers=20, niter=1000, use_fsat=False):
    param_names = ["sigma", "alpha"]
    if use_fsat:
        param_names.append("fsat")

    hod = model["hod"]
    guess = [hod.param_dict[p] for p in param_names]
    ndim = len(guess)

    lnlike = stats.multivariate_normal(mean=mean_wp, cov=cov_wp).logpdf
    lnprob = LnProbHOD(lnlike, config.lnprior, model)

    # Initialize the sampler
    if newrun:
        initial = np.array(guess)[None, :] + np.random.randn(
            nwalkers, ndim) * 0.1
    else:
        initial = None

    backend = emcee.backends.HDFBackend(backend_fn, name=name)
    if newrun:
        backend.reset(nwalkers, ndim)

    blobtype = [("wp", float, 6), ("N", int),
                ("logMmin", float), ("logM1", float),
                ("numcen", float), ("numsat", float)]
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, backend=backend, blobs_dtype=blobtype)

    sampler.run_mcmc(initial, niter, progress=True)

    return sampler


def calc_wp_and_n_from_hod(model, **params):
    hod, rp_edges, boxsize, redshift = [
        model[s] for s in ["hod", "rp_edges", "boxsize", "redshift"]]

    data = hod.populate_mock(**params)
    pos = htmo.return_xyz_formatted_array(
        *ms.util.xyz_array(data).T, boxsize, ms.bplcosmo, redshift,
        velocity=data["vz"], velocity_distortion_dimension="z")

    return ms.stats.cf.wp_rp(pos, None, rp_edges, boxsize=boxsize), len(data)


def runmcmc(gridname, niter=1000, backend_fn="mcmc.h5", newrun=True,
            which_runs=None, wpdata_file="wpreal_grids.npy",
            use_fsat=False, use_cfcmr=False, random_real=False,
            set_all_means_same=True):
    # Load data we already computed
    if random_real and isinstance(random_real, bool):
        length = np.load(wpdata_file, allow_pickle=True).shape[1]
        random_real = np.random.randint(length)
    mean_grid, cov_grid = load_wpdata(filename=wpdata_file,
                                      get_random_real=random_real,
                                      set_all_means_same=set_all_means_same)
    params = config.ModelConfig(gridname)

    rp_edges = params.rp_edges
    boxsize = params.boxsize
    redshift = params.redshift
    threshold = params.threshold

    # Selecting M_halo > 1e10 removes ~70% of the halos which is a big speedup
    halocat, closest_redshift = ms.UMConfig().load(
        redshift, thresh=lambda cat: cat["m"] > 1e10)
    if use_cfcmr:
        assert not use_fsat, "fsat parameter not allowed in the CFCMR HOD"
        hod = ms.diffhod.CFCMRHOD(halocat, threshold=threshold)
    else:
        hod = ms.diffhod.ConservativeHOD(halocat, threshold=threshold)
    print(f"{'All' if which_runs is None else which_runs} of {gridname} MCMC")
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
        name = f"{gridname}.{i}"
        if not isinstance(random_real, bool):
            name += f".real{random_real}"
        run_num += 1
        print(f"Beginning run {run_num}/{num_runs}.", flush=True)
        mcmc_from_wpdata(model, mean_grid[i], cov_grid[i], backend_fn,
                         name, niter=niter, newrun=newrun,
                         use_fsat=use_fsat)


class LightConeWpCalculator:
    def __init__(self, nrand, rp_edges, recycle_rands=True,
                 rands=None, rr=None):
        self.nrand = nrand
        self.rp_edges = rp_edges
        self.recycle_rands = recycle_rands

        self.rands, self.RR = rands, rr

    def __call__(self, lightcone, loader):
        pos = ms.util.xyz_array(lightcone, ["ra", "dec", "redshift"])
        pos[:, 2] = ms.util.comoving_disth(pos[:, 2], cosmo=ms.bplcosmo)

        if self.rands is not None:
            rands = self.rands
        else:
            rands = loader.selector.make_rands(self.nrand, rdz=True)
            rands[:, 2] = ms.util.comoving_disth(rands[:, 2], cosmo=ms.bplcosmo)
            if self.recycle_rands:
                self.rands = rands

        paircounts = ms.stats.cf.paircount_rp_pi(
            pos, rands, self.rp_edges, is_celestial_data=True,
            precomputed=(None, None, self.RR))
        if self.recycle_rands:
            self.RR = paircounts.RR

        wp = ms.stats.cf.counts_to_wp(paircounts)
        return ms.stats.DataDict({"wp": wp, "N": len(lightcone)})

    def get_rands_and_rr(self, selector):
        if self.rands is not None:
            rands = self.rands
        else:
            rands = selector.make_rands(self.nrand, rdz=True)
            rands[:, 2] = ms.util.comoving_disth(rands[:, 2], cosmo=ms.bplcosmo)
            if self.recycle_rands:
                self.rands = rands

        rr = ms.stats.cf.paircount_rp_pi(
            np.zeros((0, 3)), rands, self.rp_edges, is_celestial_data=True,
            precomputed=(None, None, self.RR)).RR

        return rands, rr


def wpreals(gridname, mockname, save=True, nrand=int(5e5), newrun=True,
            nreal=None, recycle_rands=True, randfile="", just_make_rands=False):
    def wp_reals_from_survey_params(completeness, sqdeg, rands=None, rr=None):
        assert 0 <= completeness <= 1, f"Completeness must be between 0 and 1"
        assert max_sqdeg >= sqdeg, f"Area is too big. Must be <= {max_sqdeg}"

        selector2 = ms.LightConeSelector(zlim[0], zlim[1], sqdeg=sqdeg,
                                         sample_fraction=completeness)
        calc = LightConeWpCalculator(nrand=nrand, rp_edges=rp_edges,
                                     recycle_rands=recycle_rands,
                                     rands=rands, rr=rr)
        if just_make_rands:
            return calc.get_rands_and_rr(selector2)

        wp_reals = loader.apply(calc, progress=True,
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

    # Get survey parameter grids
    # ==========================
    completeness_grid = params.completeness_grid
    sqdeg_grid = params.sqdeg_grid
    print(f"{gridname} completeness = {completeness_grid}", flush=True)

    if just_make_rands:
        assert randfile, "Must provide filename for randoms/RR array"
        randrr = np.array([
            wp_reals_from_survey_params(c, s)
            for c, s in tqdm.tqdm(list(zip(
                completeness_grid, sqdeg_grid)), desc=f"{gridname} ({mockname})")
        ], dtype=object)
        np.save(randfile, randrr)
        return

    if randfile:
        randrr = np.load(randfile, allow_pickle=True)
    else:
        n = len(completeness_grid) + len(sqdeg_grid)
        randrr = np.full((n, 2), None)

    # Calculate wp realizations
    # =========================
    wp_grid = np.array([
        wp_reals_from_survey_params(c, s, *r)
        for c, s, r in tqdm.tqdm(list(zip(
            completeness_grid, sqdeg_grid, randrr)), desc=gridname)
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
