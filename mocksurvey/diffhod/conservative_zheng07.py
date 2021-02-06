import numpy as np
from scipy import optimize, special
import halotools.empirical_models as htem

from . import halocat as hc


# noinspection PyPep8Naming
class HODZheng07:
    def __init__(self, halocat, model=None, **params):
        self.model = htem.PrebuiltHodModelFactory(
            "zheng07", redshift=halocat.redshift) if model is None else model
        self.halocat = halocat
        if not set(params.keys()).issubset(self.model.param_dict.keys()):
            raise KeyError(f"Not all kwargs {list(params)} are allowed "
                           f"model params {list(self.model.param_dict)}")
        self.model.param_dict.update(params)

    def mean_num_cens(self, logMmin=None, sigma_logM=None):
        tmp = htem.PrebuiltHodModelFactory("zheng07")
        logMmin = self.model.param_dict["logMmin"] if logMmin is None else logMmin
        sigma_logM = self.model.param_dict["sigma_logM"] \
            if sigma_logM is None else sigma_logM
        tmp.param_dict.update(logMmin=logMmin, sigma_logM=sigma_logM)
        mvir = self.halocat.halo_table["halo_mvir"][
            self.halocat.halo_table["halo_upid"] == -1]
        return tmp.mean_occupation_centrals(prim_haloprop=mvir).sum()

    def mean_num_sats(self, alpha=None, logM1=None, logM0=None):
        tmp = htem.PrebuiltHodModelFactory("zheng07")
        alpha = self.model.param_dict["alpha"] if alpha is None else alpha
        logM1 = self.model.param_dict["logM1"] if logM1 is None else logM1
        logM0 = self.model.param_dict["logM0"] if logM0 is None else logM0
        tmp.param_dict.update(alpha=alpha, logM1=logM1, logM0=logM0)
        mvir = self.halocat.halo_table["halo_mvir"][
            self.halocat.halo_table["halo_upid"] == -1]
        return tmp.mean_occupation_satellites(prim_haloprop=mvir).sum()

    def populate_mock(self):
        self.model.populate_mock(self.halocat)
        return self.model.mock.galaxy_table

    @staticmethod
    def cen_occ(logM, logMmin, sigma_logM):
        return 0.5 * (1 + special.erf((logM - logMmin) / sigma_logM))

    @staticmethod
    def sat_occ(logM, alpha, logM1, logM0):
        return ((10 ** logM - 10 ** logM0) / 10 ** logM1) ** alpha


class BaseConservativeHODZheng07:
    def __init__(self, halocat, num_gals, **kwargs):
        # Default HOD parameters
        # ======================
        self.param_dict = dict(
            logMmin=(x := kwargs.get("logMmin", 13.0)),
            sigma_logM=kwargs.get("sigma_logM", 1.0),
            alpha=kwargs.get("alpha", 1.0),
            logM1=kwargs.get("logM1", x + 1.0),
            logM0=kwargs.get("logM0", x - 1.0),
        )
        self.min_dict = dict(
            logMmin=0, sigma_logM=0.1,
            alpha=0.1, logM1=0, logM0=0
        )
        self.max_dict = dict(
            logMmin=20, sigma_logM=3.0,
            alpha=3.0, logM1=20, logM0=20
        )

        # Initialize the HOD and store number of galaxies to be conserved
        # ===============================================================
        self.hod = HODZheng07(halocat, **self.param_dict)
        self.num_gals = num_gals

    def mean_num_gals(self, **kwargs):
        raise NotImplementedError("Must be implemented by child class")

    def solve(self, param_name, guess=None):
        x0 = self.param_dict[param_name] if guess is None else guess
        low, high = self.min_dict[param_name], self.max_dict[param_name]

        def zero(x):
            return self.mean_num_gals(**{param_name: x}) - self.num_gals

        def zero_sq(x):
            return zero(x) ** 2

        try:
            return optimize.newton(zero, x0, tol=1e-3)
        except (RuntimeError, RuntimeWarning):
            verbose = False
            if verbose:
                print("Newton didn't work. Let's try Brent")
            return optimize.minimize_scalar(
                zero_sq, bracket=[low, high]).x


# noinspection PyPep8Naming
class ConservativeHODZheng07Cen(BaseConservativeHODZheng07):
    def mean_num_gals(self, **kwargs):
        return self.hod.mean_num_cens(**kwargs)

    def solve_logMmin(self):
        # This guess assumes perfect correlation between
        # stellar and halo mass (i.e., step function HOD)
        # and that the halo mass function dn/dlogM = 2e18*M^-1.1
        guess = (2e18 / 1.1 / np.log(10) / self.num_gals) ** (1 / 1.1)
        return self.solve("logMmin", guess=np.log10(guess))

    def solve_sigma_logM(self):
        return self.solve("sigma_logM")


# noinspection PyPep8Naming
class ConservativeHODZheng07Sat(BaseConservativeHODZheng07):
    def mean_num_gals(self, **kwargs):
        return self.hod.mean_num_sats(**kwargs)

    def solve_alpha(self):
        return self.solve("alpha")

    def solve_logM1(self, guess=True):
        # This guess assumes that between M0 and Mhigh=4e13,
        # the halo mass function dn/dlogM = 2e18*M^-1.1
        alpha = self.param_dict["alpha"]
        if guess and alpha < 2:
            Mhigh = 4e13
            logM0 = self.param_dict["logM0"]
            M0 = 10 ** logM0
            if alpha == 1.1:
                intgrnd = np.log(Mhigh) - logM0
            else:
                exp = alpha - 1.1
                intgrnd = (Mhigh ** exp - M0 ** exp) / exp
            guess = (2e18 / np.log(10) * intgrnd / self.num_gals) ** (1 / alpha)
            return self.solve("logM1", guess=np.log10(guess))
        else:
            return self.solve("logM1")

    def solve_logM0(self):
        return self.solve("logM0")


def fit_hod_cen(primary_halocat, num_cens, mhalo_edges, plot=False):
    mean_occupation_cen, mean_occupation_cen_err = hc.measure_cen_occ(
        primary_halocat.halo_table["halo_mvir"], num_cens, mhalo_edges)
    mhalo_cens = np.sqrt(mhalo_edges[:-1] * mhalo_edges[1:])
    x = np.log10(mhalo_cens)

    def hod_cen(logm, sigma_logm):
        logmmin = ConservativeHODZheng07Cen(
            primary_halocat, num_cens.sum(),
            sigma_logM=sigma_logm).solve_logMmin()
        return HODZheng07.cen_occ(logm, logmmin, sigma_logm)

    def cost(sigma):
        z = (hod_cen(x, sigma) - mean_occupation_cen) / mean_occupation_cen_err
        return np.sum(z ** 2)

    bracket = [0.1, 1.5]
    result = optimize.minimize_scalar(cost, bracket=bracket, tol=1e-3)
    sigma_logm_fit = result.x
    logmmin_fit = ConservativeHODZheng07Cen(
        primary_halocat, num_cens.sum(), sigma_logM=sigma_logm_fit).solve_logMmin()

    if plot:
        # noinspection PyPackageRequirements
        import matplotlib.pyplot as plt

        print(bracket)
        print([cost(x) for x in bracket])
        print(result)

        plt.fill_between(x, mean_occupation_cen + mean_occupation_cen_err,
                         mean_occupation_cen - mean_occupation_cen_err, color="grey")
        plt.plot(x, hod_cen(x, sigma_logm_fit), "k--")

        label = (f"$\\rm \\log M_{{min}} = {logmmin_fit:.2f}$\n"
                 f"$\\rm \\sigma_{{\\log M}} = {sigma_logm_fit:.2f}$")
        plt.plot([], [], "k--", label=label)
        plt.legend(frameon=False, fontsize=16)

        plt.xlabel("$\\rm M_{halo}\\; [M_\\odot]$", fontsize=14)
        plt.ylabel("$\\rm \\langle N_{cen} \\rangle$", fontsize=14)
        plt.show()

    return sigma_logm_fit, logmmin_fit


def fit_hod_sat(primary_halocat, num_sats, mhalo_edges,
                guess_alpha_logm0=(1.15, 11.8), plot=False):
    mean_occupation_sat, mean_occupation_sat_err = hc.measure_sat_occ(
        primary_halocat.halo_table["halo_mvir"], num_sats, mhalo_edges)
    mhalo_cens = np.sqrt(mhalo_edges[:-1] * mhalo_edges[1:])
    x = np.log10(mhalo_cens)
    y = np.log10(mean_occupation_sat)
    y[~(y >= -5)] = -5
    yerr = (mean_occupation_sat_err / 10 ** y) / np.log(10)

    def hod_sat(logm, alpha, logm0, min_value=0.0):
        logm1 = ConservativeHODZheng07Sat(
            primary_halocat, num_sats.sum(), alpha=alpha,
            logM0=logm0).solve_logM1()
        ans = HODZheng07.sat_occ(logm, alpha, logm1, logm0)
        ans[~(ans > min_value)] = min_value
        return ans

    def cost(params):
        alpha, logm0 = params
        ans = hod_sat(x, alpha, logm0, min_value=1e-5)
        z = (np.log10(ans.astype(np.float64)) - y) / yerr
        return np.sum(z ** 2)

    p0 = guess_alpha_logm0
    # logm0_bounds = np.log10(np.quantile(
    #     primary_halocat.halo_table["halo_mvir"], [0.2, 0.95]))
    # logm0_low = min(p0[1], logm0_bounds[0])
    # logm0_high = max(p0[1], logm0_bounds[1])
    bounds = [(0.1, 11.4), (5, 12.4)]

    result = optimize.minimize(cost, p0, method="Nelder-Mead",
                               options=dict(ftol=1e-3))
    alpha_fit, logm0_fit = result.x
    logm1_fit = ConservativeHODZheng07Sat(
        primary_halocat, num_sats.sum(), alpha=alpha_fit,
        logM0=logm0_fit).solve_logM1()

    if plot:
        # noinspection PyPackageRequirements
        import matplotlib.pyplot as plt

        print(p0, bounds)
        print([cost(x) for x in [p0, *bounds]])
        print(result)

        plt.fill_between(x, mean_occupation_sat + mean_occupation_sat_err,
                         mean_occupation_sat - mean_occupation_sat_err, color="grey")
        plt.semilogy(x, hod_sat(x, alpha_fit, logm0_fit), "k--")
        # plt.semilogy(x, hod_sat(x, *p0), "r--")

        label = (f"$\\rm \\alpha = {alpha_fit:.2f}$\n"
                 f"$\\rm \\log M_1 = {logm1_fit:.2f}$\n"
                 f"$\\rm \\log M_0 = {logm0_fit:.2f}$")
        plt.plot([], [], "k--", label=label)
        plt.legend(frameon=False, fontsize=16)

        plt.xlabel("$\\rm M_{halo}\\; [M_\\odot]$", fontsize=14)
        plt.ylabel("$\\rm \\langle N_{sat} \\rangle$", fontsize=14)
        plt.show()

    return alpha_fit, logm0_fit, logm1_fit


def measure_hod(halos, threshold, plot=False):
    # Construct halotools catalog and count the occupation in each halo
    halocat = hc.make_primary_halocat_from_um(halos, redshift=0)
    primaries, num_cens, num_sats = hc.count_sats_and_cens(
        hc.separate_pos_column(halos), threshold=threshold)

    # Fit central and satellite HOD parameters to the halo catalog
    mhalo_edges = np.logspace(11.25, 14, 31)
    sigma_logm, logmmin = fit_hod_cen(halocat, num_cens,
                                      mhalo_edges, plot=plot)
    guess = (1.15, logmmin + 0.1)
    alpha, logm0, logm1 = fit_hod_sat(halocat, num_sats,
                                      mhalo_edges,
                                      guess_alpha_logm0=guess,
                                      plot=plot)
    params = dict(sigma_logM=sigma_logm, logMmin=logmmin,
                  alpha=alpha, logM0=logm0, logM1=logm1)
    return (ConservativeHODZheng07Cen(halocat, num_cens.sum(), **params),
            ConservativeHODZheng07Sat(halocat, num_sats.sum(), **params))
