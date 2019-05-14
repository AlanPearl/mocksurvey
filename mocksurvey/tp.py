import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from . import cf

def plot_shade_err(x, y, err=None, color=None, label=None, alpha=0.8, saf=0.5, faf=0.5, line_kwa={}, scatter_kwa={}, fill_kwa={}, errorbar_kwa={}, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, color=color, alpha=alpha, **line_kwa)
    ax.scatter(x, y, color=color, label=label, alpha=alpha*saf, **scatter_kwa)
    if not err is None:
        if len(np.shape(err)) == 2:
            lower,upper = err
        else:
            lower = upper = err
        ax.fill_between(x, y-lower, y+upper, color=color, alpha=alpha*faf, **fill_kwa)
        ax.errorbar(x, y, yerr=(lower,upper), color=color, ecolor=color, alpha=alpha, **errorbar_kwa)
    return ax

def plot_halo_mass(field, nbins=50, from_gals=False, fontsize=14, ax=None):
    """Plot the halo mass function dN/dM_halo"""
    if from_gals:
        mass = field.simbox.gals["halo_mvir"]
    else:
        mass = field.simbox.halos["halo_mvir"]
    lims = np.log10(min(mass)), np.log10(max(mass))
    bins = np.logspace(lims[0], lims[1], nbins+1)
    
    if ax is None:
        ax = plt.gca()
    ax.hist(mass, bins=bins, density=True)
    ax.set_xlabel("$M_{\\rm halo}$ $(h^{-1} M_{\\ast})$", fontsize=fontsize)
    ax.set_ylabel("$dN/dM_{\\rm halo}$", fontsize=fontsize)
    ax.loglog()
    return ax

def plot_pos_scatter(field, s=0.1, fontsize=14, ax=None, realspace=False, plot_vel=False, axes=[0,2], **scatter_kwargs):
    data = field.get_data(realspace=realspace)
    rand = field.get_rands()
    
    s_rand = s / field.rand_density_factor
    
    if ax is None:
        ax = plt.gca()
    if plot_vel:
        vel = field.get_vel()
        ax.quiver(data[:,axes[0]], data[:,axes[1]], vel[:,axes[0]], vel[:,axes[1]])
    ax.scatter(rand[:,axes[0]], rand[:,axes[1]], s=s_rand, **scatter_kwargs)
    ax.scatter(data[:,axes[0]], data[:,axes[1]], s=s, **scatter_kwargs)
    axstr = ["x","y","z"]
    ax.set_xlabel("$%s$ ($h^{-1}$ Mpc)" %axstr[axes[0]], fontsize=fontsize)
    ax.set_ylabel("$%s$ ($h^{-1}$ Mpc)" %axstr[axes[1]], fontsize=fontsize)
    return ax

def plot_sky_scatter(field, s=0.1, ax=None, fontsize=14, **scatter_kwargs):
    data = field.get_data(rdz=True) * 180./np.pi
    rand = field.get_rands(rdz=True) * 180./np.pi
    s_rand = s / field.rand_density_factor
    
    if ax is None:
        ax = plt.gca()
    ax.scatter(rand[:,0], rand[:,1], s=s_rand, **scatter_kwargs)
    ax.scatter(data[:,0], data[:,1], s=s, **scatter_kwargs)
    ax.set_xlabel("$\\alpha$ (deg)", fontsize=fontsize)
    ax.set_ylabel("$\\delta$ (deg)", fontsize=fontsize)
    return ax

def plot_xi_rp_pi(field, realspace=False, rmax=25, fontsize=14, ax=None):
    rpbins = pibins = np.linspace(1e-5,rmax,31)
    pimax = pibins[-1]
    contour_levels = np.logspace(-2, 3, 21) # every 0.25 in log space
    linewidths = np.array([.2]*len(contour_levels))
    linewidths[contour_levels==1] = .6
    
    norm = mpl.colors.LogNorm()
    sm = mpl.cm.ScalarMappable(norm=norm); sm.set_array([])
    
    data = field.get_data(realspace=realspace)
    rand = field.get_rands()
    pc = cf.paircount_rp_pi(data, rand, rpbins, pimax=pimax)
    xi = cf.counts_to_xi(pc)
    xi[xi < 0] = 0 # don't allow xi(rp, pi) to go negative for log scale
    
    if ax is None:
        ax = plt.gca()
    lrbt = rpbins[0], rpbins[-1], pibins[0], pibins[-1]
    ax.contour(xi.T, contour_levels, colors='black', linewidths=linewidths)
    ax.imshow(xi.T, extent=lrbt, origin='lower', interpolation='bilinear', norm=norm)
    ax.set_xlabel("$r_{\\rm p}$ ($h^{-1}$ Mpc)", fontsize=fontsize)
    ax.set_ylabel("$\\pi$ ($h^{-1}$ Mpc)", fontsize=fontsize)
    plt.colorbar(sm, ax=ax).set_label("$\\xi(r_{\\rm p}, \\pi)$", fontsize=fontsize)
    return ax

def plot_wp_rp(field, realspace=False, fontsize=14, ax=None):
    rpbins = np.logspace(-0.87, 1.73, 14) # these bins approximately match those of Zehavi 2011
    rp = np.sqrt(rpbins[:-1]*rpbins[1:])
    
    data = field.get_data(realspace=realspace)
    rand = field.get_rands()
    pc = cf.paircount_rp_pi(data, rand, rpbins)
    wp = cf.counts_to_wp(pc)
    wp[wp < 0] = 0 # don't allow wp(rp) to go negative for log scale
    
    if ax is None:
        ax = plt.gca()
    ax.plot(rp, wp)
    ax.set_xlabel("$r_{\\rm p}$ ($h^{-1}$ Mpc)", fontsize=fontsize)
    ax.set_ylabel("$w_{\\rm p}$ ($h^{-1}$ Mpc)", fontsize=fontsize)
    return ax
    
def plot_hod_occupation(simbox, mass=None, **haloprop_kwargs):
    mass = np.logspace(11,15,501) if mass is None else mass
    ntot = simbox.get_halo_moments(mass, **haloprop_kwargs)
    ncen = simbox.get_halo_moments(mass, "central", **haloprop_kwargs)
    nsat = simbox.get_halo_moments(mass, "satellites", **haloprop_kwargs)
    
    from scipy import stats
    nsat_q = stats.poisson.ppf(q=np.array([.16,.5,.84])[:,None], mu=nsat[None,:])
    ncen_q = stats.binom.ppf(q=np.array([.16,.5,.84])[:,None], n=1, p=ncen[None,:])
    ntot_q = ncen_q + nsat_q
    
    ns = {"scatter_kwa": {"s":0}, "errorbar_kwa": {"lw":0}}
    fig,ax = plt.subplots(figsize=(16,4), ncols=3)
    lab = lambda string: "$\\langle N_{\\rm %s} \\rangle$"%string

    plot_shade_err(mass, np.array(ncen_q)[1], np.diff(np.array(ncen_q), axis=0), color="k", ax=ax[0], line_kwa={"lw":0}, **ns)
    plot_shade_err(mass, ncen, color="b", ax=ax[0], line_kwa={"label":lab("cen"), "lw":2}, **ns)
    ax[0].semilogx()

    plot_shade_err(mass, np.array(nsat_q)[1], np.diff(np.array(nsat_q), axis=0), color="k", ax=ax[1], line_kwa={"lw":0},  **ns)
    plot_shade_err(mass, nsat, color="b", ax=ax[1], line_kwa={"label":lab("sat"), "lw":2},**ns)
    ax[1].loglog()

    plot_shade_err(mass, np.array(ntot_q)[1], np.diff(np.array(ntot_q), axis=0), color="k", ax=ax[2], line_kwa={"lw":0},  **ns)
    plot_shade_err(mass, ntot, color="b", ax=ax[2], line_kwa={"label":lab("tot"), "lw":2},**ns)
    ax[2].loglog()

    [a.legend(loc=4 if a is ax[0] else 2, fontsize=14) for a in ax]
    [a.set_ylim([1e-2,50]) for a in ax[1:]]
    [a.set_xlabel("$M_{\\rm halo} \\;(M_\\odot)$", fontsize=14) for a in ax]