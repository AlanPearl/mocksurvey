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
        ax.fill_between(x, y-err, y+err, color=color, alpha=alpha*faf, **fill_kwa)
        ax.errorbar(x, y, yerr=err, color=color, ecolor=color, alpha=alpha, **errorbar_kwa)

def plot_halo_mass(field, nbins=50, from_gals=False, fontsize=14, ax=None, plot=True):
    """Plot the halo mass function dN/dM_halo"""
    if from_gals:
        mass = field.simbox.gals["halo_mvir"]
    else:
        mass = field.simbox.halos["halo_mvir"]
    lims = np.log10(min(mass)), np.log10(max(mass))
    bins = np.logspace(lims[0], lims[1], nbins+1)
    
    if plot:
        if ax is None:
            ax = plt.gca()
        ax.hist(mass, bins=bins, density=True)
        ax.set_xlabel("$M_{\\rm halo}$ $(h^{-1} M_{\\ast})$", fontsize=fontsize)
        ax.set_ylabel("$dN/dM_{\\rm halo}$", fontsize=fontsize)
        ax.loglog()
    return mass

def plot_pos_scatter(field, s=0.1, fontsize=14, ax=None, plot=True, realspace=False, plot_vel=False, axes=[0,2], **scatter_kwargs):
    data = field.get_data(realspace=realspace)
    rand = field.get_rands()
    
    s_rand = s / field.rand_density_factor
    
    if plot:
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
    return data, rand

def plot_sky_scatter(field, s=0.1, ax=None, fontsize=14, plot=True, **scatter_kwargs):
    data = field.get_data(rdz=True) * 180./np.pi
    rand = field.get_rands(rdz=True) * 180./np.pi
    s_rand = s / field.rand_density_factor
    
    if plot:
        if ax is None:
            ax = plt.gca()
        ax.scatter(rand[:,0], rand[:,1], s=s_rand, **scatter_kwargs)
        ax.scatter(data[:,0], data[:,1], s=s, **scatter_kwargs)
        ax.set_xlabel("$\\alpha$ (deg)", fontsize=fontsize)
        ax.set_ylabel("$\\delta$ (deg)", fontsize=fontsize)
    return data, rand

def plot_xi_rp_pi(field, realspace=False, fontsize=14, ax=None, plot=True):
    rpbins = pibins = np.linspace(1e-5,30,31)
    dpi = pibins[1] - pibins[0]; pimax = pibins[-1]
    contour_levels = np.logspace(-2, 3, 21) # every 0.25 in log space
    linewidths = np.array([.2]*len(contour_levels))
    linewidths[contour_levels==1] = .6
    
    norm = mpl.colors.LogNorm()
    sm = mpl.cm.ScalarMappable(norm=norm); sm.set_array([])
    
    data = field.get_data(realspace=realspace)
    rand = field.get_rands()
    pc = cf.paircount_rp_pi(data, rand, rpbins, pimax=pimax, dpi=dpi)
    xi = cf.counts_to_xi(pc)
    xi[xi < 0] = 0 # don't allow xi(rp, pi) to go negative for log scale
    
    if plot:
        if ax is None:
            ax = plt.gca()
        lrbt = rpbins[0], rpbins[-1], pibins[0], pibins[-1]
        ax.contour(xi.T, contour_levels, colors='black', linewidths=linewidths)
        ax.imshow(xi.T, extent=lrbt, origin='lower', interpolation='bilinear', norm=norm)
        ax.set_xlabel("$r_{\\rm p}$ ($h^{-1}$ Mpc)", fontsize=fontsize)
        ax.set_ylabel("$\\pi$ ($h^{-1}$ Mpc)", fontsize=fontsize)
        plt.colorbar(sm, ax=ax)
    return xi

def plot_wp_rp(field, realspace=False, fontsize=14, ax=None, plot=True):
    rpbins = np.logspace(-0.87, 1.73, 14) # these bins approximately match those of Zehavi 2011
    rp = np.sqrt(rpbins[:-1]*rpbins[1:])
    
    data = field.get_data(realspace=realspace)
    rand = field.get_rands()
    pc = cf.paircount_rp_pi(data, rand, rpbins)
    wp = cf.counts_to_wp(pc)
    wp[wp < 0] = 0 # don't allow wp(rp) to go negative for log scale
    
    if plot:
        if ax is None:
            ax = plt.gca()
        ax.plot(rp, wp)
        ax.set_xlabel("$r_{\\rm p}$ ($h^{-1}$ Mpc)", fontsize=fontsize)
        ax.set_ylabel("$w_{\\rm p}$ ($h^{-1}$ Mpc)", fontsize=fontsize)
    return wp
    
