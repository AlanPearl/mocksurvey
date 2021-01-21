import numpy as np
import pandas as pd
from scipy import stats
import halotools.sim_manager as htsim

from .. import mocksurvey as ms


def halo_conc(halos, vmax="v", mvir="m", rvir="r"):
    """
    Calculate NFW concentration parameter from Vmax, Mvir, and Rvir
    """
    gconst = 6.673e-11
    vmax = np.array(halos[vmax], dtype=np.float64) * 1e3
    mvir = np.array(halos[mvir], dtype=np.float64) * 1.989e30
    rvir = np.array(halos[rvir], dtype=np.float64) * (3.086e19 / ms.bplcosmo.h)
    rhs = 1.64 ** 2 * gconst * mvir / 4 / np.pi / rvir / vmax ** 2

    x = np.geomspace(3, 5000, 500000)[::-1]
    lhs = 1 / x * (np.log(1 + x) - x / (1 + x))
    return np.interp(rhs, lhs, x)


def separate_pos_column(halos):
    pos = halos["pos"].copy()
    pos[:, :3] %= 250.0
    orig_names, orig_vals = zip(*[(name, halos[name]) for name in
                                  halos.dtype.names if name != "pos"])
    names = ("x", "y", "z", "vx", "vy", "vz") + orig_names
    vals = (*pos.T,) + orig_vals
    return ms.util.make_struc_array(
        names, vals, ["<f4"] * 6 + [x[1] for x in halos.dtype.descr])


def make_primary_halocat_from_um(halos, redshift):
    halos = separate_pos_column(halos)
    is_cen = halos["upid"] == -1

    return htsim.UserSuppliedHaloCatalog(
        Lbox=250.0, particle_mass=1.55e8, redshift=redshift,
        halo_mvir=halos["m"][is_cen],
        halo_rvir=halos["r"][is_cen] / 1e3,
        halo_hostid=halos["upid"][is_cen],
        halo_nfw_conc=halo_conc(halos)[is_cen] / 5,
        **{f"halo_{x}": halos[x][is_cen] for x in
           ["x", "y", "z", "vx", "vy", "vz", "id", "upid"]}
    )


def get_hostid(halos, get_host_value="", drop_duplicates=True):
    # Construct hostid array with the ID of each halo's primary host
    hostid = np.array(halos["upid"], copy=True)
    primary_ids = np.asarray(halos["id"][hostid == -1])
    hostid[hostid == -1] = primary_ids

    halos = pd.DataFrame(halos, index=halos["id"])
    # If duplicates are not dropped, they will likely cause an error
    if drop_duplicates:
        halos = halos.drop_duplicates(subset=["id"])
    while True:
        is_orphan = np.isin(hostid, np.asarray(halos["id"]), invert=True)
        done = np.isin(hostid, primary_ids) | is_orphan
        if np.all(done):
            break
        else:
            hostid[~done] = halos["upid"].loc[hostid[~done]]

    if get_host_value:
        col = np.full(len(hostid), np.nan)
        col[~is_orphan] = halos[get_host_value].loc[hostid[~is_orphan]]
        return col
    else:
        return hostid


def count_sats_and_cens(halos, threshold):
    halos = pd.DataFrame(halos, index=halos["id"])
    hostid = get_hostid(halos)
    is_orphan = np.isin(hostid, np.asarray(halos["id"]), invert=True)
    is_primary = np.asarray(halos["upid"]) == -1
    primary_ids = np.asarray(halos["id"])[is_primary]

    # Count number of centrals
    over_thresh = np.asarray(halos["obs_sm"]) > threshold
    num_cens = over_thresh[is_primary].astype(int)

    # Count number of satellites
    counter = pd.value_counts(
        hostid[(~is_orphan) & (~is_primary) & over_thresh])
    num_sats = pd.Series(np.zeros_like(primary_ids), index=primary_ids)
    num_sats.loc[counter.index] = counter.values

    # Returns:
    #     - structured array of primary halos only (satellites/orphans removed)
    #     - number of central galaxies in each primary halo
    #     - number of satellites galaxies in each primary halo
    return halos[is_primary], num_cens, num_sats.values


def measure_cen_occ(halo_mass, num_cens, mhalo_edges, return_err=True):
    mhalo_hist = np.histogram(halo_mass, bins=mhalo_edges)[0]

    # Bin the central occupation by halo mass
    mean_occupation_cen = stats.binned_statistic(halo_mass, num_cens,
                                                 bins=mhalo_edges).statistic
    mean_occupation_cen_err = np.max(np.broadcast_arrays(np.sqrt(
        (1 - mean_occupation_cen) / mhalo_hist), 1e-5), axis=0)
    mean_occupation_cen_err = np.sqrt(mean_occupation_cen_err ** 2 +
                                      1 / mhalo_hist)

    # Return mean occupation of centrals [and uncertainties]
    ans = mean_occupation_cen
    if return_err:
        ans = ans, mean_occupation_cen_err
    return ans


def measure_sat_occ(halo_mass, num_sats, mhalo_edges, return_err=True):
    mhalo_hist = np.histogram(halo_mass, bins=mhalo_edges)[0]

    # Bin the satellite occupation by halo mass
    mean_occupation_sat = stats.binned_statistic(halo_mass, num_sats,
                                                 bins=mhalo_edges).statistic
    mean_occupation_sat_err = np.max(np.broadcast_arrays(np.sqrt(
        mean_occupation_sat / mhalo_hist), 1e-5), axis=0)

    # Return mean occupation of satellites [and uncertainties]
    ans = mean_occupation_sat
    if return_err:
        ans = ans, mean_occupation_sat_err
    return ans
