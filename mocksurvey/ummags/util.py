import os
import pathlib
import json
from packaging.version import parse as vparse
import numpy as np
import pandas as pd
import halotools as ht
import halotools.utils as ht_utils
import halotools.empirical_models as ht_empirical_models
import halotools.sim_manager as ht_sim_manager

from . import ummags
from .. import mocksurvey as ms
from . import kcorrect


def convert_ascii_to_npy_and_json(asciifile, outfilebase=None,
                                  remove_ascii_file=False, **kwargs):
    if outfilebase is None:
        outfilebase = ".".join(asciifile.split(".")[:-1])

    data = lightcone_from_ascii(asciifile, **kwargs)
    metadict = metadict_from_ascii(asciifile, **kwargs)

    np.save(outfilebase + ".npy", data)
    json.dump(metadict, open(outfilebase + ".json", "w"), indent=4)

    if remove_ascii_file:
        # Save disk space by deleting the huge ascii file
        os.remove(asciifile)


def lightcone_from_ascii(filename, photbands=None, obs_mass_limit=8e8,
                         true_mass_limit=0, cosmo=None):
    """
    Takes the ascii output given by UniverseMachine's `lightcone` code,
    and returns it as a numpy structured array, removing entries with
    mass lower than the specified limit. Reading the ascii table may take
    up to 20 minutes for large lightcones.

    Several new columns are added, calculated using functions in this
    module. Velocity-distorted positions replace the x, y, and z
    columns, and the old ones are renamed x_real, y_real, and z_real.
    Additionally, apparent magnitudes and distance modulus are calculated.
    Note that h-scaling is applied to all distances (including distance
    modulus), and therefore M = m - distmod + 5log(h).
    """
    if cosmo is None:
        cosmo = ms.bplcosmo
    photbands = get_photbands(photbands)

    cols = {"id": (5, "<i8"), "upid": (7, "<i8"),
            "x_real": (10, "<f4"), "y_real": (11, "<f4"),
            "z_real": (12, "<f4"), "vx": (13, "<f4"), "vy": (14, "<f4"),
            "vz": (15, "<f4"), "ra": (0, "<f4"), "dec": (1, "<f4"),
            "redshift": (2, "<f4"), "redshift_cosmo": (3, "<f4"),
            "scale_snapshot": (4, "<f4"), "obs_sm": (28, "<f4"),
            "obs_sfr": (29, "<f4"), "true_sm": (25, "<f4"),
            "true_sfr": (27, "<f4"), "halo_mvir": (16, "<f4"),
            "halo_mvir_peak": (18, "<f4"), "halo_vmax": (17, "<f4"),
            "halo_vmax_peak": (19, "<f4"), "halo_rvir": (20, "<f4"),
            "halo_delta_vmax_rank": (21, "<f4")}

    # Read in the ASCII table, make mass cut (this takes a while)
    masslimit = {"obs_sm": obs_mass_limit, "true_sm": true_mass_limit}
    reader = ht_sim_manager.tabular_ascii_reader.TabularAsciiReader(
                             filename, cols, row_cut_min_dict=masslimit)
    lc_data = reader.read_ascii()

    # Limit RA to the range [-180,180) (column = "ra")
    lc_data["ra"] = (lc_data["ra"] + 180) % 360 - 180
    # Calculate redshift-space-distorted positions (columns = "x","y","z")
    xyz_real = ms.util.xyz_array(lc_data, keys=["x_real", "y_real", "z_real"])
    vel = ms.util.xyz_array(lc_data, keys=["vx", "vy", "vz"])
    rdz = ms.util.ra_dec_z(xyz_real, vel, cosmo=cosmo)
    xyz = ms.util.rdz2xyz(rdz, cosmo=cosmo)

    # Calculate distance modulus (column = "distmod")
    dlum = cosmo.luminosity_distance(lc_data["redshift"]).value * cosmo.h
    distmod = 5 * np.log10(dlum * 1e5)
    dlum_true = cosmo.luminosity_distance(lc_data["redshift_cosmo"]
                                          ).value * cosmo.h
    distmod_cosmo = 5 * np.log10(dlum_true * 1e5)

    # Calculate apparent magnitudes (column = "m_g", "m_r", etc.)
    uvdat = ummags.UVData(photbands=photbands)
    umdat = ummags.UMData(lc_data, uvdat=uvdat)
    sfr_uv = 10 ** umdat.logssfr_uv * lc_data["obs_sm"]
    magdf = pd.DataFrame({k: umdat.abs_mag[k] + distmod_cosmo
                          for k in umdat.abs_mag.columns})

    # Name the new columns and specify their dtypes
    xyz_dtype = [(s, "f4") for s in ("x", "y", "z")]
    mag_dtype = [(f"m_{s}", "f4") for s in magdf.columns]
    other_dtype = [("sfr_uv", "<f4"),
                   ("distmod", "<f4"), ("distmod_cosmo", "<f4")]

    full_dtype = (xyz_dtype + lc_data.dtype.descr + mag_dtype +
                  other_dtype)

    # Copy all columns into a new structured numpy array
    final_lightcone_array = np.zeros(lc_data.shape, full_dtype)
    for i, (name, dtype) in enumerate(xyz_dtype):
        final_lightcone_array[name] = xyz[:, i]
    for (name, dtype) in mag_dtype:
        final_lightcone_array[name] = magdf[name[-1].lower()]
    for (name, dtype) in lc_data.dtype.descr:
        final_lightcone_array[name] = lc_data[name]
    final_lightcone_array["sfr_uv"] = sfr_uv
    final_lightcone_array["distmod"] = distmod
    final_lightcone_array["distmod_cosmo"] = distmod_cosmo

    return final_lightcone_array


def metadict_from_ascii(filename, photbands=None, obs_mass_limit=8e8,
                        true_mass_limit=0):
    photbands = get_photbands(photbands)

    with open(filename) as f:
        [f.readline() for _ in range(1)]
        cmd = " ".join(f.readline().split()[2:])
        seed = eval(f.readline().split()[-1])
        origin = [float(s) for s in f.readline().split()[-3:]]
        [f.readline() for _ in range(30)]
        rot_matrix = eval(("".join([f.readline()[1:].strip().replace(" ", ",")
                                    for _ in range(3)]))[:-1])

    (executable, config, z_low, z_high,
     x_arcmin, y_arcmin, samples) = cmd.split()[:7]

    header = """# Structured array of mock galaxy/halo properties
# Load via 
# >>> data = np.load("filename.npy")
# Cosmology: FlatLambdaCDM(H0=67.8, Om0=0.307, Ob0=0.048)
#
# Column descriptions:
# (Note: Observer at origin, and x-axis is approximately line-of-sight)
# ====================
# x_real, y_real, z_real - true comoving position in Mpc/h
# x,y,z - velocity-distorted comoving position in Mpc/h
# vx, vy, vz - Velocity in km/s
# ra,dec - celestial coords in degrees: range [-180,180) and [-90,90]
# redshift[_cosmo] - distorted [and cosmological] redshift
# m_{u,b,v,g,r,i,z,y,j,ch1,ch2} - app. magnitudes fit to UltraVISTA photometry
# obs_sm - "observed" stellar mass from UniverseMachine in Msun
# obs_sfr - "observed" SFR from UniverseMachine in Msun/yr
# true_{sm,sfr} - same, but UniverseMachine "truth" (not recommended to use)
# sfr_uv - ultraviolet SFR (abundance-matched to UltraVISTA) in Msun/yr
# distmod[_cosmo] - distorted [and cosmological] dist. modulus = m - M + 5logh
# Host halo information:
# =====================
# id - ID of host halo
# upid - ID of the central, or -1 if central
# halo_mvir[_peak] - mass in Msun [and the peak value over all time]
# halo_vmax[_peak] - maximum circular velocity in km/s [and peak value]
# halo_rvir - virial radius 
# halo_delta_vmax_rank - z-score of a proxy for halo accretion rate --> sSFR
# scale_snapshot - the scale-factor of the snapshot this halo was taken from
"""
    return dict(header=header, z_low=float(z_low), z_high=float(z_high),
                x_arcmin=float(x_arcmin), y_arcmin=float(y_arcmin),
                samples=int(samples), photbands=photbands,
                obs_mass_limit=obs_mass_limit, true_mass_limit=true_mass_limit,
                Rmatrix=rot_matrix, seed=seed, origin=origin,
                config=config, executable=executable, cmd=cmd)


def metadict_with_specprop(meta):
    if "header_specprop" in meta:
        del meta["header_specprop"]

    header_specprop = """# Structured array of spectral features.
# Nearest-neighbor matched to the synthetic spectra library compiled
# by Sean Johnson for galaxies in the UltraVISTA photometric survey.
# These spectra assume solar metallicity (Z = 0.02), the stellar mass
# given in the mock catalog, and an exponential SFH given by ltau/lage.
#
# Load via 
# >>> data = np.load("filename.specprop")
#
# Column descriptions:
# ====================
# m_HSC_{g,r,i,z,y} - Apparent magnitude integrated over the given HSC filter
# L_<line> - Intrinsic luminosity of the given <line> in erg/s
# f_<line> - Observed flux of the given <line> in erg/s/cm^2/Ang
# Wr_<line> - Equivalent width of the given <line> in Angstroms
# uvista_id - ID of the UltraVISTA galaxy this spectrum was simulated for
# uvista_{ra,dec} - Celestial coordinate of this UltraVISTA galaxy in degrees
# uvista_{ltau,lage,Av} - Properties (other than mass) used to generate spectra
"""
    return dict(header_specprop=header_specprop, **meta)


def metadict_with_spec(meta, ngal):
    if "header_spec" in meta:
        del meta["header_spec"]
    if "Ngal" in meta:
        del meta["Ngal"]
    if "Nwave" in meta:
        del meta["Nwave"]

    nwave = ms.SeanSpectraConfig().wavelength().size
    header_spec = """# Binary array of the spectrum of each galaxy in the mock.
# Nearest-neighbor matched to the synthetic spectra library compiled
# by Sean Johnson for galaxies in the UltraVISTA photometric survey.
#
# Flux units: nJy
# Wavelength grid [units: nm] = np.geomspace(
#     380.0, 1259.9885444552458, 74370)
#
# Loading instructions:
# =====================
# First, load the meta data to get the array's shape
# >>> meta = json.load(open("filename.json"))
# >>> shape = meta["Ngal"], meta["Nwave"]
#
# (Method A) Load the whole array
# >>> spec = np.fromfile("filename.spec", dtype="<f4").reshape(shape)
#
# (Method B) Load a single spectrum or masked selection
# >>> i = 123
# >>> spec = np.memmap("filename.spec", dtype="<f4", shape=shape)[i]
"""
    return dict(Ngal=ngal, Nwave=nwave, header_spec=header_spec, **meta)


def get_photbands(photbands, default=None):
    if default is None:
        default = ["u", "b", "v", "g", "r", "i", "z",
                   "y", "j", "h", "k", "ch1", "ch2"]
    if photbands is None:
        photbands = [s.lower() for s in default]
    else:
        photbands = [s.lower() for s in photbands]

    return photbands


def cam_const_z(m, prop, m2, prop2, z2, z_avg, dz, nwin=501):
    assert (vparse(ht.__version__) >= vparse("0.7dev"))
    z_sel = (z_avg - dz/2 <= z2) & (z2 <= z_avg + dz/2)

    nwin1 = min([nwin, z_sel.sum() // 2 * 2 - 1])
    if nwin1 < 2:
        print(f"Warning: Only {z_sel.sum()} galaxies in the z"
              f"={z_avg} +/- {dz}/2 bin. You should use a larger"
              f" value of dz than {dz}")

    new_prop = ht_empirical_models.conditional_abunmatch(
        m, prop, m2[z_sel], prop2[z_sel], nwin=nwin1)
    return new_prop


def cam_binned_z(m, z, prop, m2, z2, prop2, nwin=501, dz=0.05,
                 min_counts_in_z2_bins=None, seed=None):
    assert (vparse(ht.__version__) >= vparse("0.7dev"))
    assert (dz > 0)
    assert (np.shape(m) == np.shape(z)) and (np.shape(z) == np.shape(prop))

    if np.size(prop) == 0:
        return np.zeros_like(prop)
    if min_counts_in_z2_bins is None:
        min_counts_in_z2_bins = nwin+1

    zrange = z.min() - dz/20, z.max() + dz/20
    nz = int((zrange[1] - zrange[0]) / dz)
    if nz:
        centroids = np.linspace(*zrange, nz+1)
    else:
        # nz = 1
        centroids = np.array([-0.5, 0.5])*dz + np.mean(zrange)

    # noinspection PyArgumentList
    zmin, zmax = centroids.min(), centroids.max()
    s, s2 = (zmin < z) & (z < zmax), (zmin < z2) & (z2 < zmax)
    assert np.all(s)

    m2 = m2[s2]
    z2 = z2[s2]
    prop2 = prop2[s2]

    inds2 = ht_utils.fuzzy_digitize(z2, centroids, seed=seed,
                                    min_counts=min_counts_in_z2_bins)
    centroids: np.ndarray
    centroids, inds2 = ms.util.correction_for_empty_bins(centroids, inds2)
    inds = ms.util.fuzzy_digitize_improved(z, centroids, seed=seed,
                                           min_counts=min_counts_in_z2_bins)

    new_prop = np.full_like(prop, np.nan)
    for i in range(len(centroids)):
        s, s2 = inds == i, inds2 == i
        nwin1 = min([nwin, s2.sum()//2*2-1])
        if nwin1 < 2:
            print(f"Warning: Only {s2.sum()} galaxies in the z"
                  f"={centroids[i]} bin. You should use a larger"
                  f"value of dz than {dz}")
        new_prop[s] = ht_empirical_models.conditional_abunmatch(
            m[s], prop[s], m2[s2], prop2[s2], nwin1)

    return new_prop


def execute_lightcone_code(z_low, z_high, x_arcmin, y_arcmin,
                           executable=None, umcfg=None, samples=1,
                           id_tag="", do_collision_test=False, ra=0.,
                           dec=0., theta=0., rseed=None):
    # homedir = str(pathlib.Path.home()) + "/"
    if executable is None:
        # executable = homedir + "local/src/universemachine/lightcone"
        executable = ms.UMConfig().get_lightcone_executable()
    if umcfg is None:
        # umcfg = homedir + "data/LightCone/um-lightcone.cfg"
        umcfg = ms.UMConfig().get_lightcone_config()
    if rseed is not None:
        assert ms.util.is_int(rseed), "Random seed must be an integer"
    assert ms.util.is_int(samples), "Number of samples must be an integer"

    args = ["'"+str(id_tag)+"'", str(int(do_collision_test)),
            str(float(ra)), str(float(dec)), str(float(theta)), str(rseed)]

    if rseed is None:
        args.pop()
        if not theta:
            args.pop()
            if not dec:
                args.pop()
                if not ra:
                    args.pop()
                    if not do_collision_test:
                        args.pop()
                        if not id_tag:
                            args.pop()

    cmd = f"{str(executable)} {str(umcfg)} {float(z_low)} {float(z_high)} "\
          f"{float(x_arcmin)} {float(y_arcmin)} {samples} {' '.join(args)}"
    print(cmd)
    return os.system(cmd)


def default_lightcone_filenames(z_low, z_high, x_arcmin, y_arcmin,
                                samples=1, id_tag=""):
    if id_tag:
        id_tag += "_"
    return [f"survey_{id_tag}z{z_low:.2f}-{z_high:.2f}_" 
            f"x{x_arcmin:.2f}_y{y_arcmin:.2f}_{i}.dat"
            for i in range(samples)]


def generate_lightcone_filenames(args, outfilepath=None,
                                 outfilebase=None):
    fake_id = args.pop()
    z_low, z_high, x_arcmin, y_arcmin, samples, id_tag = args
    args[-1] = "" if args[-1] is None else args[-1]

    # If id_tag is provided AND outfilepath is not
    # then store in the default location
    if (id_tag is not None) and (outfilepath is None):
        try:
            outfilepath = ms.UMConfig().get_path()
        except ValueError:
            raise ValueError("To use an id-tag, you must first choose "
                             "a UMConfig directory to store the files.")

        outfilepath = os.path.join(outfilepath, "lightcones", id_tag)
        if pathlib.Path(outfilepath).is_dir():
            outfilepath = None
            raise IsADirectoryError(
                f"Lightcone with id-tag={id_tag} already exists at "
                f"{outfilepath}. Delete this directory or use a "
                f"different id-tag")

    # If id_tag is NOT provided, then make sure outfilepath is valid
    else:
        outfilepath = "" if outfilepath is None else outfilepath
        if not pathlib.Path(outfilepath).is_dir():
            raise NotADirectoryError(f"outfilepath={outfilepath} "
                                     f"must be a directory.")

    # If outfilebase NOT provided, use default universemachine naming
    if outfilebase is None:
        outfilebase = default_lightcone_filenames(*args)[0][:-6]

    # Make all names of files generated by the lightcone code
    outfilebase = os.path.join(outfilepath, outfilebase)
    asciifiles = default_lightcone_filenames(*args[:-1], fake_id)
    moved_asciifiles = [outfilebase + "_" + f.split("_")[-1]
                        for f in asciifiles]

    return asciifiles, moved_asciifiles


def progress_iterator(progress):
    if isinstance(progress, str):
        assert progress.lower() == "notebook", \
            f"progress={progress}. Must be one of {{True, False, 'notebook'}}"
        import tqdm.notebook as tqdm
        iterator = tqdm.trange
    elif progress:
        import tqdm
        iterator = tqdm.trange
    else:
        iterator = range

    return iterator


def fill_nan_spec(spec, window_len=1000, inplace=False):
    assert np.ndim(spec) < 3
    newdim = np.ndim(spec) < 2
    if inplace:
        assert isinstance(spec, np.ndarray)
    else:
        spec = np.array(spec)
    if newdim:
        spec = spec[None, :]
    shape = spec.shape

    # All NaNs must be one one side or the other
    isnan = np.isnan(spec)
    last_on_left = np.where(np.diff(isnan, axis=-1))
    assert np.all(np.arange(shape[0]) == last_on_left[0])
    last_on_left = last_on_left[1]

    window = np.arange(window_len)[None, :] + last_on_left[:, None]
    window[isnan[:, 0], :] += 1
    window[~isnan[:, 0], :] += 1 - window_len
    window[window < 0] = 0
    window[window >= shape[1]] = shape[1] - 1

    fill = spec[np.arange(shape[0])[:, None], window].mean(axis=1)
    fill = np.tile(fill[:, None], (1, spec.shape[1]))
    spec[isnan] = fill[isnan]

    if newdim:
        spec = spec[0, :]
    return spec


def count_nan(spec):
    assert np.ndim(spec) < 3
    newdim = np.ndim(spec) < 2
    if newdim:
        spec = spec[None, :]

    isnan = np.isnan(spec)
    last_on_left = np.where(np.diff(isnan, axis=-1))
    assert np.all(np.arange(spec.shape[0]) == last_on_left[0])

    counts = last_on_left[1] + 1
    counts[~isnan[:, 0]] -= spec.shape[1]
    if newdim:
        counts = counts[0]
    return counts


def fix_specprops_columns(nfinder, um_redshift, specprops, cosmo,
                          max_gband_nan=5000, specmap=None, progress=True):
    nbr_id = specprops["id"]
    redshifts = um_redshift
    masscorr = nfinder.masscorr(nbr_id)
    distcorr = ms.util.redshift_rest_flux_correction(
        specprops["redshift"], um_redshift, cosmo)
    wavelength = ms.SeanSpectraConfig().wavelength()
    hcorr = cosmo.h / 0.7

    # For calculating mag_vals (apparent magnitudes)
    def get_appmags(integrator, show_progress=True, save_specmap=None):
        iterator = ms.ummags.util.progress_iterator(show_progress)
        filternames = integrator.filter_strings

        sub_index = nfinder.sublist_indices(len(nbr_id))
        ans = np.empty((len(nbr_id), len(filternames)), dtype="<f8")
        for sub_i in iterator(len(sub_index)):
            sub_i = sub_index[sub_i]
            spectra = nfinder.avg_spectrum(nbr_id[sub_i],
                                           masscorr[sub_i],
                                           redshifts[sub_i])
            if save_specmap is not None:
                save_specmap[sub_i] = spectra
            fix_nans = np.abs(count_nan(spectra)) <= max_gband_nan
            spectra[fix_nans] = fill_nan_spec(spectra[fix_nans])
            for j in range(len(filternames)):
                ans[sub_i, j] = integrator.appmag_nm_njy(
                    spectra, filternames[j]
                )
        return [*np.transpose(ans)]

    # For calculating line_vals
    def get_line(linename):
        ans = specprops[linename]
        if linename.startswith("Wr_"):
            # Equivalent widths are not affected by redshift/distance
            return ans
        elif linename.startswith("L_"):
            # Luminosity is affected by the mass and assumed cosmology
            return ans * masscorr / hcorr ** 2
        elif linename.startswith("f_"):
            # Flux is affected by 1/(1+z)d^2 and mass but NOT cosmology
            return ans * distcorr * masscorr
        # elif linename.startswith("continuum_"): <-- these are all zeros
        else:
            raise ValueError(f"Invalid linename={linename}")

    # Collect data from UltraVISTA
    # ============================
    uvista_names = ["uvista_" + x for x in
                    ["id", "ra", "dec", "ltau", "lage", "Av"]]
    uvista_vals = [specprops[x[7:]] for x in uvista_names]
    uvista_dtypes = ["<i8" if x == "uvista_id"
                     else "<f4" for x in uvista_names]

    # Collect apparent magnitudes integrated from the spectra
    # =======================================================
    mag_names = [
        # "m_CFHT_U",
        "m_HSC_g", "m_HSC_r", "m_HSC_i", "m_HSC_z", "m_HSC_Y",
        # "m_VISTA_Y", "m_VISTA_J", "m_VISTA_H", "m_VISTA_Ks"
    ]
    fintegrator = kcorrect.OptimizedFilterIntegrator(
        wavelength, [x[2:] for x in mag_names])
    mag_vals = get_appmags(fintegrator, show_progress=progress,
                           save_specmap=specmap)
    mag_dtypes = ["<f4"] * len(mag_names)

    # Collect lines integrated from the spectra
    # =========================================
    line_names = [x for x in
                  ["L_SII_tot", "f_SII_tot", "L_SII6733", "f_SII6733", "Wr_SII6733",
                   "L_SII6718", "f_SII6718", "Wr_SII6718", "L_NII6585", "f_NII6585",
                   "Wr_NII6585", "L_NII6549", "f_NII6549", "Wr_NII6549", "L_Ha", "f_Ha",
                   "Wr_Ha", "L_OIII5008", "f_OIII5008", "Wr_OIII5008", "L_OIII4960",
                   "f_OIII4960", "Wr_OIII4960", "L_Hb", "f_Hb", "Wr_Hb", "L_Hg", "f_Hg",
                   "Wr_Hg", "L_Hd", "f_Hd", "Wr_Hd", "L_He", "f_He", "Wr_He", "L_Hz",
                   "f_Hz", "Wr_Hz", "L_NeIII3870", "f_NeIII3870", "Wr_NeIII3870",
                   "L_OII_tot", "f_OII_tot", "Wr_OII_tot", "L_OII3729", "f_OII3729",
                   "Wr_OII3729", "L_OII3727", "f_OII3727", "Wr_OII3727", "Wr_MgI2852",
                   "Wr_MgII2803", "Wr_MgII2796", "Wr_FeII2600", "Wr_FeII2586", "Wr_FeII2382",
                   "Wr_FeII2374", "Wr_FeII2344", "Wr_AlII1670", "Wr_FeII1608", "Wr_CIV1548",
                   "Wr_SiII1526", "Wr_SiIV1402", "Wr_SiIV1393", "Wr_CII1334", "Wr_OI1302",
                   "Wr_SiII1260", "continuum_Lya", "L_HI1215", "f_HI1215", "Wr_HI1215",
                   "Wr_SiII1190"] if np.any(specprops[x])]
    line_vals = [get_line(x) for x in line_names]
    line_dtypes = ["<f8" if x.startswith("L_") else "<f4" for x in line_names]

    newprops = ms.util.make_struc_array([*mag_names, *line_names, *uvista_names],
                                        [*mag_vals, *line_vals, *uvista_vals],
                                        [*mag_dtypes, *line_dtypes, *uvista_dtypes])
    return newprops
