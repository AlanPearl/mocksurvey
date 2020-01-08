import os
import pathlib
import shutil
import json
from packaging.version import parse as vparse
import numpy as np
import pandas as pd
import halotools as ht
from .. import main as ms

def makelightcones(z_low, z_high, x_arcmin, y_arcmin,
                   executable=None, umcfg=None, samples=1,
                   photbands=None, keep_ascii_files=False,
                   obs_mass_limit=8e8, true_mass_limit=0,
                   outfilepath=None, outfilebase=None, id_tag=None,
                   do_collision_test=False, ra=0.,
                   dec=0., theta=0., rseed=None):

    assert(vparse(ht.version.version) >= vparse("0.7dev"))
    if not ms.UVISTACache().are_all_files_cached():
        raise IOError("You have not specified paths to all UltraVISTA data. "
            "Please use UVISTACache('path/to/dir').auto_add()")
    if not ms.UMCache().is_lightcone_ready():
        raise IOError("You must set paths to the lightcone executable and "
            "config files via UMCache('path/to/dir').set_lightcone_"
            "<executable/config>('path/to/file')")

    if executable is None:
        executable = ms.UMCache().get_lightcone_executable()
    if umcfg is None:
        umcfg = ms.UMCache().get_lightcone_config()

    # Predict/generate filenames
    fake_id = "_tmp_file_made_by__mocksurvey.makelightcones_"
    args = [z_low, z_high, x_arcmin, y_arcmin, samples, id_tag, fake_id]
    files, moved_files = _generate_lightcone_filenames(args, outfilepath,
                                                       outfilebase)
    # Execute the lightcone code in the UniverseMachine package
    if _execute_lightcone_code(*args[:4], executable, umcfg, samples,
                fake_id, do_collision_test, ra, dec, theta, rseed):
        raise RuntimeError("lightcone code failed")

    # Move lightcone files to their desired locations
    for origin, destination in zip(files, moved_files):
        shutil.move(origin, destination)

    # Convert the enormous ascii file into a binary table + meta data
    for filename in moved_files:
        convert_ascii_to_npy_and_json(filename, photbands=photbands,
                                        obs_mass_limit=obs_mass_limit,
                                        true_mass_limit=true_mass_limit)

        # Save disk space by deleting the huge ascii files
        if not keep_ascii_files:
            os.remove(filename)

def convert_ascii_to_npy_and_json(asciifile, outfilebase=None,
                                  *args, **kwargs):
    if outfilebase is None:
        outfilebase = ".".join(asciifile.split(".")[:-1])

    lightcone = lightcone_from_ascii(asciifile, *args, **kwargs)
    metadict = metadict_from_ascii(asciifile)

    np.save(outfilebase + ".npy", lightcone)
    json.dump(metadict, open(outfilebase + ".json", "w"))

def metadict_from_ascii(filename):
    with open(filename) as f:
        [f.readline() for i in range(1)]
        cmd = " ".join(f.readline().split()[2:])
        seed = eval(f.readline().split()[-1])
        origin = [float(s) for s in f.readline().split()[-3:]]
        [f.readline() for i in range(30)]
        Rmatrix = eval(("".join([f.readline()[1:].strip().replace(" ", ",")
                                 for i in range(3)]))[:-1])

    return dict(Rmatrix=Rmatrix, seed=seed, origin=origin, cmd=cmd)

def lightcone_from_ascii(filename, photbands=None, obs_mass_limit=8e8,
                         true_mass_limit=0):
    """
    Takes the ascii output given by UniverseMachine's `lightcone` code,
    and returns it as a numpy structured array, removing entries with
    mass lower than the specified limit. Reading the ascii table may take
    up to 20 minutes for large lightcones.

    Several new columns are added, using calculations performed by functions
    in this module. Velocity-distorted positions replace the x, y, and z
    columns, and the old ones are renamed x_real, y_real, and z_real.
    Additionally, absolute magnitudes (- 5logh) and distance modulus
    (+ 5logh) are calculated. Note that the h-scaling cancels out if
    you are interested in relative magnitudes. M_V + distmod = m_v
    as expected.
    """
    photbands = _get_photbands(photbands)

    cols = {"id": (5, "i8"), "upid": (7, "i8"),
            "x_real": (10, "f4"), "y_real": (11, "f4"),
            "z_real": (12, "f4"), "vx": (13, "f4"), "vy": (14, "f4"),
            "vz": (15, "f4"), "ra": (0, "f4"), "dec": (1, "f4"),
            "redshift": (2, "f4"), "redshift_cosmo": (3, "f4"),
            "scale_snapshot": (4, "f4"), "obs_sm": (28, "f4"),
            "obs_sfr": (29, "f4"), "true_sm": (25, "f4"),
            "true_sfr": (27, "f4"), "halo_mvir": (16, "f4")}

    masslimit = {"obs_sm":obs_mass_limit, "true_sm":true_mass_limit}
    reader = ht.sim_manager.tabular_ascii_reader.TabularAsciiReader(
                             filename, cols, row_cut_min_dict=masslimit)
    lightcone = reader.read_ascii()

    xyz_real = ms.hf.xyz_array(lightcone, keys=["x_real","y_real","z_real"])
    vel = ms.hf.xyz_array(lightcone, keys=["vx", "vy", "vz"])
    rdz = ms.hf.ra_dec_z(xyz_real, vel, cosmo=ms.bplcosmo)

    xyz = ms.hf.rdz2xyz(rdz, cosmo=ms.bplcosmo)
    distmod = 5 * np.log10(np.linalg.norm(xyz_real, axis=1) * 1e5)
    magdf = get_lightcone_UMmags(lightcone, photbands=photbands)

    # Copy all data into a new structured numpy array
    xyz_dtype = [(s, "f4") for s in ("x", "y", "z")]
    mag_dtype = [("M_" + s.upper(), "f4") for s in magdf.columns]
    full_dtype = xyz_dtype + lightcone.dtype.descr + mag_dtype
    full_dtype.append(("distmod", "f4"))
    final_lightcone_array = np.zeros(lightcone.shape, full_dtype)

    for i, (name, dtype) in enumerate(xyz_dtype):
        final_lightcone_array[name] = xyz[:, i]
    for (name, dtype) in mag_dtype:
        final_lightcone_array[name] = magdf[name[-1].lower()]
    for (name, dtype) in lightcone.dtype.descr:
        final_lightcone_array[name] = lightcone[name]
    final_lightcone_array["distmod"] = distmod

    return final_lightcone_array



def get_lightcone_UMmags(UMhalos, photbands=None, nwin=501, zbin_min=0.05):
    logm = np.log10(UMhalos["obs_sm"])
    logssfr = np.log10(UMhalos["obs_sfr"]) - logm
    z = UMhalos["redshift"]

    reg, photbands, (uvista_z, uvista_logm,
        uvista_logssfr_uv) = setup_uvista_mag_regressor(photbands)

    logssfr_uv = cam_binned_z(m=logm, z=z, prop=logssfr, m2=uvista_logm,
        z2=uvista_z, prop2=uvista_logssfr_uv, nwin=nwin, zbin_min=zbin_min)

    predictor = _make_predictor_UMmags(reg, logm, logssfr_uv, photbands)
    return predictor(slice(None), redshifts=z)


def make_predictor_UMmags(UMhalos, z_avg, photbands=None, nwin=501, dz=0.2):
    logm = np.log10(UMhalos["obs_sm"])
    logssfr = np.log10(UMhalos["obs_sfr"]) - logm

    reg, photbands, (uvista_z, uvista_logm,
        uvista_logssfr_uv) = setup_uvista_mag_regressor(photbands)

    logssfr_uv = cam_const_z(m=logm, prop=logssfr, m2=uvista_logm,
        z2=uvista_z, prop2=uvista_logssfr_uv, z_avg=z_avg, dz=dz)

    predictor = _make_predictor_UMmags(reg, logm, logssfr_uv, photbands)
    return predictor


def _get_photbands(photbands):
    if photbands is None:
        photbands = ["j", "y", "g", "r"]
    else:
        photbands = [s.lower() for s in photbands]
    #         if not ("i" in photbands):
    #             photbands.append("i")
    #         if not ("k" in photbands):
    #             photbands.append("k")

    return photbands


def setup_uvista_mag_regressor(photbands):
    photbands = _get_photbands(photbands)

    UVISTA = ms.UVISTACache()
    UVISTA.PHOTBANDS = {k: UVISTA.PHOTBANDS[k]
                        for k in set(photbands) | {"k"}}
    UVISTAcat = UVISTA.load()

    uvista_z = UVISTAcat["z"]
    uvista_logm = UVISTAcat["logm"]
    uvista_logssfr_uv = np.log10(UVISTAcat["sfr_uv"]) - uvista_logm

    names = ["M_" + key.upper() for key in photbands]
    uvista_m2l = [uvista_logm + UVISTAcat[name] / 2.5 for name in names]

    s = np.isfinite(uvista_logssfr_uv)
    x = np.array([uvista_logssfr_uv, uvista_z]).T[s]
    y = np.array(uvista_m2l).T[s]

    from sklearn import ensemble
    reg = ensemble.RandomForestRegressor(n_estimators=10)
    reg.fit(x, y)

    return reg, photbands, (uvista_z, uvista_logm, uvista_logssfr_uv)


def cam_const_z(m, prop, m2, prop2, z2, z_avg, dz):
    assert (vparse(ht.version.version) >= vparse("0.7dev"))
    z_sel = (z_avg - dz/2 <= z2) & (z2 <= z_avg + dz/2)
    new_prop = empirical_models.conditional_abunmatch(m, prop,
                                m2[z_sel], prop2[z_sel], nwin=nwin)
    return logssfr_uv


def cam_binned_z(m, z, prop, m2, z2, prop2, nwin=501, zbin_min=0.05):
    assert (vparse(ht.version.version) >= vparse("0.7dev"))
    assert (zbin_min > 0)
    zrange = z.min() - zbin_min/20, z.max() + zbin_min/20
    nz = int((zrange[1] - zrange[0]) / zbin_min)
    bin_edges = np.linspace(*zrange, nz+1)

    new_prop = np.zeros_like(prop)
    for i in range(nz):
        lower, upper = bin_edges[i:i+2]
        s, s2 = ((lower <= a) & (a < upper) for a in (z, z2))
        new_prop[s] = ht.empirical_models.conditional_abunmatch(
            m[s], prop[s], m2[s2], prop2[s2], nwin)

    return new_prop


def _make_predictor_UMmags(regressor, logm, logssfr_uv, photbands):
    """Define predictor in another function to
    reduce the memory required for closure"""

    def predictor(selection, redshifts):
        """
        predict absolute magnitudes for UniverseMachine galaxies,
        given individual observed redshifts for each galaxy
        """
        x = np.array([logssfr_uv[selection], redshifts]).T
        y = regressor.predict(x)
        Mag = -2.5 * (np.asarray(logm)[selection, None] - y)
        return pd.DataFrame(Mag, columns=photbands)

    return predictor


def _execute_lightcone_code(z_low, z_high, x_arcmin, y_arcmin,
                            executable=None, umcfg=None, samples=1,
                            id_tag="",do_collision_test=False, ra=0.,
                            dec=0., theta=0., rseed=None):
    homedir = str(pathlib.Path.home()) + "/"
    if executable is None:
        executable = homedir + "local/src/universemachine/lightcone"
    if umcfg is None:
        umcfg = homedir + "data/LightCone/um-lightcone.cfg"
    if not rseed is None:
        assert(isinstance(rseed, int)), "Random seed must be an integer"
    assert(isinstance(samples, int)), "Number of samples must be an integer"

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

    cmd = f"{str(executable)} {str(umcfg)} {float(z_low)} {float(z_high)} " \
          f"{float(x_arcmin)} {float(y_arcmin)} {samples} {' '.join(args)}"
    print(cmd)
    return os.system(cmd)

def _default_lightcone_filenames(z_low, z_high, x_arcmin, y_arcmin,
                                samples=1, id_tag=""):
    if id_tag: id_tag += "_"
    return [f"survey_{id_tag}z{z_low:.2f}-{z_high:.2f}_" 
            f"x{x_arcmin:.2f}_y{y_arcmin:.2f}_{i}.dat"
            for i in range(samples)]


def _generate_lightcone_filenames(args, outfilepath=None, outfilebase=None):
    fake_id = args.pop()
    z_low, z_high, x_arcmin, y_arcmin, samples, id_tag = args

    outfilepath = "" if outfilepath is None else outfilepath
    args[-1] = "" if args[-1] is None else args[-1]
    if outfilebase is None:
        outfilebase = _default_lightcone_filenames(*args)[0][:-6]

    outfilebase = os.path.join(outfilepath, outfilebase)
    asciifiles = _default_lightcone_filenames(*args[:-1], fake_id)
    moved_asciifiles = [outfilebase + "_" + f.split("_")[-1]
                        for f in asciifiles]

    return asciifiles, moved_asciifiles
