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
from .. import mocksurvey as ms

def lightcone(z_low, z_high, x_arcmin, y_arcmin,
                   executable=None, umcfg=None, samples=1,
                   photbands=None, keep_ascii_files=False,
                   obs_mass_limit=8e8, true_mass_limit=0,
                   outfilepath=None, outfilebase=None, id_tag=None,
                   do_collision_test=False, ra=0.,
                   dec=0., theta=0., rseed=None):

    # Predict/generate filenames
    fake_id = "_tmp_file_made_by_mocksurvey_lightcone_"
    args = [z_low, z_high, x_arcmin, y_arcmin, samples, id_tag, fake_id]
    files, moved_files = _generate_lightcone_filenames(args, outfilepath,
                                                       outfilebase)
    # Check prerequisites
    assert(vparse(ht.version.version) >= vparse("0.7dev"))
    if not ms.UVISTAConfig().are_all_files_stored():
        raise ValueError("You have not specified paths to all "
                      "UltraVISTA data. "
            "Please use UVISTAConfig('path/to/dir').auto_add()")
    if not ms.UMConfig().is_lightcone_ready():
        raise ValueError("You must set paths to the lightcone executable and"
            " config files via UMConfig('path/to/dir').set_lightcone_"
            "executable/config('path/to/file')")

    if executable is None:
        executable = ms.UMConfig().get_lightcone_executable()
    if umcfg is None:
        umcfg = ms.UMConfig().get_lightcone_config()

    # Execute the lightcone code in the UniverseMachine package
    if _execute_lightcone_code(*args[:4], executable, umcfg, samples,
                fake_id, do_collision_test, ra, dec, theta, rseed):
        raise RuntimeError("lightcone code failed")

    # Move lightcone files to their desired locations
    pathlib.Path(moved_files[0]).parent.mkdir(parents=True, exist_ok=True)
    for origin, destination in zip(files, moved_files):
        pathlib.Path(origin).rename(destination)

    # Convert the enormous ascii file into a binary table + meta data
    for filename in moved_files:
        convert_ascii_to_npy_and_json(filename,
            remove_ascii_file=not keep_ascii_files, photbands=photbands,
            obs_mass_limit=obs_mass_limit, true_mass_limit=true_mass_limit)

    # If we used the id-tag functionality, update the config file
    try:
        ms.UMConfig().auto_add_lightcones()
    except ValueError:
        pass


def convert_ascii_to_npy_and_json(asciifile, outfilebase=None,
                    remove_ascii_file=False, **kwargs):
    if outfilebase is None:
        outfilebase = ".".join(asciifile.split(".")[:-1])

    lightcone = lightcone_from_ascii(asciifile, **kwargs)
    metadict = metadict_from_ascii(asciifile, **kwargs)

    np.save(outfilebase + ".npy", lightcone)
    json.dump(metadict, open(outfilebase + ".json", "w"))

    if remove_ascii_file:
        # Save disk space by deleting the huge ascii file
        os.remove(asciifile)

def metadict_from_ascii(filename, photbands=None, obs_mass_limit=8e8,
                         true_mass_limit=0):
    photbands = _get_photbands(photbands)

    with open(filename) as f:
        [f.readline() for i in range(1)]
        cmd = " ".join(f.readline().split()[2:])
        seed = eval(f.readline().split()[-1])
        origin = [float(s) for s in f.readline().split()[-3:]]
        [f.readline() for i in range(30)]
        Rmatrix = eval(("".join([f.readline()[1:].strip().replace(" ", ",")
                                 for i in range(3)]))[:-1])

    (executable, config, z_low, z_high,
     x_arcmin, y_arcmin, samples) = cmd.split()[:7]

    return dict(Rmatrix=Rmatrix, seed=seed, origin=origin, cmd=cmd,
                photbands=photbands, obs_mass_limit=obs_mass_limit,
                true_mass_limit=true_mass_limit, executable=executable,
                config=config, z_low=float(z_low), z_high=float(z_high),
                x_arcmin=float(x_arcmin), y_arcmin=float(y_arcmin),
                samples=int(samples))

def lightcone_from_ascii(filename, photbands=None, obs_mass_limit=8e8,
                         true_mass_limit=0):
    """
    Takes the ascii output given by UniverseMachine's `lightcone` code,
    and returns it as a numpy structured array, removing entries with
    mass lower than the specified limit. Reading the ascii table may take
    up to 20 minutes for large lightcones.

    Several new columns are added, calculated using functions in this
    module. Velocity-distorted positions replace the x, y, and z
    columns, and the old ones are renamed x_real, y_real, and z_real.
    Additionally, absolute magnitudes (- 5logh) and distance modulus
    (+ 5logh) are calculated. Note that the h-scaling cancels out if
    you are interested in relative magnitudes. M_V + distmod = m_v
    as expected.
    """
    photbands = _get_photbands(photbands)

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
    masslimit = {"obs_sm":obs_mass_limit, "true_sm":true_mass_limit}
    reader = ht_sim_manager.tabular_ascii_reader.TabularAsciiReader(
                             filename, cols, row_cut_min_dict=masslimit)
    lightcone = reader.read_ascii()

    # Limit RA to the range [-180,180) (column = "ra")
    lightcone["ra"] = (lightcone["ra"] + 180) % 360 - 180
    # Calculate redshift-space-distorted positions (columns = "x","y","z")
    xyz_real = ms.hf.xyz_array(lightcone,keys=["x_real","y_real","z_real"])
    vel = ms.hf.xyz_array(lightcone, keys=["vx", "vy", "vz"])
    rdz = ms.hf.ra_dec_z(xyz_real, vel, cosmo=ms.bplcosmo)
    xyz = ms.hf.rdz2xyz(rdz, cosmo=ms.bplcosmo)

    # Calculate distance modulus (column = "distmod")
    dlum = ms.bplcosmo.luminosity_distance(lightcone["redshift"]
                                           ).value * ms.bplcosmo.h
    distmod = 5 * np.log10(dlum * 1e5)
    dlum_true = ms.bplcosmo.luminosity_distance(lightcone["redshift_cosmo"]
                                           ).value * ms.bplcosmo.h
    distmod_cosmo = 5 * np.log10(dlum_true * 1e5)

    # Calculate apparent magnitudes (column = "m_j", "m_y", etc.)
    reg = MagRegressor(lightcone, photbands=photbands)
    magdf = reg.um_abs_mag + distmod_cosmo[:,None]

    # Name the new columns and specify their dtypes
    xyz_dtype = [(s, "f4") for s in ("x", "y", "z")]
    mag_dtype = [(f"m_{s}", "f4") for s in magdf.columns]
    distmod_dtype = [("distmod", "<f4"), ("distmod_cosmo", "<f4")]

    full_dtype = (xyz_dtype + lightcone.dtype.descr + mag_dtype +
                  distmod_dtype)

    # Copy all columns into a new structured numpy array
    final_lightcone_array = np.zeros(lightcone.shape, full_dtype)
    for i, (name, dtype) in enumerate(xyz_dtype):
        final_lightcone_array[name] = xyz[:, i]
    for (name, dtype) in mag_dtype:
        final_lightcone_array[name] = magdf[name[-1].lower()]
    for (name, dtype) in lightcone.dtype.descr:
        final_lightcone_array[name] = lightcone[name]
    final_lightcone_array["distmod"] = distmod
    final_lightcone_array["distmod_cosmo"] = distmod_cosmo

    return final_lightcone_array

def get_lightcone_UMmags(UMhalos, logssfr_uv, photbands=None,
                         nwin=501, dz=0.05):
    logm = np.log10(UMhalos["obs_sm"])
    logssfr = np.log10(UMhalos["obs_sfr"]) - logm
    z = UMhalos["redshift_cosmo"]

    reg, photbands, (uvista_z, uvista_logm,
        uvista_logssfr_uv) = setup_uvista_mag_regressor(photbands)

    logssfr_uv = cam_binned_z(m=logm, z=z, prop=logssfr, m2=uvista_logm,
        z2=uvista_z, prop2=uvista_logssfr_uv, nwin=nwin, dz=dz)

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

    UVISTA = ms.UVISTAConfig(photbands=photbands)
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


def cam_binned_z(m, z, prop, m2, z2, prop2, nwin=501, dz=0.05,
                 min_counts_in_z2_bins=None):
    assert (vparse(ht.version.version) >= vparse("0.7dev"))
    assert (dz > 0)
    if min_counts_in_z2_bins is None:
        min_counts_in_z2_bins = nwin+1

    zrange = z.min() - dz/20, z.max() + dz/20
    nz = int((zrange[1] - zrange[0]) / dz)
    if nz:
        centroids = np.linspace(*zrange, nz+1)
    else:
        nz = 1
        centroids = np.mean(zrange) + dz*np.array([-0.5,0.5])

    zmin, zmax = centroids.min(), centroids.max()
    s, s2 = (zmin < z) & (z < zmax), (zmin < z2) & (z2 < zmax)
    assert np.all(s)

    m2 = m2[s2]
    z2 = z2[s2]
    prop2 = prop2[s2]

    inds2 = ht_utils.fuzzy_digitize(z2, centroids,
                                    min_counts=min_counts_in_z2_bins)
    centroids, inds2 = ms.hf.correction_for_empty_bins(centroids, inds2)
    inds = ms.hf.fuzzy_digitize_improved(z, centroids,
                                    min_counts=min_counts_in_z2_bins)


    new_prop = np.full_like(prop, np.nan)
    for i in range(len(centroids)):
        s, s2 = inds==i, inds2==i
        nwin1 = min([nwin, s2.sum()//2*2-1])
        if nwin1 < 2:
            print(f"Warning: Only {s2.sum()} galaxies in the z"
                  f"={centroids[i]} bin. You should use a larger"
                  f"value of dz than {dz}")
        new_prop[s] = ht_empirical_models.conditional_abunmatch(
            m[s], prop[s], m2[s2], prop2[s2], nwin1)

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
    # homedir = str(pathlib.Path.home()) + "/"
    if executable is None:
        # executable = homedir + "local/src/universemachine/lightcone"
        executable = ms.UMConfig().config["lightcone_executable"]
    if umcfg is None:
        # umcfg = homedir + "data/LightCone/um-lightcone.cfg"
        umcfg = ms.UMConfig().config["lightcone_config"]
    if not rseed is None:
        assert(isinstance(rseed, int)), "Random seed must be an integer"
    assert(isinstance(samples, int)), "Number of samples must be " \
                                      "an integer"

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

    cmd=f"{str(executable)} {str(umcfg)} {float(z_low)} {float(z_high)} "\
        f"{float(x_arcmin)} {float(y_arcmin)} {samples} {' '.join(args)}"
    print(cmd)
    return os.system(cmd)

def _default_lightcone_filenames(z_low, z_high, x_arcmin, y_arcmin,
                                samples=1, id_tag=""):
    if id_tag: id_tag += "_"
    return [f"survey_{id_tag}z{z_low:.2f}-{z_high:.2f}_" 
            f"x{x_arcmin:.2f}_y{y_arcmin:.2f}_{i}.dat"
            for i in range(samples)]


def _generate_lightcone_filenames(args, outfilepath=None,
                                  outfilebase=None):
    fake_id = args.pop()
    z_low, z_high, x_arcmin, y_arcmin, samples, id_tag = args
    args[-1] = "" if args[-1] is None else args[-1]

    # If id_tag is provided AND outfilepath is not
    # then store in the default location
    if (not id_tag is None) and (outfilepath is None):
        try:
            outfilepath = ms.UMConfig().config["data_dir"]
        except ValueError:
            raise ValueError("To use an id-tag, you must first choose "
                             "a UMConfig directory to store the files.")

        outfilepath = os.path.join(outfilepath, "lightcones", id_tag)
        if pathlib.Path(outfilepath).is_dir():
            outfilepath = None
            raise IsADirectoryError(f"Lightcone with id-tag={id_tag} "
                                  f"already exists at {outfilepath}")

    # If id_tag is NOT provided, then make sure outfilepath is valid
    else:
        outfilepath = "" if outfilepath is None else outfilepath
        if not pathlib.Path(outfilepath).is_dir():
            raise NotADirectoryError(f"outfilepath={outfilepath} "
                                     f"must be a directory.")

    # If outfilebase NOT provided, use default universemachine naming
    if outfilebase is None:
        outfilebase = _default_lightcone_filenames(*args)[0][:-6]

    # Make all names of files generated by the lightcone code
    outfilebase = os.path.join(outfilepath, outfilebase)
    asciifiles = _default_lightcone_filenames(*args[:-1], fake_id)
    moved_asciifiles = [outfilebase + "_" + f.split("_")[-1]
                        for f in asciifiles]

    return asciifiles, moved_asciifiles


class MagRegressor:
    def __init__(self, UMhalos, photbands=None, snapshot_redshift=None,
                 nwin=501, dz=0.05):
        """
        This is just a namespace of UltraVISTA and UniverseMachine parameters
        used in the mass-to-light ratio fitting process.

        Parameters
        ----------
        UMhalos : dictionary or structured array
            Must contain the columns 'obs_sm', 'obs_sfr', 'redshift'
        photbands
        snapshot_redshift
        nwin
        dz
        """
        self.photbands = _get_photbands(photbands)
        self.names = ["M_" + key.upper() for key in self.photbands]

        # Load UltraVISTA columns: mass, sSFR_UV, redshift, mass-to-light
        (self.uvista_z, self.uvista_logm, self.uvista_logssfr_uv,
         self.uvista_m2l, self.uvista_id) = self.load_UVISTA()

        # Extract UniverseMachine columns: mass, sSFR, and redshift
        self.um_logm = np.log10(UMhalos["obs_sm"])
        self.um_logssfr = np.log10(UMhalos["obs_sfr"]) - self.um_logm
        if snapshot_redshift is None:
            # default functionality: assume redshift column is in lightcone
            self.um_z = UMhalos["redshift"]
        else:
            self.um_z = np.full_like(self.um_logm, snapshot_redshift)

        # Map UniverseMachine sSFR --> sSFR_UV via abundance matching
        if snapshot_redshift is None:
            self.um_logssfr_uv = cam_binned_z(m=self.um_logm, z=self.um_z,
                        prop=self.um_logssfr, m2=self.uvista_logm,
                        z2=self.uvista_z, prop2=self.uvista_logssfr_uv,
                        nwin=nwin, dz=dz)
        else:
            self.um_logssfr_uv = cam_const_z(m=self.um_logm,
                        prop=self.um_logssfr, m2=self.uvista_logm,
                        prop2=self.uvista_logssfr_uv, z2=self.uvista_z,
                        z_avg=snapshot_redshift, dz=dz)

        # Map (redshift, sSFR_UV) --> (mass-to-light,) via Random Forest
        self.um_abs_mag = self.fit_mass_to_light()

    def fit_mass_to_light(self):
        from sklearn import ensemble

        s = np.isfinite(self.uvista_logssfr_uv)
        x = np.array([self.uvista_logssfr_uv, self.uvista_z]).T
        y = self.uvista_m2l

        regressor = ensemble.RandomForestRegressor(n_estimators=10)
        regressor.fit(x[s], y[s])

        s = np.isfinite(self.um_logssfr_uv)
        x = np.array([self.um_logssfr_uv, self.um_z]).T
        y = regressor.predict(x[s])

        um_mag = np.full((x.shape[0],y.shape[1]), np.nan, np.float32)
        um_mag[s] = -2.5 * (np.asarray(self.um_logm)[s,None] - y)
        return pd.DataFrame(um_mag, columns=self.photbands)

    def load_UVISTA(self):
        UVISTAcat = ms.UVISTAConfig(photbands=self.photbands).load()

        uvista_z = UVISTAcat["z"].values
        uvista_logm = UVISTAcat["logm"].values
        uvista_logssfr_uv = (np.log10(UVISTAcat["sfr_uv"])
                                  - UVISTAcat["logm"]).values
        uvista_m2l = np.array([uvista_logm + UVISTAcat[name].values / 2.5
                                for name in self.names]).T
        uvista_id = UVISTAcat["id"].values

        return (uvista_z, uvista_logm, uvista_logssfr_uv, uvista_m2l,
                uvista_id)

class NeighborSpectra:
    def __init__(self, UMhalos, photbands=None, snapshot_redshift=None,
                 nwin=501, dz=0.05):
        self.Reg = MagRegressor(UMhalos, photbands=photbands,
                                snapshot_redshift=snapshot_redshift,
                                nwin=nwin, dz=dz)

        self.config = ms.SeanSpectraConfig()
        self.wave = self.config.wavelength()
        self.specid = self.config.specid()
        self.isnan = self.config.isnan()
        self.get_specmap = self.config.specmap()

        redshifts = pd.DataFrame(dict(z=self.Reg.uvista_z,
                        uvista_idx=np.arange(len(self.Reg.uvista_z))),
                         index=self.Reg.uvista_id)
        fullmapper = pd.DataFrame(dict(idx=np.arange(len(self.specid)),
                        hex=[hex(s) for s in self.specid]),
                                  index=self.specid)
        self.mapper = pd.merge(fullmapper.iloc[~self.isnan], redshifts,
                          left_index=True, right_index=True)

    def write_specmap(self, outfile, num_nearest, metric_weights=None,
             cosmo=None, verbose=1, progress=True):
        if isinstance(progress, str) and progress.lower() == "notebook":
            import tqdm.notebook as tqdm
            iterator = tqdm.trange
        elif progress:
            import tqdm
            iterator = tqdm.trange
        else:
            iterator = range

        redshift = self.Reg.um_z
        f = np.memmap(outfile, self.get_specmap().dtype, "w+",
            shape=(len(redshift), self.get_specmap().shape[1]))

        nearest, lumcorr = self.find_nearest_specid(num_nearest,
            metric_weights=metric_weights, verbose=verbose)

        for i in iterator(len(nearest)):
            f[i] = self.avg_spectrum(nearest[i], lumcorr[i],
                                     redshift[i], cosmo=cosmo)

    def avg_spectrum(self, specids, lumcorr, redshift, cosmo=None):
        if cosmo is None:
            cosmo = ms.bplcosmo
        idx = self.mapper["idx"].loc[specids].values
        z = self.mapper["z"].loc[specids].values

        lumcorr = (lumcorr * (1+z) / (1+redshift) *
                (cosmo.luminosity_distance(z).value/
                 cosmo.luminosity_distance(redshift).value)**2)

        specs = self.get_specmap()[idx, :] * lumcorr[:, None]
        waves = self.wave[None,:] / (1 + z[:,None])
        truewave = self.wave / (1 + redshift)

        truespec = []
        for spec,wave in zip(specs,waves):
            truespec.append(np.interp(truewave, wave, spec,
                                      left=np.nan, right=np.nan))
        return np.mean(truespec, axis=0)


    def find_nearest_specid(self, num_nearest, metric_weights=None,
                            verbose=1):
        if not num_nearest:
            return None
        else:
            from sklearn import neighbors
            num_nearest = int(num_nearest)
            if metric_weights is None:
                metric_weights = [1/1.0, 1/0.75, 1/2.0]

            # Select UltraVISTA galaxies with simulated spectra
            specid = ms.SeanSpectraConfig().specid()[
                ~ms.SeanSpectraConfig().isnan()]
            uvsel = np.isin(self.Reg.uvista_id, specid)
            # Instantiate the nearest-neighbor regressor with metric
            # d^2 = (logM/1)^2 + (logsSFR/0.75)^2 + (z/2)^2
            reg = neighbors.NearestNeighbors(num_nearest,
                    metric="wminkowski",
                    metric_params=dict(w=metric_weights))

            # Predictors:
            uv_x = np.array([self.Reg.uvista_logm,
                             self.Reg.uvista_logssfr_uv,
                             self.Reg.uvista_z]).T
            um_x = np.array([self.Reg.um_logm,
                             self.Reg.um_logssfr_uv,
                             self.Reg.um_z]).T
            uvsel &= np.isfinite(uv_x).all(axis=1)
            umsel = np.isfinite(um_x).all(axis=1)

            # Find nearest neighbor index --> SpecID
            reg.fit(uv_x[uvsel])
            if verbose:
                print("Assigning nearest-neighbor SpecIDs...")
            neighbor_index = reg.kneighbors(um_x[umsel])[1]
            if verbose:
                print("SpecIDs assigned successfully.")
            neighbor_id = np.full((len(self.Reg.um_logm),num_nearest), -99)
            neighbor_id[umsel] = self.Reg.uvista_id[uvsel][neighbor_index]

            # Calculate specific luminosity, averaged over all available
            # bands in each UniverseMachine and UVISTA galaxy
            uv_lum = 10**(self.Reg.uvista_logm[:,None]
                          - self.Reg.uvista_m2l)
            uv_lum = np.mean(uv_lum, axis=1)
            um_lum = np.mean(10**(-0.4*self.Reg.um_abs_mag), axis=1)
            # Return the correction factor, which is just a constant
            # we have to multiply into the stacked spectrum
            lumcorr = np.full((len(umsel),num_nearest), np.nan)
            lumcorr[umsel,:] = um_lum[umsel,None
                               ]/uv_lum[uvsel][neighbor_index]

            return neighbor_id, lumcorr
