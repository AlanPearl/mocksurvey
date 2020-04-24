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
    assert(vparse(ht.__version__) >= vparse("0.7dev"))
    if not ms.UVISTAConfig().are_all_files_stored():
        raise ValueError("You have not specified paths to all "
                         "UltraVISTA data. "
                         "Please use UVISTAConfig('path/to/dir').auto_add()")
    if not ms.UMConfig().is_lightcone_ready():
        raise ValueError("You must set paths to the lightcone executable and"
                         " config files via UMConfig('path/to/dir').set_light"
                         "cone_executable/config('path/to/file')")

    if executable is None:
        executable = ms.UMConfig().get_lightcone_executable()
    if umcfg is None:
        umcfg = ms.UMConfig().get_lightcone_config()

    # Execute the lightcone code in the UniverseMachine package
    if _execute_lightcone_code(*args[:4], executable, umcfg, samples,
                               id_tag=fake_id,
                               do_collision_test=do_collision_test,
                               ra=ra, dec=dec, theta=theta, rseed=rseed):
        raise RuntimeError("lightcone code failed")

    # Move lightcone files to their desired locations
    pathlib.Path(moved_files[0]).parent.mkdir(parents=True, exist_ok=True)
    for origin, destination in zip(files, moved_files):
        pathlib.Path(origin).rename(destination)

    # Convert the enormous ascii file into a binary table + meta data
    for filename in moved_files:
        convert_ascii_to_npy_and_json(
            filename,
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

    data = lightcone_from_ascii(asciifile, **kwargs)
    metadict = metadict_from_ascii(asciifile, **kwargs)

    np.save(outfilebase + ".npy", data)
    json.dump(metadict, open(outfilebase + ".json", "w"))

    if remove_ascii_file:
        # Save disk space by deleting the huge ascii file
        os.remove(asciifile)


def metadict_from_ascii(filename, photbands=None, obs_mass_limit=8e8,
                        true_mass_limit=0):
    photbands = _get_photbands(photbands)

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

    return dict(Rmatrix=rot_matrix, seed=seed, origin=origin, cmd=cmd,
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
    Additionally, apparent magnitudes and distance modulus are calculated.
    Note that h-scaling is applied to all distances (including distance
    modulus), and therefore M = m - distmod + 5log(h).
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
    masslimit = {"obs_sm": obs_mass_limit, "true_sm": true_mass_limit}
    reader = ht_sim_manager.tabular_ascii_reader.TabularAsciiReader(
                             filename, cols, row_cut_min_dict=masslimit)
    lc_data = reader.read_ascii()

    # Limit RA to the range [-180,180) (column = "ra")
    lc_data["ra"] = (lc_data["ra"] + 180) % 360 - 180
    # Calculate redshift-space-distorted positions (columns = "x","y","z")
    xyz_real = ms.hf.xyz_array(lc_data, keys=["x_real", "y_real", "z_real"])
    vel = ms.hf.xyz_array(lc_data, keys=["vx", "vy", "vz"])
    rdz = ms.hf.ra_dec_z(xyz_real, vel, cosmo=ms.bplcosmo)
    xyz = ms.hf.rdz2xyz(rdz, cosmo=ms.bplcosmo)

    # Calculate distance modulus (column = "distmod")
    dlum = ms.bplcosmo.luminosity_distance(lc_data["redshift"]
                                           ).value * ms.bplcosmo.h
    distmod = 5 * np.log10(dlum * 1e5)
    dlum_true = ms.bplcosmo.luminosity_distance(lc_data["redshift_cosmo"]
                                                ).value * ms.bplcosmo.h
    distmod_cosmo = 5 * np.log10(dlum_true * 1e5)

    # Calculate apparent magnitudes (column = "m_g", "m_r", etc.)
    uvdat = UVData(photbands=photbands)
    umdat = UMData(lc_data, uvdat=uvdat)
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


def _get_photbands(photbands):
    if photbands is None:
        photbands = ["g", "r", "y", "j"]
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
    if min_counts_in_z2_bins is None:
        min_counts_in_z2_bins = nwin+1

    zrange = z.min() - dz/20, z.max() + dz/20
    nz = int((zrange[1] - zrange[0]) / dz)
    if nz:
        centroids = np.linspace(*zrange, nz+1)
    else:
        # nz = 1
        centroids = np.mean(zrange) + dz*np.array([-0.5, 0.5])

    # noinspection PyArgumentList
    zmin, zmax = centroids.min(), centroids.max()
    s, s2 = (zmin < z) & (z < zmax), (zmin < z2) & (z2 < zmax)
    assert np.all(s)

    m2 = m2[s2]
    z2 = z2[s2]
    prop2 = prop2[s2]

    inds2 = ht_utils.fuzzy_digitize(z2, centroids, seed=seed,
                                    min_counts=min_counts_in_z2_bins)
    centroids, inds2 = ms.hf.correction_for_empty_bins(centroids, inds2)
    inds = ms.hf.fuzzy_digitize_improved(z, centroids, seed=seed,
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


def _execute_lightcone_code(z_low, z_high, x_arcmin, y_arcmin,
                            executable=None, umcfg=None, samples=1,
                            id_tag="", do_collision_test=False, ra=0.,
                            dec=0., theta=0., rseed=None):
    # homedir = str(pathlib.Path.home()) + "/"
    if executable is None:
        # executable = homedir + "local/src/universemachine/lightcone"
        executable = ms.UMConfig().config["lightcone_executable"]
    if umcfg is None:
        # umcfg = homedir + "data/LightCone/um-lightcone.cfg"
        umcfg = ms.UMConfig().config["lightcone_config"]
    if rseed is not None:
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

    cmd = f"{str(executable)} {str(umcfg)} {float(z_low)} {float(z_high)} "\
          f"{float(x_arcmin)} {float(y_arcmin)} {samples} {' '.join(args)}"
    print(cmd)
    return os.system(cmd)


def _default_lightcone_filenames(z_low, z_high, x_arcmin, y_arcmin,
                                 samples=1, id_tag=""):
    if id_tag:
        id_tag += "_"
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
    if (id_tag is not None) and (outfilepath is None):
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


class UVData:
    def __init__(self, photbands=None):
        self.photbands = _get_photbands(photbands)
        self.names = ["M_" + key.upper() for key in self.photbands]

        # Load UltraVISTA columns: mass, sSFR_UV, redshift, mass-to-light
        (self.z, self.logm, self.logssfr_uv,
         self.abs_mag, self.id) = self.load_uvista()

    @property
    def m2l(self):
        return self.logm[:, None] + self.abs_mag.values / 2.5

    def load_uvista(self):
        uvista_cat = ms.UVISTAConfig(photbands=self.photbands).load()

        z = uvista_cat["z"].values
        logm = uvista_cat["logm"].values
        logssfr_uv = (np.log10(uvista_cat["sfr_uv"])
                      - uvista_cat["logm"]).values
        abs_mag = pd.DataFrame({
            name[-1].lower(): uvista_cat[name].values for name in self.names})
        # m2l = np.array([logm + uvista_cat[name].values / 2.5
        #                         for name in self.names]).T
        uvista_id = uvista_cat["id"].values

        return z, logm, logssfr_uv, abs_mag, uvista_id


class UMData:
    def __init__(self, umhalos, uvdat=None, snapshot_redshift=None,
                 nwin=501, dz=0.05, seed=None):
        """
        This is just a namespace of UniverseMachine observables
        used in the mass-to-light ratio fitting process.

        Parameters
        ----------
        umhalos : dictionary or structured array
            Must contain the columns 'obs_sm', 'obs_sfr', 'redshift'
        uvdat
        snapshot_redshift
        nwin
        dz
        seed : int | None
            Don't bother setting this. You cannot set a seed for
            halotools.empirical_models.conditional_abunmatch()...
        """
        self.seed = seed
        self.uvdat = uvdat
        self.snapshot_redshift = snapshot_redshift
        self.nwin = nwin
        self.dz = dz
        if self.uvdat is None:
            self.uvdat = UVData()

        # Extract UniverseMachine columns: mass, sSFR, and redshift
        self.logm = np.log10(umhalos["obs_sm"])
        self.logssfr = np.log10(umhalos["obs_sfr"]) - self.logm
        if snapshot_redshift is None:
            # default functionality: assume redshift column is in lightcone
            self.z = umhalos["redshift"]
        else:
            self.z = np.full_like(self.logm, snapshot_redshift)

        # Setup default values (usually None) for the properties
        try:
            self._abs_mag = pd.DataFrame(
                {band: umhalos[f"m_{band}"] - umhalos["distmod"]
                 for band in self.uvdat.photbands})
        except (KeyError, ValueError):
            self._abs_mag = None
        try:
            self._logssfr_uv = np.log10(umhalos["sfr_uv"]
                                        / umhalos["obs_sm"])
        except (KeyError, ValueError):
            self._logssfr_uv = None

    @property
    def m2l(self):
        return self.logm[:, None] + self.abs_mag.values / 2.5

    # Map UniverseMachine sSFR --> sSFR_UV via abundance matching
    @property
    def logssfr_uv(self):
        if self._logssfr_uv is not None:
            return self._logssfr_uv
        if self.snapshot_redshift is None:
            self._logssfr_uv = cam_binned_z(
                        m=self.logm,
                        z=self.z, prop=self.logssfr,
                        m2=self.uvdat.logm,
                        z2=self.uvdat.z,
                        prop2=self.uvdat.logssfr_uv,
                        nwin=self.nwin, dz=self.dz,
                        seed=self.seed)
        else:
            self._logssfr_uv = cam_const_z(
                        m=self.logm,
                        prop=self.logssfr,
                        m2=self.uvdat.logm,
                        prop2=self.uvdat.logssfr_uv,
                        z2=self.uvdat.z,
                        z_avg=self.snapshot_redshift, dz=self.dz)
        return self._logssfr_uv

    # Map (redshift, sSFR_UV) --> (mass-to-light,) via Random Forest
    @property
    def abs_mag(self):
        if self._abs_mag is None:
            with ms.hf.temp_seed(self.seed):
                self._abs_mag = self.fit_mass_to_light()
        return self._abs_mag

    def fit_mass_to_light(self):
        from sklearn import ensemble

        s = np.isfinite(self.uvdat.logssfr_uv)
        x = np.array([self.uvdat.logssfr_uv,
                      self.uvdat.z]).T
        y = self.uvdat.m2l

        regressor = ensemble.RandomForestRegressor(n_estimators=10)
        regressor.fit(x[s], y[s])

        s = np.isfinite(self.logssfr_uv)
        x = np.array([self.logssfr_uv, self.z]).T
        y = regressor.predict(x[s])

        abs_mag = np.full((x.shape[0], y.shape[1]), np.nan, np.float32)
        abs_mag[s] = 2.5 * (y - np.asarray(self.logm)[s, None])
        return pd.DataFrame(abs_mag, columns=self.uvdat.photbands)


class SeanSpecStacker:
    def __init__(self, uvista_z, uvista_id):
        self.config = ms.SeanSpectraConfig()
        self.wave = self.config.wavelength()
        self.specid = self.config.specid()
        self.isnan = self.config.isnan()
        self.get_specmap = self.config.specmap()

        redshifts = pd.DataFrame(dict(
                        z=uvista_z,
                        uvista_idx=np.arange(len(uvista_z))),
                        index=uvista_id)
        fullmapper = pd.DataFrame(dict(
                        idx=np.arange(len(self.specid)),
                        hex=[hex(s) for s in self.specid]),
                        index=self.specid)
        self.mapper = pd.merge(fullmapper.iloc[~self.isnan], redshifts,
                               left_index=True, right_index=True)

    def avg_spectrum(self, specids, lumcorr, redshift, cosmo=None,
                     return_each_spec=False):
        if cosmo is None:
            cosmo = ms.bplcosmo
        idx = self.id2idx_specmap(specids)
        z = self.id2redshift(specids)

        lumcorr = (lumcorr * (1+z) / (1+redshift) *
                   (cosmo.luminosity_distance(z).value /
                    cosmo.luminosity_distance(redshift).value)**2)

        specs = self.get_specmap()[idx, :] * lumcorr[:, None]
        waves = self.wave[None, :] / (1 + z[:, None])
        truewave = self.wave / (1 + redshift)

        eachspec = []
        for spec, wave in zip(specs, waves):
            eachspec.append(np.interp(truewave, wave, spec,
                                      left=np.nan, right=np.nan))
        avgspec = np.mean(eachspec, axis=0)
        if return_each_spec:
            return avgspec, eachspec
        else:
            return avgspec

    def id2redshift(self, uvista_id):
        return np.reshape(self.mapper["z"].loc[
                        np.ravel(uvista_id)].values, np.shape(uvista_id))

    def id2idx_uvista(self, uvista_id):
        return np.reshape(self.mapper["uvista_idx"].loc[
                        np.ravel(uvista_id)].values, np.shape(uvista_id))

    def id2idx_specmap(self, uvista_id):
        return np.reshape(self.mapper["idx"].loc[
                        np.ravel(uvista_id)].values, np.shape(uvista_id))


class NeighborSeanSpecFinder:
    def __init__(self, umhalos, photbands=None, snapshot_redshift=None,
                 nwin=501, dz=0.05, seed=None):
        self.uvdat = UVData(photbands=photbands)
        self.umdat = UMData(umhalos, uvdat=self.uvdat,
                            snapshot_redshift=snapshot_redshift,
                            nwin=nwin, dz=dz, seed=seed)
        self.stacker = SeanSpecStacker(self.uvdat.z,
                                       self.uvdat.id)

    def id2redshift(self, uvista_id):
        return self.stacker.id2redshift(uvista_id)

    def id2idx_uvista(self, uvista_id):
        return self.stacker.id2idx_uvista(uvista_id)

    def id2idx_specmap(self, uvista_id):
        return self.stacker.id2idx_specmap(uvista_id)

    def avg_spectrum(self, specids, lumcorr, redshift, cosmo=None):
        return self.stacker.avg_spectrum(specids, lumcorr, redshift,
                                         cosmo=cosmo)

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

        redshift = self.umdat.z
        f = np.memmap(outfile, self.stacker.get_specmap().dtype, "w+",
                      shape=(len(redshift), self.stacker.get_specmap().shape[1]))

        nearest, lumcorr = self.find_nearest_specid(
            num_nearest, metric_weights=metric_weights, verbose=verbose)

        for i in iterator(len(nearest)):
            f[i] = self.avg_spectrum(nearest[i], lumcorr[i],
                                     redshift[i], cosmo=cosmo)

    def find_nearest_specid(self, num_nearest, metric_weights=None, ummask=None, verbose=1):
        if not num_nearest:
            return None
        from sklearn import neighbors
        num_nearest = int(num_nearest)
        if metric_weights is None:
            metric_weights = [1/1.0, 1/0.75, 1/2.0]
        if ummask is None:
            ummask = np.ones(self.umdat.z.shape, dtype=bool)

        # Select UltraVISTA galaxies with simulated spectra
        specid = ms.SeanSpectraConfig().specid()[
            ~ms.SeanSpectraConfig().isnan()]
        uvsel = np.isin(self.uvdat.id, specid)
        # Instantiate the nearest-neighbor regressor with metric
        # d^2 = (logM/1)^2 + (logsSFR/0.75)^2 + (z/2)^2
        reg = neighbors.NearestNeighbors(
                num_nearest,
                metric="wminkowski",
                metric_params=dict(w=metric_weights))

        # Predictors:
        uv_x = np.array([self.uvdat.logm,
                         self.uvdat.logssfr_uv,
                         self.uvdat.z]).T
        um_x = np.array([self.umdat.logm,
                         self.umdat.logssfr_uv,
                         self.umdat.z]).T
        uvsel &= np.isfinite(uv_x).all(axis=1)
        umsel = ummask
        assert np.isfinite(um_x).all()

        # Find nearest neighbor index --> SpecID
        reg.fit(uv_x[uvsel])
        if verbose:
            print("Assigning nearest-neighbor SpecIDs...")
        neighbor_index = reg.kneighbors(um_x[umsel])[1]
        if verbose:
            print("SpecIDs assigned successfully.")
        neighbor_id = np.full((len(self.umdat.logm), num_nearest), -99)
        neighbor_id[umsel] = self.uvdat.id[uvsel][neighbor_index]

        # # Calculate specific luminosity, averaged over all available
        # # bands in each UniverseMachine and UVISTA galaxy
        # uv_lum = np.mean(10**(-0.4*self.uvdat.abs_mag.values), axis=1)
        # um_lum = np.mean(10**(-0.4*self.umdat.abs_mag.values), axis=1)
        # # Return the correction factor, which is just a constant
        # # we have to multiply into the stacked spectrum
        # lumcorr = np.full((len(umsel), num_nearest), np.nan)
        # lumcorr[umsel, :] = um_lum[umsel, None
        #                            ] / uv_lum[uvsel][neighbor_index]

        return neighbor_id[ummask]

    def lumcorr(self, spec_id, ummask=None):
        if ummask is None:
            ummask = np.ones(self.umdat.z.shape, dtype=bool)

        # Calculate specific luminosity, averaged over all available
        # bands in each UniverseMachine and UVISTA galaxy
        uv_lum = np.mean(10 ** (-0.4 * self.uvdat.abs_mag.values), axis=1)
        um_lum = np.mean(10 ** (-0.4 * self.umdat.abs_mag.values), axis=1)
        # Return the correction factor, which is just a constant
        # we have to multiply into the stacked spectrum
        # lumcorr = np.full(spec_id.shape, np.nan)
        uv_lum = uv_lum[self.id2idx_uvista(spec_id)]
        um_lum = um_lum[(ummask, *[None, ]*(spec_id.ndim-1))]
        return um_lum / uv_lum

    def masscorr(self, spec_id, ummask=None):
        if ummask is None:
            ummask = np.ones(self.umdat.z.shape, dtype=bool)

        uv_logm = self.uvdat.logm[self.id2idx_uvista(spec_id)]
        um_logm = self.umdat.logm[(ummask, *[None, ]*(spec_id.ndim-1))]

        return 10 ** (um_logm - uv_logm)

    def best_color(self, spec_id, ummask=None):
        if ummask is None:
            ummask = np.ones(self.umdat.z.shape, dtype=bool)

        um_colors = -np.diff(self.umdat.abs_mag[ummask])
        nbr_colors = -np.diff(self.uvdat.abs_mag.values[
                                  self.id2idx_uvista(spec_id)])

        best_neighbor = (np.abs(um_colors[:, None, :] - nbr_colors)
                         ).sum(axis=2).argmin(axis=1)
        return spec_id[np.arange(len(spec_id)), best_neighbor]
