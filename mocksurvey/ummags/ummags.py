import os
import pathlib
from typing import Union, Sequence, Optional
import json
from packaging.version import parse as vparse
import numpy as np
import pandas as pd
import halotools as ht

from . import util
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
    files, moved_files = util.generate_lightcone_filenames(
        args, outfilepath, outfilebase)
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
    if util.execute_lightcone_code(
            *args[:4], executable, umcfg, samples, id_tag=fake_id,
            do_collision_test=do_collision_test,
            ra=ra, dec=dec, theta=theta, rseed=rseed):
        raise RuntimeError("lightcone code failed")

    # Move lightcone files to their desired locations
    pathlib.Path(moved_files[0]).parent.mkdir(parents=True, exist_ok=True)
    for origin, destination in zip(files, moved_files):
        pathlib.Path(origin).rename(destination)

    # Convert the enormous ascii file into a binary table + meta data
    for filename in moved_files:
        util.convert_ascii_to_npy_and_json(
            filename,
            remove_ascii_file=not keep_ascii_files, photbands=photbands,
            obs_mass_limit=obs_mass_limit, true_mass_limit=true_mass_limit)

    # If we used the id-tag functionality, update the config file
    try:
        ms.UMConfig().auto_add_lightcones()
    except ValueError:
        pass


def selected_lightcone(output_dir: str,
                       selector: Optional[object] = None,
                       input_dir: Union[str, object] = ".",
                       outfile: Optional[str] = None,
                       input_realization_index: Union[
                           str, int, Sequence[int]] = "all",
                       nblocks_per_dim: int = 1) -> None:
    """
    Take an input lightcone and perform a selection and optionally
    break it up into sky regions. The resulting lightcone is saved
    in a new directory.
    Parameters
    ----------
    output_dir : str
        Directory name to save the new lightcone. This directory goes
        into the default lightcone storage location, unless the path
        is explicit (starts with '.', '..', or '/') and already exists
    selector : LightConeSelector (default taken from meta data)
        Specifies the lightcone selection function
    input_dir : str | LightConeConfig
        Directory of the input lightcone
    outfile : str (default=None)
        Base of output file names. By default, use the same naming
        convention as the input lightcone
    input_realization_index : int | str | array-like (default="all")
        Specify realization index(es) from the input to create
        the selected lightcone(s). All realizations used by default
    nblocks_per_dim : int (default=1)
        Integer greater than 1 will break up the lightcone into
        data into nblocks_per_dim^2 different equal-sized sky regions

    Returns
    -------
    None (files are written)
    """
    if not ms.util.explicit_path(output_dir, assert_dir=True):
        output_dir = ms.UMConfig().get_path("lightcones", output_dir)
        pathlib.Path(output_dir).mkdir(parents=True)
    if isinstance(input_dir, ms.LightConeConfig):
        config = input_dir
    else:
        config = ms.LightConeConfig(input_dir, is_temp=True)
    with ms.util.suppress_stdout():
        config.auto_add()
    if selector is None:
        selector = ms.util.selector_from_meta(config.load_meta(0))

    nblocks = nblocks_per_dim ** 2
    if input_realization_index == "all":
        input_realization_index = range(len(config["files"]))
    input_realization_index = np.atleast_1d(input_realization_index)

    assert isinstance(nblocks_per_dim, int)
    assert input_realization_index.dtype.kind == "i"

    for index in input_realization_index:
        num = f"_{index}" if len(input_realization_index) > 1 else ""
        base_fn = f"{config['files'][index]}"[:-4] \
            if outfile is None else f"{outfile}{num}"
        base_fn = os.path.join(output_dir, base_fn)

        cat, meta = config.load(index)
        cat = cat[selector(cat)]
        block_digits = selector.block_digitize(cat, (nblocks_per_dim,
                                                     nblocks_per_dim))
        for i in range(nblocks):
            num = f"-{i}" if nblocks > 1 else ""
            phot_fn = f"{base_fn}{num}.npy"
            meta_fn = f"{base_fn}{num}.json"

            mask = block_digits == i
            cat_block = cat[mask]

            selector_num = max([1, *(int(key.split("_")[-1]) + 1
                                     for key in meta.keys()
                                     if key.startswith("selector_"))])
            np.save(phot_fn, cat_block)
            json.dump({**meta, f"selector_{selector_num}":
                       repr(selector)}, open(meta_fn, "w"), indent=4)


def neighbor_spectrum(input_dir: str = ".",
                      input_realization_index: Union[
                          str, int, Sequence[int]] = "all",
                      make_specmap: bool = False,
                      best_of: int = 6,
                      photbands: Sequence[str] = None):
    """
    Take an input lightcone and perform a selection and optionally break it
    up into sky regions. The resulting data is saved in the same directory,
    but with the same names, but with extensions '.spec' and '.specprop'
    Parameters
    ----------
    input_dir : str | LightConeConfig
        Directory of the lightcone
    input_realization_index : int | str | array-like (default="all")
        Specify realization index(es) from the input to create
        the selected lightcone(s). All realizations used by default
    make_specmap : bool (default = False)
        If True, binary files of the raw synthetic spectra will be
        written ('.spec' extension) in addition to the spectral
        property files ('.specprop' extension)'
    best_of : int (default = 6)
        Number of nearest neighbors to match in mass/redshift/sSFR
        space prior to choosing the nearest color neighbor
    photbands : list[str] (default = ['g', 'r', 'y', 'j'])
        Photometric bands used for nearest-neighbor color matching
        of the nearest `best_of` neighbors

    Returns
    -------
    None (files are written)
    """
    photbands = util.get_photbands(photbands, "gryj")
    if isinstance(input_dir, ms.LightConeConfig):
        config = input_dir
    else:
        config = ms.LightConeConfig(input_dir, is_temp=True)
    input_dir = config.get_path()
    with ms.util.suppress_stdout():
        config.auto_add()

    if input_realization_index == "all":
        input_realization_index = range(len(config["files"]))
    input_realization_index = np.atleast_1d(input_realization_index)

    assert input_realization_index.dtype.kind == "i"

    for index in input_realization_index:
        base_fn = f"{config['files'][index]}"[:-4]
        base_fn = os.path.join(input_dir, base_fn)

        meta_fn = f"{base_fn}.json"
        prop_fn = f"{base_fn}.specprop"
        if make_specmap:
            spec_fn = f"{base_fn}.spec"
        else:
            spec_fn = None

        cat, meta = config.load(index)
        ngal = len(cat)
        meta = util.metadict_with_spec(meta, ngal)

        nfinder = NeighborSeanSpecFinder(cat, photbands=photbands)
        nearest = nfinder.find_nearest_specid(
            num_nearest=best_of, bestcolor=True)
        propcat = nfinder.specprops(nearest, cosmo=ms.bplcosmo,
                                    specmap_filename=spec_fn,
                                    progress=True)

        json.dump(meta, open(meta_fn, "w"), indent=4)
        np.save(prop_fn, propcat)
        os.rename(prop_fn + ".npy", prop_fn)
        # if make_specmap:
        #     nfinder.write_specmap(spec_fn, nearest,
        #                           corr="mass", progress=True)


class UVData:
    def __init__(self, photbands=None):
        self.photbands = util.get_photbands(photbands)
        self.names = ["M_" + key for key in self.photbands]

        # Load UltraVISTA columns: mass, sSFR_UV, redshift, mass-to-light
        (self.z, self.logm, self.logssfr_uv,
         self.abs_mag, self.id) = self.load_uvista()

    @property
    def m2l(self):
        return self.logm[:, None] + self.abs_mag.values / 2.5

    def load_uvista(self):
        uvista_cat = ms.UVISTAConfig(photbands=self.photbands).load()

        z = uvista_cat["redshift"].values
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
            self._logssfr_uv = util.cam_binned_z(
                        m=self.logm,
                        z=self.z, prop=self.logssfr,
                        m2=self.uvdat.logm,
                        z2=self.uvdat.z,
                        prop2=self.uvdat.logssfr_uv,
                        nwin=self.nwin, dz=self.dz,
                        seed=self.seed)
        else:
            self._logssfr_uv = util.cam_const_z(
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
            with ms.util.temp_seed(self.seed):
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

    @property
    def bytes_per_spec(self):
        specmap = self.get_specmap()
        return specmap.itemsize * specmap.shape[-1]

    def avg_spectrum(self, specids, lumcorrs, redshift, cosmo=None,
                     return_each_spec=False):
        if cosmo is None:
            cosmo = ms.bplcosmo
        is_scalar = not ms.util.is_arraylike(specids)
        specids = np.atleast_2d(np.transpose(specids)).T
        lumcorrs = np.atleast_2d(np.transpose(lumcorrs)).T
        redshift = np.atleast_1d(redshift)
        assert specids.ndim == 2 and lumcorrs.ndim == 2 and redshift.ndim == 1

        idx = self.id2idx_specmap(specids)
        z = self.id2redshift(specids)

        lumcorrs = lumcorrs * ms.util.redshift_rest_flux_correction(
            from_z=z, to_z=redshift[:, None], cosmo=cosmo)

        specs = self.get_specmap()[idx, :] * lumcorrs[:, :, None]
        waves = self.wave[None, None, :] / (1 + z[:, :, None])
        truewave = self.wave[None, :] / (1 + redshift[:, None])

        eachspec = [[np.interp(truewave_i, wave_ij, spec_ij,
                               left=np.nan, right=np.nan)
                     for wave_ij, spec_ij in zip(waves_i, specs_i)]
                    for truewave_i, waves_i, specs_i in zip(truewave, waves, specs)]
        avgspec = np.mean(eachspec, axis=1)
        if is_scalar:
            avgspec, eachspec = avgspec[0], eachspec[0]
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
        photbands = util.get_photbands(photbands, "gryj")
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

    def avg_spectrum(self, specids, lumcorrs, redshift, cosmo=None):
        return self.stacker.avg_spectrum(specids, lumcorrs, redshift,
                                         cosmo=cosmo)

    def sublist_indices(self, length, max_mem_usage=int(1e8)):
        sublength = int(max_mem_usage / self.stacker.bytes_per_spec)
        return list(ms.util.generate_sublists(
            list(range(length)), sublength))

    def init_specmap(self, filename, num_spectra):
        f = np.memmap(filename, self.stacker.get_specmap().dtype, "w+",
                      shape=(num_spectra, len(self.stacker.wave)))
        return f

    def write_specmap(self, outfile, neighbor_id, ummask=None,
                      cosmo=None, corr="mass", progress=True,
                      max_mem_usage=int(1e8)):
        if ummask is None:
            ummask = np.ones(self.umdat.z.shape, dtype=bool)
        iterator = util.progress_iterator(progress)

        um_redshift = self.umdat.z[ummask]
        f = self.init_specmap(outfile, len(um_redshift))

        if np.ndim(neighbor_id) == 1:
            neighbor_id = np.asarray(neighbor_id)[:, None]

        if corr is None:
            masscorr = np.ones_like(neighbor_id)
        elif corr == "mass":
            masscorr = self.masscorr(neighbor_id, ummask=ummask)
        elif corr == "lum":
            masscorr = self.lumcorr(neighbor_id, ummask=ummask)
        else:
            raise ValueError(f"Invalid corr={corr}")

        sub_index = self.sublist_indices(len(neighbor_id), max_mem_usage)
        for i in iterator(len(sub_index)):
            i = sub_index[i]
            f[i] = self.avg_spectrum(neighbor_id[i], masscorr[i],
                                     um_redshift[i], cosmo=cosmo)

    def find_nearest_specid(self, num_nearest=6, bestcolor=True,
                            metric_weights=None, ummask=None, verbose=1):
        """
        Default metric_weights = [1/1.0, 1/0.5, 1/2.0]
        Specifies weights for logM, logsSFR_UV, redshift
        """
        if not num_nearest:
            return None
        from sklearn import neighbors
        num_nearest = int(num_nearest)
        if metric_weights is None:
            metric_weights = [1/1.0, 1/0.5, 1/2.0]
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

        ans = neighbor_id[ummask]
        if bestcolor:
            ans = self.best_color(ans, ummask=ummask)  # .reshape(-1, 1)
        return ans

    def specprops(self, specid, ummask=None, cosmo=None,
                  specmap_filename=None, progress=True):
        """
        Return structured array of spectral properties derived from
        Sean's synthetic spectra
        Parameters
        ----------
        specid : array
            One-dimensional array containing the UltraVISTA ID of
            the galaxies to include in the returned array
        ummask : numpy mask (default = None)
            Mask you can apply to UniverseMachine redshifts
        cosmo : astropy.Cosmology (default = Bolshoi-Planck)
            Used to calculate distance discrepancy, which is used
            as a flux correction factor
        specmap_filename : str (default = None)
            If supplied, save the spectra as they are generated to
            a memory map at this filename (alternative to write_specmap)
        progress : bool | str (default = True)
            If True, use tqdm progress bar. If 'notebook', use
            tqdm.notebook progress bar. If False, no progress bar

        Returns
        -------
        propcat : structured array
            All spectral properties for each specified galaxy
        """
        if cosmo is None:
            cosmo = ms.bplcosmo
        if ummask is None:
            ummask = np.ones(self.umdat.z.shape, dtype=bool)
        if specmap_filename is None:
            specmap = None
        else:
            specmap = self.init_specmap(specmap_filename, len(specid))
        props = self.stacker.config.load()
        props.index = props["id"].values
        props = props.loc[specid]

        dtypes = list(zip(props.columns.values, props.dtypes.values))
        propcat = np.empty(len(props), dtype=dtypes)
        for key in props.columns.values:
            propcat[key] = props[key]

        um_redshift = self.umdat.z[ummask]
        propcat = util.fix_specprops_columns(self, um_redshift, propcat,
                                             cosmo, max_gband_nan=5000,
                                             specmap=specmap,
                                             progress=progress)
        return propcat

    def lumcorr(self, spec_id, band=None, ummask=None):
        if ummask is None:
            ummask = np.ones(self.umdat.z.shape, dtype=bool)

        if band is None:
            available = self.uvdat.photbands
            band = available[-1]
            if "y" in available:
                band = "y"
            elif "j" in available:
                band = "j"
            elif "i" in available:
                band = "i"

        # Normalize specified photometric band
        uv_lum = 10 ** (-0.4 * self.uvdat.abs_mag[band].values)
        um_lum = 10 ** (-0.4 * self.umdat.abs_mag[band].values)

        # Return the correction factor, which is just a constant
        # we have to multiply into the stacked spectrum
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
