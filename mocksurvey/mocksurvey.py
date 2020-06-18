"""
mocksurvey.py
Author: Alan Pearl

A collection of useful tools for coducting mock surveys of galaxies
populated by `halotools` and `UniverseMachine` models.
"""

import os
import pathlib
import warnings
import json
import inspect
from typing import Tuple

import numpy as np
import pandas as pd
import astropy.table as astropy_table
import tarfile
from natsort import natsorted

# Local modules
from . import util
# Local packages
from .stats import cf
from . import httools
from . import ummags
from . import filechunk
# Default cosmology (Bolshoi-Planck)
from .httools.httools import bplcosmo


uvista_sqdeg = 1.62
pfs_sqdeg = 15.0
pfs_max_dict = dict(m_y=22.5, m_j=22.8)


def mass_complete_pfs_selector(lightcone, zlim, compfrac=0.95, fieldshape="sq",
                               masslim=None, randfrac=0.7, sqdeg=15.0,
                               masscol="obs_sm"):
    z_low, z_high = zlim
    max_dict = pfs_max_dict.copy()

    if masslim is None:
        incompsel = LightConeSelector(z_low, z_high, 15., "fullsky",
                                      max_dict=max_dict)
        comptest = CompletenessTester(lightcone, incompsel)
        masslim = comptest.limit(compfrac, column=masscol)
    min_dict = {masscol: masslim}

    compsel = LightConeSelector(
        z_low, z_high, sqdeg, fieldshape, randfrac,
        max_dict=max_dict, min_dict=min_dict)
    return compsel


class RealizationLoader:
    def __init__(self, name="PFS", selector=None, nreal=None):
        if isinstance(name, LightConeConfig):
            self.config = name
            self.name = self.config.get_path()
        else:
            self.config = LightConeConfig(name)
            self.name = name
        self.nreal = self.nreal_avail = len(self.config["files"])
        self.meta = [self.config.load_meta(i) for i in range(self.nreal)]
        if nreal is not None:
            assert nreal <= self.nreal, f"There are only {self.nreal}" \
                                        f"realizations of {name} available"
            self.nreal = nreal

        if selector is None:
            selector = util.selector_from_meta(self.meta[0])

        self.initial_selector = selector
        self.secondary_selector = None
        self.cosmo = selector.cosmo

        # noinspection PyUnusedLocal
        def _null_selector(*args, **kwargs):
            return slice(None)
        self._null_selector = _null_selector
        self._all_catalogs = None

    @property
    def generator(self):
        if self._all_catalogs is None:
            return (self.load(i) for i in range(self.nreal))
        else:
            selector = self.get_secondary_selector()
            return (cat[selector(cat)] for cat in self._all_catalogs)

    @property
    def volume(self):
        if self.secondary_selector is None:
            return self.initial_selector.volume
        else:
            return self.secondary_selector.volume

    def get_secondary_selector(self):
        return self._null_selector if self.secondary_selector is None \
            else self.secondary_selector

    def load_all(self):
        if self._all_catalogs is None:
            self._all_catalogs = []
            for i in range(self.nreal):
                cat = self.config.load(i)[0]
                cat = cat[self.initial_selector(cat)]
                self._all_catalogs.append(cat)

        selector = self.get_secondary_selector()
        return [cat[selector(cat)] for cat in self._all_catalogs]

    def load(self, index):
        s1, s2 = self.initial_selector, self.get_secondary_selector()
        if self._all_catalogs is None:
            return (v := self.config.load(index)[0])[s1(v)][s2(v)]
        else:
            return (v := self._all_catalogs[index])[s2(v)]

    def cosmic_var(self, statfuncs, take_var=False):
        """
        Parameters
        ----------
        statfuncs : callable | list of callables
            Functions that return the desired statistic. They must take two
            positional arguments: a lightcone and this RealizationLoader

        take_var : bool (default=False)
            If true, this function will return the variance of
            the desired statistic instead of an array of values
            across realizations (which you could then take the variance of)

        Returns
        -------
        results : array
            A list of results with same the length as statfuncs. Or just
            one result if statfuncs is a callable rather than a list.
            If take_var=True, each result = tuple(stat, stat_variance).
            Else, each result is an array of stat realizations
        """
        is_arraylike = util.is_arraylike(statfuncs)
        if not is_arraylike:
            statfuncs = [statfuncs]

        # noinspection PyTypeChecker
        stats: list = np.full((len(statfuncs), self.nreal), np.nan).tolist()

        for j, real in enumerate(self.generator):
            for i, statfunc in enumerate(statfuncs):
                stats[i][j] = statfunc(real, self)

        results = np.array(stats)
        if take_var:
            results = [(stat.mean(axis=0), stat.std(axis=0, ddof=1) ** 2)
                       for stat in results]
        if not is_arraylike:
            results = results[0]

        return results


class LightConeSelector:
    def __init__(self, z_low, z_high, sqdeg=None, fieldshape="sq",
                 sample_fraction=1., min_dict=None, max_dict=None,
                 cosmo=None, center_radec=None, realspace=False, deg=True):
        if cosmo is None:
            cosmo = bplcosmo
        assert isinstance(fieldshape, str)
        fieldshape = "full_sky" if sqdeg is None else fieldshape.lower()
        if fieldshape.startswith("full"):
            fieldshape = "full_sky"
        elif fieldshape.startswith("sq"):
            fieldshape = "square"
        elif fieldshape.startswith("cir"):
            fieldshape = "circle"
        elif fieldshape.startswith("hex"):
            fieldshape = "hexagon"
        else:
            raise ValueError(
                "fieldshape must be a string and it must"
                "start with one of: 'sq', 'cir', 'hex', or 'full'")

        self.z_low, self.z_high = z_low, z_high
        self.sqdeg = 41_252.9612494 if fieldshape.startswith("full") else sqdeg
        self.fieldshape = fieldshape
        self.sample_fraction = sample_fraction
        self.min_dict = {} if min_dict is None else min_dict
        self.max_dict = {} if max_dict is None else max_dict
        self.realspace = realspace
        self.cosmo, self.deg = cosmo, deg

        z, dz = (z_high+z_low)/2., z_high - z_low

        simbox = httools.SimBox(redshift=z, empty=True)
        if self.fieldshape.startswith("full"):
            fieldshape = "circle"
        self.field = simbox.field(empty=True, sqdeg=sqdeg, scheme=fieldshape,
                                  center_rdz=center_radec, delta_z=dz)
        self.field_selector = self.field.field_selector

    def __call__(self, lightcone, seed=None):
        conditions = [self.field_selection(lightcone),
                      self.redshift_selection(lightcone),
                      self.rand_selection(lightcone, seed=seed),
                      self.dict_selection(lightcone)]
        return np.all(conditions, axis=0)

    def __repr__(self):
        d = self.__dict__.copy()
        d["center_radec"] = self.field.center_rdz.tolist()[:2]
        kw = "".join([f", {key}={repr(d[key])}" for key in
                      inspect.getfullargspec(self.__init__).args[1:]])
        kw = kw.replace(" km / (Mpc s)", "").replace(" K", "")

        return f"{type(self).__name__}({kw})"

    def __str__(self):
        return f"{type(self).__name__}(z_low={self.z_low}, " \
               f"z_high={self.z_high}, sqdeg={self.sqdeg}, **kw)"

    def __and__(self, other):
        assert isinstance(other, LightConeSelector)
        center_radec = self.field.center_rdz.tolist()[:2]
        assert center_radec == other.field.center_rdz.tolist()[:2]
        assert self.cosmo.h == other.cosmo.h
        assert self.cosmo.Om0 == other.cosmo.Om0
        assert self.deg == other.deg

        # Field selection
        if self.sqdeg < other.sqdeg:
            sqdeg, fieldshape = self.sqdeg, self.fieldshape
        else:
            sqdeg, fieldshape = other.sqdeg, other.fieldshape
            if self.sqdeg == other.sqdeg:
                assert self.fieldshape == other.fieldshape

        # Redshift selection
        if self.z_low > other.z_low and self.z_high < other.z_high:
            realspace = self.realspace
        elif self.z_low < other.z_low and self.z_high > other.z_high:
            realspace = other.realspace
        else:
            assert self.realspace == other.realspace
            realspace = self.realspace
        z_low = max(self.z_low, other.z_low)
        z_high = min(self.z_high, other.z_high)

        # Miscellaneous
        sample_fraction = self.sample_fraction * other.sample_fraction
        min_dict = {**self.min_dict, **other.min_dict}
        max_dict = {**self.max_dict, **other.max_dict}
        for key in min_dict.keys():
            if key in self.min_dict and self.min_dict[key] > min_dict[key]:
                min_dict[key] = self.min_dict[key]
        for key in max_dict.keys():
            if key in self.max_dict and self.max_dict[key] < max_dict[key]:
                max_dict[key] = self.max_dict[key]

        return LightConeSelector(z_low, z_high, sqdeg, fieldshape,
                                 sample_fraction, min_dict, max_dict,
                                 cosmo=self.cosmo, center_radec=center_radec,
                                 realspace=realspace, deg=self.deg)

    @property
    def volume(self):
        return util.volume(self.sqdeg, [self.z_low, self.z_high],
                           cosmo=self.cosmo)

    def field_selection(self, lightcone):
        if self.fieldshape.startswith("full"):
            return np.ones(len(lightcone), dtype=bool)
        else:
            rd = util.xyz_array(lightcone, ["ra", "dec"])
            return self.field_selector(rd, deg=self.deg)

    def redshift_selection(self, lightcone):
        z = lightcone["redshift_cosmo"] if self.realspace \
            else lightcone["redshift"]
        return (self.z_low <= z) & (z <= self.z_high)

    def rand_selection(self, lightcone, seed=None):
        with util.temp_seed(seed):
            if self.sample_fraction < 1:
                return np.random.random(len(lightcone)) < self.sample_fraction
            else:
                return np.ones(len(lightcone), dtype=bool)

    def dict_selection(self, lightcone):
        ones = np.ones(len(lightcone), dtype=bool)

        cond1 = ones
        if self.min_dict:
            cond1 = np.all([lightcone[key] >= self.min_dict[key]
                            for key in self.min_dict], axis=0)
        cond2 = ones
        if self.max_dict:
            cond2 = np.all([lightcone[key] <= self.max_dict[key]
                            for key in self.max_dict], axis=0)

        return cond1 & cond2

    def make_rands(self, n, rdz=False, seed=None):
        # Calculate limits in ra, dec, and distance
        fieldshape = self.field.get_shape(rdz=True)[:2, None]
        rdlims = self.field.center_rdz[:2, None] + np.array([[.5, -.5]]) \
            * fieldshape
        distlim = util.comoving_disth([self.z_low, self.z_high], self.cosmo)

        rands = util.rand_rdz(n, *rdlims, distlim, seed=seed)
        # This only works perfectly if the fieldshape is a square
        # Cut randoms that fall outside the shape of the field
        # if not self.fieldshape == "square":
        rands = rands[self.field_selector(rands)]
        # Convert to Cartesian coordinates
        if not rdz:
            rands = util.rdz2xyz(rands, cosmo=None, use_um_convention=True)
        else:
            rands[:, 2] = util.distance2redshift(rands[:, 2], cosmo=self.cosmo, vr=None)
            # Convert radians to degrees
            if self.deg:
                rands[:, :2] *= 180 / np.pi
        return rands

    def block_digitize(self, lightcone, nbins=(2, 2, 1)):
        data = util.xyz_array(lightcone, ["ra", "dec", "redshift"])
        fieldshape = self.field.get_shape(rdz=True, deg=self.deg)
        center = self.field.center_rdz

        # noinspection PyProtectedMember
        return cf._assign_block_indices(data, None, center,
                                        fieldshape, nbins)[0]


class CompletenessTester:
    def __init__(self, lightcone, selector):
        self.lightcone = lightcone
        self.selector = selector

    def limit(self, quantile=0.95, column="obs_sm", max_val=False):
        mass, completeness = self.completeness(column, max_val)
        return np.interp(quantile, completeness, mass)

    def completeness(self, column="obs_sm", max_val=False):
        if isinstance(column, str):
            column = self.lightcone[column].copy()
        elif util.is_arraylike(column):
            column = np.array(column)
        else:
            raise ValueError(f"column={column} but must "
                             f"be a string or an array")

        # Get all galaxies selected by redshift only
        no_selection = self.selector.redshift_selection(self.lightcone)
        gals = self.lightcone[no_selection]
        column = column[no_selection]

        order = np.argsort(column)
        # gals, column = gals[order], column[order]
        # This is 3x faster for some reason...
        gals, column = np.take(gals, order), np.take(column, order)
        gals = gals if max_val else gals[::-1]
        column = column if max_val else column[::-1]

        selection = self.selector.dict_selection(gals)
        frac = np.cumsum(selection) / (np.arange(1, len(gals)+1))

        mass = column[::-1]
        completeness = frac[::-1]

        return mass, completeness


class BaseConfig(dict):
    """
    Abstract template class. Do not instantiate.
    """
    def __init__(self, config_file, data_dir=None, is_temp=False):
        dict.__init__(self)
        self.is_temp = is_temp
        self._init_dict(data_dir=data_dir)
        self._read_config(config_file)
        if data_dir:
            self["data_dir"] = os.path.abspath(data_dir)
        if data_dir is None and self["data_dir"] is None:
            raise ValueError("Your first time using "
                             f"{type(self).__name__}, you "
                             "must set the path to where you "
                             "will be storing the files.")

    def _init_dict(self, data_dir=None):
        self.update({"data_dir": data_dir, "files": []})

    def clear_files(self):
        self.clear_keys(["files"])

    def __repr__(self):
        return f"{type(self).__name__}(\"{self.get_path()}\")"

    def dict(self):
        return dict(self)

    def get_path(self, filename="", *args):
        return os.path.join(self["data_dir"], filename, *args)

    def auto_add(self):
        """
        Automatically try to add all files contained in the data directory.

        Takes no arguments and returns None.
        """
        self.clear_files()
        d = self.get_path()
        # self.clear()

        files = [f for f in natsorted(os.listdir(d))
                 if os.path.isfile(os.path.join(d, f))]

        n = len(files)
        for f in files:
            try:
                self.add_file(f)
            except ValueError:
                n -= 1

        print(f"Total of {n} files stored in {self}")
        self.save()

    def save(self):
        """
        Update the config file to account for any changes that have
        been made to this object. For example:
            -Files could be added via config.add_file()
            -Files could be removed via config.remove_file()
            -Data directory could be changed by instantiating this
            object via config = SomeConfig(data_dir="path/to/new/dir")

        Takes no arguments and returns None.
        """

        # Don't write the file if this is a temporary config
        if not self.is_temp and not self["data_dir"] is None:
            json.dump(self, open(self._filepath, "w"), indent=4)

    def add_file(self, filename):
        """
        Add a new file to store

        Parameters
        ----------
        filename : str
            The name of the file (do not include path)

        Returns
        -------
        None
        """
        if filename in self["files"]:
            raise ValueError("That file is already stored")

        fullpath = os.path.join(self.get_path(), filename)
        if not os.path.isfile(fullpath):
            raise FileNotFoundError(f"{fullpath} does not exist.")

        self["files"].append(filename)

    def remove_file(self, filename):
        """
        Remove a data file from a config. Note this does NOT delete the file.
        It is the child config's responsibility to call self.save() after this

        Parameters
        ----------
        filename : str
            The name of the file (do not include path)

        Returns
        -------
        i : int
            The index of the file being removed
        """
        try:
            i = self["files"].index(filename)
        except ValueError:
            raise ValueError(f"Cannot remove {filename}, as it is not"
                             f" currently stored. Currently stored "
                             f"files: {self['files']}")

        del self["files"][i]
        return i

    def delete(self):
        self.clear_keys(keep=[])
        os.remove(self._filepath)

    def clear_keys(self, keys=None, keep=None):
        """
        Erases the config file for this object; all files are forgotten
        """
        if keys is None:
            if keep is None:
                keep = ["data_dir"]
        else:
            keep = [key for key in self.keys() if key not in keys]
        keep = {key: self[key] for key in keep}

        self.clear()
        self._init_dict()
        self.update(keep)

    def _read_config(self, filename):
        dirpath = util.config_file_directory()
        filepath = os.path.join(dirpath, filename)

        self._filepath = filepath

        #  Don't read the file if this is a temporary config
        if not self.is_temp:
            if not os.path.isdir(dirpath):
                os.mkdir(dirpath)

            if os.path.isfile(filepath):
                config = json.load(open(filepath))
                self.update(config)


class UMConfig(BaseConfig):
    """
    Keeps track of the locations of locally saved binary files that
    come from the UniverseMachine data release.

    Parameters
    ----------
    data_dir : str (required on first run)
        The path to the directory where you plan on saving all of the binary files.
        If the directory is moved, then you must provide this argument again.
    """
    is_temp = False

    def __init__(self, data_dir=None):
        config_file = "config-um.json"
        BaseConfig.__init__(self, config_file, data_dir)

    def _init_dict(self, data_dir=None):
        BaseConfig._init_dict(self, data_dir)
        self.update({"z": [], "lightcones": [],
                     "lightcone_config": None,
                     "lightcone_executable": None})

    def clear_files(self):
        # BaseConfig.clear_files()
        self.clear_keys(["files", "z"])

    def set_lightcone_config(self, filepath):
        assert(os.path.isfile(filepath)), f"file does not exist: {filepath}"
        self["lightcone_config"] = os.path.abspath(filepath)

    def set_lightcone_executable(self, filepath):
        assert (os.path.isfile(filepath)), f"file does not exist: {filepath}"
        self["lightcone_executable"] = os.path.abspath(filepath)

    def get_lightcone_config(self):
        return self["lightcone_config"]

    def get_lightcone_executable(self):
        return self["lightcone_executable"]

    def is_lightcone_ready(self):
        return (self["lightcone_config"] is not None and
                self["lightcone_executable"] is not None and
                os.path.isfile(self["lightcone_config"]) and
                os.path.isfile(self["lightcone_executable"]))

    def load(self, redshift=0, thresh=None, ztol=0.05):
        """
        Load a halo table into memory, at a given snapshot in redshift.

        Parameters
        ----------
        redshift : float (default = 0)
            Desired redshift of the snapshot

        thresh : callable, None, or "none" (optional)
            Callable which takes a halo catalog as input and returns a boolean
            array to select the halos on before loading them into memory.
            By default, ``thresh = lambda cat: cat["obs_sm"] > 3e9``.
            "none" loads the entire table and is equivalent to
            ``thresh = lambda cat: slice(None)``

        ztol : float (default = 0.005)
            A match must be within redshift +/- ztol

        Returns
        -------
        halos : np.ndarray
            Structured array of the requested halo catalog
        """
        if thresh is None:
            def thresh(cat):
                return cat["obs_sm"] > 3e9
        dtype = np.dtype([('id', 'i8'), ('descid', 'i8'), ('upid', 'i8'),
                          ('flags', 'i4'), ('uparent_dist', 'f4'),
                          ('pos', 'f4', 6), ('vmp', 'f4'), ('lvmp', 'f4'),
                          ('mp', 'f4'), ('m', 'f4'), ('v', 'f4'), ('r', 'f4'),
                          ('rank1', 'f4'), ('rank2', 'f4'), ('ra', 'f4'),
                          ('rarank', 'f4'), ('A_UV', 'f4'), ('sm', 'f4'),
                          ('icl', 'f4'), ('sfr', 'f4'), ('obs_sm', 'f4'),
                          ('obs_sfr', 'f4'), ('obs_uv', 'f4'),
                          ('empty', 'f4')], align=True)

        filename, true_z = self._get_file_at_redshift(
            redshift, ztol)
        fullpath = self.get_path(filename)

        if isinstance(thresh, str) and thresh.lower() == "none":
            # all 12 million halos
            return np.fromfile(fullpath, dtype=dtype), true_z
        else:
            # don't load halos into memory until after the selection
            mm = np.memmap(fullpath, dtype=dtype)
            return np.array(mm[thresh(mm)]), true_z

    def auto_add(self):
        """
        In addition to that below, this searches for available lightcones
        """
        self.auto_add_lightcones()
        self.setup_snaps_txt()
        self.setup_lightcone_cfg()
        BaseConfig.auto_add(self)
    auto_add.__doc__ += "\n" + BaseConfig.auto_add.__doc__

    def add_file(self, filename, redshift=None):
        """
        Add a new binary file containing a UniverseMachine snapshot

        Parameters
        ----------
        filename : str
            The name of the file (do not include path)
        redshift : float (optional)
            If given, the redshift of this snapshot. If not given, redshift
            will be inferred from the filename, by assuming that the
            filename is of form "*_{scalefactor}.bin"

        Returns
        -------
        None
        """
        if redshift is None:
            redshift = self._infer_redshift(filename)

        BaseConfig.add_file(self, filename)

        self["z"].append(redshift)

    def remove_file(self, filename):
        """
        Remove a binary file from our records. Note this does NOT delete the file.

        Parameters
        ----------
        filename : str
            The name of the file (do not include path)

        Returns
        -------
        i : int
            The index of the file being removed
        """
        i = BaseConfig.remove_file(self, filename)
        del self["z"][i]
        self.save()
        return i

    def setup_snaps_txt(self):
        sfr_cats = UMWgetter().sfr_cats
        scales = [i.split("_")[-1][:-4] for i in sfr_cats]
        indices = list(range(len(scales)))
        lines = []
        for fname, scale, index in zip(sfr_cats, scales, indices):
            if fname in self["files"]:
                line = f"{index} {scale}"
                lines.append(line)
        snaps_txt = "\n".join(lines)
        snaps_file = self.get_path("snaps.txt")
        open(snaps_file, mode="w").write(snaps_txt)

    def setup_lightcone_cfg(self, cosmo=bplcosmo):
        lightcone_txt = f"""#Input/output locations and details
INBASE = {self.get_path()} # directory with snaps.txt
OUTBASE = {self.get_path()} # directory with sfr_catalogs
NUM_BLOCKS = 144 #The number of cat.box* files

#Box size / cosmology
BOX_SIZE = {250: <16} #In Mpc/h
Om = {cosmo.Om0: <22} #Omega_matter
Ol = {cosmo.Ode0: <22} #Omega_lambda
h0 = {cosmo.h: <22} #h0 = H0 / (100 km/s/Mpc)
fb = {cosmo.Ob0/cosmo.Om0: <22} #cosmic baryon fraction

#Parallel node setup
NUM_NODES = 48           #Total number of nodes used
BLOCKS_PER_NODE = 24    #Parallel tasks per node
#This will generate 8 universes in parallel:
#24 x 48 = 1152 = 144 (NUM_BLOCKS) x 8
#A minimum of NUM_NODES = 6 should be used to generate one universe if
#BLOCKS_PER_NODE = 24, since 144 = 24 x 6.

#Option to calculate ICL
CALC_ICL = 1
"""
        lightcone_file = self.get_path("lightcone.cfg")
        open(lightcone_file, mode="w").write(lightcone_txt)
        self.set_lightcone_config(lightcone_file)

    def auto_add_lightcones(self):
        """
        Automatically add all lightcones found via add_lightcone
        """
        # path = self.get_path("lightcones")
        # pathlib.Path(path).mkdir(exist_ok=True)
        # candidates = [os.path.join(path, name) for name in os.listdir(path)]
        # dirs = [c for c in candidates if os.path.isdir(c)]
        self.clear_keys(["lightcones"])
        self.save()  # this config must exist to find available lightcones
        names = LightConeConfig.available_lightcones()
        for name in names:
            self.add_lightcone(name)
        self.save()

    def add_lightcone(self, name):
        """
        Given the name of a lightcone sample, add it to our records.

        Parameters
        ----------
        name : str
            The name of the lightcone sample. It is also the name
            of the directory located at {data_dir}/lightcones/{name}

        Returns
        -------
        None
        """
        data_dir = self.get_path()
        path = os.path.join(data_dir, "lightcones", name)
        LightConeConfig(path).auto_add()
        if path not in self["lightcones"]:
            self["lightcones"].append(path)

    def remove_lightcone(self, path):
        raise NotImplementedError()

    def _get_file_at_redshift(self, redshift, ztol):
        i = util.choose_close_index(
            redshift, self["z"], ztol, permit_multiple=True)
        return self["files"][i], self["z"][i]

    @staticmethod
    def _infer_redshift(filename):
        a = ".".join(filename.split("_")[-1].split(".")[:-1])
        try:
            a = float(a)
            return (1 - a) / a
        except ZeroDivisionError:
            return np.inf
        except (SystemExit, KeyboardInterrupt):
            raise
        except:
            raise ValueError("Cannot infer a redshift from this filename.")


class LightConeConfig(BaseConfig):
    """
    Keeps track of the locations of locally saved binary files
    that come from the UniverseMachine data release.

    Parameters
    ----------
    data_dir : str (required on first run)
        The path to the directory where you plan on saving all of the binary files.
        If the directory is moved, then you must provide this argument again.
    """
    def __init__(self, data_dir=None, is_temp=False):
        self.is_temp = is_temp
        if data_dir is None:
            raise ValueError("Must supply the name/path to the lightcone.\n"
                             f"Stored paths: {self.stored_lightcones()}\n"
                             f"Available names: {self.available_lightcones()}")

        # If path is explicit and exists, interpret argument as actual path
        if util.explicit_path(data_dir, assert_dir=True):
            data_dir = os.path.abspath(data_dir)
        # Otherwise, name is a folder in standard data path
        else:
            data_dir = UMConfig().get_path("lightcones", data_dir)
            assert(os.path.isdir(data_dir)), f"{data_dir} does not exist"

        config_file = self._path_to_filename(data_dir)
        BaseConfig.__init__(self, config_file, data_dir, is_temp)

    @staticmethod
    def stored_lightcones():
        available = [os.path.join(util.config_file_directory(), x)
                     for x in os.listdir(util.config_file_directory())
                     if x.startswith("config-lightcone-")]
        return [json.load(open(x))["data_dir"] for x in available]

    @staticmethod
    def available_lightcones():
        try:
            lcdir = UMConfig().get_path("lightcones")
        except ValueError:
            return []
        else:
            return [x for x in os.listdir(lcdir) if os.path.isdir(
                os.path.join(lcdir, x))]

    @staticmethod
    def _path_to_filename(path):
        parts = pathlib.Path(path).absolute().parts
        pathname = "&&" + "&&".join(parts[1:])
        return f"config-lightcone-{pathname}.json"

    def _check_load_index(self, index, check_ext=None):
        n = len(self["files"])
        assert util.is_int(index), "index must be an integer"
        assert (0 <= index <= n-1), f"index={index} but -1 < index < {n}"
        if check_ext is not None:
            filename = util.change_file_extension(
                self.get_path(self["files"][index]), check_ext)
            assert os.path.isfile(
                filename
            ), f"File {filename} does not exist"

    def load(self, index: int) -> Tuple[np.ndarray, dict]:
        """
        Load a lightcone catalog, along with its corresponding meta data.

        Parameters
        ----------
        index : int
            Number specifies which lightcone realization to load

        Returns
        -------
        lightcone : np.ndarray
            Structured array of the requested halo catalog

        meta : dict
            Dictionay storing additional information about this lightcone
        """
        self._check_load_index(index)

        datafile = self.get_path(self["files"][index])
        metafile = util.change_file_extension(datafile, "json")

        data = np.load(datafile)
        meta = json.load(open(metafile))
        return data, meta

    def load_meta(self, index):
        """
        Load the meta data corresponding to a lightcone catalog.

        Parameters
        ----------
        index : int
            Number specifies which lightcone realization to load

        Returns
        -------
        meta : dict
            Dictionay storing additional information about this lightcone
        """
        self._check_load_index(index)

        datafile = self.get_path(self["files"][index])
        metafile = util.change_file_extension(datafile, "json")
        return json.load(open(metafile))

    def load_specprop(self, index):
        """Load neighbor-matched spectroscopic properties catalog"""
        self._check_load_index(index, "specprop")

        datafile = self.get_path(self["files"][index])
        specpropfile = util.change_file_extension(datafile, "specprop")
        return np.load(specpropfile)

    def load_specmap(self, index):
        """Load memory-map of spectra data cube"""
        self._check_load_index(index, "spec")

        datafile = self.get_path(self["files"][index])
        specfile = util.change_file_extension(datafile, "spec")

        meta = self.load_meta(index)
        shape = meta["Ngal"], meta["Nwave"]
        return np.memmap(specfile, dtype="<f4", shape=shape)

    def add_file(self, filename):
        """
        Add a new file containing a lightcone

        Parameters
        ----------
        filename : str
            The name of the file (do not include path)

        Returns
        -------
        None
        """
        meta_filename = util.change_file_extension(filename, "json")
        metapath = self.get_path(meta_filename)

        if not filename.endswith(".npy"):
            raise ValueError(f"lightcone file {filename} must end in '.npy'")
        if not os.path.isfile(metapath):
            raise ValueError(f"metadata file {metapath} does not exist")

        BaseConfig.add_file(self, filename)

    def remove_file(self, filename):
        """
        Remove a file from our records. Note this does NOT delete the file.

        Parameters
        ----------
        filename : str
            The name of the file (do not include path)

        Returns
        -------
        i : int
            The index of the file being removed
        """
        i = BaseConfig.remove_file(self, filename)
        self.save()
        return i


class UVISTAConfig(BaseConfig):
    """
    Keeps track of the locations of locally saved UVISTA files.

    Parameters
    ----------
    data_dir : str (required on first run)
        The path to the directory where you plan on saving all of the files.
        If the directory is moved, then you must provide this argument again.
    """
    is_temp = False
    UVISTAFILES = {
        "p": "UVISTA_final_v4.1.cat",  # photometric catalog
        "z": "UVISTA_final_v4.1.zout",  # EAZY output
        "f": "UVISTA_final_BC03_v4.1.fout",  # FAST output
        "s": "UVISTA_final_colors_sfrs_v4.1.dat",  # IR+UV lum/sfr
        "uv": "UVISTA_final_v4.1.153-155.rf",  # rest-frame U,V
        "vj": "UVISTA_final_v4.1.155-161.rf",  # rest-frame V,J
    }

    def _msg1(self, ftype):
        return f"File type {repr(ftype)} not recognized. " \
               f"Must be one of {set(self.UVISTAFILES.keys())}"

    PHOTBANDS = {
        "u": "u", "b": "B", "v": "V",
        "g": "gp", "r": "rp", "i": "ip", "z": "zp",
        "y": "Y", "j": "J", "h": "H", "k": "Ks",
        "ch1": "ch1", "ch2": "ch2", "ch3": "ch3", "ch4": "ch4"
    }

    def __init__(self, data_dir=None, photbands=None):
        config_file = "config-uvista.json"
        BaseConfig.__init__(self, config_file, data_dir)

        photbands = ummags.util.get_photbands(photbands)
        self.PHOTBANDS = {k: self.PHOTBANDS[k] for k in photbands}

    def get_filepath(self, filetype):
        """
        Returns the absolute path to the requested file.
        Filetype options are "p", "z", "f", and "s" where
        each option corresponds to:
        """
        filename = self.UVISTAFILES[filetype]
        return BaseConfig.get_path(self, filename)
    get_filepath.__doc__ += "\n" + repr(UVISTAFILES)

    def add_file(self, filename):
        if filename not in self.UVISTAFILES.values():
            raise ValueError("That's not a UVISTA file")

        BaseConfig.add_file(self, filename)

    def remove_file(self, filename):
        BaseConfig.remove_file(self, filename)
        self.save()

    def get_names(self, filetype):
        if filetype == "p":
            return [
                'id', 'ra', 'dec', 'xpix', 'ypix', 'Ks_tot', 'eKs_tot', 'Ks',
                'eKs', 'H', 'eH', 'J', 'eJ', 'Y', 'eY', 'ch4', 'ech4', 'ch3',
                'ech3', 'ch2', 'ech2', 'ch1', 'ech1', 'zp', 'ezp', 'ip', 'eip',
                'rp', 'erp', 'V', 'eV', 'gp', 'egp', 'B', 'eB', 'u', 'eu',
                'IA484', 'eIA484', 'IA527', 'eIA527', 'IA624', 'eIA624',
                'IA679', 'eIA679', 'IA738', 'eIA738', 'IA767', 'eIA767',
                'IB427', 'eIB427', 'IB464', 'eIB464', 'IB505', 'eIB505',
                'IB574', 'eIB574', 'IB709', 'eIB709', 'IB827', 'eIB827', 'fuv',
                'efuv', 'nuv', 'enuv', 'mips24', 'emips24', 'K_flag', 'K_star',
                'K_Kron', 'apcor', 'z_spec', 'z_spec_cc', 'z_spec_id', 'star',
                'contamination', 'nan_contam', 'orig_cat_id', 'orig_cat_field',
                'USE']
        elif filetype == "z":
            return ['id', 'z_spec', 'z_a', 'z_m1', 'chi_a', 'z_p', 'chi_p',
                    'z_m2', 'odds', 'l68', 'u68', 'l95', 'u95', 'l99', 'u99',
                    'nfilt', 'q_z', 'z_peak', 'peak_prob', 'z_mc']
        elif filetype == "f":
            return ['id', 'z', 'ltau', 'metal', 'lage', 'Av', 'lmass',
                    'lsfr', 'lssfr', 'la2t', 'chi2']
        elif filetype == "s":
            return ['ID', 'z_peak', 'UmV', 'VmJ', 'L_UV', 'L_IR', 'SFR_UV',
                    'SFR_IR', 'SFR_tot', 'SFR_SED', 'LMASS', 'USE']
        elif filetype == "uv":
            return ['id', 'z', 'DM', 'nfilt_fit', 'chi2_fit', 'L153', 'L155']
        elif filetype == "vj":
            return ['id', 'z', 'DM', 'nfilt_fit', 'chi2_fit', 'L155', 'L161']
        else:
            raise ValueError(self._msg1(filetype))

    def names_to_keep(self, filetype):
        if filetype == "p":
            return list(
                {"id", "ra", "dec", "Ks_tot", *self.PHOTBANDS.values(),
                 "star", "K_flag", "zp", "ip", "contamination",
                 "nan_contam", "Ks"})
        elif filetype == "z":
            return ["z_peak"]
        elif filetype == "f":
            return ["lmass", "lssfr"]
        elif filetype == "s":
            return ["SFR_tot", "SFR_UV", "SFR_IR"]
        elif filetype == "uv":
            return ["L153", "L155"]
        elif filetype == "vj":
            return ["L155", "L161"]
        else:
            raise ValueError(self._msg1(filetype))

    def get_skips(self, filetype):
        if filetype == "p" or filetype == "z":
            return 1
        elif filetype == "f":
            return 17
        elif filetype == "s":
            return 2
        elif filetype == "uv" or filetype == "vj":
            return 11
        else:
            raise ValueError(self._msg1(filetype))

    def load_filetype(self, ftype):
        return pd.read_csv(
            self.get_filepath(ftype), delim_whitespace=True,
            names=self.get_names(ftype), skiprows=self.get_skips(ftype),
            usecols=self.names_to_keep(ftype))

    def are_all_files_stored(self):
        return len(self["files"]) == len(self.UVISTAFILES)

    def load(self, include_rel_mags=False, cosmo=None):
        """
        Load catalog of UltraVISTA data. Masses and SFRs
        are not h-scaled (units of Msun and yr), but
        distances and absolute magnitudes are h-scaled
        (units of Mpc/h and +5logh)

        Parameters
        ----------
        include_rel_mags : bool
            If true, include columns with apparent magnitudes in each
            photband. Columns use the same name as absolute magnitudes
            but in lower case (e.g., 'm_g')
        cosmo : astropy.cosmology.Cosmology
            Cosmology object used to calculate distances and specify
            a little h value (default = Bolshoi-Planck cosmology)

        Returns
        -------
        catalog : DataFrame
            Contains all observational data for each UltraVISTA galaxy
            in Adam Muzzin's K-selected catalog
        """

        # Define a little h correction factor. Muzzin used h = 0.7
        # Dependence on Luminosity or Stellar Mass = h^-2
        # Dependence on Time or Distance = h^-1
        # Dependence on SFR = Mass/Time = h^-1
        # Dependence on sSFR = 1/Time = h^1
        if cosmo is None:
            cosmo = bplcosmo
        h_corr = cosmo.h / 0.7

        if not self.are_all_files_stored():
            raise ValueError("Can't load until all files are stored")

        ftypes = ["p", "f", "z", "s", "uv", "vj"]
        dat = [self.load_filetype(s) for s in ftypes]

        uvrest = -2.5*np.log10(dat[4]["L153"]/dat[4]["L155"])
        vjrest = -2.5*np.log10(dat[5]["L155"]/dat[5]["L161"])
        z = dat[2]["z_peak"]
        sfr_tot = dat[3]["SFR_tot"] / h_corr
        sfr_uv = dat[3]["SFR_UV"] / h_corr
        sfr_ir = dat[3]["SFR_IR"] / h_corr
        logm = dat[1]["lmass"] - 2 * np.log10(h_corr)
        logssfr = dat[1]["lssfr"] + np.log10(h_corr)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            d_com = cosmo.comoving_distance(z.values).value * cosmo.h
            d_lum = cosmo.luminosity_distance(z.values).value * cosmo.h

            aperture_factor = dat[0]["Ks_tot"] / dat[0]["Ks"]
            relative_mags = {
                key: -2.5 * np.log10(dat[0][val] * aperture_factor) + 25
                for (key, val) in self.PHOTBANDS.items()
            }
            if "k" in relative_mags.keys():
                kmag = relative_mags["k"]
            else:
                kmag = -2.5 * np.log10(dat[0]["Ks_tot"]) + 25
            distmod = 5 * np.log10(d_lum * 1e5)
            absolute_mags = {
                key: val - distmod
                for (key, val) in relative_mags.items()
            }

        selection = np.all([
            *(np.isfinite(x) for x in list(absolute_mags.values())),
            np.isfinite(logm), z > 1.1e-2, kmag < 23.4,
            dat[0]["star"] == 0, dat[0]["K_flag"] < 4,
            dat[0]["contamination"] == 0, dat[0]["nan_contam"] < 3],
            axis=0)

        # rel_keys, rel_vals = zip(*relative_mags.items())
        rel_keys, rel_vals = relative_mags.keys(), relative_mags.values()
        rel_keys = ["m_" + key for key in rel_keys]
        rel_vals = rel_vals if include_rel_mags else []
        rel_keys = rel_keys if include_rel_mags else []

        # abs_keys, abs_vals = zip(*absolute_mags.items())
        abs_keys, abs_vals = absolute_mags.keys(), absolute_mags.values()
        abs_keys = ["M_" + key for key in abs_keys]

        names = ["id", "ra", "dec", "redshift", "logm", "logssfr",
                 "d_com", "d_lum", "UVrest", "VJrest", *rel_keys,
                 *abs_keys, "sfr_tot", "sfr_uv", "sfr_ir"]
        cols = [dat[0]["id"], dat[0]["ra"], dat[0]["dec"], z, logm,
                logssfr, d_com, d_lum, uvrest, vjrest, *rel_vals,
                *abs_vals, sfr_tot, sfr_uv, sfr_ir]

        data = dict(zip(names, cols))
        data = pd.DataFrame(data)
        data = data[selection]
        data.index = np.arange(len(data))

        return data


class SeanSpectraConfig(BaseConfig):
    """
    Keeps track of the locations of locally saved files storing
    information about Sean's simulated spectra.

    Parameters
    ----------
    data_dir : str (required on first run)
        The path to the directory where you plan on saving all of the files.
        If the directory is moved, then you must provide this argument again.
    """
    is_temp = False
    SEANFILES = ["cosmos_V17.fits", "specid.npy", "wavelength.npy",
                 "spectra.mmap", "isnan.npy", "cosmos_V17_old.fits", "PYOBS"]

    def __init__(self, data_dir=None):
        config_file = "config-seanspec.json"
        BaseConfig.__init__(self, config_file, data_dir)

    def get_filepath(self, index=0):
        """
        Returns the absolute path to the requested file.
        Just give the index of the following list:
        """
        filename = self.SEANFILES[index]
        return BaseConfig.get_path(self, filename)
    get_filepath.__doc__ += "\n" + repr(SEANFILES)

    def add_file(self, filename):
        if filename not in self.SEANFILES:
            raise ValueError("Invalid SeanSpectra file")

        BaseConfig.add_file(self, filename)

    def remove_file(self, filename):
        BaseConfig.remove_file(self, filename)
        self.save()

    @staticmethod
    def names_to_keep(index=0):
        assert index == 0
        return ["id", "redshift", "L_UV", "L_IR", "SFR_UV", "SFR_IR",
                "SFR_tot", "SFR_SED", "ltau", "metal", "lage", "Av",
                "lmass", "lsfr", "lssfr", "m_CFHT_u", "m_Subaru_g",
                "m_Subaru_B", "m_Subaru_V", "m_Subaru_r", "m_Subaru_i",
                "m_Subaru_z", "m_VISTA_Y", "m_VISTA_J", "m_VISTA_H",
                "m_VISTA_Ks"]

    def are_all_files_stored(self):
        return set(self["files"]) == set(self.SEANFILES)

    def load(self, old_version=False, keep_all_columns=True):
        i = 5 if old_version else 0
        dat = astropy_table.Table.read(self.get_filepath(i))
        if not keep_all_columns:
            dat.keep_columns(self.names_to_keep())
        return dat.to_pandas()

    def specid(self):
        return np.load(self.get_filepath(1))

    def wavelength(self):
        return np.load(self.get_filepath(2))

    def isnan(self):
        return np.load(self.get_filepath(4))

    def specmap(self):
        path = self.get_filepath(3)
        shape = (self.specid().size, self.wavelength().size)
        return lambda: np.memmap(path, dtype="<f4", shape=shape)


class UMWgetter:
    url = "http://behroozi.users.hpc.arizona.edu/UniverseMachine/DR1/SFR/"
    webpage = """[   ] sfr_catalog_0.055623.bin          01-Dec-2019 10:16  11K
    [   ] sfr_catalog_0.060123.bin          01-Dec-2019 10:16  126K
    [   ] sfr_catalog_0.062373.bin          01-Dec-2019 10:16  296K
    [   ] sfr_catalog_0.064623.bin          01-Dec-2019 10:16  632K
    [   ] sfr_catalog_0.066873.bin          01-Dec-2019 10:16  1.2M
    [   ] sfr_catalog_0.069123.bin          01-Dec-2019 10:16  2.2M
    [   ] sfr_catalog_0.071373.bin          01-Dec-2019 10:16  3.7M
    [   ] sfr_catalog_0.073623.bin          01-Dec-2019 10:16  6.1M
    [   ] sfr_catalog_0.075873.bin          01-Dec-2019 10:16  9.4M
    [   ] sfr_catalog_0.078123.bin          01-Dec-2019 10:16  14M
    [   ] sfr_catalog_0.080373.bin          01-Dec-2019 10:16  20M
    [   ] sfr_catalog_0.082623.bin          01-Dec-2019 10:16  27M
    [   ] sfr_catalog_0.085998.bin          01-Dec-2019 10:16  42M
    [   ] sfr_catalog_0.089373.bin          01-Dec-2019 10:16  62M
    [   ] sfr_catalog_0.092748.bin          01-Dec-2019 10:16  88M
    [   ] sfr_catalog_0.096123.bin          01-Dec-2019 10:16  120M
    [   ] sfr_catalog_0.099498.bin          01-Dec-2019 10:16  159M
    [   ] sfr_catalog_0.102873.bin          01-Dec-2019 10:16  203M
    [   ] sfr_catalog_0.106248.bin          01-Dec-2019 10:16  253M
    [   ] sfr_catalog_0.109623.bin          01-Dec-2019 10:16  308M
    [   ] sfr_catalog_0.112998.bin          01-Dec-2019 10:16  368M
    [   ] sfr_catalog_0.116373.bin          01-Dec-2019 10:16  432M
    [   ] sfr_catalog_0.119748.bin          01-Dec-2019 10:16  498M
    [   ] sfr_catalog_0.123123.bin          01-Dec-2019 10:17  565M
    [   ] sfr_catalog_0.126498.bin          01-Dec-2019 10:17  634M
    [   ] sfr_catalog_0.129873.bin          01-Dec-2019 10:17  705M
    [   ] sfr_catalog_0.133248.bin          01-Dec-2019 10:17  777M
    [   ] sfr_catalog_0.136623.bin          01-Dec-2019 10:17  850M
    [   ] sfr_catalog_0.141685.bin          01-Dec-2019 10:17  957M
    [   ] sfr_catalog_0.146748.bin          01-Dec-2019 10:18  1.0G
    [   ] sfr_catalog_0.151810.bin          01-Dec-2019 10:18  1.1G
    [   ] sfr_catalog_0.156873.bin          01-Dec-2019 10:18  1.3G
    [   ] sfr_catalog_0.161935.bin          01-Dec-2019 10:18  1.4G
    [   ] sfr_catalog_0.166998.bin          01-Dec-2019 10:19  1.5G
    [   ] sfr_catalog_0.172060.bin          01-Dec-2019 10:19  1.5G
    [   ] sfr_catalog_0.177123.bin          01-Dec-2019 10:20  1.6G
    [   ] sfr_catalog_0.182185.bin          01-Dec-2019 10:20  1.7G
    [   ] sfr_catalog_0.187248.bin          01-Dec-2019 10:20  1.8G
    [   ] sfr_catalog_0.192310.bin          01-Dec-2019 10:21  1.8G
    [   ] sfr_catalog_0.197373.bin          01-Dec-2019 10:21  1.9G
    [   ] sfr_catalog_0.202435.bin          01-Dec-2019 10:22  1.9G
    [   ] sfr_catalog_0.207498.bin          01-Dec-2019 10:22  2.0G
    [   ] sfr_catalog_0.212560.bin          01-Dec-2019 10:23  2.0G
    [   ] sfr_catalog_0.217623.bin          01-Dec-2019 10:23  2.1G
    [   ] sfr_catalog_0.222685.bin          01-Dec-2019 10:24  2.1G
    [   ] sfr_catalog_0.227748.bin          01-Dec-2019 10:24  2.1G
    [   ] sfr_catalog_0.232810.bin          01-Dec-2019 10:24  2.1G
    [   ] sfr_catalog_0.237873.bin          01-Dec-2019 10:25  2.2G
    [   ] sfr_catalog_0.242935.bin          01-Dec-2019 10:25  2.2G
    [   ] sfr_catalog_0.247998.bin          01-Dec-2019 10:26  2.2G
    [   ] sfr_catalog_0.253060.bin          01-Dec-2019 10:27  2.2G
    [   ] sfr_catalog_0.258123.bin          01-Dec-2019 10:27  2.2G
    [   ] sfr_catalog_0.263185.bin          01-Dec-2019 10:28  2.2G
    [   ] sfr_catalog_0.268248.bin          01-Dec-2019 10:28  2.3G
    [   ] sfr_catalog_0.273310.bin          01-Dec-2019 10:29  2.3G
    [   ] sfr_catalog_0.278373.bin          01-Dec-2019 10:29  2.3G
    [   ] sfr_catalog_0.283435.bin          01-Dec-2019 10:30  2.3G
    [   ] sfr_catalog_0.288498.bin          01-Dec-2019 10:30  2.3G
    [   ] sfr_catalog_0.293560.bin          01-Dec-2019 10:31  2.3G
    [   ] sfr_catalog_0.298623.bin          01-Dec-2019 10:32  2.3G
    [   ] sfr_catalog_0.303685.bin          01-Dec-2019 10:32  2.3G
    [   ] sfr_catalog_0.308748.bin          01-Dec-2019 10:33  2.3G
    [   ] sfr_catalog_0.313810.bin          01-Dec-2019 10:33  2.3G
    [   ] sfr_catalog_0.318873.bin          01-Dec-2019 10:33  2.3G
    [   ] sfr_catalog_0.323935.bin          01-Dec-2019 10:34  2.3G
    [   ] sfr_catalog_0.328997.bin          01-Dec-2019 10:34  2.3G
    [   ] sfr_catalog_0.334060.bin          01-Dec-2019 10:35  2.3G
    [   ] sfr_catalog_0.339122.bin          01-Dec-2019 10:35  2.3G
    [   ] sfr_catalog_0.344185.bin          01-Dec-2019 10:35  2.3G
    [   ] sfr_catalog_0.349247.bin          01-Dec-2019 10:36  2.3G
    [   ] sfr_catalog_0.354310.bin          01-Dec-2019 10:36  2.3G
    [   ] sfr_catalog_0.359372.bin          01-Dec-2019 10:37  2.3G
    [   ] sfr_catalog_0.364435.bin          01-Dec-2019 10:37  2.3G
    [   ] sfr_catalog_0.369497.bin          01-Dec-2019 10:38  2.3G
    [   ] sfr_catalog_0.374560.bin          01-Dec-2019 10:39  2.3G
    [   ] sfr_catalog_0.379622.bin          01-Dec-2019 10:39  2.2G
    [   ] sfr_catalog_0.384685.bin          01-Dec-2019 10:40  2.2G
    [   ] sfr_catalog_0.389747.bin          01-Dec-2019 10:40  2.2G
    [   ] sfr_catalog_0.394810.bin          01-Dec-2019 10:41  2.2G
    [   ] sfr_catalog_0.399872.bin          01-Dec-2019 10:41  2.2G
    [   ] sfr_catalog_0.404935.bin          01-Dec-2019 10:42  2.2G
    [   ] sfr_catalog_0.409997.bin          01-Dec-2019 10:42  2.2G
    [   ] sfr_catalog_0.415060.bin          01-Dec-2019 10:43  2.2G
    [   ] sfr_catalog_0.420122.bin          01-Dec-2019 10:43  2.2G
    [   ] sfr_catalog_0.425185.bin          01-Dec-2019 10:44  2.2G
    [   ] sfr_catalog_0.430247.bin          01-Dec-2019 10:44  2.2G
    [   ] sfr_catalog_0.435310.bin          01-Dec-2019 10:45  2.2G
    [   ] sfr_catalog_0.440372.bin          01-Dec-2019 10:45  2.2G
    [   ] sfr_catalog_0.445435.bin          01-Dec-2019 10:46  2.1G
    [   ] sfr_catalog_0.450497.bin          01-Dec-2019 10:47  2.1G
    [   ] sfr_catalog_0.455560.bin          01-Dec-2019 10:47  2.1G
    [   ] sfr_catalog_0.460622.bin          01-Dec-2019 10:48  2.1G
    [   ] sfr_catalog_0.465685.bin          01-Dec-2019 10:48  2.1G
    [   ] sfr_catalog_0.470747.bin          01-Dec-2019 10:48  2.1G
    [   ] sfr_catalog_0.475810.bin          01-Dec-2019 10:49  2.1G
    [   ] sfr_catalog_0.480872.bin          01-Dec-2019 10:49  2.1G
    [   ] sfr_catalog_0.485935.bin          01-Dec-2019 10:50  2.1G
    [   ] sfr_catalog_0.490997.bin          01-Dec-2019 10:50  2.1G
    [   ] sfr_catalog_0.496060.bin          01-Dec-2019 10:51  2.1G
    [   ] sfr_catalog_0.501122.bin          01-Dec-2019 10:51  2.0G
    [   ] sfr_catalog_0.506185.bin          01-Dec-2019 10:52  2.0G
    [   ] sfr_catalog_0.511247.bin          01-Dec-2019 10:52  2.0G
    [   ] sfr_catalog_0.516310.bin          01-Dec-2019 10:53  2.0G
    [   ] sfr_catalog_0.521372.bin          01-Dec-2019 10:53  2.0G
    [   ] sfr_catalog_0.526435.bin          01-Dec-2019 10:53  2.0G
    [   ] sfr_catalog_0.531497.bin          01-Dec-2019 10:54  2.0G
    [   ] sfr_catalog_0.536560.bin          01-Dec-2019 10:54  2.0G
    [   ] sfr_catalog_0.541622.bin          01-Dec-2019 10:55  2.0G
    [   ] sfr_catalog_0.546685.bin          01-Dec-2019 10:55  2.0G
    [   ] sfr_catalog_0.551747.bin          01-Dec-2019 10:56  2.0G
    [   ] sfr_catalog_0.556810.bin          01-Dec-2019 10:57  2.0G
    [   ] sfr_catalog_0.561872.bin          01-Dec-2019 10:57  1.9G
    [   ] sfr_catalog_0.566935.bin          01-Dec-2019 10:58  1.9G
    [   ] sfr_catalog_0.571997.bin          01-Dec-2019 10:58  1.9G
    [   ] sfr_catalog_0.577060.bin          01-Dec-2019 10:58  1.9G
    [   ] sfr_catalog_0.582123.bin          01-Dec-2019 10:59  1.9G
    [   ] sfr_catalog_0.587185.bin          01-Dec-2019 10:59  1.9G
    [   ] sfr_catalog_0.592248.bin          01-Dec-2019 11:00  1.9G
    [   ] sfr_catalog_0.597310.bin          01-Dec-2019 11:00  1.9G
    [   ] sfr_catalog_0.602373.bin          01-Dec-2019 11:01  1.9G
    [   ] sfr_catalog_0.607435.bin          01-Dec-2019 11:01  1.9G
    [   ] sfr_catalog_0.612498.bin          01-Dec-2019 11:02  1.9G
    [   ] sfr_catalog_0.617560.bin          01-Dec-2019 11:02  1.9G
    [   ] sfr_catalog_0.622623.bin          01-Dec-2019 11:03  1.9G
    [   ] sfr_catalog_0.627685.bin          01-Dec-2019 11:03  1.9G
    [   ] sfr_catalog_0.632748.bin          01-Dec-2019 11:04  1.8G
    [   ] sfr_catalog_0.637810.bin          01-Dec-2019 11:04  1.8G
    [   ] sfr_catalog_0.642873.bin          01-Dec-2019 11:05  1.8G
    [   ] sfr_catalog_0.647935.bin          01-Dec-2019 11:05  1.8G
    [   ] sfr_catalog_0.652998.bin          01-Dec-2019 11:05  1.8G
    [   ] sfr_catalog_0.658060.bin          01-Dec-2019 11:06  1.8G
    [   ] sfr_catalog_0.663123.bin          01-Dec-2019 11:06  1.8G
    [   ] sfr_catalog_0.668185.bin          01-Dec-2019 11:07  1.8G
    [   ] sfr_catalog_0.673248.bin          01-Dec-2019 11:07  1.8G
    [   ] sfr_catalog_0.678310.bin          01-Dec-2019 11:07  1.8G
    [   ] sfr_catalog_0.683373.bin          01-Dec-2019 11:08  1.8G
    [   ] sfr_catalog_0.690967.bin          01-Dec-2019 11:08  1.8G
    [   ] sfr_catalog_0.698560.bin          01-Dec-2019 11:09  1.8G
    [   ] sfr_catalog_0.706154.bin          01-Dec-2019 11:09  1.8G
    [   ] sfr_catalog_0.713748.bin          01-Dec-2019 11:09  1.8G
    [   ] sfr_catalog_0.721342.bin          01-Dec-2019 11:10  1.7G
    [   ] sfr_catalog_0.728935.bin          01-Dec-2019 11:10  1.7G
    [   ] sfr_catalog_0.736529.bin          01-Dec-2019 11:10  1.7G
    [   ] sfr_catalog_0.744123.bin          01-Dec-2019 11:11  1.7G
    [   ] sfr_catalog_0.751717.bin          01-Dec-2019 11:11  1.7G
    [   ] sfr_catalog_0.759310.bin          01-Dec-2019 11:12  1.7G
    [   ] sfr_catalog_0.766904.bin          01-Dec-2019 11:12  1.7G
    [   ] sfr_catalog_0.774498.bin          01-Dec-2019 11:13  1.7G
    [   ] sfr_catalog_0.782092.bin          01-Dec-2019 11:13  1.7G
    [   ] sfr_catalog_0.789685.bin          01-Dec-2019 11:13  1.7G
    [   ] sfr_catalog_0.797279.bin          01-Dec-2019 11:14  1.7G
    [   ] sfr_catalog_0.804873.bin          01-Dec-2019 11:14  1.7G
    [   ] sfr_catalog_0.812467.bin          01-Dec-2019 11:14  1.7G
    [   ] sfr_catalog_0.820060.bin          01-Dec-2019 11:15  1.7G
    [   ] sfr_catalog_0.827654.bin          01-Dec-2019 11:15  1.7G
    [   ] sfr_catalog_0.835248.bin          01-Dec-2019 11:16  1.7G
    [   ] sfr_catalog_0.842842.bin          01-Dec-2019 11:16  1.7G
    [   ] sfr_catalog_0.850435.bin          01-Dec-2019 11:16  1.6G
    [   ] sfr_catalog_0.858029.bin          01-Dec-2019 11:17  1.6G
    [   ] sfr_catalog_0.865623.bin          01-Dec-2019 11:17  1.6G
    [   ] sfr_catalog_0.873217.bin          01-Dec-2019 11:18  1.6G
    [   ] sfr_catalog_0.880810.bin          01-Dec-2019 11:18  1.6G
    [   ] sfr_catalog_0.888404.bin          01-Dec-2019 11:19  1.6G
    [   ] sfr_catalog_0.895998.bin          01-Dec-2019 11:19  1.6G
    [   ] sfr_catalog_0.903592.bin          01-Dec-2019 11:20  1.6G
    [   ] sfr_catalog_0.911185.bin          01-Dec-2019 11:20  1.6G
    [   ] sfr_catalog_0.918779.bin          01-Dec-2019 11:21  1.6G
    [   ] sfr_catalog_0.926373.bin          01-Dec-2019 11:21  1.6G
    [   ] sfr_catalog_0.933967.bin          01-Dec-2019 11:21  1.6G
    [   ] sfr_catalog_0.941560.bin          01-Dec-2019 11:22  1.6G
    [   ] sfr_catalog_0.949154.bin          01-Dec-2019 11:22  1.6G
    [   ] sfr_catalog_0.956748.bin          01-Dec-2019 11:23  1.6G
    [   ] sfr_catalog_0.964342.bin          01-Dec-2019 11:24  1.6G
    [   ] sfr_catalog_0.971935.bin          01-Dec-2019 11:25  1.6G
    [   ] sfr_catalog_0.979529.bin          01-Dec-2019 11:26  1.6G
    [   ] sfr_catalog_0.987123.bin          01-Dec-2019 11:26  1.6G
    [   ] sfr_catalog_0.994717.bin          01-Dec-2019 11:26  1.6G
    [   ] sfr_catalog_1.002310.bin          01-Dec-2019 11:27  1.6G"""

    def __init__(self):
        """
        Class designed to handle downloading UniverseMachine SFR catalogs
        from peterbehroozi.com/data
        """

        self.sfr_cats = [i.split()[2] for i in self.webpage.split("\n")]
        self.scales = np.array([float(i.split("_")[-1][:-4]) for i in self.sfr_cats])
        self.redshifts = 1/self.scales - 1

        unitsize = dict(K=2 ** 10, M=2 ** 20, G=2 ** 30)
        sizes = [i.split()[-1] for i in self.webpage.split("\n")]
        self.sizes = [float(i[:-1]) * unitsize[i[-1]] for i in sizes]

    def download_sfrcat_index(self, i, overwrite=False):
        file_url = self.url + self.sfr_cats[i]
        outfile = UMConfig().get_path(self.sfr_cats[i])
        util.wget_download(file_url, outfile, overwrite=overwrite)

    def download_sfrcat_redshift(self, redshift, overwrite=False):
        zmin, zmax = np.min(redshift), np.max(redshift)
        imin = util.choose_close_index(zmax, self.redshifts, "none")
        imax = util.choose_close_index(zmin, self.redshifts, "none")
        for i in range(imin, imax+1):
            self.download_sfrcat_index(i, overwrite=overwrite)
        UMConfig().auto_add()


class UVISTAWgetter:
    def __init__(self):
        """
        Class designed to handle downloading UltraVISTA and SeanSpectra files
        """
        self.uvista_config = UVISTAConfig()
        self.sean_config = SeanSpectraConfig()

        self.uvista_url = "https://pitt.box.com/shared/static/" \
                          "rfcmakle4i6xm5s4uuquzerw9a7sjqf1.gz"
        self.sean_url = "https://pitt.box.com/shared/static/" \
                        "tfrah1t7rslp1jsqiccwxcwvfwy7ngw4.gz"
        self.spec_urls = ["https://pitt.box.com/shared/static/"
                          "2sxv6neh4533t9hiej0zj8wk997bnee6.0",
                          "https://pitt.box.com/shared/static/"
                          "lxdqnf1n77c8m6f3m6s44tyxr3nehu3j.1"]

        self.uvista_path = self.uvista_config.get_path()
        self.sean_path = self.sean_config.get_path()
        self.spec_path = self.sean_path

        self.uvista_tarf = self.uvista_config.get_path("UVISTA_data.tar.gz")
        self.sean_tarf = self.sean_config.get_path("SeanSpectra_data.tar.gz")
        self.spec_tarf = self.sean_config.get_path("spectra.mmap.tar.gz")

    @staticmethod
    def decompress_tar(tarf):
        tar = tarfile.open(tarf)
        tar.extractall(path=os.path.dirname(tarf))
        os.remove(tarf)

    @staticmethod
    def wget_and_join_chunks(urls, tarf, overwrite=False):
        for i, url in enumerate(urls):
            util.wget_download(url, outfile=f"{tarf}.chunk.{i}",
                               overwrite=overwrite)
        filechunk.joinchunks(tarf, rmchunks=True)

    @staticmethod
    def check_already_downloaded(wget_files, config):
        stored_files = os.listdir(config.get_path())
        if ans := set(wget_files).issubset(stored_files):
            print("All files already downloaded")
            config.auto_add()
        return ans

    def download_uvista(self, overwrite=False):
        if not overwrite:
            wget_files = ["UVISTA_final_BC03_v4.1.fout",
                          "UVISTA_final_colors_sfrs_v4.1.dat",
                          "UVISTA_final_v4.1.153-155.rf",
                          "UVISTA_final_v4.1.155-161.rf",
                          "UVISTA_final_v4.1.cat",
                          "UVISTA_final_v4.1.zout"]
            if self.check_already_downloaded(wget_files, self.uvista_config):
                return

        util.wget_download(self.uvista_url, outfile=self.uvista_tarf,
                           overwrite=overwrite)
        self.decompress_tar(self.uvista_tarf)
        UVISTAConfig().auto_add()

    def download_sean_specprops(self, overwrite=False):
        if not overwrite:
            wget_files = ["specid.npy", "wavelength.npy",
                          "isnan.npy", "cosmos_V17.fits",
                          "PYOBS"]
            if self.check_already_downloaded(wget_files, self.sean_config):
                return

        util.wget_download(self.sean_url, outfile=self.sean_tarf,
                           overwrite=overwrite)
        self.decompress_tar(self.sean_tarf)
        SeanSpectraConfig().auto_add()

    def download_sean_specmap(self, overwrite=False):
        if not overwrite:
            wget_files = ["spectra.mmap"]
            if self.check_already_downloaded(wget_files, self.sean_config):
                return

        self.wget_and_join_chunks(self.spec_urls, self.spec_tarf,
                                  overwrite=overwrite)
        self.decompress_tar(self.spec_tarf)
        SeanSpectraConfig().auto_add()
