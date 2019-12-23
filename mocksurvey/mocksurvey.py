"""
mocksurvey.py
Author: Alan Pearl

Some useful classes for coducting mock surveys of galaxies populated by `halotools` and `UniverseMachine`.

Classes
-------
SimBox: (and subclasses HaloBox and GalBox)
   Contains information about the simulation box (e.g., the halo data), and populates galaxies given an HOD model available from `halotools`.

BoxField:
    Basic class used to select a rectangular prism of galaxies (or all galaxies by default) populated by the SimBox. Data and randoms can be accessed via methods get_data and get_rands

MockField:
    A more sophisticated version of BoxField, with identical data access methods, in which galaxies are selected by celestial coordinates by a given scheme (shape) on the sky. Data access methods work in the same way as BoxField.

MockSurvey:
    A collection of MockFields, centered at nearby places on the sky. Data access methods work in the same way as BoxField.
"""

from . import hf
from . import cf
from . import tp
from . import ummags

import os
import gc
import warnings
import math
import scipy
import numpy as np
import pandas as pd
import functools
from inspect import getfullargspec
from halotools import sim_manager, empirical_models
from halotools.mock_observables import return_xyz_formatted_array
from astropy import cosmology, table as astropy_table

__all__ = ["Observable", "RedshiftSelector", "FieldSelector", "CartesianSelector", "CelestialSelector", "SimBox", "HaloBox", "GalBox", "BoxField", "MockField", "MockSurvey", "PFSSurvey"]

class Observable:
    def __init__(self, funcs, names=None, args=None, kwargs=None):
        N = len(funcs)
        self.funcs = funcs
        self.names = range(N) if names is None else names
        self.funcdic = dict(zip(self.names,self.funcs))
        if isinstance(args, dict):
            self.argsdic = args.copy()
        else:
            self.argsdic = dict(zip(self.names,[()]*N)) if args is None else dict(zip(names,args))
        if isinstance(kwargs, dict):
            self.kwargsdic = kwargs.copy()
        else:
            self.kwargsdic = dict(zip(self.names,[{}]*N)) if kwargs is None else dict(zip(names,kwargs))
        self.indexdic = {}
        self.lendic = {}

        self.mean = None
        self.mean_jack = None
        self.covar_jack = None
        self.mean_real = None
        self.covar_real = None
        self.mean_rand = None
        self.covar_rand = None

    def get_jackknife(self, name=None):
        return self.get_data(name, method="jackknife")

    def get_realization(self, name=None):
        return self.get_data(name, method="realization")

    def get_random_realization(self, name=None):
        return self.get_data(name, method="random_realization")

    def get_data(self, name=None, method=None):
        accepted = ["jackknife", "realization", "random_realization"]
        if not method is None and not (method in accepted):
            raise ValueError(f"`method` must be one of {accepted}")
        if method is None:
            mean, covar = self.mean, None
        else:
            mean = self.__dict__["mean_"+method[:4]]
            covar = self.__dict__["covar_"+method[:4]]
        if mean is None:
            if method is None:
                raise ValueError("Must calculate the observables first.\n"
                        "Use <Observable object>.obs_func(data,rands)")
            else:
                raise ValueError("Must run this method first.\n"
                    f"Use <Observable object>.{method}(data,rands,...)")


        if name is None:
            return mean if (method is None) else (mean, covar)
        else:
            index0 = self.indexdic[name]
            index1 = index0 + self.lendic[name]
            s = slice(index0, index1)
            return mean[s] if (method is None) else (mean[s], covar[s,s])

    def jackknife(self, data, rands, centers, fieldshape, nbins=(2,2,1), data_to_bin=None, rands_to_bin=None, **kwargs):

        self.mean_jack, self.covar_jack = cf.block_jackknife(data, rands, centers, fieldshape, nbins, data_to_bin, rands_to_bin, self.obs_func, [], {"store": False}, **kwargs)
        return self.mean_jack, self.covar_jack

    def realization(self, rands, field, nrealization=25, **get_data_kw):
        data = field.get_data(**get_data_kw)
        samples = [self.obs_func(data, rands, store=False)]
        if len(samples[0]) >= nrealization:
            print("`nrealization` should probably be greater than the number of observables", flush=True)

        for i in range(nrealization-1):
            field.simbox.populate_mock()
            data = type(field)(**field._kwargs_).get_data(**get_data_kw)
            samples.append(self.obs_func(data, rands, store=False))

        samples = np.array(samples)
        self.mean_real = np.mean(samples, axis=0)
        self.covar_real = np.cov(samples, rowvar=False)
        return self.mean_real, self.covar_real

    def random_realization(self, data, field, nrealization=25, **get_rands_kw):
        rands = field.get_rands(**get_rands_kw)
        samples = [self.obs_func(data, rands, store=False)]
        if len(samples[0]) >= nrealization:
            print("`nrealization` should probably be greater than the number of observables", flush=True)

        for i in range(nrealization-1):
            field.make_rands()
            data = field.get_rands(**get_rands_kw)
            samples.append(self.obs_func(data, rands, store=False))

        samples = np.array(samples)
        self.mean_real = np.mean(samples, axis=0)
        self.covar_real = np.cov(samples, rowvar=False)
        return self.mean_rand, self.covar_rand

    def obs_func(self, data, rands=None, store=True, param_dict=None):
        if param_dict is None:
            param_dict = {}
        supported_params = {"icc"}
        if not set(param_dict.keys()).issubset(supported_params):
            raise ValueError(f"param_dict={param_dict} contains illegal keys."
                             f"\nAllowed keys must be in: {supported_params}")
        answers = []
        i = 0
        for name in self.names:
            func = self.funcdic[name]
            args = self.argsdic[name]
            kwargs = self.kwargsdic[name]

            ans = np.atleast_1d(func(data,rands,*args,**kwargs))
            # Integral constraint constant
            # ============================
            if "icc" in param_dict and name.lower().startswith("wp"):
                ans -= param_dict["icc"]

            answers.append(ans)

            l = len(ans)
            if not name in self.indexdic:
                self.indexdic[name] = i
                self.lendic[name] = l
            i += l

        answer = np.concatenate(answers)
        if store:
            self.mean = answer
        return answer

class RedshiftSelector:
    def __init__(self, mockfield):
        self.mean_redshift = mockfield.simbox.redshift
        self.cosmo = mockfield.simbox.cosmo
        self.delta_z = mockfield.delta_z
        self.zlim = self.mean_redshift + np.array([-.5,.5])*self.delta_z
        self.dlim = hf.comoving_disth(self.zlim, self.cosmo)

    def make_selection(self, redshift, input_is_distance=False):
        lower, upper = self.dlim if input_is_distance else self.zlim
        return (lower < redshift) & (redshift < upper)

class FieldSelector:
    def __init__(self, mockfield):
        self.mean_redshift = mockfield.simbox.redshift
        self.sqdeg = mockfield.sqdeg
        self.center = mockfield.center
        self.center_rdz = mockfield.center_rdz
        self.cosmo = mockfield.simbox.cosmo
        self.delta_z = mockfield.delta_z
        self.scheme = mockfield.scheme
        self.make_selection, self.get_fieldshape = self.choose_selector()

    def choose_selector(self):
        scheme = self.scheme
        if scheme.lower().startswith("cir"):
            return self.circle_selector, self.circle_fieldshape
        if scheme.lower().startswith("sq"):
            return self.square_selector, self.square_fieldshape
        if scheme.lower().startswith("hex"):
            return self.hexagon_selector, self.hexagon_fieldshape
        if scheme.lower().startswith("npoly"):
            self.n_vertices = int(scheme.split("-")[-1])
            return self.npoly_selector, self.npoly_fieldshape
        else:
            raise ValueError("scheme = `%s` is invalid." %scheme)

    def npoly_sqdeg2radius(self, n, return_angle=False):
        """
        Input: solid angle of entire field (in sq degrees)
        Output: The radius (of circumscribed circle, in Mpc/h OR radians)
        """
        omega = self.sqdeg * np.pi**2 / 180.**2
        f = lambda angle: hf.make_npoly(angle, n).area() - omega

        angle = scipy.optimize.brentq(f, 0, np.pi/2.)

        if return_angle:
            return angle
        else:
            return hf.angle_lim_to_dist(angle, self.mean_redshift, self.cosmo)


    def circle_sqdeg2radius(self, return_angle=False):
        """
        Input: solid angle of entire field (in sq degrees)
        Output: The radius of a circular field (in Mpc/h OR radians)
        """
        omega = self.sqdeg * np.pi**2 / 180.**2
        angle = math.acos(1 - omega/2./np.pi)

        if return_angle:
            return angle
        else:
            return hf.angle_lim_to_dist(angle, self.mean_redshift, self.cosmo)

    def square_sqdeg2apothem(self, return_angle=False):
        """
        Input: solid angle of entire field (in sq degrees)
        Output: The apothem of a square field (in Mpc/h OR radians)
        """
        angle0 = math.sqrt(self.sqdeg)
        omega = self.sqdeg * np.pi**2 / 180.**2
        f = lambda angle: 2*angle*math.sin(angle/2.) - omega
        fp = lambda angle: 2*math.sin(angle/2.) + angle*math.cos(angle/2.)
        if angle0 < np.pi/6.:
            angle = scipy.optimize.newton(f, fprime=fp, x0=angle0)/2.
        else:
            angle = scipy.optimize.brentq(f, 0, np.pi)/2.

        if return_angle:
            return angle
        else:
            return hf.angle_lim_to_dist(angle, self.mean_redshift, self.cosmo)

    def hexagon_sqdeg2apothem(self, return_angle=False):
        """
        Input: solid angle of entire field (in sq degrees)
        Output: The apothem of a hexagonal field (in Mpc/h OR radians)
        """
        angle0 = self.circle_sqdeg2radius(return_angle=True)
        omega = self.sqdeg * np.pi**2 / 180.**2
        cnst = 1 - math.sqrt(3)/4 * omega
        f = lambda angle: angle*math.sin(angle) - math.cos(angle) + cnst
        fp = lambda angle: angle*math.cos(angle) + 2*math.sin(angle)
        if angle0 < np.pi/6.:
            angle = scipy.optimize.newton(f, fprime=fp, x0=angle0)
        else:
            angle = scipy.optimize.brentq(f, 0, np.pi/2.)

        if return_angle:
            return angle
        else:
            return hf.angle_lim_to_dist(angle, self.mean_redshift, self.cosmo)

    def _z_length(self):
        return hf.redshift_lim_to_dist(self.delta_z, self.mean_redshift, self.cosmo)

class CartesianSelector(FieldSelector):
    def __init__(self, mockfield):
        FieldSelector.__init__(self, mockfield)

        zlim = (self.mean_redshift - self.delta_z/2.,
                self.mean_redshift + self.delta_z/2.)
        omega = self.sqdeg * (np.pi / 180.)**2
        d1,d2 = hf.comoving_disth(zlim, self.cosmo)

        volume = omega/3. * (d2**3 - d1**3)
        depth = d2-d1

        # Calculate area such that volume is preserved in
        # the conversion from Celestial -> Cartesian
        self.area = volume / depth

    def circle_selector(self, xyz):
        """Select galaxies in a circle centered at x,y = field.center"""
        field_radius = np.sqrt(self.area / np.pi)
        xy = xyz[:,:2] - self.center[np.newaxis,:2]
        rad2 = np.sum(xy**2, axis=1)
        return rad2 < field_radius**2

    def square_selector(self, xyz):
        """Select galaxies in a square centered at x,y = field.center"""
        field_apothem = np.sqrt(self.area / 4.)
        xy = xyz[:,:2] - self.center[np.newaxis,:2]
        b1 = xy[:,0] < field_apothem
        b2 = xy[:,0] > -field_apothem
        b3 = xy[:,1] < field_apothem
        b4 = xy[:,1] > -field_apothem
        return b1 & b2 & b3 & b4

    def hexagon_selector(self, xyz):
        """Select galaxies in a hexagon centered at x,y = field.center"""
        field_apothem = np.sqrt(self.area / (2*np.sqrt(3)))
        xy = xyz[:,:2] - self.center[np.newaxis,:2]
        diagonal = math.sqrt(3.) * xy[:,0]
        b1 = xy[:,1] < field_apothem
        b2 = xy[:,1] > -field_apothem
        b3 = xy[:,1] < 2*field_apothem - diagonal
        b4 = xy[:,1] < 2*field_apothem + diagonal
        b5 = xy[:,1] > -2*field_apothem - diagonal
        b6 = xy[:,1] > -2*field_apothem + diagonal
        return b1 & b2 & b3 & b4 & b5 & b6

    def circle_fieldshape(self, rdz=False):
        if rdz: raise NotImplementedError("Why would you need to know the Celestial field shape of a Cartesian field?...")
        field_radius = self.circle_sqdeg2radius()
        return np.array([2.*field_radius]*2 + [self._z_length()], dtype=np.float32)

    def square_fieldshape(self, rdz=False):
        if rdz: raise NotImplementedError("Why would you need to know the Celestial field shape of a Cartesian field?...")
        field_apothem = self.square_sqdeg2apothem()
        return np.array([2.*field_apothem]*2 + [self._z_length()], dtype=np.float32)

    def hexagon_fieldshape(self, rdz=False):
        if rdz: raise NotImplementedError("Why would you need to know the Celestial field shape of a Cartesian field?...")
        field_apothem = self.hexagon_sqdeg2apothem()
        return np.array([4./math.sqrt(3.)*field_apothem, 2.*field_apothem, self._z_length()], dtype=np.float32)

class CelestialSelector(FieldSelector):
    def __init__(self, mockfield):
        FieldSelector.__init__(self, mockfield)

    def circle_selector(self, rdz):
        """Select galaxies in a circle centered at ra,dec = (0,0) radans"""
        field_radius = self.circle_sqdeg2radius(return_angle=True)
        rd = rdz[:,:2] - self.center_rdz[np.newaxis,:2]
        z = np.cos(rd[:,0]) * np.cos(rd[:,1])
        y = np.sin(rd[:,0]) * np.cos(rd[:,1])
        x = -np.sin(rd[:,1])
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        return theta < field_radius


    def square_selector(self, rdz):
        """Select galaxies in a square centered at ra,dec = (0,0) radians"""
        field_apothem = self.square_sqdeg2apothem(return_angle=True)
        rd = rdz[:,:2] - self.center_rdz[np.newaxis,:2]
        b1 = rd[:,0] < field_apothem
        b2 = rd[:,0] > -field_apothem
        b3 = rd[:,1] < field_apothem
        b4 = rd[:,1] > -field_apothem
        return b1 & b2 & b3 & b4

    def hexagon_selector(self, rdz):
        """Select galaxies in a hexagon centered at ra,dec = (0,0) radians"""
        field_apothem = self.hexagon_sqdeg2apothem(return_angle=True)
        rd = rdz[:,:2] - self.center_rdz[np.newaxis,:2]
        diagonal = np.sqrt(3.) * rd[:,0]
        b1 = rd[:,1] < field_apothem
        b2 = rd[:,1] > -field_apothem
        b3 = rd[:,1] < 2*field_apothem - diagonal
        b4 = rd[:,1] < 2*field_apothem + diagonal
        b5 = rd[:,1] > -2*field_apothem - diagonal
        b6 = rd[:,1] > -2*field_apothem + diagonal
        return b1 & b2 & b3 & b4 & b5 & b6

    def circle_fieldshape(self, rdz=False):
        field_radius = self.circle_sqdeg2radius(return_angle=True)
        if rdz:
            return np.array([2.*field_radius]*2 + [self.delta_z], dtype=np.float32)
        else:
            field_radius = hf.angle_lim_to_dist(field_radius, self.mean_redshift+self.delta_z/2., self.cosmo)
            return np.array([2.*field_radius]*2 + [self._z_length()], dtype=np.float32)

    def square_fieldshape(self, rdz=False):
        field_apothem = self.square_sqdeg2apothem(return_angle=True)
        if rdz:
            return np.array([2.*field_apothem]*2 + [self.delta_z], dtype=np.float32)
        else:
            field_apothem = hf.angle_lim_to_dist(field_apothem, self.mean_redshift+self.delta_z/2., self.cosmo)
            return np.array([2.*field_apothem]*2 + [self._z_length()], dtype=np.float32)

    def hexagon_fieldshape(self, rdz=False):
        field_apothem = self.hexagon_sqdeg2apothem(return_angle=True)
        if rdz:
            return np.array([4./math.sqrt(3.)*field_apothem, 2.*field_apothem, self.delta_z], dtype=np.float32)
        else:
            field_apothem = hf.angle_lim_to_dist(field_apothem, self.mean_redshift+self.delta_z/2., self.cosmo)
            return np.array([4./math.sqrt(3.)*field_apothem, 2.*field_apothem, self._z_length()], dtype=np.float32)



class BoxField:
    """
    BoxField(simbox, **kwargs)
    
    Conduct a mock observation of a single field of the populated galaxies of a simbox via Cartesian selection (i.e. a box of specified shape)
    
    Default Values:
        - **center** = `simbox.Lbox`/2.
        - **shape** = None (select all, equivalent to `shape=simbox.Lbox`)
        - **realspace** = False
        - **collision_fraction** = 0.
        - **realspace_selection** = False
        - **empty** = `simbox.empty` (default = False)
        - **rand_density_factor** = 10.
        - **zprec** = 1e-3
    Arguments
    ---------
    simbox : SimBox object
        SimBox object containing halos and galaxies from which to select and observe
    
    Keyword Arguments
    -----------------
    center : array_like, with shape (3,)
        Cartesian coordinate between [0,0,0] and `simbox.Lbox` which specifies the position at which [ra,dec,z] = [0, 0, `simbox.redshift`].
        
    shape : float or array_like, with shape (3,)
        Length of the field in the x, y, and z directions, i.e. select all galaxies within `center` :math:`\pm` 0.5 * `shape`. If float is given, then each dimension will be the same length.
    
    realspace : boolean
        If true, do not apply redshift distortion to galaxies.
    
    collision_fraction : float, between 0 and 1
        Fraction of galaxies to randomly exclude from the observation.
        
    realspace_selection : boolean
        If true, select galaxies before applying velocity distortion.
    
    empty : boolean
        If true, don't actually select any galaxies. Necessary if the simbox has not been populated.
    
    rand_density_factor : float
        If randoms are generated by self.make_rands(), then generate this many times more data than galaxies.

    zprec : float
        The precision of the redshift cubic interpolation grid. Smaller values provide more accurate redshifts, but this can be expensive.
    
    Useful Methods
    --------------
    - get_data(rdz=False, realspace=False)
    - get_rands(rdz=False, realspace=False)
    - get_vel()
    - get_redshift(realspace=False)
    - get_dist(realspace=False)
    - get_mgid()
    - get_shape(rdz=False)
    - make_rands()
    """
    def __init__(self, simbox, **kwargs):
        self._kwargs_ = {**kwargs, "simbox":simbox}

        self.simbox = simbox
        self.center = self.simbox.Lbox/2.
        self.shape = None
        self.collision_fraction = 0.
        self.realspace_selection = False
        self.empty = simbox.empty
        self.rand_density_factor = 10.
        self.zprec = 1e-3
        self.halo_vel_factor = None
        self.gal_vel_factor = None

        hf.kwargs2attributes(self, kwargs)

        self.center = np.asarray(self.center, dtype=np.float32)
        self.selection = slice(None)
        if (not self.shape is None) and (not self.empty):
            self.shape = np.atleast_1d(self.shape).astype(np.float32)
            if len(self.shape) == 1: self.shape = np.tile(self.shape, 3)
            self.selection = self._make_selection()

        # For BoxField, Cartesian selection/distortion is ALWAYS true.
        self.center_rdz = np.array([0.,0.,simbox.redshift], dtype=np.float32)
        self.cartesian_distortion = True
        self.cartesian_selection = True

        # Saved data
        dist = simbox.cosmo.comoving_distance(simbox.redshift).value * simbox.cosmo.h
        self.origin = self.center - np.array([0,0,dist])
        self._xyz = None
        self._xyz_real = None
        self._xyz_rands = None
        self._rdz = None
        self._rdz_real = None
        self._rdz_rands = None
        #gc.collect()

    def get(self, key):
        """
        Catch-all function for returning a column from simbox.gals under
        the selection function of this field.
        
        Returns
        -------
        data : ndarray/astropy column of shape (N,)
            Array containing the column specified by `key`.
        
        Parameters
        ----------
        key : string
            String of the key specifying the column we want to grab
        """
        return self.simbox.gals[key][self.selection]

    def get_data(self, rdz=False, realspace=False) -> np.ndarray:
        """
        Returns the positions of all galaxies selected by this object.
        
        Returns
        -------
        data : ndarray of shape (N,3)
            Array containing columns x, y, z (units of Mpc/h). If `rdz` is ``True`` then columns are instead ra, dec, redshift. To separate the columns, do one of the following:
                
            >>> x = data[:,0]; y = data[:,1]; z = data[:,2]
            
            >>> x,y,z = data.T
        
        Parameters
        ----------
        rdz : boolean (default = False)
            If ``True``, return columns of ra, dec, z (ra,dec in radians)
        
        realspace : boolean (default = False)
            If ``True``, return galaxies with their original positions, before velocity distortion
        """
        if rdz:
            xyz = self._rdz_real if realspace else self._rdz
        else:
            xyz = self._xyz_real if realspace else self._xyz
        if xyz is None:
            xyz = self._get_gals(rdz=rdz, realspace=realspace)
            self._set_data(xyz, rdz=rdz, realspace=realspace)
        return xyz

    def get_rands(self, rdz=False, realspace=True):
        """
        Returns the positions of uniform random data selected by this object.
        
        Returns
        -------
        rands : ndarray of shape (N,3)
            Array containing columns x, y, z (units of Mpc/h). If `rdz` is ``True`` then columns are instead ra, dec, redshift. To separate the columns, do one of the following:
            
            >>> x = rands[:,0]; y = rands[:,1]; z = rands[:,2]
            
            >>> x,y,z = rands.T
        
        Parameters
        ----------
        rdz : boolean (default = False)
            If ``True``, return columns of ra, dec, z (ra,dec in radians)
        
        realspace : boolean (default = True)
            This argument is IGNORED and only included for consistency between get_data() and get_rands()
        """
        del realspace
        if self._xyz_rands is None:
            self.make_rands()
        if rdz and self._rdz_rands is None:
            self._rdz_rands = hf.ra_dec_z(self._xyz_rands-self.origin, np.zeros_like(self._xyz_rands), self.simbox.cosmo, self.zprec)
        return self._rdz_rands if rdz else self._xyz_rands

    def get_vel(self, halo_vel_factor=None, gal_vel_factor=None):
        """
        Returns the velocity of each galaxy selected by this object.
        
        Returns
        -------
        vel : ndarray of shape (N,3)
            Array containing columns vx, vy, vz (units of km/s). To separate the columns, do one of the following:
            
            >>> vx = vel[:,0]; vy = vel[:,1]; vz = vel[:,2]
            
            >>> vx,vy,vz = vel.T
        """
        if halo_vel_factor is None:
            halo_vel_factor = self.halo_vel_factor
        if gal_vel_factor is None:
            gal_vel_factor = self.gal_vel_factor

        if (not halo_vel_factor is None) or (not gal_vel_factor is None):
            return hf.factor_velocity(
                hf.xyz_array(self.simbox.gals,
                        ["vx","vy","vz"])[self.selection],
                hf.xyz_array(self.simbox.gals,
                        ["halo_vx", "halo_vy", "halo_vz"])[self.selection],
                halo_vel_factor=halo_vel_factor,
                gal_vel_factor=gal_vel_factor,
                inplace=True)
        else:
            return hf.xyz_array(self.simbox.gals,
                                ["vx","vy","vz"])[self.selection]

    def get_redshift(self, realspace=False):
        """
        Returns the redshift of each galaxy selected by this object. Equivalent to ``get_data(rdz=True)[:,2]``.
        
        Returns
        -------
        redshift : ndarray of shape (N,)
            Array containing the redshift of each galaxy (unitless)
        
        Parameters
        ----------
        realspace : boolean (default = False)
            If ``True``, return cosmological redshift only, ignoring any velocity distortion
        """
        rdz = self._rdz_real if realspace else self._rdz
        if rdz is None:
            rdz = self.get_data(rdz=True, realspace=realspace)

        return rdz[:,2]

    def get_dist(self, realspace=False):
        """
        Returns the comoving distance of each galaxy selected by this object.
        
        Returns
        -------
        distance : ndarray of shape (N,3)
            Array containing the distance of each galaxy (units of Mpc/h)
        
        Parameters
        ----------
        realspace : boolean (default = False)
            If ``True``, return true distance, ignoring any velocity distortion
        """
        xyz = self.get_data(realspace=realspace)
        return np.sqrt(np.sum(xyz**2, axis=1))


    def get_mgid(self):
        """
        Returns the Mock Galaxy ID of each galaxy selected by this object.
        
        Returns
        -------
        mgid : ndarray of shape (N,)
            Array containing a unique number for each galaxy, useful for cross-matching galaxies between fields
        """
        return np.asarray(self.simbox.gals["mgid"][self.selection])

    def get_shape(self, rdz=False):
        """
        Returns the length along each dimension necessary to fully contain this field
        
        Returns
        -------
        shape : ndarray of shape (3,)
            Either the Cartesian or Celestial shape of this field
        
        Parameters
        ----------
        rdz : boolean (default = False)
            If true, return Celestial shape (length along ra, dec, redshift) instead of the Cartesian shape (length along x, y, z)
        """
        if rdz:
            raise ValueError("Sorry, I haven't implemented get_shape(rdz=True) for a BoxField yet.")
        if self.shape is None:
            return self.simbox.Lbox
        else:
            return self.shape

    def make_rands(self, density_factor=None, seed=None):
        """
        Generate a uniform distribution of random data to trace the selection function of this object.
        
        Parameters
        ----------
        density_factor : float (default = self.rand_density_factor)
            Generate this many times more data than the number of galaxies. A value of ~20 or higher is encouraged if randoms are going to be used for correlation functions down to radii under 1 Mpc/h
        
        seed : int (default = None)
            Seed for the random generation so it may be reproduced
        """
        self._xyz_rands = self._rdz_rands = None
        if density_factor is None:
            density_factor = self.rand_density_factor

        Nrand = int(density_factor * len(self.get_data()) + 0.5)
        lower = self.center - self.get_shape()/2.

        if not seed is None: np.random.seed(seed)
        self._xyz_rands = (np.random.random((Nrand,3))*self.get_shape()[None,:] + lower[None,:]).astype(np.float32)
        if not seed is None: np.random.seed(None)

    def _make_selection(self, xyz=None):
        if xyz is None:
            xyz = self._get_gals(realspace=self.realspace_selection)

        lower = self.center - self.get_shape()/2.
        upper = self.center + self.get_shape()/2.
        selection = np.all((lower[None,:] <= xyz) &
                           (xyz <= upper[None,:]), axis=1)

        if self.collision_fraction > 0.:
            collisions = hf.sample_fraction(len(selection), self.collision_fraction)
            selection[collisions] = False

        self._set_data(xyz[selection], realspace=self.realspace_selection)
        return selection

    def _set_data(self, data, rdz=False, realspace=False):
        if realspace:
            if rdz:
                self._rdz_real = data
            else:
                self._xyz_real = data
        else:
            if rdz:
                self._rdz = data
            else:
                self._xyz = data

    def _get_gals(self, rdz=False, realspace=False):
        if rdz:
            xyz = self.get_data(realspace=realspace)
            xyz = hf.ra_dec_z(xyz-self.origin, np.zeros_like(xyz),
                              self.simbox.cosmo, self.zprec)
        else:
            xyz = hf.xyz_array(self.simbox.gals)[self.selection]
            if not realspace:
                if (self.halo_vel_factor is None) and (
                        self.gal_vel_factor is None):
                    vz = self.simbox.gals["vz"][self.selection]
                else:
                    vz = hf.factor_velocity(
                        self.simbox.gals["vz"][self.selection],
                        self.simbox.gals["halo_vz"][self.selection],
                        halo_vel_factor=self.halo_vel_factor,
                        gal_vel_factor=self.gal_vel_factor,
                        inplace=False)
                xyz = self._apply_distortion(xyz, vz)
        return xyz

    def _apply_distortion(self, xyz, vz):
        return return_xyz_formatted_array(*xyz.T, self.simbox.Lbox, self.simbox.cosmo, self.simbox.redshift, velocity=vz, velocity_distortion_dimension="z")




class MockField:
    """
    MockField(simbox, **kwargs)
    
    Conduct a mock observation of a single field of the populated galaxies of a simbox via celestial selection (i.e., galaxies selected by ra, dec, redshift)
    
    Default Values:
        - **center** = `simbox.Lbox`/2.
        - **center_rdz** = [0., 0., `simbox.redshift`]
        - **scheme** = "square"
        - **sqdeg** = 15.
        - **delta_z** = 0.1
        - **collision_fraction** = 0.
        - **realspace_selection** = False
        - **cartesian_selection** = False
        - **cartesian_distortion** = False
        - **empty** = False
        - **rand_density_factor** = 20.
        - **zprec** = 1e-3
    
    Arguments
    ---------
    simbox : SimBox object
        SimBox object containing halos and galaxies from which to select and observe
    
    Keyword Arguments
    -----------------
    center : array_like
        Cartesian coordinate between [0,0,0] and `simbox.Lbox` which specifies the position of [ra,dec,z] = [0, 0, `simbox.redshift`].
        
    center_rdz : array_like, with shape (2,) or (3,)
        [ra,dec] at which to center the field around (redshift index ignored).
        
    scheme : string
        Shape of the field, e.g. "circle", "square", "hexagon".
        
    sqdeg : float
        Field size in square degrees.
        
    delta_z : float
        Select redshift within `simbox.redshift` :math:`\pm` 0.5* `delta_z`.
        
    collision_fraction : float, between 0 and 1
        Fraction of galaxies to randomly exclude from the observation.
        
    realspace_selection : boolean
        If true, select galaxies before applying velocity distortion.
        
    cartesian_selection : boolean
        If true, select galaxies by Cartesian coordinates, with x-y selections applied as if the entire sample was at the redshift of the simbox.
        
    cartesian_distortion : boolean
        If true, apply velocity distortion along the :math:`\\hat{z}` direction instead of the exact line-of-sight direction.
    
    empty : boolean
        If true, don't actually select any galaxies. Necessary if the simbox has not been populated.
    
    rand_density_factor : float
        If randoms are generated by self.make_rands(), then generate this many times more data than galaxies.
    
    zprec : float
        The precision of the redshift cubic interpolation grid. Smaller values provide more accurate redshifts, but this can be expensive.
    
    Useful Methods
    --------------
    - get_data(rdz=False, realspace=False)
    - get_rands(rdz=False, realspace=True)
    - get_vel()
    - get_redshift(realspace=False)
    - get_dist(realspace=False)
    - get_mgid()
    - get_shape(rdz=False)
    - make_rands()
    """
    defaults = {
        "cartesian_distortion": False,
        "cartesian_selection": False,
        "realspace_selection": False,
        "collision_fraction": 0.,
        "scheme": "square",
        "sqdeg": 15.,
        "delta_z": 0.1,
        "zprec": 1e-3,
        "rand_density_factor": 20.,
    }
    def __init__(self, simbox, **kwargs):
        self._kwargs_ = {**kwargs, "simbox":simbox}
        self.simbox = simbox
        self.center = self.simbox.Lbox/2.
        self.center_rdz = np.array([0.,0.,simbox.redshift])
        self.empty = simbox.empty
        self.__dict__.update(self.defaults)
        self.halo_vel_factor = None
        self.gal_vel_factor = None


#        self.cartesian_distortion = False
#        self.cartesian_selection = False
#        self.realspace_selection = False
#        self.collision_fraction = 0.
#        self.scheme = "square"
#        self.sqdeg = 15.
#        self.delta_z = 0.1
#        self.zprec = 1e-3
#        self.rand_density_factor = 20.

        # Update default parameters with any keyword arguments
        hf.kwargs2attributes(self, kwargs)

        self._gals = {}
        self._rands = {}
        self.origin, self.Lbox_rdz = self._centers_to_origin()
        self.field_selector, self.redshift_selector = self.get_selectors()

        # Create field selection from FieldSelector, given sqdeg and scheme
        if not self.empty:
            self._initialize()

    def _initialize(self):
        if hasattr(self.simbox, "gals"):
            self.selection = self._make_selection()
        else:
            self.selection = slice(None)

        for key in self._gals:
            self._gals[key] = self._gals[key][self.selection]
        #gc.collect()

    def get_selectors(self):
        if self.cartesian_selection:
            return CartesianSelector(self).make_selection, RedshiftSelector(self).make_selection
        else:
            return CelestialSelector(self).make_selection, RedshiftSelector(self).make_selection


# Public member functions for data access
# =======================================

    def get_data(self, rdz=False, realspace=False):
        if rdz:
            return self._get_rdz(dataset=self._gals, realspace=realspace)
        else:
            return self._get_xyz(dataset=self._gals, realspace=realspace)

    def get_rands(self, rdz=False, realspace=True):
        del realspace
        if rdz:
            return self._get_rdz(dataset=self._rands)
        else:
            return self._get_xyz(dataset=self._rands)

    def get_vel(self, halo_vel_factor=None, gal_vel_factor=None):
        return self._get_vel(halo_vel_factor=halo_vel_factor,
                             gal_vel_factor=gal_vel_factor)

    def get_redshift(self, realspace=False):
        return self._get_redshift(realspace=realspace)

    def get_dist(self, realspace=False):
        xyz = self.get_data(realspace=realspace)
        return np.sqrt(np.sum(xyz**2, axis=1))

    def get_mgid(self):
        return np.asarray(self.simbox.gals["mgid"][self.selection])

    def get_shape(self, rdz=False):
        if self.cartesian_selection:
            selector = CartesianSelector(self)
        else:
            selector = CelestialSelector(self)

        return selector.get_fieldshape(rdz=rdz)

    def get_lims(self, rdz=False, overestimation_factor=1.):
        shape = self.get_shape(rdz=rdz) * overestimation_factor
        center = self.center_rdz if rdz else self.center

        xlim = [center[0]-shape[0]/2., center[0]+shape[0]/2.]
        ylim = [center[1]-shape[1]/2., center[1]+shape[1]/2.]
        zlim = [center[2]-shape[2]/2., center[2]+shape[2]/2.]

        return np.array([xlim, ylim, zlim], dtype=np.float32)

    def make_rands(self, density_factor=None, seed=None):
        if density_factor is None:
            density_factor = self.rand_density_factor
        else:
            self.rand_density_factor = density_factor
        density_gals = self.simbox.get_density()
        density_rands = density_factor * density_gals

        if self.cartesian_selection:
            volume = np.product(self.get_shape(rdz=False))
            Nran = int(density_rands * volume + 0.5)
            # Cartesian selection
            if not seed is None:
                np.random.seed(seed)
            rands = (np.random.random((Nran, 3)).astype(np.float32) - 0.5) * self.get_shape(rdz=False)[None,:] + self.center[None,:]

        else:
            # Celestial (ra,dec) selection
            ralim, declim, _ = self.get_lims(rdz=True, overestimation_factor=1.02)
            zlim = self.get_lims(rdz=False, overestimation_factor=1.02)[2] - self.origin[2]

            volume = hf.volume_rdz(ralim, declim, zlim)
            Nran = int(density_rands * volume + 0.5)

            rands = hf.rand_rdz(Nran, ralim, declim, zlim, seed)
            rands = self.rdz2xyz(rands, input_is_distance=True)


        self._rands = {
            "x_real": rands[:,0],
            "y_real": rands[:,1],
            "z_real": rands[:,2]
        }

        selection = self._make_selection(dataset='rands') & self._select_within_simbox(rands)
        for key in self._rands:
            self._rands[key] = self._rands[key][selection]

    get = BoxField.get
    get_data.__doc__ = BoxField.get_data.__doc__
    get_rands.__doc__ = BoxField.get_rands.__doc__
    get_vel.__doc__ = BoxField.get_vel.__doc__
    get_redshift.__doc__ = BoxField.get_redshift.__doc__
    get_dist.__doc__ = BoxField.get_dist.__doc__
    get_mgid.__doc__ = BoxField.get_mgid.__doc__
    get_shape.__doc__ = BoxField.get_shape.__doc__
    make_rands.__doc__ = BoxField.make_rands.__doc__

    def xyz2rdz(self, xyz, vel=None):
        return hf.ra_dec_z(xyz-self.origin, vel, cosmo=self.simbox.cosmo, zprec=self.zprec)

    def rdz2xyz(self, rdz, input_is_distance=False):
        cosmo = None if input_is_distance else self.simbox.cosmo
        return hf.rdz2xyz(rdz, cosmo=cosmo) + self.origin

    def volume(self):
        (_,_,(d1,d2)),v,_ = self._measure_volume_setup(oef=1)
        if self.cartesian_selection:
            return v
        else:
            omega = self.sqdeg * (np.pi/180.)**2
            return omega/3. * (d2**3 - d1**3)


    def measure_volume(self, precision=1e-3, recursion_lim=10):
        lims, volume, rand_generator = self._measure_volume_setup()

        N0 = 10000
        N = 0
        n = 0
        r = 0
        # order of magnitude estimation only
        sigma = lambda: np.sqrt( (1-r)/(r*(N-1)) )
        get_Nmore = lambda: min([
            int( N0 - N + 1 + (1-r)/(r*precision**2) ),
            2**24])

        Nmore = N0
        i = 0
        while (not r) or (sigma() > precision):
            i += 1
            if i > recursion_lim:
                s = sigma() if r else np.inf
                raise RecursionError(f"After {recursion_lim} "
                f"iterations, {N} randoms were generated. " 
                f"precision={s:e}, short of the required "
                f"precision of {precision:e}.")
            rand = rand_generator(Nmore, *lims)

            n += self.apply_selection(rand, input_is_distance=True).sum()
            N += Nmore

            r = n/N

            Nmore = get_Nmore() if r else N0

        return r * volume

    def apply_selection(self, data, input_is_distance=False):
        return (self.field_selector(data[:,:2]) &
                self.redshift_selector(data[:,2], input_is_distance))


    def _measure_volume_setup(self, oef=1.5):
        if self.cartesian_selection:
            lims = self.get_lims(rdz=False, overestimation_factor=oef)
            lims[-1] -= self.origin[-1]
            volume = np.product(np.diff(lims, axis=1))
            rand_generator = (lambda N, xlim, ylim, zlim:
                              (np.random.random((N,3)) *
                              np.diff(lims, axis=1).T) +
                              lims[:,:1].T)
        else:
            lims = self.get_lims(rdz=True, overestimation_factor=oef)
            lims[2] = self.simbox.redshift2distance(lims[2])
            volume = hf.volume_rdz(*lims)
            rand_generator = hf.rand_rdz

        return lims, volume, rand_generator

    def _get_rdz(self, dataset=None, realspace=False):
        dataset, datanames, _ = self._get_dataset(dataset)
        if dataset is self._rands:
            realspace = True

        if realspace:
            zkey = 'redshift_real'
        else:
            zkey = 'redshift'

        already_done = {'ra', 'dec', zkey}.issubset(datanames)
        if not already_done:
            rdz = self._redshift_distortion_rdz(realspace=realspace, dataset=dataset)
            if self.cartesian_distortion:
                rdz2 = self._get_redshift(realspace=realspace, dataset=dataset)
            else:
                rdz2 = rdz[:,2]
            hf.update_table(dataset, {'ra': rdz[:,0], 'dec': rdz[:,1], zkey: rdz2})

        rdz = hf.xyz_array(dataset, keys=['ra', 'dec', zkey])
        return rdz

    def _get_xyz(self, dataset=None, realspace=False):
        dataset, datanames, selection = self._get_dataset(dataset)
        if dataset is self._rands:
            realspace = True

        if realspace:
            if dataset is self._gals:
                return hf.xyz_array(self.simbox.gals)[selection]
            elif dataset is self.simbox.gals:
                return hf.xyz_array(self.simbox.gals)
            # update_table(dataset, {"x_real":x, "y_real":y, "z_real":z})

        if realspace:
            xkey, ykey, zkey = "x_real", "y_real", "z_real"
        elif self.cartesian_distortion:
            xkey, ykey, zkey = "x_real", "y_real", "z"
            if dataset is self.simbox.gals:
                zkey = "z_red"
        else:
            xkey, ykey, zkey = "x", "y", "z"
            if dataset is self.simbox.gals:
                xkey, ykey, zkey = "x_red", "y_red", "z_red"


        if realspace or self.cartesian_distortion:
            already_done = {xkey, ykey, zkey}.issubset(datanames)
            if not already_done:
                xyz = self._cartesian_distortion_xyz(realspace=realspace, dataset=dataset)
                hf.update_table(dataset, {xkey: xyz[:,0], ykey: xyz[:,1], zkey: xyz[:,2]})

        else:
            already_done = {xkey, ykey, zkey}.issubset(datanames)
            if not already_done:
                rdz = self._get_rdz(realspace=realspace, dataset=dataset)
                xyz = self.rdz2xyz(rdz)
                hf.update_table(dataset, {xkey: xyz[:,0], ykey: xyz[:,1], zkey: xyz[:,2]})

        xyz = hf.xyz_array(dataset, keys=[xkey, ykey, zkey])
        return xyz

    def _get_vel(self, realspace=False, dataset=None,
                 halo_vel_factor=None, gal_vel_factor=None):
        dataset, datanames, selection = self._get_dataset(dataset)
        if halo_vel_factor is None:
            halo_vel_factor = self.halo_vel_factor
        if gal_vel_factor is None:
            gal_vel_factor = self.gal_vel_factor

        if realspace or dataset is self._rands:
            if len(datanames) == 0:
                length = 1
            else:
                length = len(dataset[ list(datanames)[0] ])
            return np.zeros((length, 3))
        else:
            if not(halo_vel_factor is None) or not(gal_vel_factor is None):
                return hf.factor_velocity(
                    hf.xyz_array(self.simbox.gals,
                                 keys=['vx', 'vy', 'vz'])[selection],
                    hf.xyz_array(self.simbox.gals,
                                 keys=['halo_vx', 'halo_vy', 'halo_vz'])[selection],
                    halo_vel_factor=halo_vel_factor,
                    gal_vel_factor=gal_vel_factor,
                    inplace=True)
            else:
                return hf.xyz_array(self.simbox.gals,
                                    keys=['vx', 'vy', 'vz'])[selection]

    def _get_redshift(self, realspace=False, dataset=None):
        dataset, datanames, selection = self._get_dataset(dataset)
        if dataset is self._rands:
            realspace = True

        if realspace:
            zkey = 'redshift_real'
        else:
            zkey = 'redshift'

        already_done = zkey in datanames
        if not already_done:
            xyz = self._get_xyz(realspace=realspace, dataset=dataset)
            if self.cartesian_distortion:
                dist = xyz[:,2] - self.origin[None,2]
            else:
                dist = np.sqrt(np.sum((xyz-self.origin[None,:])**2, axis=1))

            vr = np.zeros(dist.shape)
            redshift = hf.distance2redshift(dist, vr, self.simbox.cosmo, self.zprec)
            hf.update_table(dataset, {zkey: redshift})

        redshift = dataset[zkey]
        return redshift

    def _make_selection(self, dataset=None):
        if self.cartesian_selection:
            data = self._get_xyz(realspace=self.realspace_selection, dataset=dataset)
        else:
            data = self._get_rdz(realspace=self.realspace_selection, dataset=dataset)

        selection = self.apply_selection(data, input_is_distance=self.cartesian_selection)
        collisions = hf.sample_fraction(len(selection), self.collision_fraction)
        selection[collisions] = False
        return selection

    def _get_dataset_helper(self, dataset):
        if not isinstance(dataset, str):
            return dataset
        elif dataset.lower().startswith("gal"):
            return self._gals
        elif dataset.lower().startswith("sim"):
            return self.simbox.gals
        elif dataset.lower().startswith("rand"):
            return self._rands
        else:
            return dataset

    def _get_dataset(self, dataset):
        dataset = self._get_dataset_helper(dataset)
        if dataset is None or dataset is self._gals:
            dataset = self._gals
            datanames = dataset.keys()
            if 'selection' in self.__dict__:
                selection = self.selection
            else:
                selection = slice(None)
        elif dataset is self.simbox.gals:
            datanames = dataset.colnames
            selection = slice(None)
        elif dataset is self._rands:
            if dataset == {}:
                self.make_rands()
            dataset = self._rands
            datanames = dataset.keys()
            selection = slice(None)
        else:
            raise ValueError("Cannot interpret dataset=%s; must be in {'gals', 'simbox', 'rands'}"% dataset)

        return dataset, datanames, selection

    def _select_within_simbox(self, data):
        """Construct boolean selection mask to ensure no data is outside the SimBox"""
        lower = np.array([0.,0.,0.])
        upper = self.simbox.Lbox - lower

        select_lower = np.all(data >= lower[None,:], axis=1)
        select_upper = np.all(data <= upper[None,:], axis=1)

        selection = select_lower & select_upper
# =============================================================================
#         if np.any(~selection):
#             print(np.where(~selection), "which is a fraction of", np.where(~selection)[0].size / selection.size, flush=True)
#             print("WARNING: Attempting to make a selection beyond the extents of the SimulationBox.", flush=True)
# =============================================================================
        return selection

    def _cartesian_distortion_xyz(self, realspace=False, dataset=None):
        xyz = self._get_xyz(realspace=True, dataset=dataset)
        v = self._get_vel(realspace=realspace, dataset=dataset)[:,2]
        xyz_red = return_xyz_formatted_array(xyz[:,0], xyz[:,1], xyz[:,2], velocity=v, velocity_distortion_dimension="z",
            cosmology=self.simbox.cosmo, redshift=self.simbox.redshift, period=self.simbox.Lbox)

        return xyz_red.astype(np.float32)

    def _redshift_distortion_rdz(self, realspace=False, dataset=None):
        xyz = self._get_xyz(realspace=True, dataset=dataset)
        vel = None if realspace else self._get_vel(dataset=dataset)
        return self.xyz2rdz(xyz, vel)

    def _centers_to_origin(self):
        # Cast to numpy arrays
        self.center = np.asarray(self.center, dtype=np.float32)
        self.center_rdz = np.array([*self.center_rdz[:2], self.simbox.redshift], dtype=np.float32)

        # Set the (cartesian) center as if center_rdz = [0,0,redshift]
        close_dist = self.simbox.cosmo.comoving_distance(self.simbox.redshift - self.delta_z/2.).value * self.simbox.cosmo.h
        center_dist = close_dist + self.get_shape()[2]/2.
        origin = self.center - np.array([0., 0., center_dist], dtype=np.float32)

        points = np.array([ [0.,0.,0.], [self.simbox.Lbox[0], 0., 0.] ]) + origin
        ra = hf.ra_dec_z(points, np.zeros_like(points), self.simbox.cosmo)[:,0]
        ra = abs(ra[1]-ra[0]) * math.sqrt(2.)

        Lbox_rdz = np.array([ra, ra, self.delta_z], dtype=np.float32)

        return origin, Lbox_rdz


class MockSurvey:
    """
    MockSurvey(simbox, rdz_centers, **kwargs)
    
    Conduct a mock observation of a multi-field survey of galaxies contained in a SimBox, via celestial selection ONLY (i.e., galaxies selected by ra, dec, redshift)
    
    Arguments
    ---------
    simbox : SimBox object
        SimBox object containing halos and galaxies from which to select and observe
    
    rdz_centers : ndarray of shape (ncenters,2) or (ncenters,3)
        List of ra,dec coordinates to place the centers of the fields of this survey
    
    All keyword arguments are passed to MockField object (**see MockField documentation below**)
    """
    def __init__(self, simbox, rdz_centers, **kwargs):
        self._kwargs_ = {**kwargs, "simbox":simbox,
                         "rdz_centers":rdz_centers}

        # Initialize the MockFields in their specified positions
        self.fields = [simbox.field(center_rdz=c, **kwargs) for c in rdz_centers]

        # Create selection function that only counts each galaxy ONCE
        mgid = []
        for field in self.fields: mgid += field.get_mgid().tolist()
        self.mgid, self.selection = np.unique(np.asarray(mgid), return_index=True)

        # Hacky/lazy way of defining the methods get_data, get_vel, etc.
        # using the corresponding MockField methods
        # accessors = [x for x in dir(MockField) if ((not x.startswith("_")) and (not x in ["make_rands","get_shape","get_rands"]))]
        accessors = ["get_data", "get_dist", "get_mgid", "get_redshift", "get_vel"]
        for accessor in accessors:
            unbound = (lambda accessor: lambda self, rdz=False, realspace=False: self._field2survey(accessor, rdz, realspace))(accessor)
            unbound.__doc__ = self.fields[0].__getattribute__(accessor).__doc__
            # noinspection PyArgumentList
            self.__setattr__(accessor, unbound.__get__(self))

        self.simbox = simbox
        self.rand_density_factor = self.fields[0].rand_density_factor
        self.origin = self.fields[0].origin

        self.center = kwargs.get("center", simbox.Lbox/2.)
        self.center_rdz = np.array([0., 0., simbox.redshift], dtype=np.float32)

    def field_selector(self, rdz):
        s = [field.field_selector(rdz) for field in self.fields]
        return np.any(s, axis=0)

    def redshift_selector(self, redshift):
        s = [field.redshift_selector(redshift) for field in self.fields]
        return np.any(s, axis=0)

    def get_rands(self, rdz=False):
        if not hasattr(self, "rand_rdz"):
            self.make_rands()

        if rdz:
            return self.rand_rdz
        else:
            return self.rand_xyz

    def get_shape(self, rdz=False, return_lims=False):
        centers = np.array([field.center_rdz if rdz else field.center for field in self.fields])
        shapes = np.array([field.get_shape(rdz=rdz) for field in self.fields])
        lower,upper = centers-shapes/2., centers+shapes/2.
        lower,upper = np.min(lower, axis=0), np.max(upper, axis=0)
        if return_lims:
            return lower,upper
        else:
            return upper - lower

    def make_rands(self, density_factor=None, seed=None):
        if density_factor is None:
            density_factor = self.rand_density_factor

        density = density_factor * self.simbox.get_density()
        lims = np.asarray(self.get_shape(rdz=True, return_lims=True)).T
        lims[2] = np.asarray(self.get_shape(rdz=False, return_lims=True)).T[2]

        N = density * hf.volume_rdz(*lims)
        self.rand_rdz = hf.rand_rdz(N, *lims, seed=seed).astype(np.float32)

        selections = [field.apply_selection(self.rand_rdz, input_is_distance=True) for field in self.fields]
        selection = np.any(selections, axis=0)

        self.rand_rdz = self.rand_rdz[selection]
        self.rand_xyz = self.rdz2xyz(self.rand_rdz, input_is_distance=True)


    def _field2survey(self, funcstring, rdz, realspace):
        ans = []
        kwargs = [("rdz", rdz), ("realspace", realspace)]
        kwargs = {kwkey:kwval for kwkey,kwval in kwargs if kwkey in getfullargspec(self.fields[0].__getattribute__(funcstring))[0]}
        for field in self.fields: ans += field.__getattribute__(funcstring)(**kwargs).tolist()
        ans = np.asarray(ans, dtype=np.float32)[self.selection]
        return ans

class PFSSurvey(MockSurvey):
    def __init__(self, simbox=None, center=None, threshold=-21):

        simname = "smdpl"
        redshift = 1.35
        Nbox = (1,1,3)
        threshold = threshold

        scheme = "hexagon"
        delta_z = 0.7
        sqdeg_total = 15.
        collision_fraction = 0.3
        numfield = 10
        sqdeg_each = sqdeg_total / numfield

        if simbox is None:
            simbox = SimBox(simname=simname, redshift=redshift, Nbox=Nbox, threshold=threshold)
        if center is None:
            center = simbox.Lbox/2.

        emptyfield = simbox.field(delta_z=delta_z, sqdeg=sqdeg_each, scheme=scheme, empty=True)
        w,h,_ = emptyfield.get_shape(rdz=True)
        w *= 3./4.; h *= 0.5
        centers = [[-3*w,0], [-2*w,h], [-2*w,-h], [-w,0], [0,h], [0,-h],
                   [w,0], [2*w,h], [2*w,-h], [3*w,0]]

        assert(len(centers) == numfield)

        MockSurvey.__init__(self, simbox, scheme=scheme, rdz_centers=centers, center=center, delta_z=delta_z, sqdeg=sqdeg_each, collision_fraction=collision_fraction)

    def wp_jackknife(self):
        data = self.get_data()
        rands = self.get_rands()
        data2b = self.get_data(rdz=True)
        rand2b = self.get_rands(1)
        centers = [field.center_rdz for field in self.fields]
        boxshape = self.fields[0].get_shape(1)
        nbins = (2,2,1)
        func = cf.wp_rp
        args = np.logspace(-0.87, 1.73, 14), 50. # rpbins, pimax

        wp,covar = cf.block_jackknife(data, rands, centers, boxshape, nbins, data2b, rand2b, func, args, rdz_distance=False, debugging_plots=True)

        return wp,covar
MockSurvey.__doc__ += MockField.__doc__


class SimBox:
    """
    SimBox(**kwargs)
    
    Stores all data for the halos, galaxies, and the model dictating their connection.
    
    Default Values:
        - **simname** = "smdpl"
        - **version_name** = None
        - **hodname** = "zheng07"
        - **cosmo** = cosmology.FlatLambdaCDM(name="WMAP5", H0=70.2, Om0=0.277, Tcmb0=2.725, Neff=3.04, Ob0=0.0459)
        - **redshift** = 1.0
        - **threshold** = 10.5
        - **populate_on_instantiation** = True
        - **dz_tol** = 0.1
        - **Nbox** = [1, 1, 1]
        - **rotation** = None
        - **empty** = False
        - **Lbox** = [400., 400., 400.] (if empty=True)
        - **volume** = None

    Keyword Arguments
    -----------------
    simname : string
        Identifies the halo table to use; options: {"smdpl", "bolshoi", "bolplanck", "multidark", "consuelo"}
    
    version_name : string
        Identifies the version of the halo table to use (version string is set during creation of the Cached Halo Catalog)
    
    hodname : string
        Identifies the HOD model to dictate the galaxy-halo connection
    
    cosmo : cosmology.Cosmology object
        Used primarily to convert comoving distance to redshift
    
    redshift : float
        The redshift of the dark matter simulation snapshot
    
    threshold : float
        Only galaxies larger than this value (of stellar mass/absolute magnitude) will be populated
    
    populate_on_instantiation : bool
        If False, the galaxies will not be populated upon instantiation, which saves a few seconds
    
    dz_tol : float
        The precision to which the Cosmology object will calculate its interpolation grid of redshifts
    
    Nbox : length-3 array-like (int, int, int)
        Number of periodic halo cubes that are instantiated along each axis
        
    rotation : int or None
        Default (``None``) produces random orientation, while an integer 0, 1, or 2 (mod 3) indicates that the z, y, or x axis should be the one along our line-of-sight respectively. ``None`` is helpful for producing a LITTLE bit more sample variance in statistics which depend on line-of-sight, like the projected correlation function, :math:`w_{\\rm p}(r_{\\rm p})`. However, for consistent results, you must always pass rotation=0 (or any consistent integer).
    
    empty : boolean
        If true, don't load the halo table or populate galaxies, which saves a lot of time, but produces an empty SimBox object which is almost useless
    
    Lbox : length-3 array-like (float, float, float)
        Length of each axis of the simulation box (taken automatically from halo catalog unless empty=True)
    
    volume : float
        Volume of the simulation box region where halos are selected from (not needed unless calculating density of a position-masked set of halos)
    
    Useful Methods
    --------------
    - populate_mock(seed=None, masking_function=None, rotation=0, Nbox=None)
    - update_param_dict(param_dict, param_names=None)
    - redshift2distance(redshift)
    - get_density()
    - get_volume()
    """
    defaults = {
        "simname": "smdpl", # Small Multidark Planck; also try bolshoi, bolplanck, multidark, consuelo...
        "version_name": None, # Version of halo catalog: "halotools_v0p4" or "my_cosmosim_halos"
        "hodname": "zheng07", # Specifies the HOD model to populate galaxies; also try Zheng07
        "cosmo": cosmology.FlatLambdaCDM(name="WMAP5", H0=70.2, Om0=0.277, Tcmb0=2.725, Neff=3.04, Ob0=0.0459),
        "redshift": 1.0, # Redshift of simulation
        "threshold": -21., # Galaxy mass threshold is 10**[threshold] M_sun
        "populate_on_instantiation": True, # Populate the galaxies upon instantiation
        "dz_tol": 0.1, # Make sure only one halo catalog exists within +/- dz_tol of given redshift
        "Nbox": None,
        "rotation": None,
        "Lbox": np.array([400.]*3, dtype=np.float32),
        "empty": False,
        "volume": None,
    }
    def __init__(self, **kwargs):
        self._kwargs_ = kwargs.copy()
        self.__dict__.update(self.defaults)

        # Update default parameters with any keyword arguments
        hf.kwargs2attributes(self, kwargs)
        # Initialize model and get halos
        self.populated = False
        self.construct_model()
        self.Lbox = np.asarray(self.Lbox, dtype=np.float32)
        self.mgid_counter = 0 # assign mock galaxy id's starting from zero
        if not self.empty:
            self.get_halos()
            if self.populate_on_instantiation:
                # Paint galaxies into the dark matter halos
                self.populate_mock()

    def get_density(self, dataset="gals"):
        volume = self.get_volume()
        if dataset.startswith("gal"):
            number = len(self.gals)
        elif dataset.startswith("halo"):
            number = len(self.halos)
        else:
            raise ValueError("dataset=%s invalid. Must be in {`gal*`, `halo*`}"%dataset)
        return number/volume

    def get_volume(self):
        return np.product(self.Lbox) if self.volume is None else self.volume

    def redshift2distance(self, redshift):
        return hf.comoving_disth(redshift, self.cosmo)
# Conduct a mock observation (a single field or a multi-field survey)
# =========================================

    def field(self, **kwargs):
        """
        Returns a MockField object from this simbox (**see MockField documentation below**)
        """

        field = MockField(self, **kwargs)
        return field

    def survey(self, rdz_centers, **kwargs):
        """
        Returns a MockSurvey object from this simbox (**see MockSurvey documentation below**)
        """
        survey = MockSurvey(self, rdz_centers, **kwargs)
        return survey

    def boxfield(self, **kwargs):
        """
        Returns a BoxField object from this simbox (**see BoxField documentation below**)
        """
        field = BoxField(self, **kwargs)
        return field

    field.__doc__ += MockField.__doc__
    survey.__doc__ += MockSurvey.__doc__
    boxfield.__doc__ += BoxField.__doc__

# Helper member functions
# =======================
    
    def rotate(self, rotation, gals=None):
        if rotation is None:
            rotation = np.random.randint(3)
        if rotation:
            gals = self.gals if gals is None else gals
            k0 = [["x","y","z"], ["vx","vy","vz"],
            ["halo_x","halo_y","halo_z"], ["halo_vx","halo_vy","halo_vz"]]

            k1 = [[s+"_tmp" for s in keys] for keys in k0]
            k2 = [np.roll(a,rotation).tolist() for a in k0]
            keys = [np.ravel(k).tolist() for k in [k0,k1,k2]]

            for olds,news in [[keys[0],keys[1]],[keys[1],keys[2]]]:
                for old,new in zip(olds,news):
                    gals.rename_column(old,new)
    
    def populate_mock(self, seed=None, masking_function=None, rotation=None, Nbox=None):
        """Implement HOD model over halos to populate mock galaxy sample"""
        gc.collect()
        if Nbox is None:
            Nbox = self.Nbox
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not Nbox is None:
                self._populate_periodically(Nbox, seed, masking_function)
            else:
                self._populate(seed, masking_function)
        
        if rotation is None:
            rotation = self.rotation
        self.rotate(rotation, self.gals)

        # TODO: make this assertion work. And then set gals["mgid"] here instead of asserting it
        # assert(np.all(self.gals["mgid"] == np.arange(len(self.gals))))

    def get_halos(self):
        """Get halo catalog from <simname> dark matter simulation"""
        self._set_version_name()
        def get_halos():
            self.halocat = sim_manager.CachedHaloCatalog(
                simname=self.simname, redshift=self.redshift,
                halo_finder='rockstar', dz_tol=self.dz_tol,
                version_name=self.version_name)

        # if at first you don't succeed . . .
        try:
            try:
                try:
                    get_halos()
                except OSError:
                    get_halos()
            except OSError:
                get_halos()
        except OSError:
            raise

        self.halos = self.halocat.halo_table

        # Set the side lengths of the box
        if self.Nbox is None:
            self.Lbox = self.halocat.Lbox.astype(np.float32)
        else:
            self.Nbox = np.asarray(self.Nbox)
            self.Lbox = (self.halocat.Lbox * self.Nbox).astype(np.float32)

    def update_model_params(self, param_dict, param_names=None):
        if not param_names is None:
            param_dict = dict(zip(param_names, param_dict))
        if not set(param_dict).issubset(self.model.param_dict):
            raise ValueError("`param_dict`=%s has incompatible keys with model: %s"
                             %(param_dict, self.model.param_dict))
        else:
            self.model.param_dict.update(param_dict)

    def construct_model(self):
        """Use HOD Model to construct mock galaxy sample"""
        self.model = empirical_models.PrebuiltHodModelFactory(self.hodname, redshift=self.redshift, cosmo=self.cosmo, threshold=self.threshold)

    def get_halo_mass(self):
        return self.halos["halo_mvir"]

    def get_halo_occupation(self, galaxy_type="all"):
        if not self.populated:
            self.populate_mock()
        halos = self.model.mock.halo_table

        if galaxy_type.lower() == "all":
            n = halos["halo_num_centrals"] + halos["halo_num_satellites"]
        elif galaxy_type.lower().startswith("cen"):
            n = halos["halo_num_centrals"]
        elif galaxy_type.lower().startswith("sat"):
            n = halos["halo_num_satellites"]
        else:
            raise ValueError(f"galaxy_type={galaxy_type} not understood. Must be one of {'both', 'central', 'satellite'}")

        return n

    def get_halo_moments(self, prim_haloprop, galaxy_type="all", **kwargs):
        if galaxy_type.lower() == "all":
            n = lambda **k: (self.model.mean_occupation_centrals(**k)
                          + self.model.mean_occupation_satellites(**k))
        elif galaxy_type.lower().startswith("cen"):
            n = self.model.mean_occupation_centrals
        elif galaxy_type.lower().startswith("sat"):
            n = self.model.mean_occupation_satellites
        else:
            raise ValueError(f"galaxy_type={galaxy_type} not understood. Must be one of {'both', 'cen*', 'sat*'}")

        return n(prim_haloprop=prim_haloprop, **kwargs)


# Generate a function to select halos before population, if performance is crucial
# ================================================================================
    @staticmethod
    def get_halo_selection_function(field):
        selection_function = field.field_selector
        def halo_selection_function(halos):
            xyz = hf.xyz_array(halos, ["halo_x", "halo_y", "halo_z"])
            rdz = hf.ra_dec_z(xyz-field.origin, np.zeros_like(xyz), cosmo=None)
            return selection_function(rdz)
        return halo_selection_function

# Private functions
# =================

    def _populate(self, seed=None, masking_function=None):
        # self.model.param_dict.update(self.model_params)
        if self.populated:
            self.model.mock.populate(seed=seed, masking_function=masking_function)
        else:
            self.model.populate_mock(self.halocat, seed=seed, masking_function=masking_function)
            self.populated = True

        # Assign variables
        self.gals = self.model.mock.galaxy_table  # galaxy table
        stop = self.mgid_counter + len(self.gals)
        mgid = np.arange(self.mgid_counter, stop)
        self.gals["mgid"] = mgid
        self.mgid_counter = stop

    def _populate_periodically(self, Nbox, seed=None, masking_function=None):
        """Effectively duplicate the halos (nx,ny,nz) times before populating"""
        nx,ny,nz = Nbox
        Nbox = np.asarray(Nbox); assert(Nbox.shape==(3,))
        sim_x, sim_y, sim_z = self.halocat.Lbox
        duplications = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    self._populate(seed, masking_function=masking_function)
                    xadd, yadd, zadd = ix*sim_x, iy*sim_y, iz*sim_z

                    dup = self.gals
                    dup["x"] += xadd; dup["y"] += yadd; dup["z"] += zadd
                    duplications.append(dup)

        self.gals = astropy_table.vstack(duplications)
        self.Lbox = (self.halocat.Lbox * Nbox).astype(np.float32)



    def _set_version_name(self):
        if self.version_name is None:
            if self.simname in {"multidark", "bolshoi", "bolplanck", "consuelo"}:
                self.version_name = "halotools_v0p4"
            else:
                self.version_name = "my_cosmosim_halos"


class HaloBox(SimBox):
    """
    HaloBox(**kwargs)
    
    Subclass of SimBox, but this class only contains halo data, and no galaxy data. Using several GalBoxes and only one HaloBox is recommended for multiprocessing.
    
    Acceptable kwargs
    -----------------
        - **simname**
        - **version_name**
        - **cosmo**
        - **redshift**
        - **dz_tol**
        - **Nbox**
        - **Lbox**
        - **empty**
        
    See SimBox documentation below
    """
    accepted_kwargs = ["simname", "version_name", "cosmo", "redshift", "dz_tol", "Nbox", "Lbox", "empty"]

    # noinspection PyMissingConstructor
    def __init__(self, **kwargs):
        self._kwargs_ = kwargs.copy()
        self.__dict__.update({s:self.defaults[s] for s in self.accepted_kwargs})


        # Update default parameters with any keyword arguments
        hf.kwargs2attributes(self, kwargs)
        # Initialize model and get halos
        self.Lbox = np.asarray(self.Lbox, dtype=np.float32)
        if not self.empty:
            self.get_halos()

class GalBox(SimBox):
    """
    HaloBox(**kwargs)
    
    Subclass of SimBox, but this class only contains HOD model/galaxy data, and no halo data. Using several GalBoxes and only one HaloBox is recommended for multiprocessing.
    
    Positional Arguments
    ---------------------
    halobox : HaloBox/SimBox object
        Object which stores the Cached Halo Table, so that the GalBox doesn't have to.
    
    Acceptable kwargs
    -----------------
        - **populate_on_instantiation**
        - **hodname**
        - **threshold**
        - **empty**
        - **volume**
        - **Nbox**
        - **rotation**
        
    See SimBox documentation below
    """
    accepted_kwargs = ["populate_on_instantiation", "hodname", "threshold", "empty", "volume", "Nbox", "rotation"]

    # noinspection PyMissingConstructor
    def __init__(self, halobox=None, **kwargs):
        self._kwargs_ = kwargs.copy()
        self.halobox = HaloBox(empty=True) if halobox is None else halobox
        self.__dict__.update({s:self.defaults[s] for s in self.accepted_kwargs})

        self.empty = self.halobox.empty
        self.Nbox = self.halobox.Nbox

        hf.kwargs2attributes(self, kwargs)

        self.halos = None if self.empty else self.halobox.halos
        self.halocat = None if self.empty else self.halobox.halocat
        self.redshift = self.halobox.redshift
        self.Lbox = self.halobox.Lbox
        self.cosmo = self.halobox.cosmo

        self.populated = False
        self.construct_model()
        # Populate galaxies inside halos
        self.mgid_counter = 0 # assign mock galaxy id's starting from zero
        if not self.empty and self.populate_on_instantiation:
            if self.populate_on_instantiation:
                self.populate_mock()
HaloBox.__doc__ = GalBox.__doc__ = SimBox.__doc__


class BaseCache:
    """
    Abstract template class. Do not instantiate.
    """
    def __init__(self, config_dir, config_file, cache_dir=None):
        self._read_config(config_dir, config_file)
        if not cache_dir is None:
            self.config["cache_dir"] = cache_dir
        elif not "cache_dir" in self.config:
            raise ValueError("Your first time running this, you "
                             "must provide the path to where you "
                             "will be storing the files.")

        if not "files" in self.config:
            self.config["files"] = []

    def get_filepath(self, filename):
        return os.path.join(self.config["cache_dir"], filename)

    def auto_add(self):
        """
        Automatically try to add all binary files contained in the cache directory.

        Takes no arguments and returns None.
        """
        d = self.config["cache_dir"]
        files = [f for f in os.listdir(d)
                 if os.path.isfile(os.path.join(d,f))]

        n = len(files)
        for f in files:
            try:
                self.add(f)
            except ValueError:
                n -= 1

        print(f"Successfully cached {n} files")

    def update(self):
        """
        Update the config file to account for any changes that have been made to this object. For example:
            -Files could be added via UMCache.add()
            -Files could be removed via UMCache.remove()
            -Cache directory could be changed by instantiating this object via UMCache(cache_dir="path/to/new/dir")

        Takes no arguments and returns None.
        """
        lines = []
        for key in self.config:
            val = self.config[key]
            line = f"{key} = {repr(val)}"
            lines.append(line)

        with open(self.filename, "w") as f:
            f.write("\n".join(lines))

    def add(self, filename):
        """
        Add a new file to cache

        Parameters
        ----------
        filename : str
            The name of the file (do not include path)

        Returns
        -------
        None
        """
        if filename in self.config["files"]:
            raise ValueError("That file is already cached")

        fullpath = os.path.join(self.config["cache_dir"], filename)
        if not os.path.isfile(fullpath):
            raise FileNotFoundError(f"{fullpath} does not exist.")

        self.config["files"].append(filename)

    def remove(self, filename):
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
        try:
            i = self.config["files"].index(filename)
        except ValueError:
            raise ValueError(f"Cannot remove {filename}, as it is not"
                              " currently stored. Currently stored "
                             f"files: {self.config['files']}")

        del self.config["files"][i]
        return i

    def remove_all(self):
        """
        Remove all files from our records. Simply erases the entire config file
        """
        self.config = {}
        self.update()

    def _read_config(self, dirname, filename):
        dirpath = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), dirname)
        filepath = os.path.join(dirpath, filename)

        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        if not os.path.isfile(filepath):
            open(filepath, "a").close()

        with open(filepath) as f:
            code = f.read()

        config, empty = {}, {}
        exec(code, config)
        exec("", empty); [config.pop(i) for i in empty]

        self.config = config.copy()
        self.filename = filepath

class UMCache(BaseCache):
    """
    Keeps track of the locations of locally saved binary files that come from the UniverseMachine data release.

    Parameters
    ----------
    cache_dir : str (required on first run)
        The path to the directory where you plan on saving all of the binary files. If the directory is moved, then you must provide this argument again.
    """
    def __init__(self, cache_dir=None):
        config_dir, config_file = ".um-cache", "um-config.py"
        BaseCache.__init__(self, config_dir, config_file, cache_dir)

        if not "z" in self.config:
            self.config["z"] = []

    def load(self, redshift=0, thresh=None, ztol=0.05):
        """
        Load a halo table into memory, at a given snapshot in redshift.

        Parameters
        ----------
        redshift : float (default = 0)
            Desired redshift of the snapshot

        thresh : callable, None, or "none" (optional)
            Callable which takes a halo catalog as input and returns a boolean array to select the halos on before loading them into memory. By default, ``thresh = lambda cat: cat["obs_sm"] > 3e9``. "none" loads the entire table and is equivalent to ``thresh = lambda cat: slice(None)``

        ztol : float (default = 0.05)
            A match must be within redshift +/- ztol

        Returns
        -------
        halos : np.ndarray
            Structured array of the requested halo catalog
        """
        if thresh is None:
            thresh = lambda cat: cat["obs_sm"] > 3e9
        dtype = np.dtype([('id','i8'),('descid','i8'),('upid','i8'),
                          ('flags','i4'),('uparent_dist','f4'),
                          ('pos','f4',(6)),('vmp','f4'),('lvmp','f4'),
                          ('mp','f4'),('m','f4'),('v','f4'),('r','f4'),
                          ('rank1','f4'),('rank2','f4'),('ra','f4'),
                          ('rarank','f4'),('A_UV','f4'),('sm','f4'),
                          ('icl','f4'),('sfr','f4'),('obs_sm','f4'),
                          ('obs_sfr','f4'),('obs_uv','f4'),
                          ('empty','f4')], align=True)

        filename, true_z = self._get_file_at_redshift(redshift, ztol)
        fullpath = self.get_filepath(filename)

        if isinstance(thresh, str) and thresh.lower() == "none":
            # all 12 million halos
            return np.fromfile(fullpath, dtype=dtype), true_z
        else:
            # don't load halos into memory until after the selection
            mm = np.memmap(fullpath, dtype=dtype)
            return np.array(mm[thresh(mm)]), true_z

    def add(self, filename, redshift=None):
        """
        Add a new binary file containing a UniverseMachine snapshot

        Parameters
        ----------
        filename : str
            The name of the file (do not include path)
        redshift : float (optional)
            If given, the redshift of this snapshot. If not given, redshift will be inferred from the filename, by assuming that the filename is of form "*_{scalefactor}.bin"

        Returns
        -------
        None
        """
        if redshift is None:
            redshift = self._infer_redshift(filename)

        BaseCache.add(self, filename)

        self.config["z"].append(redshift)

        self.update()

    def remove(self, filename):
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
        i = BaseCache.remove(self, filename)
        del self.config["z"][i]
        self.update()
        return i

    def _get_file_at_redshift(self, redshift, ztol):
        wh = np.where(np.isclose(self.config["z"], redshift,
                                 rtol=0, atol=ztol))[0]
        if len(wh) < 1:
            raise ValueError(f"No redshifts matching {redshift}. Try "
                             f"increasing ztol from {ztol}. Available "
                             f"redshifts: {self.config['z']}")
        if len(wh) > 1:
            raise ValueError("Multiple matching redshifts:" 
                             f"{self.config['z'][wh]}")
        return self.config["files"][wh[0]], self.config["z"][wh[0]]

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

class UVISTACache(BaseCache):
    """
    Keeps track of the locations of locally saved UVISTA files.

    Parameters
    ----------
    cache_dir : str (required on first run)
        The path to the directory where you plan on saving all of the files. If the directory is moved, then you must provide this argument again.
    """
    UVISTAFILES = {
        "p": "UVISTA_final_v4.1.cat",  # photometric catalog
        "z": "UVISTA_final_v4.1.zout",  # EAZY output
        "f": "UVISTA_final_BC03_v4.1.fout",  # FAST output
        "s": "UVISTA_final_colors_sfrs_v4.1.dat", # IR+UV lum/sfr
        "uv": "UVISTA_final_v4.1.153-155.rf", # rest-frame U,V
        "vj": "UVISTA_final_v4.1.155-161.rf", # rest-frame V,J
    }
    PHOTBANDS = {"k": "Ks", "h": "H", "j": "J", "y": "Y",
                 "z": "zp", "i": "ip", "r": "rp", "v": "V",
                 "g": "gp", "b": "B", "u": "u"}
    def __init__(self, cache_dir=None):
        config_dir, config_file = ".um-cache", "uvista-config.py"
        BaseCache.__init__(self, config_dir, config_file, cache_dir)


    def get_filepath(self, filetype):
        """
        Returns the absolute path to the requested file.
        Filetype options are "p", "z", "f", and "s" where
        each option corresponds to:
        """
        filename = self.UVISTAFILES[filetype]
        return BaseCache.get_filepath(self, filename)
    get_filepath.__doc__ += "\n" + repr(UVISTAFILES)

    def add(self, filename):
        if not filename in self.UVISTAFILES.values():
            raise ValueError("That's not a UVISTA file")

        BaseCache.add(self, filename)
        self.update()

    def remove(self, filename):
        BaseCache.remove(self, filename)
        self.update()

    @staticmethod
    def get_names(filetype):
        if filetype == "p":
         return ['id','ra','dec','xpix','ypix','Ks_tot','eKs_tot','Ks',
         'eKs', 'H', 'eH', 'J', 'eJ', 'Y', 'eY', 'ch4', 'ech4', 'ch3',
         'ech3', 'ch2', 'ech2', 'ch1', 'ech1', 'zp', 'ezp', 'ip', 'eip',
         'rp', 'erp', 'V', 'eV', 'gp', 'egp', 'B', 'eB', 'u', 'eu',
         'IA484', 'eIA484', 'IA527', 'eIA527', 'IA624', 'eIA624',
         'IA679', 'eIA679', 'IA738', 'eIA738', 'IA767', 'eIA767',
         'IB427', 'eIB427', 'IB464', 'eIB464', 'IB505', 'eIB505', 'IB574',
         'eIB574', 'IB709', 'eIB709', 'IB827', 'eIB827', 'fuv', 'efuv',
         'nuv', 'enuv', 'mips24', 'emips24', 'K_flag', 'K_star',
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
            raise ValueError(f"filetype {filetype} not recognized")

    def names_to_keep(self, filetype):
        if filetype == "p":
            return ["id", "ra", "dec", "Ks_tot",
                    *self.PHOTBANDS.values(), "star", "K_flag", "zp",
                    "ip", "contamination", "nan_contam"]
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
            raise ValueError(f"filetype {filetype} not recognized")

    @staticmethod
    def get_skips(filetype):
        if filetype == "p" or filetype == "z":
            return 1
        elif filetype == "f":
            return 17
        elif filetype == "s":
            return 2
        elif filetype == "uv" or filetype == "vj":
            return 11
        else:
            raise ValueError(f"filetype {filetype} not recognized")

    def load(self, include_rel_mags=False):
        if not len(self.config["files"]) == len(self.UVISTAFILES):
            raise ValueError("Can't load until all files are in cache")

        types = ["p", "f", "z", "s", "uv", "vj"]
        dat = [pd.read_csv(self.get_filepath(s), delim_whitespace=True,
                names=self.get_names(s), skiprows=self.get_skips(s),
                usecols=self.names_to_keep(s)) for s in types]

        UVrest = -2.5*np.log10(dat[4]["L153"]/dat[4]["L155"])
        VJrest = -2.5*np.log10(dat[5]["L155"]/dat[5]["L161"])
        cosmo = cosmology.Planck13
        z = dat[2]["z_peak"]
        sfr_tot = dat[3]["SFR_tot"]
        sfr_uv = dat[3]["SFR_UV"]
        sfr_ir = dat[3]["SFR_IR"]
        logm = dat[1]["lmass"]
        logssfr = dat[1]["lssfr"]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            d_com = cosmo.comoving_distance(z).value * cosmo.h
            d_lum = cosmo.luminosity_distance(z).value * cosmo.h

            aperture_factor = dat[0]["Ks_tot"] / dat[0]["Ks"]
            relative_mags = {
                key: -2.5 * np.log10(dat[0][val] * aperture_factor) + 25
                    for (key,val) in self.PHOTBANDS.items()
            }
            absolute_mags = {
                key: val - 5 * np.log10(d_lum * 1e5)
                    for (key,val) in relative_mags.items()
            }

        selection = np.all([
            np.isfinite(list(absolute_mags.values())).all(axis=0),
            np.isfinite(logm), z > 1.5e-2, relative_mags["k"] < 23.4,
            dat[0]["star"] == 0, dat[0]["K_flag"] < 4,
            dat[0]["contamination"] == 0, dat[0]["nan_contam"] < 3],
            axis=0)

        rel_keys, rel_vals = zip(*relative_mags.items())
        rel_keys = [key + "_AB" for key in rel_keys]
        rel_vals = rel_vals if include_rel_mags else []
        rel_keys = rel_keys if include_rel_mags else []

        abs_keys, abs_vals = zip(*absolute_mags.items())
        abs_keys = ["M_" + key.upper() for key in abs_keys]

        names = ["id", "ra", "dec", "z", "logm", "sfr_tot", "logssfr",
                 "d_com", "d_lum", "UVrest", "VJrest", *rel_keys,
                 *abs_keys, "sfr_uv", "sfr_ir"]
        cols = [dat[0]["id"], dat[0]["ra"], dat[0]["dec"], z, logm,
                sfr_tot, logssfr, d_com, d_lum, UVrest, VJrest, *rel_vals,
                *abs_vals, sfr_uv, sfr_ir]

        data = dict(zip(names,cols))
        data = pd.DataFrame(data)
        data = data[selection]
        data.index = np.arange(len(data))

        return data



class UniverseMachine(GalBox):
    def __init__(self, redshift=0, thresh=None, ztol=0.05):
        GalBox.__init__(self)
        posnames = []
        for s in ["", "halo_"]:
            posnames += [f"{s}x", f"{s}y", f"{s}z",
                         f"{s}vx", f"{s}vy", f"{s}vz"]

        h,z = UMCache().load(redshift=redshift, thresh=thresh, ztol=ztol)
        pos = pd.DataFrame(np.concatenate([h["pos"]] * 2, axis=1),
                           columns=posnames)

        othernames = list(h.dtype.names)
        othernames.remove("pos")
        halos = pd.DataFrame(h[othernames])

        halos = pd.concat([pos, halos], axis=1, copy=False)
        self.halos = self.gals = halos

        self.redshift = z
        self.populated = True
        self.simname, self.Lbox, self.empty = ("bolplanck",
                            np.array([250.,250.,250.]), False)

    def field(self, rotation=None, **kwargs):
        return UMMockField(self, rotation=None, **kwargs)

    def boxfield(self, rotation=None, **kwargs):
        return UMBoxField(self, rotation=None, **kwargs)

    def survey(self, *args, **kwargs):
        raise NotImplementedError()

    # Use @functools.cached_property (requires Python 3.8)
    @property
    @functools.lru_cache(maxsize=None)
    def mag_predictor(self, photbands=None):
        return ummags.make_predictor_UMmags(self.halos, self.redshift, photbands)

class UMField:
    """
    Abstract template class. Do not instantiate
    """
    def get_abs_mag(self, band):
        band = band.lower()
        data = self._get_abs_mags()

        try:
            return data[band]
        except KeyError:
            raise KeyError(f"{band} is not an allowed band. "
                             f"Use one of {data.columns.tolist()}.")

    def get_rel_mag(self, band):
        d_on_10pc = self._get_lum_dist()
        return self.get_abs_mag(band) + 5. * np.log10(d_on_10pc)

    @functools.lru_cache(maxsize=None)
    def _get_lum_dist(self):
        return 1e5 * self.simbox.cosmo.luminosity_distance(
            self.get_redshift(realspace=True)).value

    @functools.lru_cache(maxsize=None)
    def _get_abs_mags(self):
        return self.simbox.mag_predictor(self.selection,
                        self.get_redshift(realspace=False))

class UMMockField(MockField, UMField):
    def __init__(self, um, rotation=None, **kwargs):
        um.rotate(rotation)
        MockField.__init__(self, um, **kwargs)
UMMockField.__doc__ = MockField.__doc__

class UMBoxField(BoxField, UMField):
    def __init__(self, um, rotation=None, **kwargs):
        um.rotate(rotation)
        BoxField.__init__(self, um, **kwargs)

    def mag_predictor(self, z):
        um.mag_predictor(self.selection, z)
UMBoxField.__doc__ = BoxField.__doc__