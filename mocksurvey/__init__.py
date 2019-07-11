"""
mocksurvey.py
Author: Alan Pearl

Some useful classes for coducting mock surveys of galaxies populated by `halotools`.

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

import gc
import warnings
import math
import scipy
import numpy as np
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
    
    def obs_func(self, data, rands=None, store=True, param_dict={}):
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
        
    def make_selection(self, redshift, convert_distance_to_redshift=False):
        upper, lower = self.mean_redshift + self.delta_z/2., self.mean_redshift - self.delta_z/2.
        if convert_distance_to_redshift:
            redshift = hf.distance2redshift(redshift, 0, self.cosmo)
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
    
    def circle_selector(self, xyz):
        """Select galaxies in a circle centered at ra,dec = (0,0) Mpc"""
        field_radius = self.circle_sqdeg2radius()
        xy = xyz[:,:2] - self.center[np.newaxis,:2]
        rad2 = np.sum(xy**2, axis=1)
        return rad2 < field_radius**2
    
    def square_selector(self, xyz):
        """Select galaxies in a square centered at ra,dec = (0,0) Mpc"""
        field_apothem = self.square_sqdeg2apothem()
        xy = xyz[:,:2] - self.center[np.newaxis,:2]
        b1 = xy[:,0] < field_apothem
        b2 = xy[:,0] > -field_apothem
        b3 = xy[:,1] < field_apothem
        b4 = xy[:,1] > -field_apothem
        return b1 & b2 & b3 & b4
    
    def hexagon_selector(self, xyz):
        """Select galaxies in a hexagon centered at ra,dec = (0,0) Mpc"""
        field_apothem = self.hexagon_sqdeg2apothem()
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
        
    def get_data(self, rdz=False, realspace=False):
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
        
        lower, upper = self.center - self.get_shape()/2., self.center + self.get_shape()/2.
        selection = np.all((lower[None,:] <= xyz) & (xyz <= upper[None,:]), axis=1)
        
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
            xyz = hf.ra_dec_z(xyz-self.origin, np.zeros_like(xyz), self.simbox.cosmo, self.zprec)
        else:
            xyz = hf.xyz_array(self.simbox.gals)[self.selection]
            if not realspace:
                if (self.halo_vel_factor is None) and (self.gal_vel_factor is None):    
                    vz = self.simbox.gals["vz"][self.selection]
                else:
                    vz = hf.factor_velocity(self.simbox.gals["vz"]
                                                [self.selection],
                                        self.simbox.gals["halo_vz"]
                                                [self.selection],
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
            rands = self.rdz2xyz(rands, input_distance=True)
            
            
        self._rands = {
            "x_real": rands[:,0],
            "y_real": rands[:,1],
            "z_real": rands[:,2]
        }
        
        selection = self._make_selection(dataset='rands') & self._select_within_simbox(rands)
        for key in self._rands:
            self._rands[key] = self._rands[key][selection]
    
    
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
    
    def rdz2xyz(self, rdz, input_distance=False):
        cosmo = None if input_distance else self.simbox.cosmo
        return hf.rdz2xyz(rdz, cosmo=cosmo) + self.origin
    
    def measure_volume(self, precision=1e-3):
        rdz_lims = self.get_lims(rdz=True, overestimation_factor=1.02)
        rdz_lims[2] = self.simbox.redshift2distance(rdz_lims[2])
        N0 = 10000
        N = 0
        n = 0
        r = 0
        sigma = lambda: np.sqrt( (1-r)/(r*(N-1)) )
        get_Nmore = lambda: int( N0 - N + 1 + (1-r)/(r*precision**2) )
        
        Nmore = N0
        i = 0
        while (not r) or (sigma() > precision):
            i += 1
            if i > 10:
                raise RecursionError("sigma=%f > required=%f after 10 iterations" %(sigma, precision))
            rand_rdz = hf.rand_rdz(Nmore, *rdz_lims)
            
            n += self._rdz_selection(rand_rdz, input_distance=True).sum()
            N += Nmore
            
            r = n/N
            
            Nmore = get_Nmore() if r else N0
        
        return r * hf.volume_rdz(*rdz_lims)
    
# Private member functions
# ========================
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
    
    def _get_vel(self, realspace=False, dataset=None, halo_vel_factor=None, gal_vel_factor=None):
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
            return hf.factor_velocity(
                hf.xyz_array(self.simbox.gals, 
                             keys=['vx', 'vy', 'vz'])[selection],
                hf.xyz_array(self.simbox.gals, 
                             keys=['halo_vx', 'halo_vy', 'halo_vz'])[selection],
                halo_vel_factor=halo_vel_factor,
                gal_vel_factor=gal_vel_factor,
                inplace=True)
    
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
        redshift = self._get_redshift(realspace=self.realspace_selection, dataset=dataset)
        
        field_selector = self.field_selector
        redshift_selector = self.redshift_selector
        
        selection = field_selector(data) & redshift_selector(redshift)
        collisions = hf.sample_fraction(len(selection), self.collision_fraction)
        selection[collisions] = False
        return selection
    
    def _rdz_selection(self, data, input_distance=False):
        zlim = self.get_lims(rdz=not input_distance)[2]
        if input_distance: zlim -= self.origin[2]
        
        red_select = (zlim[0] < data[:,2]) & (data[:,2] < zlim[1])
        field_select = self.field_selector(data[:,:2])
        
        return field_select & red_select

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
        accessors = [x for x in dir(MockField) if ((not x.startswith("_")) and (not x in ["make_rands","get_shape","get_rands"]))]
        for accessor in accessors:
            unbound = (lambda accessor: lambda self, rdz=False, realspace=False: self._field2survey(accessor, rdz, realspace))(accessor)
            unbound.__doc__ = self.fields[0].__getattribute__(accessor).__doc__
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
        
        selections = [field._rdz_selection(self.rand_rdz, input_distance=True) for field in self.fields]
        selection = np.any(selections, axis=0)
        
        self.rand_rdz = self.rand_rdz[selection]
        self.rand_xyz = self.rdz2xyz(self.rand_rdz, input_distance=True)
        
    
    def _field2survey(self, funcstring, rdz, realspace):
        ans = []
        kwargs = [("rdz", rdz), ("realspace", realspace)]
        kwargs = {kwkey:kwval for kwkey,kwval in kwargs if kwkey in getfullargspec(self.fields[0].__getattribute__(funcstring))[0]}
        for field in self.fields: ans += field.__getattribute__(funcstring)(**kwargs).tolist()
        ans = np.asarray(ans, dtype=np.float32)[self.selection]
        return ans

class PFSSurvey(MockSurvey):
    def __init__(self, simbox=None, center=None, threshold=10.5):
        
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
        args = rpbins, pimax = np.logspace(-0.87, 1.73, 14), 50.
        
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
        dist = self.cosmo.comoving_distance(redshift).value * self.cosmo.h
        return dist.astype(redshift.dtype)
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
        if rotation is None:
            rotation = np.random.randint(3)
        if rotation:
            k0 = [["x","y","z"], ["vx","vy","vz"],
            ["halo_x","halo_y","halo_z"], ["halo_vx","halo_vy","halo_vz"]]
            
            k1 = [[s+"_tmp" for s in keys] for keys in k0]
            k2 = [np.roll(a,rotation).tolist() for a in k0]
            keys = [np.ravel(k).tolist() for k in [k0,k1,k2]]

            for olds,news in [[keys[0],keys[1]],[keys[1],keys[2]]]:
                for old,new in zip(olds,news):
                    self.gals.rename_column(old,new)

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
    def get_halo_selection_function(self, field):
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