"""
mocksurvey
Author: Alan Pearl

Some useful classes for conducting mock surveys of galaxies populated by `halotools` and `UniverseMachine`.

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

from .main import *
