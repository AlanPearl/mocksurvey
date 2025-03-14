"""
Excerpt from kcorrect code by Sean Johnson.
Modified by Alan Pearl to work the code into my package.
"""

import os

import numpy as np
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
# from astropy.table import Table

# from .cosmography import distancemodulus
from ... import mocksurvey as ms

c_A = 2.9979e18
c_nm = 2.9979e17


class OptimizedFilterIntegrator:
    """
    Provides the functionality of appmag_nm_njy more efficiently
    by saving the rebinned filter transmission curves instead of
    loading them and rebinning for every single spectrum
    """

    def __init__(self, wave_o_nm, filter_strings):
        wave_o = wave_o_nm * 10.0  # nm to Ang.
        self.nu = c_A / wave_o
        self.filter_strings = filter_strings

        self.saved_filters = {
            s: rebinfilter(readfilter(s), self.nu)
            for s in self.filter_strings
        }
        self.selections = {
            s: self.saved_filters[s]["transmission"] != 0
            for s in self.filter_strings
        }

        # Apply selection on filter transmission curve.
        # This still needs to be applied on the spectrum itself
        self.saved_filters = {
            s: self.saved_filters[s][self.selections[s]]
            for s in self.filter_strings
        }

    def appmag_nm_njy(self, fnu_njy, filter_string):
        rebinned_filter = self.saved_filters[filter_string]
        selection = self.selections[filter_string]

        fnu = fnu_njy[..., selection] / 1e9  # nJy to Jy
        nu = self.nu[selection]

        numerator = simps(fnu * rebinned_filter['transmission'] / nu, nu)
        denominator = simps(3631.0 * rebinned_filter['transmission'] / nu, nu)

        m = -2.5 * np.log10(numerator / denominator)
        return m


# Helper function that returns filter transmission curve
def readfilter(filtername):
    pyobs = ms.SeanSpectraConfig().get_path("PYOBS")
    filename = os.path.join(pyobs, "kcorrect", "filters",
                            filtername + ".npy")
    return np.load(filename)


# Rebin a filter response function to a new frequency grid
def rebinfilter(filtername, nu_new):
    filter_interp = interp1d(filtername['nu'], filtername['transmission'],
                             bounds_error=False, fill_value=0.0)

    filter_rebinned = np.zeros(len(nu_new),
                               dtype={'names': ('wave', 'nu', 'transmission'),
                                      'formats': (float, float, float)})
    filter_rebinned['nu'] = nu_new
    filter_rebinned['wave'] = 3e18 / filter_rebinned['nu']
    filter_rebinned['transmission'] = filter_interp(nu_new)

    return filter_rebinned
