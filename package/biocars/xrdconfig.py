from utils import utils
from mu import mu
import numpy as np

def make_attenuation_function(element, scale):
    """
    element: an element identifier (either atomic number or 1 or 2 letter abbreviation)
    scale: a fit parameter that corresponds (but is not equal) to the thickness
    """
    return lambda energies: np.exp(-(mu.ElementData(element).sigma)(energies)/scale)

@utils.eager_persist_to_file("cache/make_attenuation_function_from_transmission/")
def make_attenuation_function_from_transmission(element, transmission):
    energies, intensities = pinkbeam_spectrum() # unattenuated beam profile
    myfunc = lambda scale: np.exp(-(mu.ElementData(element).sigma)(energies)/scale) 
    deviation = lambda scale: abs(transmission - np.sum(myfunc(scale) * intensities)/np.sum(intensities))
    res = minimize_scalar(deviation)
    scale = res.x
    print 'scale factor for ', element, ':', scale
    return make_attenuation_function(element, scale)

def make_attenuation_function_from_thickness(element, thickness):
    """
    Return an attenuation function given an element and a thickness (in cm).
    """
    eltdat = mu.ElementData(element)
    return lambda energies: np.exp(-(eltdat.sigma)(energies) * eltdat.density * thickness)

# map nominal attenuation values to actual
actual_attenuations =\
    {1: 1.,
    1: 1.,
    1.5: 1.36,# revise
    2.5: 2.29,# revise
    4.5: 4.8,# revise
    # 6 layers of UHV Al. This value is based on attenuation at 12 
    #keV (not a proper average over the incident spectrum
    2: 1.957,
    15: 18.88,
    20: 24.5,
    300: 563.,# Ag
    10: 12.50005461,
    74: 103.5,
    32: 42.3,
    3: 3.66758697,
    7: 8.0360173}

# TODO: check this
Alfoil_thickness = 25e-4
# TODO: make a factory function for these
ATTENUATION_FUNCTIONS = {}
# Deprecated (from Nov 2015 run)
ATTENUATION_FUNCTIONS[2] = make_attenuation_function_from_thickness('Al', 6 * Alfoil_thickness)
ATTENUATION_FUNCTIONS[1.5] = make_attenuation_function_from_thickness('Al', 3 * Alfoil_thickness)
ATTENUATION_FUNCTIONS[2.5] = make_attenuation_function_from_thickness('Al', 8 * Alfoil_thickness)
ATTENUATION_FUNCTIONS[300] = make_attenuation_function_from_thickness('Ag', 75e-4)
ATTENUATION_FUNCTIONS[4.5] = lambda energies: make_attenuation_function_from_thickness('Al', 4 * Alfoil_thickness)(energies) *\
    ATTENUATION_FUNCTIONS[3](energies)
ATTENUATION_FUNCTIONS[32] = lambda energies: ATTENUATION_FUNCTIONS[3](energies) * ATTENUATION_FUNCTIONS[10](energies)
ATTENUATION_FUNCTIONS[74] = lambda energies: ATTENUATION_FUNCTIONS[7](energies) * ATTENUATION_FUNCTIONS[10](energies)

# Revised Jul 2016
ATTENUATION_FUNCTIONS[10] = lambda energies: make_attenuation_function_from_thickness('Al', 15 * 25e-4)(energies) *\
    make_attenuation_function_from_thickness('Ti', 25e-4)(energies)
ATTENUATION_FUNCTIONS[7] = lambda energies: make_attenuation_function_from_thickness('Al', 11 * 25e-4)(energies) *\
    make_attenuation_function_from_thickness('Ti', 25e-4)(energies)
ATTENUATION_FUNCTIONS[3] = lambda energies: make_attenuation_function_from_thickness('Al', 11 * 25-4)(energies) 

ATTENUATION_FUNCTIONS[20] = lambda energies: ATTENUATION_FUNCTIONS[2](energies) * ATTENUATION_FUNCTIONS[10](energies)
ATTENUATION_FUNCTIONS[15] = lambda energies: ATTENUATION_FUNCTIONS[1.5](energies) * ATTENUATION_FUNCTIONS[10](energies)

from utils.mpl_plotly import plt
