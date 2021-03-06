import sys
import numpy as np
import numpy.ma as ma
import scipy
import os
import pdb
import glob
from PIL import Image
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from scipy import arange, array, exp
from scipy.optimize import minimize_scalar
import operator
import fnmatch
#from  libtiff import TIFF
import random
import string
import re
#from matplotlib import cm

import rldeconvolution

from mu import mu
from utils import utils
from xrdconfig import actual_attenuations, ATTENUATION_FUNCTIONS, plt

import pickle
import atexit, dill

# TODO: move these data to a different module
# TODO: attenuation factors for the Al filters
# TODO: get rid of explicit caching (using the radial_integrations directories)

# From Nov 2015 beam run
## map nominal attenuation values to actual
#actual_attenuations =\
#    {1: 1.,
#    1: 1.,
#    1.5: 1.51,# revise
#    2.5: 2.29,# revise
#    4.5: 4.8,# revise
#    # 6 layers of UHV Al. This value is based on attenuation at 12 
#    #keV (not a proper average over the incident spectrum
#    2: 1.957,
#    300: 563.,# Ag
#    10: 12.,# Ti
#    74: 103.5,
#    32: 42.3,
#    3: 3.2,
#    7: 7.4}

nominal_attenuations = {v: k for k, v in actual_attenuations.items()}

# combinations of number of pulses and nominal attenuation for which we've
# measured air scatter
#measured_background_params = [(100, 300), (250, 300), (400, 300), (500, 300),
#    (100, 74), (50, 74), (50, 32), (10, 10), (1, 3), (3, 3), (2, 3), (4, 3),
#    (8, 3), (16, 3), (32, 3), (64, 3), (128, 3), (1, 1), (5, 1), (10, 1),
#    (2, 1), (4, 1), (8, 1), (1, 1.5), (1, 4.5), (1, 7), (1, 2.5), (2, 2.5), (4, 2.5), (8, 2.5),
#    (2, 4.5), (4, 4.5), (8, 4.5)]
measured_background_params = [
     (10, 10), (1, 3), 
     (1, 1),  (10, 1),
     (1, 7)]
AIR_SCATTERN_PATTERN = 'air_scatter_copied/%dp_%sx_as*' 

def open_mccd(filepath):
    """
    Open an mccd file, perform basic conditioning, and return it as a numpy
    array.
    """
    #img = TIFF.open(filepath, 'r').read_image()
    img = np.array(Image.open(filepath))
    # get rid of outying pixel values
    # TODO: make the threshold paramater adjustable?
    threshold = 20. * np.std(img)
    mean = np.mean(img)
    img[img > threshold] = mean
    return img

def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            return self[key]
        def __missing__(self, key):
            ret = self[key] = self.f(*key[0], **{k: v for k, v in key[1]})
            return ret
    return memodict(f)

def default_bgsubtraction(x, y, endpoint_size = 10, endpoint_right = 10):
    """
    Return a function that linearly interpolates between the lower and upper
    x values of the dataset specified by x and y
    """
    bgx = np.concatenate((x[:endpoint_size], x[-endpoint_right:]))
    bgy = np.concatenate((y[:endpoint_size], y[-endpoint_right:]))
    interp_func = rldeconvolution.extrap1d(interp1d(bgx, bgy))
    return interp_func

def pinkbeam_spectrum():
    """
    Spectrum of the un-attenuated pink beam
    """
    dat = np.genfromtxt("mda/14IDB_15067.txt")
    beam = [dat.T[1], dat.T[21]]
    bgsub = default_bgsubtraction(*beam)(beam[0])
    beam[1] = beam[1] - bgsub
    beam[0] = 1000 * beam[0]
    return beam


#center coords of the beam on the ccd
CENTER = [1984, 1967]

PHOTON_ENERGY = 12000. # NOMINAL incident photon energy
HBARC = 1973. #eV * Angstrom

# From Nov 2015 run. Filters were changed thereafter
## Define filter transmission functions
#Alfoil_thickness = 17e-4 # speculative
#ATTENUATION_FUNCTIONS = {}
#ATTENUATION_FUNCTIONS[2] = make_attenuation_function_from_thickness('Al', 6 * Alfoil_thickness)
#ATTENUATION_FUNCTIONS[1.5] = make_attenuation_function_from_thickness('Al', 4 * Alfoil_thickness)
#ATTENUATION_FUNCTIONS[2.5] = make_attenuation_function_from_thickness('Al', 8 * Alfoil_thickness)
#ATTENUATION_FUNCTIONS[300] = make_attenuation_function_from_thickness('Ag', 75e-4)
#ATTENUATION_FUNCTIONS[10] = make_attenuation_function_from_thickness('Ti', 75e-4)
#ATTENUATION_FUNCTIONS[7] = lambda energies: make_attenuation_function_from_thickness('Al', 25e-4)(energies) *\
#    make_attenuation_function_from_thickness('Ag', 25e-4)(energies)
#ATTENUATION_FUNCTIONS[3] = lambda energies: make_attenuation_function_from_thickness('Al', 100e-4)(energies) *\
#    make_attenuation_function_from_thickness('Ti', 25e-4)(energies)
#ATTENUATION_FUNCTIONS[4.5] = lambda energies: make_attenuation_function_from_thickness('Al', 4 * Alfoil_thickness)(energies) *\
#    ATTENUATION_FUNCTIONS[3](energies)
#ATTENUATION_FUNCTIONS[32] = lambda energies: ATTENUATION_FUNCTIONS[3](energies) * ATTENUATION_FUNCTIONS[10](energies)
#ATTENUATION_FUNCTIONS[74] = lambda energies: ATTENUATION_FUNCTIONS[7](energies) * ATTENUATION_FUNCTIONS[10](energies)


def mask_peaks_and_iterpolate(x, y, peak_ranges):
    for peakmin, peakmax in peak_ranges:
        good_indices = np.where(np.logical_or(x < peakmin, x > peakmax))[0]
        y = y[good_indices]
        x = x[good_indices]
    return interp1d(x, y)

def peak_sizes(x, y, peak_ranges, bg_subtract = True):
    backgnd = mask_peaks_and_iterpolate(x, y, peak_ranges)
    if bg_subtract is True:
        subtracted = y - backgnd(x)
    else:
        subtracted = y
    sizeList = []
    for peakmin, peakmax in peak_ranges:
        peakIndices = np.where(np.logical_and(x >= peakmin, x <= peakmax))[0]
        sizeList += [np.sum(subtracted[peakIndices])]
    return sizeList

# TODO: complete this
#def find_center(arr2d, start_coords):
#    """
#    In a 2d powder pattern, find a local optimal candidate for the center
#    coordinates of the radial distribution, i.e., the center coordinate for
#    which the distribution's l2 norm is maximized
#    """
#    x0, y0 = CENTER
#    def l2norm(arr):
#        theta, intensity = radialSum(arr)
#        return np.sqrt(np.sum(np.square(intensity - np.mean(intensity))))
#    to_visit = [start_coords]
#    norm_start = l2norm(radialSum(arr2d, center = CENTER)
#    while len(to_visit) > 0:
#        x, y = start_coords
#        left, right, top, down = (x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)
#        if




# TODO: write test case for sample_Au10x*
def deep_glob(pattern):
    """
    TODO: this will break if there's a glob outside of the 
    filename part of the pattern

    Pattern may be either a glob pattern or a list of file paths.
    """
    if type(pattern) == str:
        return [os.path.join(dirpath, f) \
            for dirpath, dirnames, files in os.walk(os.path.dirname(pattern)) \
                for f in fnmatch.filter(files, os.path.basename(pattern))]
    else: # Assume a list of paths
        return pattern

def radial_density_paths(directory_glob_pattern):
    """
    given a glob pattern, return full paths describing intended locations of 
    all radial density files
    """
    def radial_name(prefix):
        dirname = os.path.dirname(prefix)
        basename = os.path.basename(prefix)
        radial_file_path = dirname + "/radial_integrations/" + basename + "processed.dat"
        return radial_file_path

    if type(directory_glob_pattern) == str and directory_glob_pattern[-5:] != ".mccd":
        directory_glob_pattern = directory_glob_pattern + "*mccd"
    all_mccds = deep_glob(directory_glob_pattern)
    radial_paths = map(radial_name, all_mccds)
    return radial_paths

#@utils.eager_persist_to_file("cache/sum_radial_densities.p")
@utils.memoize(timeout = None)
def sum_radial_densities(directory_glob_pattern, average = False, give_tuple = False, ncores = None):
    """
    Sum or average radial density profiles

    Inputs:
        directory_glob_pattern, the filepath glob pattern for which to retrieve data
        average: if true, average; else, sum.
        give_tuple: if true, return a tuple (theta, intensities); else, only return the
            array of intensities
    Returns the summed radial profile
    """
    paths = radial_density_paths(directory_glob_pattern)
    generate_radial_all(directory_glob_pattern, ncores = ncores)
    extractIntensity = lambda name: np.genfromtxt(name)
    if not average:
        r, intensity = sum_many(map(extractIntensity, paths))
        if give_tuple:
            return r/len(paths), intensity
        else:
            return intensity
    else:
        return sum_many(map(extractIntensity, paths))[1]/len(paths)



@utils.eager_persist_to_file("cache/generate_radial_all/")
def generate_radial_all(directory_glob_pattern, recompute = False, center = CENTER, ncores = None):
    """
    Given a glob pattern, generate radial distributions for all the matching .mccd files.

    No value is returned. Run sum_radial_densities after this function in order to
    retrieve the data
    """
    def radial_name(prefix):
        dirname = os.path.dirname(prefix)
        basename = os.path.basename(prefix)
        radial_file_path = dirname + "/radial_integrations/" + basename + "processed.dat"
        return radial_file_path

    if type(directory_glob_pattern) == str and directory_glob_pattern[-5:] != ".mccd":
        directory_glob_pattern = directory_glob_pattern + "*mccd"
    all_mccds = deep_glob(directory_glob_pattern)
    
    @utils.eager_persist_to_file("cache/process_one_frame/")
    # TODO: why does caching fail here?
    def process_one_frame(mccd):
        """
        Radially integrate a frame and save the result to file
        """
        radial_path = radial_name(mccd)
        radial_directory = os.path.dirname(radial_path)
        if not os.path.exists(radial_directory):
            os.mkdir(radial_directory)
        if (not os.path.exists(radial_path)) or recompute:
            #radial = radialSum(np.array(Image.open(mccd)),  center = CENTER)
            img =  open_mccd(mccd)
            radial = radialSum(img,  center = center)
            np.savetxt(radial_path, radial)
    map(process_one_frame, all_mccds)


#normalization modes: peak, if a numerical value is provided it is assumed to be the angle at which
#normalization is desired.
def process_radial_distribution(radial_distribution, normalization = None,
    bgsub = None, label = None, plot = False,
    smooth = 1, scale = 'angle'):
    """
    Normalize, scale, and background-subtract a powder pattern.
    bgsub is a function that takes x and y coordinates and returns an interpolation function
    for background subtraction as a function of x.
    """
    if bgsub is None:
        #bgsub = default_bgsubtraction
        bgsub = lambda x, y: lambda z: 0

    def show():
        plt.xlabel("angle (degrees)")
        plt.ylabel("inensity (arb)")
        plt.legend()
        plt.show()

    # TODO: refactor
    def rescale_data(label):
        r, intensity = radial_distribution
        #intensity -= bgsub
        orig_intensity = intensity.copy()
        intensity = intensity - bgsub(r, intensity)(r)
        if normalization == "peak":
            intensity /= np.max(intensity)
        elif isinstance(normalization, (int, long, float)):
            interpolated = interp1d(r, intensity)
            orig_interpolated = interp1d(r, orig_intensity)
            intensity *= normalization
        if scale == 'q':
            qq = 2 * PHOTON_ENERGY * np.sin(np.deg2rad(r/2))/HBARC
            if plot:
                plt.plot(qq, gaussian_filter(intensity, smooth), label = label)
            return qq, intensity
        if scale == 'angle':
            if plot:
                plt.plot(r, gaussian_filter(intensity, smooth), label = label)
            return r, intensity
        else:
            raise ValueError("invalid key: " +  str(scale))
    return rescale_data(label)




#to sum numpy arrays without altering shape
def sum_many(arr1d):
    """
    Sum all arrays in arr1d.
    """
    return reduce(operator.add, arr1d)


def radialSum(data, distance = 140000., pixSize = 88.6, center = CENTER):
    """
    Perform an azimuthal integration of a 2d array.

    Inputs:
        data, the 2d array
        pixSize, the physical pixel size (in microns)
        center, the polar origin
    Returns: 1d numpy array, 1d numpy array-->theta, radial profile,
        where theta is in degrees
    """
    if type(data) == str:
        data = open_mccd(data)
    if center is None:
        l, h = np.shape(data)
        center = [l/2, h/2]
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    rsorted = np.sort(np.unique(r.ravel()))
    theta = np.arctan(rsorted * pixSize / distance)
    return np.rad2deg(theta), radialprofile


def process_all_globs(glob_patterns, show = False, individual = False, subArray = None,
normalization = "peak", normalizationArray = None, labelList = None, filterList
= None, recompute = False, scale = 'angle', ncores = None):
    """
    Return scaled, background-subtracted, and normalized powder patterns for data
    corresponding to each pattern in glob_patterns
    """
    if subArray is None:
        subArray = [None] * len(glob_patterns)
    if labelList is None:
        labelList = [None] * len(glob_patterns)
    if filterList is None:
        filterList = [None] * len(glob_patterns)
    if normalizationArray is None:
        normalizationArray = [None] * len(glob_patterns)
    def run_and_plot(glob, bgsub = None, label = None, data_filter = None, normalization = None):
        return process_radial_distribution(sum_radial_densities(glob, False, True, ncores = ncores), normalization =
normalization, bgsub = bgsub, label = glob, scale = scale, plot = show)
    outputs = []
    for patt, subtraction, label, one_filter, norm in zip(glob_patterns, subArray, labelList, filterList, normalizationArray):
       outputs = outputs + [run_and_plot(patt, bgsub = subtraction, label = label, data_filter = one_filter, normalization = norm)]
    if show:
        plt.legend()
        plt.show(block = False)
    #output format: angles, intensities, angles, intensities, etc.
    return np.vstack((_ for _ in outputs))

def radial_mean(filenames, sigma = 1, ncores = None):
    return gaussian_filter(sum_radial_densities(filenames, average = True), sigma = sigma)

@utils.eager_persist_to_file("cache/dark_subtraction/")
def dark_subtraction(npulses, nframes = 1):
    """
    Return the dark frame for an exposure of duration npulses.

    If npulses is a key in lookup_dict, the corresponding data is returned.
    Otherwise an interpolation between existing exposures is performed and
    returned.
    """
    #pdb.set_trace()
    # TODO: refactor this; specify dark frames in a variable at the
    # module scope. Add interpolation. (vs number of pulses)
#    lookup_dict =\
#        {1: radial_mean("dark_runs/1p_dark*"),
#        2: radial_mean("dark_runs/2p_dark*"),
#        3: radial_mean("dark_runs/3p_dark*"),
#        4: radial_mean("dark_runs/4p_dark*"),
#        5: radial_mean("dark_runs/5p_dark*"),
#        8: radial_mean("dark_runs/8p_dark*"),
#        10: radial_mean("dark_runs/10p_dark*"),
#        16: radial_mean("dark_runs/16p_dark*"),
#        50: radial_mean("dark_runs/50p_dark*"),
#        64: radial_mean("dark_runs/64p_dark*"),
#        100: radial_mean("dark_runs/100p_dark*"),
#        128: radial_mean("dark_runs/128p_dark*"),
#        250: radial_mean("dark_runs/250p_dark*"),
#        400: radial_mean("dark_runs/400p_dark*"),
#        500: radial_mean("dark_runs/500p_dark*"),
#        1000: radial_mean("background_exposures/dark_frames/dark_1000p*")}
    lookup_dict =\
        {1: radial_mean("../xrd/dark/dark_1p*"),
        #2: radial_mean("dark_runs/2p_dark*"),
        #3: radial_mean("dark_runs/3p_dark*"),
        #4: radial_mean("dark_runs/4p_dark*"),
        #5: radial_mean("dark_runs/5p_dark*"),
        #8: radial_mean("dark_runs/8p_dark*"),
        10: radial_mean("../xrd/dark/dark_10p*"),
        #16: radial_mean("dark_runs/16p_dark*"),
        #50: radial_mean("dark_runs/50p_dark*"),
        #64: radial_mean("dark_runs/64p_dark*"),
        100: radial_mean("dark_runs/100p_dark*"),
        #128: radial_mean("dark_runs/128p_dark*"),
        #250: radial_mean("dark_runs/250p_dark*"),
        #400: radial_mean("dark_runs/400p_dark*"),
        #500: radial_mean("dark_runs/500p_dark*"),
        1000: radial_mean("background_exposures/dark_frames/dark_1000p*")}
    def dark_frame(npulses):
        if npulses in lookup_dict:
            return lookup_dict[npulses]
        else:
            return (lookup_dict[1000] - lookup_dict[1]) * ((npulses - 1)/999.) + lookup_dict[1]
    return dark_frame(npulses) * nframes

@utils.eager_persist_to_file("cache/air_scatter/")
def air_scatter(npulses, attenuation, nframes = 1):
    """
    Return estimated air scatter contribution to the radial profile for a given
    number of pulses and attenuation level. Note that this background does NOT
    include dark counts.
    """
    def extract_intensity(path):
        raw =  np.genfromtxt(path)[1]
        return raw - dark_subtraction(npulses_ref)
    def interpolated(attenuation):
        return attenuation_1x_1000p / attenuation
    npulses_ref = 1000. #number of pulses in the air scatter refernce images
    attenuation_1x_1000p = extract_intensity("background_exposures/beam_on_no_sample_2/radial_integrations/1x_1000p_001.mccdprocessed.dat")
    return nframes * (npulses/npulses_ref) * interpolated(attenuation)

@utils.eager_persist_to_file("cache/air_scatter2/")
def air_scatter2(npulses, attenuation, nframes = 1, filtersize = 10):
    """
    Return estimated air scatter contribution to the radial profile for a given
    number of pulses and attenuation level. Note that this background does NOT
    include dark counts.

    attenuation is ACTUAL attenuation
    """
    # contruct the lookup dict of air scatter data
    attenuation = nominal_attenuations[attenuation]
    BACKGROUND_DICT = {}
    def add_background_dict_entry(npulses, attenuation, extractor_function):
        globpattern = AIR_SCATTERN_PATTERN % (npulses, attenuation)
        BACKGROUND_DICT[(npulses, attenuation)] = extractor_function(globpattern)
    for pair in measured_background_params:
        add_background_dict_entry(pair[0], pair[1], radial_mean)
    def extract_intensity(npulses, attenuation):
        unsubtracted = BACKGROUND_DICT[(npulses, attenuation)]
        return gaussian_filter(unsubtracted - dark_subtraction(npulses), filtersize)
    # TODO: find air scatter dataset with same attenuation level (instead of
    # defaulting to 10x)
    def interpolated():
        npulses_intepolation = 10
        attenuation_interpolation = 10
        #attenuation_interpolation = attenuation
        return nframes * (npulses/npulses_intepolation) *\
            extract_intensity(npulses_intepolation, attenuation_interpolation) * actual_attenuations[attenuation_interpolation]/ actual_attenuations[attenuation]
    if (npulses, attenuation) in BACKGROUND_DICT:
        print "found air scatter data: ", (npulses, attenuation)
        return nframes * extract_intensity(npulses, attenuation)
    else:
        print "interpolating air scatter data"
        return interpolated()

@memoize
def kapton_background(npulses, attenuation = 1, nframes = 1, subtract_air_scatter = False):
    kapton_file = "test_diffraction/radial_integrations/v2o5nano_10x_1000_kapton_background_2.mccdprocessed.dat"
    kapton_attenuation = 10
    kapton_pulses = 1000
    raw = np.genfromtxt(kapton_file)[1] - dark_subtraction(kapton_pulses)
    if subtract_air_scatter:
        air_scatter_intensity = air_scatter2(npulses, attenuation)
        raw = raw - air_scatter_intensity
    return raw * nframes * (float(npulses)/kapton_pulses) * (float(kapton_attenuation) / attenuation)

def kapton_only(npulses, attenuation = 1, nframes = 1, subtract_air_scatter = False):
    """
    returns the signal for kapton, not including dark counts or air scatter
    """
    return (kapton_background(npulses, attenuation, nframes) - air_scatter2(npulses, attenuation, nframes))



def bgsubtract(npulses, attenuation, nframes, kaptonfactor = 1., airfactor = 1.):
    darksub = dark_subtraction(npulses, nframes)
    airsub = airfactor * air_scatter2(npulses, attenuation, nframes)
    if kaptonfactor: # kaptonsub != 0
        kaptonsub = kaptonfactor * kapton_only(npulses, attenuation, nframes)
    else:
        kaptonsub = 0.0
    interp_func = lambda x, y: rldeconvolution.extrap1d(interp1d(x, kaptonsub + darksub + airsub))
    return interp_func

def beam_spectrum(attenuator):
    beam = pinkbeam_spectrum()
    if (attenuator == 1):
        return beam
    else:
        attenuation = ATTENUATION_FUNCTIONS[attenuator]
        return [beam[0], beam[1] * attenuation(beam[0])]

# A few functions for extracting information about a glob pattern
def glob_nframes(glob_pattern):
    return len(radial_density_paths(glob_pattern))


def glob_npulses(glob_pattern):
    matches = re.findall(r'.*[_\/]([0-9\.]+)p.*', glob_pattern)
    if len(matches) != 1:
        raise ValueError("malformed name: can't interpret number of pulses")
    else:
        npulses = int(matches[0])
        print 'npulses ', npulses
        return npulses

def num(s):
    """
    Parse string to float or int.
    """
    try: 
        return int(s)
    except ValueError:
        return float(s)

def glob_attenuator(glob_pattern):
    matches = re.findall(r'.*_([0-9\.]+)x.*', glob_pattern)
    if len(matches) != 1:
        raise ValueError("malformed name: can't interpret number of pulses")
    else:
        attenuation = num(matches[0])
        print 'attenuation ', attenuation
        return attenuation

#@utils.eager_persist_to_file("cache/full_process/")
@utils.memoize(timeout = None)
def full_process(glob_pattern, attenuator, norm = 1., dtheta = 1e-3, filtsize = 2, npulses = 1, center = CENTER, airfactor = 1.0, kaptonfactor = 0.0, smooth_size = 0.3, deconvolution_iterations = 100, ncores = None, **kwargs):
    """
    process image files into radial distributions if necessary, then 
    sum the distributions and deconvolve

    attenuator: numeric value
    """
    beam = beam_spectrum(attenuator)
    generate_radial_all(glob_pattern, center = center, ncores = ncores)
    nframes = glob_nframes(glob_pattern)
    if nframes == 0:
        raise ValueError(glob_pattern +  ": no matching files found")
    print "nframes: " + str(nframes)
    angles, intensities = process_all_globs([glob_pattern], subArray=[bgsubtract(npulses, actual_attenuations[attenuator], nframes, kaptonfactor =  kaptonfactor, airfactor = airfactor)], normalizationArray=[actual_attenuations[attenuator]/(npulses * nframes)], show = False, ncores = ncores)
    angles = np.deg2rad(angles)
    # zero negative values TODO: find cause for negative values here. 
    intensities[intensities < 0] = 0.
    if deconvolution_iterations == 0:
        return angles, gaussian_filter(intensities, filtsize)
    else:
        return rldeconvolution.deconvolve(angles, gaussian_filter(intensities, filtsize), beam[0], beam[1], dtheta, 'matrix', smooth_size = smooth_size, deconvolution_iterations = deconvolution_iterations)




def process_and_plot(pattern_list, deconvolution_iterations = 100, plot_powder = True, show = True, dtheta = 5e-4, filtsize = 2, center = CENTER, airfactor = 1.0, kaptonfactor = 0.0, smooth_size = 0.0, normalize = '', peak_ranges = None):
    # TODO update docstring
    """
    Process data specified by glob patterns in pattern_list, using the
    parameters extracted from the filepath prefix

    Inputs:
        normalize: if == height, normalizes spectra by peak value; if == a tuple
            (assumed to contain two ints), normalizes by the integral of the
            specified range; if bool(height) == False, no normalization is done.
        peak_ranges: a list of tuples storing min and max angles of powder peaks
            (necessary for generating a plot of integrated Bragg peak intensities)
    """
    def one_spectrum(pattern):
        npulses = glob_npulses(pattern)
        x, y = full_process(pattern, glob_attenuator(pattern), dtheta = dtheta,
            filtsize = filtsize, npulses = npulses, center = center, airfactor = airfactor,
            kaptonfactor = kaptonfactor, smooth_size = smooth_size, deconvolution_iterations = deconvolution_iterations)
        #x, y = estimator(deconvolution_iterations)
        # normalize by peak height
        if normalize == 'height':
            return [x, y/np.max(y)]
        # normalize by integral of the specified range
        elif isinstance(normalize, tuple): 
            integral = peak_sizes(x, y, [normalize], bg_subtract = False)
            return [x, y / integral]
        elif normalize:
            raise ValueError("invalid value for normalize: " + str(normalize))
        else:
            return x, y
    spectra = [one_spectrum(pattern) for pattern in pattern_list]


    def plot_curves(cmap_start = 0., cmap_max = 1.0):
        # TODO: implement this
        #cmap = plt.get_cmap('coolwarm')
        ncurves = len(spectra)
        for i, pattern, curve in zip(range(len(spectra)), pattern_list, spectra):
            #plt.plot(*curve, label = pattern, color = cmap(cmap_start + i * (cmap_max - cmap_start)/ncurves))
            plt.plot(*curve, label = pattern)
        plt.legend()
#        ax.set_xlabel('Angle (rad)')
#        ax.set_ylabel('Intensity (arb)')
        plt.xlabel('Angle (rad)')
        plt.ylabel('Intensity (arb)')

    if plot_powder:
        plot_curves()
    if show:
        plt.show()
    return spectra

