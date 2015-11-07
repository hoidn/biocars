import sys
import ipdb
import numpy as np
import scipy
import os
import pdb
import glob
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from scipy import arange, array, exp
import operator
import fnmatch
from  libtiff import TIFF
import random
import string
import re

import rldeconvolution

# TODO: install these packages as a dependencies
import mu
import utils

import pickle
import atexit, dill


#center coords of the beam on the ccd
CENTER = [1984, 1967]

DATA_DIR = "/media/sf_data/seidler_1506/script_cache"
PHOTON_ENERGY = 12000. # NOMINAL incident photon energy
HBARC = 1973. #eV * Angstrom

attfunc_Ag = lambda energies: np.exp(-(mu.ElementData(47).sigma)(energies)/12.3)
attfunc_Ti = lambda energies: np.exp(-(mu.ElementData(22).sigma)(energies)/30.)
ATTENUATION_FUNCTIONS =\
    {'Ag': attfunc_Ag,
    'Ti': attfunc_Ti,
    300: attfunc_Ag,
    10: attfunc_Ti}


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


def deep_glob(pattern):
    """
    TODO: this will break if there's a glob outside of the 
    filename part of the pattern
    """
    return [os.path.join(dirpath, f) \
        for dirpath, dirnames, files in os.walk(os.path.dirname(pattern)) \
            for f in fnmatch.filter(files, os.path.basename(pattern))]

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

    if directory_glob_pattern[-5:] != ".mccd":
        directory_glob_pattern = directory_glob_pattern + "*mccd"
    all_mccds = deep_glob(directory_glob_pattern)
    radial_paths = map(radial_name, all_mccds)
    return radial_paths

@utils.persist_to_file("cache/sum_radial_densities.p")
def sum_radial_densities(directory_glob_pattern, average = False, give_tuple = False):
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
    extractIntensity = lambda name: np.genfromtxt(name)
    if not average:
        r, intensity = sum_many(utils.parallelmap(extractIntensity, paths))
        if give_tuple:
            return r/len(paths), intensity
        else:
            return intensity
    else:
        return sum_many(utils.parallelmap(extractIntensity, paths))[1]/len(paths)


def generate_radial_all(directory_glob_pattern, recompute = False, center = CENTER):
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

    if directory_glob_pattern[-5:] != ".mccd":
        directory_glob_pattern = directory_glob_pattern + "*mccd"
    all_mccds = deep_glob(directory_glob_pattern)
    
    def process_one_frame(mccd):
        radial_path = radial_name(mccd)
        radial_directory = os.path.dirname(radial_path)
        if not os.path.exists(radial_directory):
            os.mkdir(radial_directory)
        if (not os.path.exists(radial_path)) or recompute:
            #radial = radialSum(np.array(Image.open(mccd)),  center = CENTER)
            radial = radialSum(TIFF.open(mccd, 'r').read_image(),  center = center)
            np.savetxt(radial_path, radial)
    utils.parallelmap(process_one_frame, all_mccds, 4)

def default_bgsubtraction(x, y, endpoint_size = 10, endpoint_right = 10):
    """
    Return a function that linearly interpolates between the lower and upper
    x values of the dataset specified by x and y
    """

    bgx = np.concatenate((x[:endpoint_size], x[-endpoint_right:]))
    bgy = np.concatenate((y[:endpoint_size], y[-endpoint_right:]))
    interp_func = rldeconvolution.extrap1d(interp1d(bgx, bgy))
    return interp_func

#normalization modes: peak, if a numerical value is provided it is assumed to be the angle at which
#normalization is desired.
#TODO: smooth. why doesn't it work?
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


#TODO implement filtering and parallelism
def sumImages(filename_list, filter = None):
    """
    Sum a bunch of images
    Input: a list of filenames.
    Returns the summed images as a numpy array
    """
#    def partition_list(lst):
#        stride = len(lst)/num_cores
#        sublists = [lst[i * stride: (i + 1) * stride] for i in xrange(num_cores)]
#        return sublists
    if len(filename_list) < 1:
        raise ValueError("need at least one file")
    imsum = np.array(Image.open(filename_list[0]))
    if len(filename_list) == 1:
        return imsum
    for fname in filename_list[1:]:
        imsum += np.array(Image.open(filename_list[0]))
    return imsum


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
        data = np.array(Image.open(data))
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
= None, recompute = False, scale = 'angle'):
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
        return process_radial_distribution(sum_radial_densities(glob, False, True), normalization =
normalization, bgsub = bgsub, label = glob, scale = scale, plot = show)
    outputs = []
    for patt, subtraction, label, one_filter, norm in zip(glob_patterns, subArray, labelList, filterList, normalizationArray):
       outputs = outputs + [run_and_plot(patt, bgsub = subtraction, label = label, data_filter = one_filter, normalization = norm)]
    if show:
        plt.legend()
        plt.show(block = False)
    #output format: angles, intensities, angles, intensities, etc.
    return np.vstack((_ for _ in outputs))

def radial_mean(filenames, sigma = 1):
    return gaussian_filter(sum_radial_densities(filenames, average = True), sigma = sigma)

@utils.persist_to_file("cache/dark_subtraction.p")
def dark_subtraction(npulses, nframes = 1):
    lookup_dict = {1: radial_mean("background_exposures/dark_frames/dark_1p*"), 1000: radial_mean("background_exposures/dark_frames/dark_1000p*")}
    def dark_frame(npulses):
        return (lookup_dict[1000] - lookup_dict[1]) * ((npulses - 1)/999.) + lookup_dict[1]
    return dark_frame(npulses) * nframes

@utils.persist_to_file("cache/air_scatter.p")
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
    #lookup_dict = {1:extract_intensity( "background_exposures/beam_on_no_sample_2/radial_integrations/1x_1000p_001.mccdprocessed.dat"), 10: extract_intensity("background_exposures/beam_on_no_sample_2/radial_integrations/10x_1000p.mccdprocessed.dat"), 300: extract_intensity("background_exposures/beam_on_no_sample_2/radial_integrations/300x_1000p.mccdprocessed.dat")}
    npulses_ref = 1000. #number of pulses in the air scatter refernce images
    attenuation_1x_1000p = extract_intensity( "background_exposures/beam_on_no_sample_2/radial_integrations/1x_1000p_001.mccdprocessed.dat")
    return nframes * (npulses/npulses_ref) * interpolated(attenuation)


@memoize
def kapton_background(npulses, attenuation = 1, nframes = 1, subtract_air_scatter = False):
    kapton_file = "test_diffraction/radial_integrations/v2o5nano_10x_1000_kapton_background_2.mccdprocessed.dat"
    kapton_attenuation = 10
    kapton_pulses = 1000
    raw = np.genfromtxt(kapton_file)[1] - dark_subtraction(kapton_pulses)
    if subtract_air_scatter:
        air_scatter_intensity = air_scatter(npulses, attenuation)
        raw = raw - air_scatter_intensity
    return raw * nframes * (float(npulses)/kapton_pulses) * (float(kapton_attenuation) / attenuation)

def kapton_only(npulses, attenuation = 1, nframes = 1, subtract_air_scatter = False):
    """
    returns the signal for kapton, not including dark counts or air scatter
    """
    return (kapton_background(npulses, attenuation, nframes) - air_scatter(npulses, attenuation, nframes))



def bgsubtract(npulses, attenuation, nframes, kaptonfactor = 1., airfactor = 1.):
    darksub = dark_subtraction(npulses, nframes)
    airsub = airfactor * air_scatter(npulses, attenuation, nframes)
    #kaptonsub = kaptonfactor * kapton_background(npulses, attenuation, nframes)
    if kaptonfactor: # kaptonsub != 0
        kaptonsub = kaptonfactor * kapton_only(npulses, attenuation, nframes)
    else:
        kaptonsub = 0.0
    interp_func = lambda x, y: rldeconvolution.extrap1d(interp1d(x, kaptonsub + darksub + airsub))
    return interp_func

def beam_spectrum(attenuator):
    dat = np.genfromtxt("mda/14IDB_15067.txt")
    beam = [dat.T[1], dat.T[21]]
    bgsub = default_bgsubtraction(*beam)(beam[0])
    beam[1] = beam[1] - bgsub
    beam[0] = 1000 * beam[0]
    if attenuator == 'None':
        return beam
    else:
        attenuation = ATTENUATION_FUNCTIONS[attenuator]
        return [beam[0], beam[1] * attenuation(beam[0])]
#    elif attenuator == 'Ag':
#        
#    elif attenuator == 'Ti':
#        Ti = mu.ElementData(22).sigma
#        return [beam[0], beam[1] * np.exp(-Ti(beam[0])/30.)]

# A few functions for extracting information about a glob pattern
def glob_nframes(glob_pattern):
    return len(radial_density_paths(glob_pattern))

def glob_npulses(glob_pattern):
    matches = re.findall(r'.*_([0-9]+)p.*', glob_pattern)
    if len(matches) != 1:
        raise ValueError("malformed name: can't interpret number of pulses")
    else:
        npulses = int(matches[0])
        print 'npulses ', npulses
        return npulses

def glob_attenuator(glob_pattern):
    matches = re.findall(r'.*_([0-9]+)x.*', glob_pattern)
    if len(matches) != 1:
        raise ValueError("malformed name: can't interpret number of pulses")
    else:
        attenuation = int(matches[0])
        print 'attenuation ', attenuation
        return attenuation

# TODO: does caching work here?
@utils.persist_to_file("cache/full_process.p")
def full_process(glob_pattern, attenuator, norm = 1., dtheta = 1e-3, filtsize = 2, npulses = 1, center = CENTER, airfactor = 1.0, kaptonfactor = 0.0, **kwargs):
    """
    process image files into radial distributions if necessary, then 
    sum the distributions and deconvolve

    attenuator: == 'None' or 'Ag'
    """
    #nominal_attenuations = {'None': 1, 'Ag': 300, 'Ti': 10.0}
    actual_attenuations = {'None': 1., 'Ag': 563., 'Ti': 12.}
    actual_attenuations[10] = actual_attenuations['Ti']
    actual_attenuations[300] = actual_attenuations['Ag']
    beam = beam_spectrum(attenuator)
    generate_radial_all(glob_pattern, center = center)
    nframes = glob_nframes(glob_pattern)
    if nframes == 0:
        raise ValueError(glob_pattern +  ": no matching files found")
    print "nframes: " + str(nframes)
    angles, intensities = process_all_globs([glob_pattern], subArray=[bgsubtract(npulses, actual_attenuations[attenuator], nframes, kaptonfactor =  kaptonfactor, airfactor = airfactor)], normalizationArray=[actual_attenuations[attenuator]/(npulses * nframes)], show = False)
    angles = np.deg2rad(angles)
    # zero negative values
    intensities[intensities < 0] = 0.
    est = rldeconvolution.make_estimator(angles, gaussian_filter(intensities, filtsize), beam[0], beam[1], dtheta, 'matrix')
    return est

def process_and_plot(pattern_list, deconvolution_iterations = 100, plot = True, show = True, dtheta = 1e-3, filtsize = 2, center = CENTER, airfactor = 1.0, kaptonfactor = 0.0):
    """
    Process data specified by glob patterns in pattern_list, using the
    parameters extracted from the filepath prefix
    """
    def one_spectrum(pattern):
        npulses = glob_npulses(pattern)
        estimator = full_process(pattern, glob_attenuator(pattern), dtheta = dtheta,
            filtsize = filtsize, npulses = npulses, center = center, airfactor = airfactor,
            kaptonfactor = kaptonfactor)
        return estimator(deconvolution_iterations)
    spectra = [one_spectrum(pattern) for pattern in pattern_list]
    if plot:
        for pattern, curve in zip(pattern_list, spectra):
            plt.plot(*curve, label = pattern)
        plt.legend()
    if show:
        plt.show()
    return spectra
