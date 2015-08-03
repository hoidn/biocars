import sys
from  pathos import multiprocessing 
import ipdb
import numpy as np
import scipy
import os
import pdb
import glob
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from scipy import arange, array, exp
import operator
import fnmatch
from  libtiff import TIFF

import multiprocessing as mp
import random
import string
import rldeconvolution
import mu
from scipy.ndimage.filters import gaussian_filter as filt

import pickle
import atexit, dill

DATA_DIR = "/media/sf_data/seidler_1506/script_cache"
PHOTON_ENERGY = 12000. # NOMINAL incident photon energy
HBARC = 1973. #eV * Angstrom

def mask_peaks_and_iterpolate(x, y, peak_ranges):
    for peakmin, peakmax in peak_ranges:
        good_indices = np.where(np.logical_or(x < peakmin, x > peakmax))[0]
        y = y[good_indices]
        x = x[good_indices]
    return scipy.interpolate.interp1d(x, y)

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

#TODO: these two files are good candidates for a utilities package
def persist_to_file(file_name):

    try:
        cache = dill.load(open(file_name, 'r'))
    except (IOError, ValueError):
        cache = {}

    atexit.register(lambda: dill.dump(cache, open(file_name, 'w')))

    def decorator(func):
        #check if function is a closure and if so construct a dict of its bindings
        if func.func_code.co_freevars:
            closure_dict = dict(zip(func.func_code.co_freevars, (c.cell_contents for c in func.func_closure)))
        else:
            closure_dict = {}
        def new_func(*args, **kwargs):
            key = (args, frozenset(kwargs.items()), frozenset(closure_dict.items()))
            if key not in cache:
                cache[key] = func(*key[0], **{k: v for k, v in key[1]})
            return cache[key]
        return new_func

    return decorator

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

#center coords of the beam on the ccd
CENTER = [1984, 1967]

## Define an output queue
#output = mp.Queue()
#
### define a example function
##def rand_string(length, output):
##    """ Generates a random string of numbers, lower- and uppercase chars. """
##    rand_str = ''.join(random.choice(
##                    string.ascii_lowercase
##                    + string.ascii_uppercase
##                    + string.digits)
##               for i in range(length))
##    output.put(rand_str)
#
## Setup a list of processes that we want to run
#processes = [mp.Process(target=rand_string, args=(5, output)) for x in range(4)]
#
## Run processes
#for p in processes:
#    p.start()
#
## Exit the completed processes
#for p in processes:
#    p.join()
#
## Get process results from the output queue
#results = [output.get() for p in processes]
#
#print(results)


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

#TODO: make this consistent
@memoize
def sum_radial_densities(directory_glob_pattern, average = False, give_tuple = False):
    paths = radial_density_paths(directory_glob_pattern)
    extractIntensity = lambda name: np.genfromtxt(name)
    if not average:
        r, intensity = sum_many(map(extractIntensity, paths))
        if give_tuple:
            return r/len(paths), intensity
        else:
            return intensity
    else:
        return sum_many(map(extractIntensity, paths))[1]/len(paths)


def generate_radial_all(directory_glob_pattern, recompute = False):
    """
    given a glob pattern, generate radial distributions for all the matching .mccd files
    """
    def radial_name(prefix):
        dirname = os.path.dirname(prefix)
        basename = os.path.basename(prefix)
        radial_file_path = dirname + "/radial_integrations/" + basename + "processed.dat"
        return radial_file_path

    if directory_glob_pattern[-5:] != ".mccd":
        directory_glob_pattern = directory_glob_pattern + "*mccd"
    all_mccds = deep_glob(directory_glob_pattern)
    for mccd in all_mccds:
        radial_path = radial_name(mccd)
        radial_directory = os.path.dirname(radial_path)
        if not os.path.exists(radial_directory):
            os.mkdir(radial_directory)
        if (not os.path.exists(radial_path)) or recompute:
            #radial = radialSum(np.array(Image.open(mccd)),  center = CENTER)
            radial = radialSum(TIFF.open(mccd, 'r').read_image(),  center = CENTER)
            np.savetxt(radial_path, radial)


def default_bgsubtraction(x, y, endpoint_size = 10, endpoint_right = 10):
    bgx = np.concatenate((x[:endpoint_size], x[-endpoint_right:]))
    bgy = np.concatenate((y[:endpoint_size], y[-endpoint_right:]))
    interp_func = extrap1d(interp1d(bgx, bgy))
    return interp_func

#normalization modes: peak, if a numerical value is provided it is assumed to be the angle at which
#normalization is desired.
#TODO: smooth. why doesn't it work?
def plot(radial_distribution, normalization = None, bgsub = None, label = None,
smooth = 1, scale = 'angle'):
    """
    bgsub is a function that takes x and y coordinates and returns an interpolation that
    yields background subtraction as a function of x

    filter takes an int (the total intensity in a given frame) and returns a boolean
    """
    def default_bgsubtraction(x, y, endpoint_size = 10, endpoint_right = 10):
        bgx = np.concatenate((x[:endpoint_size], x[-endpoint_right:]))
        bgy = np.concatenate((y[:endpoint_size], y[-endpoint_right:]))
        interp_func = extrap1d(interp1d(bgx, bgy))
        return interp_func

    if bgsub is None:
        #bgsub = default_bgsubtraction
        bgsub = lambda x, y: lambda z: 0

    def show():
        plt.xlabel("angle (degrees)")
        plt.ylabel("inensity (arb)")
        plt.legend()
        plt.show()

    def plotOne(label):
        r, intensity = radial_distribution
        #intensity -= bgsub
        orig_intensity = intensity.copy()
        intensity = intensity - bgsub(r, intensity)(r)
        if normalization == "peak":
            intensity /= np.max(intensity)
        elif isinstance(normalization, (int, long, float)):
            interpolated = interpolate.interp1d(r, intensity)
            orig_interpolated = interpolate.interp1d(r, orig_intensity)
            #normval = interpolated(normalization)
            #normval = orig_interpolated(normalization)
            intensity *= normalization
        if scale == 'q':
            qq = 2 * PHOTON_ENERGY * np.sin(np.deg2rad(r/2))/HBARC
            plt.plot(qq, gaussian_filter(intensity, smooth), label = label)
            return qq, intensity
        if scale == 'angle':
            plt.plot(r, gaussian_filter(intensity, smooth), label = label)
            return r, intensity
        else:
            raise ValueError("invalid key: " +  str(scale))
    return plotOne(label)



#TODO implement filtering and parallelism
def parallel_sum_images(filename_list, num_cores, filter = None):
    def partition_list(lst):
        stride = len(lst)/num_cores
        sublists = [lst[i * stride: (i + 1) * stride] for i in xrange(num_cores)]
        return sublists
    def single_threaded_sum(lst, output):
        if len(filename_list) < 1:
            raise ValueError("need at least one file")
        imsum = np.array(Image.open(filename_list[0]))
        if len(filename_list) == 1:
            return imsum
        for fname in filename_list[1:]:
            imsum += np.array(Image.open(filename_list[0]))
        output.put(imsum)
    output = mp.Queue()
    processes = [mp.Process(target=single_threaded_sum, args=(filenames, output)) for filenames in filename_list]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()
    sub_sums = np.array([output.get() for p in processes])
    #pool = Pool(processes = num_cores)
    #sub_sums = pool.map(single_threaded_sum, partition_list(filename_list))
    return sum_many(sub_sums)

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

#    def pointwise(x):
#        if x < xs[0]:
#            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
#        elif x > xs[-1]:
#            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
#        else:
#            return interpolator(x)


    def pointwise(x):
        if x < xs[0]:
            return 0.
        elif x > xs[-1]:
            return 0.
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

def accumulator(func, accum, lst):
    """
    higher-order function to perform accumulation
    """
    if len(lst) == 0:
        return accum
    else:
        return accumulator(func, func(accum, lst[0]), lst[1:])


#to sum numpy arrays without altering shape
def sum_many(arr1d):
    return accumulator(operator.add, arr1d[0], arr1d[1:])



#TODO implement filtering and parallelism
def sumImages(filename_list, num_cores, filter = None):
    def partition_list(lst):
        stride = len(lst)/num_cores
        sublists = [lst[i * stride: (i + 1) * stride] for i in xrange(num_cores)]
        return sublists
    if len(filename_list) < 1:
        raise ValueError("need at least one file")
    imsum = np.array(Image.open(filename_list[0]))
    if len(filename_list) == 1:
        return imsum
    for fname in filename_list[1:]:
        imsum += np.array(Image.open(filename_list[0]))
    return imsum


def radialSum(data, distance = 140000., pixSize = 88.6, center = None):
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


def swap(arr1d, centerindx):
    breadth = min(centerindx, len(arr1d) - centerindx)
    newarr = np.zeros(len(arr1d))
    for i in range(len(breadth)):
        newarr[centerindx - i], newarr[centerindx + i] = newarr[centerindx + i], newarr[centerindx - i]
    return np.sum(np.abs(newarr - arr1d))



def plotAll(glob_patterns, show = True, individual = False, subArray = None,
normalization = "peak", normalizationArray = None, labelList = None, filterList
= None, recompute = False, scale = 'angle'):
    if subArray is None:
        subArray = [None] * len(glob_patterns)
    if labelList is None:
        labelList = [None] * len(glob_patterns)
    if filterList is None:
        filterList = [None] * len(glob_patterns)
    if normalizationArray is None:
        normalizationArray = [None] * len(glob_patterns)
#    def run_and_plot(glob, bgsub = None, label = None, data_filter = None, normalization = None):
#        run = Run(glob, data_filter = data_filter, recompute = recompute)
#        run.plot(bgsub = bgsub, normalization = normalization, label = label)
#        return run
    def run_and_plot(glob, bgsub = None, label = None, data_filter = None, normalization = None):
        return plot(sum_radial_densities(glob, False, True), normalization =
normalization, bgsub = bgsub, label = glob, scale = scale)
#    if individual is True:
#        #TODO: support kwargs
#        for patt in glob_patterns:
#            if patt[-5:] != ".mccd":
#                patt = patt + "*.mccd"
#            fileList = glob.glob(patt)
#            runs = map(run_and_plot, fileList)
#            runs[-1].show()
#    else:
    outputs = []
    for patt, subtraction, label, one_filter, norm in zip(glob_patterns, subArray, labelList, filterList, normalizationArray):
       outputs = outputs + [run_and_plot(patt, bgsub = subtraction, label = label, data_filter = one_filter, normalization = norm)]
    plt.legend()
    plt.show(block = False)
    #output format: angles, intensities, angles, intensities, etc.
    return np.vstack((_ for _ in outputs))


def radial_mean(filenames, sigma = 1):
    return gaussian_filter(sum_radial_densities(filenames, average = True), sigma = sigma)

@persist_to_file("cache/dark_subtraction.p")
def dark_subtraction(npulses, nframes = 1):
    lookup_dict = {1: radial_mean("background_exposures/dark_frames/dark_1p*"), 3: radial_mean("background_exposures/dark_frames/dark_3p*"), 10: radial_mean("background_exposures/dark_frames/dark_10p*"), 1000: radial_mean("background_exposures/dark_frames/dark_1000p*")}
    def dark_frame(npulses):
        return (lookup_dict[1000] - lookup_dict[1]) * ((npulses - 1)/999.) + lookup_dict[1]
    return dark_frame(npulses) * nframes

@memoize
def air_scatter(npulses, attenuation, nframes = 1):
    npulses_ref = 1000. #number of pulses in the air scatter refernce images
    def extract_intensity(path):
        raw =  np.genfromtxt(path)[1]
        return raw - dark_subtraction(npulses_ref)
    if not os.path.exists(DATA_DIR + '/air_scatter.p'):
        lookup_dict = {1:extract_intensity( "background_exposures/beam_on_no_sample_2/radial_integrations/1x_1000p_001.mccdprocessed.dat"), 10: extract_intensity("background_exposures/beam_on_no_sample_2/radial_integrations/10x_1000p.mccdprocessed.dat"), 300: extract_intensity("background_exposures/beam_on_no_sample_2/radial_integrations/300x_1000p.mccdprocessed.dat")}
        air_3_interpolation = lookup_dict[1]/3.
        lookup_dict[3] = air_3_interpolation
        with open(DATA_DIR + '/air_scatter.p', 'wb') as f:
            pickle.dump(lookup_dict, f)
    else:
        lookup_dict = pickle.load(open(DATA_DIR + '/air_scatter.p', 'rb'))
    return nframes * (npulses/npulses_ref) * lookup_dict[attenuation]


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
    kaptonsub = kaptonfactor * kapton_only(npulses, attenuation, nframes)
    interp_func = lambda x, y: extrap1d(interp1d(x, kaptonsub + darksub + airsub))
    return interp_func

def beam_spectrum(attenuator):
    dat = np.genfromtxt("mda/14IDB_15067.txt")
    beam = [dat.T[1], dat.T[21]]
    bgsub = default_bgsubtraction(*beam)(beam[0])
    beam[1] = beam[1] - bgsub
    beam[0] = 1000 * beam[0]
    if attenuator == 'None':
        return beam
    elif attenuator == 'Ag':
        Ag = mu.ElementData(47).mu
        return [beam[0], beam[1] * np.exp(-Ag(beam[0])/12.3)]
    elif attenuator == 'Ti':
        Ti = mu.ElementData(22).mu
        return [beam[0], beam[1] * np.exp(-Ti(beam[0])/30.)]


@persist_to_file("cache/full_process.p")
def full_process(glob_pattern, attenuator, norm = 1., dtheta = 1e-3, filtsize = 2, npulses = 1):
    """
    process image files into radial distributions if necessary, then 
    sum the distributions and deconvolve

    attenuator: == 'None' or 'Ag'
    """
    beam = beam_spectrum(attenuator)
    nominal_attenuations = {'None': 1, 'Ag': 300, 'Ti': 10.0}
    actual_attenuations = {'None': 1., 'Ag': 563., 'Ti': 12.}
    generate_radial_all(glob_pattern)
    nframes = len(radial_density_paths(glob_pattern))
    print "nframes: " + str(nframes)
    angles, intensities = plotAll([glob_pattern], subArray=[bgsubtract(npulses, nominal_attenuations[attenuator], nframes, 0., .0)], normalizationArray=[npulses * nframes/actual_attenuations[attenuator]], show = False)
    angles = np.deg2rad(angles)
    est = rldeconvolution.make_estimator(angles, filt(intensities, filtsize), beam[0], beam[1], dtheta, 'matrix')
    return est

def process_async(f, tuples_and_dicts):
    pool = multiprocessing.Pool()
    return [pool.apply_async(f, *tupdic) for tupdic in tuples_and_dicts]

#if __name__ == '__main__':
#    cmd = ' '.join(sys.argv[1:])
#    print cmd
#    eval(cmd)
#estAu_300x, estAu10x, estAu1x = process_async(full_process, [[("Au_foil/Thursday/Au_foil_5micron_300x_1000p_*", 'Ag'), {'dtheta': 2e-4, 'npulses': 1000}], [("Au_foil/Thursday/Au_foil_5micron_10x_10p_*", 'Ag'), {'dtheta': 2e-4, 'npulses': 10}], [("Au_foil/Thursday/new_trace/Au_foil_5micron_1x_1p_*", 'Ag'), {'dtheta': 2e-4, 'npulses': 1}]])
