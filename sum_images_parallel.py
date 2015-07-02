import numpy as np
import pickle
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

import multiprocessing as mp
import random
import string

DATA_DIR = "/data/seidler_1506/script_cache"

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

#TODO: why doesn't this work?
#def generate_radial_all(directory_glob_pattern):
#    """ 
#    given a glob pattern, generate radial distributions for all the matching .mccd files
#    """
##    def radial_name(prefix):
##        dirname = os.path.dirname(prefix)
##        basename = os.path.basename(prefix)
##        radial_file_path = dirname + "/radial_integrations/" + basename + "processed.dat"
##        return radial_file_path
##
#    if directory_glob_pattern[-5:] != ".mccd":
#        directory_glob_pattern = directory_glob_pattern + "*mccd"
#    all_mccds = deep_glob(directory_glob_pattern)
##    for mccd in all_mccds:
##        radial_path = radial_name(mccd)
#    for mccd, radial_path in zip(all_mccds, radial_density_paths(directory_glob_pattern)):
#        radial_directory = os.path.dirname(radial_path)
#        if not os.path.exists(radial_directory):
#            os.mkdir(radial_directory)
#        if not os.path.exists(radial_path):
#            radial = radialSum(np.array(Image.open(mccd)),  center = CENTER)
#            np.savetxt(radial_path, radial)


def generate_radial_all(directory_glob_pattern):
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
        if not os.path.exists(radial_path):
            radial = radialSum(np.array(Image.open(mccd)),  center = CENTER)
            np.savetxt(radial_path, radial)

#normalization modes: peak, if a numerical value is provided it is assumed to be the angle at which 
#normalization is desired.
#TODO: smooth. why doesn't it work?
def plot(radial_distribution, normalization = None, bgsub = None, label = None, smooth = 1):
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
        bgsub = default_bgsubtraction

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
        plt.plot(r, gaussian_filter(intensity, smooth), label = label)
    plotOne(label)



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

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
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
    #pdb.set_trace()
    rsorted = np.sort(np.unique(r.ravel()))
    theta = np.arctan(rsorted * pixSize / distance)
    return np.rad2deg(theta), radialprofile 


##normalization modes: peak, if a numerical value is provided it is assumed to be the angle at which 
##normalization is desired.
#def plot(name, normalization = "peak"):
#    def plotOne(name):
#        r, intensity = np.genfromtxt("runs/" + self.name + "/" + name + "/processed.dat")
#        if normalization == "peak":
#            intensity /= np.max(intensity)
#        elif isinstance(normalization, (int, long, float)):
#            interpolated = interpolate.interp1d(r, intensity)
#            normval = interpolated(normalization)
#            intensity /= normval
#        plt.plot(r, intensity, label = self.name + "_" + conditionStr)
#    if condition is None:
#        plotOne(self.condition)
#    else:
#        plotOne(condition)

#def findCenter(data, searchwidth = 1000):
#    w, h = np.shape(data)
#    def difference(center):
#        arrchunk = np.sum(data[w/2 - 100, w/2 + 100, :], axis = 0)
#        difference 


def swap(arr1d, centerindx):
    breadth = min(centerindx, len(arr1d) - centerindx)
    newarr = np.zeros(len(arr1d))
    for i in range(len(breadth)):
        newarr[centerindx - i], newarr[centerindx + i] = newarr[centerindx + i], newarr[centerindx - i]
    return np.sum(np.abs(newarr - arr1d))

#class Run(object):
#    def __init__(self, name, condition = None, center = [1984, 1967]):
#    #def __init__(self, name, condition = None, center = [1984, 1873]):
#        if not os.path.isdir("runs/" + name):
#            os.mkdir("runs/" + name)
#        self.name = name
#        self.switch_condition(condition)
#        self.center = center #center coords of powder pattern on detector
#    
#    def switch_condition(self, condition):
#        if condition is None:
#            self.condition = "default"
#        else:
#            self.condition = condition
#	self.prefix = "runs/" + self.name + '/' + self.condition
#        if not os.path.exists(self.prefix):
#            os.mkdir(self.prefix)
#
#    def add_frames(self, files = None, globbing = False):
#        """
#        mode can equal names or glob
#        """
#        if files == None:
#            files = glob.glob("*.mccd")
#        if len(files) == 0:
#            print "no new files found"
#            return
#
#        globbed_files = files
##        globbed_files = []
##        for file in files:
##            globbed_files = globbed_files + glob.glob(file)
#        #pdb.set_trace()
#        newFiles = [self.prefix + "/" + fname for fname in globbed_files]
#        for fname, newfname in zip(globbed_files, newFiles):
#            os.system("mv " + fname + " " + newfname)
#        allfiles  = glob.glob(self.prefix + "/*.mccd")
#        summed = sumImages(allfiles, 1)
#        summed.astype("uint32").tofile(self.prefix + "/averaged.dat") 
#        #summed.tofile(self.prefix + "/averaged.dat") 
#        #TODO: is this center value right?
#        radial = radialSum(summed,  center = self.center)
#        np.savetxt(self.prefix + "/processed.dat", radial)
#
#    #normalization modes: peak, if a numerical value is provided it is assumed to be the angle at which 
#    #normalization is desired.
#    def plot(self, condition = None, normalization = "peak", bgsub = 0):
#        def plotOne(conditionStr):
#            r, intensity = np.genfromtxt("runs/" + self.name + "/" + conditionStr + "/processed.dat")
#            intensity -= bgsub
#            if normalization == "peak":
#                intensity /= np.max(intensity)
#            elif isinstance(normalization, (int, long, float)):
#                interpolated = interpolate.interp1d(r, intensity)
#                normval = interpolated(normalization)
#                intensity /= normval
#            plt.plot(r, intensity, label = self.name + "_" + conditionStr)
#        if condition is None:
#            plotOne(self.condition)
#        else:
#            plotOne(condition)
#
#    def show(self):
#        plt.xlabel("angle (degrees)")
#        plt.ylabel("inensity (arb)")
#        plt.legend()
#        plt.show()
#

#class Run(object):
#    def __init__(self, name, condition = None, center = CENTER, data_filter = None, recompute = False):
#        if name[-5:] != ".mccd":
#            self.glob = name + "*.mccd"
#        else:
#            self.glob = name
#        if data_filter is not None:
#            self.data_filter = lambda filename: data_filter(self.intensity_distribution_dictionary[filename])
#        else:
#            self.data_filter = lambda intensity: True
#        self.basename = name.replace("*", '')
#        self.directory = os.path.dirname(self.basename)
#        self.center = center #center coords of powder pattern on detector
#        self.sumpath = self.basename + "summed.dat"
#        self.radialpath = self.basename + "processed.dat"
#        self.allfiles = None
#        self.intensity_distribution_dictionary = None
#        allfiles = glob.glob(self.glob)
#        self.allfiles = allfiles
#        self.process(recompute = recompute)
#
#    def already_processed(self):
#        return os.path.exists(self.radialpath)
#
#    def intensity_distribution(self, reprocess = False):
#        #TODO: refactor
#        if self.intensity_distribution_dictionary is not None:
#            return self.intensity_distribution_dictionary
#        elif os.path.exists(self.basename + "intensities.p"):
#            self.intensity_distribution_dictionary = pickle.load(open(self.basename + "intensities.p", "rb"))
#            return self.intensity_distribution_dictionary
#
##        if (not self.already_processed()) or reprocess:
##            self.process(recompute = True)
#        def intensity(filename):
#            return np.sum(np.array(Image.open(filename)))
#        intensities = map(intensity, self.allfiles)
#        self.intensity_distribution_dictionary =  {name: intensity for (name, intensity) in zip(self.allfiles, intensities)}
#        with open(self.basename + "intensities.p", "wb") as f:
#            pickle.dump(self.intensity_distribution_dictionary, f)
#        return self.intensity_distribution_dictionary
#
#    def process(self, recompute = False):
#        """
#        mode can equal names or glob
#
#        if filter changes you NEED to reprocess for the changes to propagate
#        """
#        if (not self.already_processed()) or (recompute):
#            if self.intensity_distribution_dictionary is None:
#                self.intensity_distribution()
#            #intensities = [self.intensity_distribution_dictionary[name] for name in self.allfiles]
#            goodfiles = filter(self.data_filter, self.allfiles)
#            summed = sumImages(goodfiles, 1)
#            summed.astype("uint32").tofile(self.basename + "summed.dat") 
#            #summed.tofile(self.prefix + "/averaged.dat") 
#            #TODO: is this center value right?
#            radial = radialSum(summed,  center = self.center)
#            np.savetxt(self.basename + "processed.dat", radial)
#
#    #normalization modes: peak, if a numerical value is provided it is assumed to be the angle at which 
#    #normalization is desired.
#    def plot(self, normalization = "peak", bgsub = None, label = None):
#        """
#        bgsub is a function that takes x and y coordinates and returns an interpolation that 
#        yields background subtraction as a function of x
#
#        filter takes an int (the total intensity in a given frame) and returns a boolean
#        """
#        def default_bgsubtraction(x, y, endpoint_size = 10, endpoint_right = 70):
#            bgx = np.concatenate((x[:endpoint_size], x[-endpoint_right:]))
#            bgy = np.concatenate((y[:endpoint_size], y[-endpoint_right:]))
#            interp_func = extrap1d(interp1d(bgx, bgy))
#            return interp_func
#
#        if bgsub is None:
#            bgsub = default_bgsubtraction
#
#        def plotOne(label):
#            r, intensity = np.genfromtxt(self.basename + "processed.dat")
#            #intensity -= bgsub
#            intensity = intensity - bgsub(r, intensity)(r)
#            if normalization == "peak":
#                intensity /= np.max(intensity)
#            elif isinstance(normalization, (int, long, float)):
#                interpolated = interpolate.interp1d(r, intensity)
#                normval = interpolated(normalization)
#                #intensity /= normval
#                intensity *= normalization
#            plt.plot(r, intensity, label = label)
#        if label is None:
#            label = self.basename
#        plotOne(label)
#
#    def show(self):
#        plt.xlabel("angle (degrees)")
#        plt.ylabel("inensity (arb)")
#        plt.legend()
#        plt.show()

def plotAll(glob_patterns, show = True, individual = False, subArray = None, normalization = "peak", normalizationArray = None, labelList = None, filterList = None, recompute = False):
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
        plot(sum_radial_densities(glob, give_tuple = True), normalization = normalization, bgsub = bgsub, label = glob)
    if individual is True:
        #TODO: support kwargs
        for patt in glob_patterns:
            if patt[-5:] != ".mccd":
                patt = patt + "*.mccd"
            fileList = glob.glob(patt)
            runs = map(run_and_plot, fileList)
            runs[-1].show()
    else:
        for patt, subtraction, label, one_filter, norm in zip(glob_patterns, subArray, labelList, filterList, normalizationArray):
           run_and_plot(patt, bgsub = subtraction, label = label, data_filter = one_filter, normalization = norm)
        plt.legend()
        plt.show()

def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret
    return memodict(f)


def radial_mean(filenames, sigma = 1):
    return gaussian_filter(sum_radial_densities(filenames, average = True), sigma = sigma)

@memoize
def dark_subtraction(npulses, nframes = 1):
    if not os.path.exists(DATA_DIR + '/dark_sub.p'):
        lookup_dict = {1: radial_mean("background_exposures/dark_frames/dark_1p*"), 3: radial_mean("background_exposures/dark_frames/dark_3p*"), 10: radial_mean("background_exposures/dark_frames/dark_10p*"), 1000: radial_mean("background_exposures/dark_frames/dark_1000p*")}
        dark_250_interpolation = (lookup_dict[1000] - lookup_dict[1])/4 + lookup_dict[1]
        dark_500_interpolation = (lookup_dict[1000] - lookup_dict[1])/2 + lookup_dict[1]
        dark_2000_interpolation = 2 * lookup_dict[1000] - lookup_dict[1]
        lookup_dict[250] = dark_250_interpolation
        lookup_dict[500] = dark_500_interpolation
        lookup_dict[2000] = dark_2000_interpolation
        with open(DATA_DIR + '/dark_sub.p', 'wb') as f:
            pickle.dump(lookup_dict, f)
    else:
        lookup_dict = pickle.load(open(DATA_DIR + '/dark_sub.p', 'rb'))

    return lookup_dict[npulses] * nframes

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
    return (kapton_background(npulses, attenuation) - air_scatter(npulses, attenuation))



def bgsubtract(npulses, attenuation, nframes, kaptonfactor = 1.):
    darksub = dark_subtraction(npulses, nframes)
    airsub = air_scatter(npulses, attenuation, nframes)
    #kaptonsub = kaptonfactor * kapton_background(npulses, attenuation, nframes)
    kaptonsub = kaptonfactor * kapton_only(npulses, attenuation, nframes)
    interp_func = lambda x, y: extrap1d(interp1d(x, kaptonsub + darksub + airsub))
    return interp_func


