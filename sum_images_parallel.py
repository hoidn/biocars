import numpy as np
import os
import pdb
import glob
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import interpolate

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

##TODO implement filtering and parallelism
#def parallel_sum_images(filename_list, num_cores, filter = None):
#    def partition_list(lst):
#        stride = len(lst)/num_cores
#        sublists = [lst[i * stride: (i + 1) * stride] for i in xrange(num_cores)]
#        return sublists
#    def single_threaded_sum(lst):
#        if len(filename_list) < 1:
#            raise ValueError("need at least one file")
#        imsum = np.array(Image.open(filename_list[0]))
#        if len(filename_list) == 1:
#            return imsum
#        for fname in filename_list[1:]:
#            imsum += np.array(Image.open(filename_list[0]))
#        return imsum
#    pool = Pool(processes = num_cores)
#    sub_sums = pool.map(single_threaded_sum, partition_list(filename_list))
#    return np.sum(

def radialSum(data, distance = 150000., pixSize = 88.6, center = None):
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

class Run(object):
    def __init__(self, name, condition = None, center = [1984, 1967]):
        if name[-5:] != ".mccd":
            self.glob = name + "*.mccd"
        else:
            self.glob = name
        self.basename = name.replace("*", '')
        self.directory = os.path.dirname(self.basename)
        self.center = center #center coords of powder pattern on detector
        self.sumpath = self.basename + "summed.dat"
        self.radialpath = self.basename + "processed.dat"
        self.process()

    def already_processed(self):
        return os.path.exists(self.radialpath)

    def process(self, recompute = False):
        """
        mode can equal names or glob
        """
        if not self.already_processed() and not recompute:
            allfiles = glob.glob(self.glob)
            summed = sumImages(allfiles, 1)
            summed.astype("uint32").tofile(self.basename + "summed.dat") 
            #summed.tofile(self.prefix + "/averaged.dat") 
            #TODO: is this center value right?
            radial = radialSum(summed,  center = self.center)
            np.savetxt(self.basename + "processed.dat", radial)

    #normalization modes: peak, if a numerical value is provided it is assumed to be the angle at which 
    #normalization is desired.
    def plot(self, normalization = "peak", bgsub = 0):
        def plotOne(label):
            r, intensity = np.genfromtxt(self.basename + "processed.dat")
            intensity -= bgsub
            if normalization == "peak":
                intensity /= np.max(intensity)
            elif isinstance(normalization, (int, long, float)):
                interpolated = interpolate.interp1d(r, intensity)
                normval = interpolated(normalization)
                intensity /= normval
            plt.plot(r, intensity, label = label)
        plotOne(self.basename)

    def show(self):
        plt.xlabel("angle (degrees)")
        plt.ylabel("inensity (arb)")
        plt.legend()
        plt.show()

def plotAll(glob_patterns, show = True, individual = False, subArray = None, normalization = "peak"):
    if subArray is None:
        subArray = [0] * len(glob_patterns)
    def run_and_plot(glob, bgsub = 0):
        run = Run(glob)
        run.plot(bgsub = bgsub, normalization = normalization)
        return run
    if individual is True:
        for patt in glob_patterns:
            if patt[-5:] != ".mccd":
                patt = patt + "*.mccd"
            fileList = glob.glob(patt)
            runs = map(run_and_plot, fileList)
            runs[-1].show()
    else:
        for patt, subtraction in zip(glob_patterns, subArray):
            run = run_and_plot(patt, subtraction)
        if show is True:
            run.show()
