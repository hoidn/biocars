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

class Run(object):
    def __init__(self, name, condition = None, center = [1973, 1973]):
    #def __init__(self, name, condition = None, center = [1984, 1873]):
        if not os.path.isdir("runs/" + name):
            os.mkdir("runs/" + name)
        self.name = name
        self.switch_condition(condition)
        self.center = center #center coords of powder pattern on detector
    
    def switch_condition(self, condition):
        if condition is None:
            self.condition = "default"
        else:
            self.condition = condition
	self.prefix = "runs/" + self.name + '/' + self.condition
        if not os.path.exists(self.prefix):
            os.mkdir(self.prefix)

    def add_frames(self, files = None, mode = "names"):
        """
        mode can equal names or glob
        """
        if files == None:
            files = glob.glob("*.mccd")
        if len(files) == 0:
            print "no new files found"
            return
        newFiles = [self.prefix + "/" + fname for fname in files]
        for fname, newfname in zip(files, newFiles):
            os.system("mv " + fname + " " + newfname)
        allfiles  = glob.glob(self.prefix + "/*.mccd")
        summed = sumImages(allfiles, 1)
        summed.astype("uint32").tofile(self.prefix + "/averaged.dat") 
        #summed.tofile(self.prefix + "/averaged.dat") 
        #TODO: is this center value right?
        radial = radialSum(summed,  center = self.center)
        np.savetxt(self.prefix + "/processed.dat", radial)

    #normalization modes: peak, if a numerical value is provided it is assumed to be the angle at which 
    #normalization is desired.
    def plot(self, condition = None, normalization = "peak"):
        def plotOne(conditionStr):
            r, intensity = np.genfromtxt("runs/" + self.name + "/" + conditionStr + "/processed.dat")
            if normalization == "peak":
                intensity /= np.max(intensity)
            elif isinstance(normalization, (int, long, float)):
                interpolated = interpolate.interp1d(r, intensity)
                normval = interpolated(normalization)
                intensity /= normval
            plt.plot(r, intensity, label = self.name + "_" + conditionStr)
        if condition is None:
            plotOne(self.condition)
        else:
            plotOne(condition)

    def show(self):
        plt.xlabel("angle (degrees)")
        plt.ylabel("inensity (arb)")
        plt.legend()
        plt.show()

