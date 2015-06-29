import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

#TODO implement filtering and parallelism
def sumImages(filename_list, num_cores, filter = None):
    def partition_list(lst):
        stride = len(lst)/num_cores
        sublists = [lst[i * stride: (i + 1) * stride] for i in xrange(num_cores)]
        return sublists
    if len(filename_list) < 2:
        return
    imsum = np.array(Image.open(filename_list[0])
    for fname in filename_list[1:]:
        imsum += np.array(Image.open(filename_list[0]))

#TODO: check that this works
def radialSum(data, center = None):
    if center is None:
        l, h = np.shape(data)
        center = [l/2, h/2]
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile 

class Run(object):
    def __init__(self, name, condition = None):
        if not os.path.isdir("runs/" + name):
            os.system("mkdir runs/" + name)
        self.name = name
        self.switch_condition(condition)
        self.prefix = "runs/" + name + '/' + self.condition
    
    def switch_condition(self, condition):
        if condition is None:
            self.condition = "default"
        if not os.path.exists("mkdir runs/" + self.prefix)
            os.system("mkdir runs/" + self.prefix)
        self.condition = condition

    def add_frames(self, files = None):
        if files == None:
            files = glob.glob("*.tif")
        newFiles = [prefix + "/" + fname for fname in files]
        for fname, newfname in zip(files, newFiles):
            os.system("mv " + fname + " " + newfname)
        summed = sumImages(newFiles)
        np.tofile(prefix + "/averaged.dat", np.astype(summed, uint32)) 
        radial = radialSum(summed)
        np.savetxt(prefix + "/processed.dat")

    def plot(condition = None):
        def plotOne(conditionStr):
            # TODO: implement this format
            r, intensity = np.genfromtxt("runs/" + self.name + "/" + conditionStr + "processed.dat")
            plt.plot(r, intensity, conditionStr)
        if condition is None:
            plotOne(self.condition)
        else:
            plotOne(condition)

