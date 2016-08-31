from utils.mpl_plotly import plt
import processLDV as ldv
from scipy.ndimage.filters import gaussian_filter as gfilt
import numpy as np
from collections import namedtuple
import glob

LDV_DATA_DIR = '../ldv/'

def get_time_series(run_name):
    patterns = [LDV_DATA_DIR + "C3%s.*", LDV_DATA_DIR + "C4%s.*"]
    glob_matches = [glob.glob(pat) for pat in patterns]
    for glist in glob_matches:
        if len(glist) > 1:
            raise ValueError("%s: more than one match found")
    names = [matchlist[0] for matchlist in glob_matches]
    #names = [LDV_DATA_DIR + pat % run_name for pat in patterns]
    return ldv.processLDV(*names)[0]

def get_fft(run_name):
    patterns = ["C3%s.txt", "C4%s.txt"]
    names = [LDV_DATA_DIR + pat % run_name for pat in patterns]
    return ldv.processLDV(*names)[1]

def _rebin_spectrum(x, y, rebin = 5):
    """
    Rebin `x` and `y` into arrays of length `int(len(x)/rebin)`. The
    highest-x bin is dropped in case len(x) isn't a multiple of rebin.
    x is assumed to be evenly-spaced and in ascending order.
    Returns: x, y
    """
    assert type(x) == np.ndarray
    assert type(y) == np.ndarray
    def group(arr1d):
        """
        op: a function to evaluate on each new bin that returns a numeric value.
        >>> rebin = 3
        >>> group(range(10))
        [1.0, 4.0, 7.0]
        """
        newsize = (len(arr1d) / rebin)
        cropped = arr1d[:newsize * rebin]
        return np.mean(cropped.reshape((newsize, rebin)), axis = 1)
#        import itertools
#        i = itertools.count()
#        def key(dummy):
#            xindx = i.next()
#            return int(xindx/rebin)
#        return [op(list(values)) for groupnum, values in itertools.groupby(arr1d, key = key)][:-1]
    return group(x), group(y)

def resample(x, y):
    """
    arr should be an iterable of length 2 containing two arrays: [x, y]
    """
    maxsize = 20000
    dimy = len(x)
    if dimy <= maxsize:
        return x, y
    else:
        factor_reduction = dimy / maxsize
        return _rebin_spectrum(x, y, rebin = factor_reduction)


#def plot_several_time_series(*run_names):
#    [plt.plot(*get_time_series(run_name), label = run_name) for run_name in run_names]
#    plt.xlabel('Time (ns)')
#    plt.ylabel('Distance (microns)')
#    plt.show()
#    return 

#def plot_several_ffts(*run_names):
#    curves = [resample(*get_fft(run_name)) for run_name in run_names]
#    [plt.plot(*curve, label = label) for curve, label in zip(curves, run_names)]
#    plt.xlabel('Frequency (Hz)')
#    plt.show()
#    return

LDVdata = namedtuple("LDVdata", ["timeseries", "fft"])

def get_LDVdata(run_name, f_filter = None):
    patterns = [LDV_DATA_DIR + "C3%s.*", LDV_DATA_DIR + "C4%s.*"]
    glob_matches = [glob.glob(pat % run_name)for pat in patterns]
    for glist in glob_matches:
        if len(glist) != 1:
            raise ValueError("%s: more than one match found")
    names = [matchlist[0] for matchlist in glob_matches]
#    names = [matchlist[0] for matchlist in glob_matches]
#    patterns = ["C3%s.txt", "C4%s.txt"]
#    names = [LDV_DATA_DIR + pat % run_name for pat in patterns]
    x, y = ldv.processLDV(*names)
    x_times, x_positions = x
    if f_filter is not None:
        return LDVdata([x_times, f_filter(x_positions)], y)
    else:
        return LDVdata(x, y)
    #return LDVdata(*ldv.processLDV(*names))

def resample_ldvdata(ldv_data):
    return LDVdata(resample(*ldv_data.timeseries), resample(*ldv_data.fft))

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def make_bandpass_filter(lowcut = 5e3, highcut = 2e6, fs = 1e7, order = 3):
    from scipy.signal import firwin, lfilter
    def filt(x):
        return butter_bandpass_filter(x, lowcut, highcut, fs = fs, order = order)
#        h=firwin( numtaps=numtaps, cutoff=cutoff, nyq = nyq)
#        return lfilter( h, 1.0, x)
    return filt

def plot_several(run_names, f_filter = None, series = True, fft = True, plotter = plt):
    assert type(run_names) == list
    #if downsample:
    #raw = get_LDVdata(run_name, f_filter = f_filter)
    ldvdata = [resample_ldvdata(get_LDVdata(run_name, f_filter = f_filter)) for run_name in run_names]
    #else:
    #    ldvdata = [get_LDVdata(run_name) for run_name in run_names]
    time_series, ffts = zip(*ldvdata)
    if series:
        [plotter.plot(*curve, label = label) for curve, label in zip(time_series, run_names)]
        plotter.xlabel('Time (ns)')
        plotter.ylabel('Distance (microns)')
        plotter.show()
    if fft:
        plotter.yscale('log')
        [plotter.plot(*curve, label = label) for curve, label in zip(ffts, run_names)]
        plotter.xlabel('Frequency (Hz)')
        plotter.show()
    return ldvdata
