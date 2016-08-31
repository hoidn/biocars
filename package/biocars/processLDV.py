# Valenza and Hoidn, July 2016

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import pdb

from utils import utils
import readTrc

THRESHOLD = 0.5
RESOLUTION = 1e-3

RESAMPLE_RATE = 40 # 40 ns

def read_csv(path):
    return pd.read_csv(path, skiprows = 4)

def read_trc(path):
    import readTrc
    datX, datY, m = readTrc.readTrc(path)
    return pd.DataFrame(np.vstack((datX, datY)), index = ['Time', 'Ampl']).T

def trc_get_sample_time(path):
    """
    Return sample interval in ns, rounded up to the next integer.
    """
    datX, datY, m = readTrc.readTrc(path)
    return int(np.ceil(m['HORIZ_INTERVAL'] * 1e9))

def read_trace(path):
    """
    Load a .txt (csv) or .trc oscilloscope data file and return it as a pandas dataframe.
    """
    suffix = path[-3:]
    if suffix == 'txt':
        return read_csv(path)
    elif suffix == 'trc':
        return read_trc(path)
    else:
        raise ValueError("Invalid file extension: %s" % suffix)

def downsampled_trace(path):
    """
    path: path to data file

    Returns a pandas dataframe.
    """
    chan1 = read_trace(path)

    # the spacing between data points is 1 ns
    suffix = path[-3:]
    if suffix == 'txt':
        # default to 40 ns. TODO: parse this from .txt csv files.
        stride = 40
    elif suffix == 'trc':
        stride = max(RESAMPLE_RATE / trc_get_sample_time(path), 1)
    else:
        raise ValueError("Invalid file extension: %s" % suffix)

    #print 'stride: %d' % stride
    chan1=chan1[0:-1:stride] 
    return chan1


#@utils.eager_persist_to_file('cache/quad_decode/')
def quad_decode(ch1_analog, ch2_analog):
    ch1, ch2 = ch1_analog/np.max(ch1_analog) > THRESHOLD, ch2_analog/np.max(ch2_analog) > THRESHOLD
    assert type(ch1) == np.ndarray
    ch1_0 = (ch1 == 0)
    ch1_1 = (ch1 == 1)
    ch2_0 = (ch2 == 0)
    ch2_1 = (ch2 == 1)


    increments = np.zeros(len(ch1))
    state1 = ch1_0 & ch2_0
    state2 = ch1_0 & ch2_1
    state3 = ch1_1 & ch2_0
    state4 = ch1_1 & ch2_1

    state1_prev = np.roll(state1, -1)
    state2_prev = np.roll(state2, -1)
    state3_prev = np.roll(state3, -1)
    state4_prev = np.roll(state4, -1)

    increments[np.logical_and(state1, state2_prev)] += 1
    increments[np.logical_and(state1, state3_prev)] -= 1

    increments[np.logical_and(state2, state4_prev)] += 1
    increments[np.logical_and(state2, state1_prev)] -= 1

    increments[np.logical_and(state3, state1_prev)] += 1
    increments[np.logical_and(state3, state4_prev)] -= 1

    increments[np.logical_and(state4, state3_prev)] += 1
    increments[np.logical_and(state4, state2_prev)] -= 1

    return np.cumsum(increments)

# inputs: chan1, chan2 are filenames that 
# correspond to the encoder datafiles
#@output.stdout_to_file('ldv.log')
@utils.eager_persist_to_file('cache/')
def processLDV(chan1, chan2):

    chan1, chan2 = map(downsampled_trace, [chan1, chan2])

    # encoder state transition algorithm
    times = chan1[['Time']].values.T[0] * 1e9 # timestamps in ns

    counts = quad_decode(chan1['Ampl'].values, chan2['Ampl'].values)
    counts = counts * RESOLUTION

    N = counts.size
    posfft = np.fft.fft(counts)

    delt = 40e-9
    delv = 1/(delt*N)

    posfft = np.fft.fftshift(np.abs(posfft)**2)
    freq = np.arange(-N*delv/2+delv, N*delv/2, delv)
    return (times, counts), (freq, posfft)
