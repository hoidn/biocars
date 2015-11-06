import numpy as np
from scipy.ndimage.filters import gaussian_filter
import imp
import sum_images_parallel as sip
import glob

imp.reload(sip)

#def kapton_backgnd(attenuation, npulses, nframes = 1):

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
    return gaussian_filter(sip.sum_radial_densities(filenames, average = True), sigma = sigma)

@memoize
def dark_subtraction(npulses, nframes = 1):
    lookup_dict = {1: radial_mean("background_exposures/dark_frames/dark_1p*"), 3: radial_mean("background_exposures/dark_frames/dark_3p*"), 10: radial_mean("background_exposures/dark_frames/dark_10p*"), 1000: radial_mean("background_exposures/dark_frames/dark_1000p*")}
    dark_500_interpolation = (lookup_dict[1000] - lookup_dict[1])/2 + lookup_dict[1]
    lookup_dict[500] = dark_500_interpolation

    return lookup_dict[npulses] * nframes

@memoize
def air_scatter(npulses, attenuation, nframes = 1):
    npulses_ref = 1000. #number of pulses in the air scatter refernce images
    def extract_intensity(path):
        raw =  np.genfromtxt(path)[1]
        return raw - dark_subtraction(npulses_ref)
    lookup_dict = {1:extract_intensity( "background_exposures/beam_on_no_sample_2/radial_integrations/1x_1000p_001.mccdprocessed.dat"), 10: extract_intensity("background_exposures/beam_on_no_sample_2/radial_integrations/10x_1000p.mccdprocessed.dat"), 300: extract_intensity("background_exposures/beam_on_no_sample_2/radial_integrations/300x_1000p.mccdprocessed.dat")}
    return nframes * (npulses/npulses_ref) * lookup_dict[attenuation]

#@memoize
#def kapton_background(npulses, attenuation = 1, nframes = 1):
#    kapton_file = "test_diffraction/radial_integrations/v2o5nano_10x_1000_kapton_background_2.mccdprocessed.dat"
#    kapton_attenuation = 10
#    kapton_pulses = 1000
#    raw = np.genfromtxt(kapton_file)[1]
#    dark_background = dark_subtraction(kapton_pulses)
#    air_background = air_scatter(kapton_pulses, kapton_attenuation)
#    kapton_alone = raw - dark_background - air_background
#    return kapton_alone * float(npulses/kapton_pulses) * (kapton_attenuation/attenuation) * nframes


@memoize
def kapton_background(npulses, attenuation = 1, nframes = 1):
    kapton_file = "test_diffraction/radial_integrations/v2o5nano_10x_1000_kapton_background_2.mccdprocessed.dat"
    kapton_attenuation = 10
    kapton_pulses = 1000
    raw = np.genfromtxt(kapton_file)[1] - dark_subtraction(kapton_pulses)
    return raw * nframes * (float(npulses)/kapton_pulses) * (float(kapton_attenuation) / attenuation)


def bgsubtract(npulses, attenuation, intensities, nframes):
    darksub = dark_subtraction(npulses, nframes = nframes)
    kaptonsub = kapton_background(npulses, attenuation, nframes = nframes)
