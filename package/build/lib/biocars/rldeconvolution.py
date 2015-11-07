import numpy as np
#TODO: these two files are good candidates for a utilities package
import ipdb
import scipy.ndimage.filters as filt
import matplotlib.pyplot as plt
from scipy import interpolate
import atexit, dill
import hashlib
import collections
import utils

np.seterr(invalid='raise')

TINY = 1e-9

def make_hashable(obj):
    """
    return hash of an object that supports python's buffer protocol
    """
    return hashlib.sha1(obj).hexdigest()

def hashable_dict(d):
    """
    try to make a dict convertible into a frozen set by 
    replacing any values that aren't hashable but support the 
    python buffer protocol by their sha1 hashes
    """
    #TODO: replace type check by check for object's bufferability
    for k, v in d.iteritems():
        if isinstance(v, np.ndarray):
            d[k] = make_hashable(v)
    return d

##TODO: these two files are good candidates for a utilities package
#def persist_to_file(file_name):
#
#    try:
#        cache = dill.load(open(file_name, 'r'))
#    except (IOError, ValueError):
#        cache = {}
#
#    atexit.register(lambda: dill.dump(cache, open(file_name, 'w')))
#
#    def decorator(func):
#        #check if function is a closure and if so construct a dict of its bindings
#        if func.func_code.co_freevars:
#            closure_dict = hashable_dict(dict(zip(func.func_code.co_freevars, (c.cell_contents for c in func.func_closure))))
#        else:
#            closure_dict = {}
#        def new_func(*args, **kwargs):
#            key = (args, frozenset(kwargs.items()), frozenset(closure_dict.items()))
#            if key not in cache:
#                cache[key] = func(*key[0], **{k: v for k, v in key[1]})
#            return cache[key]
#        return new_func
#
#    return decorator

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return 0.
        elif x > xs[-1]:
            return 0.
        else:
            return interpolator(x)

    def ufunclike(x):
        ys = np.zeros((len(x)))
        good_indices = np.where(np.logical_and(x > xs[0], x < xs[-1]))[0]
        ys[good_indices] = interpolator(x[good_indices])
        return ys
        #return np.array(map(pointwise, np.array(xs)))

    return ufunclike

def my_convolve(arr1, arr2, padding_length = 50):
    """
    arr2 is assumed to be the (not necessarily normalized) response function
    the response function is assumed to be peaked at the central index

    length of arr1 MUST BE ODD?
    """
    return np.convolve(arr1, arr2)[len(arr1)/2:-(len(arr1)/2 - 1)]

def convolve_matrix_vector(mat1, arr):
    """
    convolve an (n + m - 1)xn matrix with a vector of length m, equivalent to
    a convolution where the kerenel varies with the independent variable. 

    The convolution samples each point of overlap, resulting in an array
    of length n + m - 1, where n is the length of the convolution kernel.
    This determines the number of rows that that the convolution matrix 
    must have
    """
    mat1, mat2 = pre_convolution_mask_matrices(mat1, arr)
    convolved = np.array([np.dot(row, arr) for row in mat1])
    return pad_spectrum(mat1, convolved)

def pre_convolution_mask_matrices(mat1, mat2):
    # TODO: figure out if this step is actually necessary
    return mat1, mat2

def pre_convolution_pad_matrices(mat1, mat2):
#    ipdb.set_trace()
    width = len(mat2) + np.shape(mat1)[1] - 1
    if np.shape(mat1)[0] != width:
        raise ValueError("wrong number of rows in convolution matrix")
    len1, len2 = np.shape(mat1)[1], np.shape(mat2)[0]
    wid1 = np.shape(mat1)[0]
    #wid1, wid2 = np.shape(mat1)[0], np.shape(mat2)[0]
    if len2 < len1:
        raise ValueError("second dimension of second matrix must be smaller")
    mat2 = np.hstack((np.zeros((len1 - 1)), mat2, np.zeros((len1 - 1))))
    mat1 = np.hstack((mat1, np.zeros((wid1, len2 + len1 - 2))))
    roll_matrix(mat1)
    return mat1, mat2

def pad_spectrum(mat1, mat2):
    """
    Return a modified copy of the vector mat2 with dimensions expanded so 
    that it can be convolved with mat1
    """
    padsize = np.shape(mat1)[1] - len(mat2)
    if padsize < 0:
        raise ValueError("mat2 is not broadcastable with mat1")
    if padsize % 2:
        return np.hstack((np.zeros((1 + padsize / 2)), mat2, np.zeros((padsize / 2))))
    else:
        return np.hstack((np.zeros((padsize / 2)), mat2, np.zeros((padsize / 2))))

def stackself(vec, num):
    """
    vstack an array with itself num times
    """
    return np.vstack((vec for _ in range(num)))

def roll_matrix(mat):
    """
    cyclically permute each row by the row index.
    """
    for i, row in enumerate(mat):
        mat[i] =  np.roll(row, i)
    return mat

def pad_data(arr, padding_length = None, endpoint_length = 10):
    if padding_length is None:
        padding_length = len(arr)
    pad_value_start = np.mean(arr[:endpoint_length])
    pad_value_end = np.mean(arr[-endpoint_length - 1:])
    arr_padded = np.concatenate((np.repeat(pad_value_start, padding_length), arr, np.repeat(pad_value_end, padding_length)))
    return arr_padded

def unpad_data(arr, padding_length = None, endpoint_length = 10, asym = False):
    if padding_length is None:
        padding_length = len(arr)/3
    if asym:
        return arr[padding_length:-padding_length + 1]
    else:
        return arr[padding_length:-padding_length]

def make_estimator(measuredx, measuredy, kernelx, kernely, grid_spacing, convolution_mode = 'vector', regrid = True, kernel_width = 1600, smooth_size = 1.):
    """
    return a function that takes a number of iterations and an optional starting estimate for the "real" spectrum and returns an improved estimate
    measuredx, measuredy: the target data
    kernel: the instrumental resolution function, takes an array of x values
    """
    xmin, xmax = measuredx[0], measuredx[-1]
    if (xmin != min(measuredx)) or (xmax != max(measuredx)):
        raise ValueError("x array must be ordered")
    if regrid:
        num_points_new_grid = int((xmax - xmin)/grid_spacing)
        num_points_new_grid += num_points_new_grid % 2 #make even
        newx = np.linspace(xmin - (xmax - xmin), xmax + (xmax - xmin), 3 * num_points_new_grid)
    else:
        newx = measuredx
    #TODO: IS THIS PROPERLY CENTERED?
    midpoint = newx[len(newx)/2]
    measured_interpolated = extrap1d(interpolate.interp1d(measuredx, measuredy))
    #newy = pad_data(measured_interpolated(newx))
    newy = measured_interpolated(newx)
    newy_padded = padrl(newy, kernel_width)
    if convolution_mode == 'vector':
        kernel_y_reversed = kernel_y[::-1]
        @utils.eager_persist_to_file("cache/estimator_vector.p")
        def estimator(num_iterations, starting_estimate = newy):
            current_estimate = starting_estimate.copy()
            for i in range(num_iterations):
                convolved_object = my_convolve(kernel_y, current_estimate)
                current_estimate = current_estimate * my_convolve(kernel_y_reversed, newy/(TINY + convolved_object))
            return newx, unpad_data(current_estimate, padding_length = len(newx))
    elif convolution_mode == 'matrix':
        kernel_mat = make_deconvolution_matrix([kernelx, kernely], [newx, newy], kernel_width)
        kernel_mat_expanded, newy_expanded = pre_convolution_pad_matrices(kernel_mat, newy)
        #TODO: ascontiguousarray: necessary?
        kernel_mat_expanded_reversed = pre_convolution_pad_matrices(np.ascontiguousarray(np.fliplr(kernel_mat)), newy)[0]
        #ipdb.set_trace()
        @utils.eager_persist_to_file("cache/estimator_matrix/prefix")
        def estimator(num_iterations, current_estimate = newy_expanded):
            for i in range(num_iterations):
                convolved_object = convolve_matrix_vector(kernel_mat_expanded, current_estimate)
                current_estimate = current_estimate * convolve_matrix_vector(kernel_mat_expanded_reversed, newy_expanded/(TINY + convolved_object))
                current_estimate = filt.gaussian_filter(current_estimate, smooth_size)
            return unpad_data(newx, padding_length = len(newx)/3), unpad_data(current_estimate, padding_length = (len(current_estimate) - len(newx)/3)/2, asym = False)
    return estimator

def smoothed_step(sigma = 10):
    y = np.concatenate((np.zeros(100), np.ones(100)))
    x = np.array(map(float, range(len(y))))
    return x, filt.gaussian_filter(y, sigma)

def smoothed_peak(sigma = 10, size = 200):
    y = np.concatenate((np.zeros(size/2 - 1), np.ones(1), np.zeros(size/2)))
    x = np.array(map(float, range(len(y))))
    return x, filt.gaussian_filter(y, sigma)

def smoothed_peak_2(mu = 0, sigma = 10):
    kern = make_gaussian_kernel(sigma = sigma, mu = mu)
    x = np.array(map(float, range(200)))
    return kern(x)


def make_gaussian_kernel(sigma = 10, mu = 0):
    func = lambda arr: np.exp((-(arr - mu)**2)/(2 * sigma**2))
    return func

def regrid(x, y, newx):
    interpolated = interpolate.interp1d(x, y)
    return extrap1d(interpolated)(newx)

def padrl(arr1d, padlen, asym = False):
    """
    given a vector and padlen, an even integer, asymmetrically pad the vector with 
    padlen/2 zeros on the left and padlen/2 - 1 zeros on the right
    """
    if asym:
        return np.concatenate((np.zeros((padlen /2)), arr1d, np.zeros((padlen/2 - 1))))
    else:
        return np.concatenate((np.zeros((padlen /2)), arr1d, np.zeros((padlen/2))))

def make_deconvolution_matrix(beam_spectrum, powder_pattern, kernel_width, nominal_energy = 12000.):
    """
    return matrix of point spread functions as a function of scattering
    angle, given an incident spectrum

    powder data is assumed to be on regular grid

    format of beam_spectrum: energy(eV), intensity
    format of powder_pattern: scattering angle (radian), intensity

    powder pattern CANNOT include theta == 0

    kernel_width is assumed to be even
    """
    
    min_angle = 0.01 #use a generic gaussian response function below this angle
    def normalize(y, x):
        normalization = np.trapz(y, x)
        return y/normalization
    if kernel_width % 2 != 0:
        raise ValueError("kernel_width must be even")
    energy, beam_intensity = beam_spectrum
    angle, powder_intensity = powder_pattern
    dtheta = angle[1] - angle[0]

    #kernel_angles = np.arange(-kernel_width/2, kernel_width/2, dtheta)
    kernel_angles = np.linspace(-kernel_width * dtheta / 2, kernel_width * dtheta / 2, kernel_width)
    num_kernel_points = len(kernel_angles)

    scale = lambda theta: 2 * np.tan(theta/2)
    #center_of_mass = np.dot(energy, beam_intensity) / np.sum(beam_intensity)
    #energy_differences = energy - center_of_mass
    energy_differences = energy - nominal_energy
    fractional_energy_differences = energy_differences / energy
    #delta = padrl(np.ones((1)), kernel_width, asym = True)
    gaussian = smoothed_peak(size = kernel_width)[1]
    gaussian = normalize(gaussian, kernel_angles)
    def make_one_kernel(ang):
        relative_angles = fractional_energy_differences * scale(ang)
        regrided =  regrid(relative_angles, beam_intensity, kernel_angles)
        #regrided =  regrid(relative_angles, beam_intensity, kernel_angles)[::-1]
        return normalize(regrided, kernel_angles)
        #return regrided/normalization
    all_kernels = np.array([make_one_kernel(ang) if ang > min_angle else gaussian for ang in angle])
    return np.vstack((np.zeros((num_kernel_points/2, num_kernel_points)),
all_kernels, np.zeros((num_kernel_points/2 - 1, num_kernel_points))))

def energy_grid(beam_spectrum, central_angle, dtheta, kernel_angles, num_kernel_points):
    energy, beam_intensity = beam_spectrum
    center_of_mass = np.dot(energy, beam_intensity) / np.sum(beam_intensity)
    energy_differences = energy - center_of_mass
    uncentered_kernel_fractional_energies = np.cumsum(1/np.tan((central_angle + kernel_angles)/2)) * dtheta
    centered_kernel_fractional_energies = uncentered_kernel_fractional_energies - uncentered_kernel_fractional_energies[num_kernel_points/2]
    beam_interp = extrap1d(interpolate.interp1d((energy - center_of_mass)[::-1] / energy, beam_intensity))
    return beam_interp(centered_kernel_fractional_energies)
