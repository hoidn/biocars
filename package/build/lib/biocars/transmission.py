import numpy as np
import rayonix_process as rp
import mu
import scipy.optimize as opt

"""
Module for analyzing energy spectra transmitted through a single-crystal target
of known composition
"""

# TODO: add a function that calculates an energy difference spectrum, given
# incident and transmitted spectra
# TODO: add a utility for converting "volume" fractions to atomic fractions

def correct_attenuation(beam_spectrum, composition_dict, density = None, thickness = None, bg_sub = False, target = None):
    """
    Inputs:
        beam_spectrum: array of shape 2 x n
        composition_dict: keys, values are element identifier, composition 
            fraction ("volume fraction" of a species based on the density of its
            elemental form)
        density: sample density, in g/cm^3
        thickness: sample thickness, in cm
        bg_sub: if True, perform background subtraction of beam_spectrum before
        attenuating it.
    Applies attenuation of beam_spectrum due to the specified target material
    and returns the resulting spectrum
    """
    if density is not None:
        if thickness is None:
            raise ValueError("both or neither of density and thickness must be provided")
    else:
        if target is None:
            raise ValueError("target spectrum must be provided if density and thickness are None")

    energies, intensities = beam_spectrum
    if bg_sub:
        intensities -= rp.default_bgsubtraction(energies, intensities)(energies)
    #if target is not None:
    partial_spectrum = np.zeros(len(energies))
    # vector for the weighted sum of attenuation coefficients
    partial_mu = np.zeros(len(energies))
    attenuated_intensities = np.zeros(len(energies))
    for elt, fraction in composition_dict.iteritems():
        mufunc = lambda e: fraction * mu.ElementData(elt).mu(e)
        partial_mu += mufunc(energies)
        if density:
            effective_thickness = thickness * density / mu.ElementData(elt).density
            attenuated_intensities += intensities * np.exp(-effective_thickness * mufunc(energies))

    if not density: # Need to fit for the sample thickness if it's not provided
        target_energies, target_intensities = target
        effective_thickness = optimize_thickness_to_match_l1(intensities, target_intensities, partial_mu)
        attenuated_intensities = intensities * np.exp(-effective_thickness * 1e-4 * partial_mu)
    return [energies, attenuated_intensities]

def scale_to_match_l1(source_intensities, target_intensities):
    """
    Evaluate the value by which to multiply source_intensities which gives the
    smallest L1 norm difference between the resulting vector and
    target_intensities.
    """
    def l1(scale):
        return np.sum(np.abs(scale * source_intensities - target_intensities))
    return opt.minimize_scalar(l1).x

def optimize_thickness_to_match_l1(source_intensities, target_intensities, mu):
    """
    Inputs:
        source_intensities: 1 x n array; original (unattenuated) intensities
            incident on the sample
        target_intensities: 1 x n array; vector of intensities to fit to
        mu: 1 x n (where n is the same here as for source_intensities and
            target_intensities) array of material attenuation coefficients,
            in 1/cm
    Returns the sample thickness for which the correponding attenuation of
    source_intensities best fits target_intensities. Units are microns
    """
    def l1(thickness):
        """
        thickness units: microns
        """
        return np.sum(np.abs(source_intensities * np.exp(-thickness * 1e-4 * mu) - target_intensities))
    return opt.minimize_scalar(l1).x
