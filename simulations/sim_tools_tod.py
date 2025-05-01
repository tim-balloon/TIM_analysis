"""
simulations/python/sim_tools_tod.py
Contains necessary function for performing TOD simulations.
"""

import numpy as np, sys, os, warnings

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def detector_noise_model(noise_level, fknee, alphaknee, total_samples, sample_freq):



    """
    TOD noise model (1/f, white noise, and total).

    .. math::
        P(f) = A^2 \left[ 1+ \left( \\frac{f_{\\rm knee}}{f}\\right)^{\\alpha_{\\rm knee}} \\right].

    Parameters
    ----------
    noise_level: float
        White noise level (A in the above equation) for the detector.
    fknee: float
        Knee frequency for 1/f  (f_knee in the above equation).
    alphaknee: float
        Slope for 1/f (alpha_knee in the above equation).
    total_samples: int
        Total TOD samples.
    sample_freq: float
        Sampling frequency of the TOD.

    Returns
    -------
    freq: array
        TOD frequencies length = total_samples
    noise_powspec: array
        Noise power spectrum.
    noise_powspec_one_over_f: array
        1/f portion of the noise power spectrum.
    noise_powspec_white: array
        White noise portion of the noise power spectrum.
    """

    freq = np.fft.fftfreq(total_samples, 1/sample_freq) #TOD frequencies.
    noise_powspec_white = np.tile( noise_level**2., len(freq) )
    noise_powspec_one_over_f = noise_level**2. * (fknee/freq)**alphaknee
    noise_powspec = noise_powspec_one_over_f + noise_powspec_white
    
    return freq, noise_powspec, noise_powspec_one_over_f, noise_powspec_white

def get_correlated_powspec(rho, powspec1, powespec2):

    """
    Returns correalted noise given rho and the auto-power spectrum of two detectors.
    
    .. math::
        P_{ij} = \\rho_{ij} \sqrt{P_{ii} P_{jj}}

    Parameters
    ----------
    rho: float
        Correlation coefficient (rho_ij).
    powspec1: array
        Power spectrum 1 (P_ii).
    powspec2: array
        Power spectrum 2 (P_jj).

    Returns
    -------
    corr_powspec: array
        Cross-power spectrum.
    """

    corr_powspec = rho * np.sqrt( powspec1 * powespec2)
    
    return corr_powspec 