"""
LPC encoding
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

from scipy.signal import resample
from scipy.signal.windows import hann 
from scipy.linalg import solve_toeplitz, toeplitz

from math import *


# -----------------------------------------------------------------------------
# Windowing
# -----------------------------------------------------------------------------


def blocks_decomposition(x, w, R = 0.5):

    """
    Performs the windowing of the signal
    
    Parameters
    ----------
    
    x: numpy array
      single channel signal
    w: numpy array
      window
    R: float (default: 0.5)
      overlapping between subsequent windows
    
    Return
    ------
    
    out: numpy array
      block decomposition of the signal
    """
    
    # TODO

    

def blocks_reconstruction(blocks, w, signal_size, R = 0.5):

    """
    Reconstruct a signal from overlapping blocks
    
    Parameters
    ----------
    
    blocks: numpy array
      signal segments. blocks[i,:] contains the i-th windowed
      segment of the speech signal
    w: numpy array
      window
    signal_size: int
      size of the original signal
    R: float (default: 0.5)
      overlapping between subsequent windows
    
    Return
    ------
    
    out: numpy array
      reconstructed signal
    """

    # TODO


# -----------------------------------------------------------------------------
# Linear Predictive coding
# -----------------------------------------------------------------------------

def autocovariance(x, k):

    """
    Estimates the autocovariance C[k] of signal x
    
    Parameters
    ----------
    
    x: numpy array
      speech segment to be encoded
    k: int
      covariance index
    """
    
    # TODO


def lpc_encode(x, p):

    """
    Linear predictive coding 
    
    Predicts the coefficient of the linear filter used to describe the 
    vocal track
    
    Parameters
    ----------
    
    x: numpy array
      segment of the speech signal
    p: int
      number of coefficients in the filter
      
    Returns
    -------
    
    out: tuple (coef, e, g)
      coefs: numpy array
        filter coefficients
      prediction: numpy array
        lpc prediction
    """
    
    # TODO

    

def lpc_decode(coefs, source):

    """
    Synthesizes a speech segment using the LPC filter and an excitation source
    
    Parameters
    ----------

    coefs: numpy array
      filter coefficients
        
    source: numpy array
      excitation signal
    
    Returns
    -------
    
    out: numpy array
      synthesized segment
    """

    # TODO



# -----------------------------------------------------------------------------
# Pitch detection
# -----------------------------------------------------------------------------

def compute_cepstrum(x):

    """
    Computes the cepstrum of the signal
    
    Parameters
    ----------
    
    x: numpy array
      signal
      
    Return
    ------
    
    out: nunmpy array
      signal cepstrum
    """

    # TODO



def cepstrum_pitch_detection(cepstrum, threshold, max_rate, sample_rate):

    """
    Cepstrum based pitch detection
    
    Parameters
    ----------
    
    cepstrum: numpy array
      cepstrum of the signal segment
    threshold: float 
      threshold used to distinguish between voiced/unvoiced segments
    max_rate: float
      maximal pitch frequency
    sample_rate: float
      sample rate of the signal
      
    Return
    ------
    
    out: int
      estimated pitch. For an unvoiced segment, the pitch is set to zero
    """

    # TODO

