import numpy as np
import noisereduce as nr
import scipy.signal
import matplotlib.pyplot as plt
def clearNoise(arr):
	reduced_noise = nr.reduce_noise(y=arr, sr=16000)
	return reduced_noise * 2


def max_normalized_cross_correlation(a, b):
    """
    Compute the maximum normalized cross-correlation and corresponding lag.

    The normalized cross-correlation is a value between -1 and 1, where:
    - 1 indicates a perfect match at a certain lag.
    - -1 indicates a perfect inverse match.
    - 0 indicates no linear relationship.
    """

    signal_a = (a - np.mean(a)) / np.std(a)
    signal_b = (b - np.mean(b)) / np.std(b)

    correlation = scipy.signal.correlate(signal_b, signal_a, mode='full')
   
    # Normalize by length to get correlation coefficient range
    norm_factor = len(signal_a)
    correlation /= norm_factor

    # Find max correlation and corresponding lag
    max_corr = np.max(correlation)
    lag = np.argmax(correlation) - len(signal_a) + 1

    return max_corr, lag
