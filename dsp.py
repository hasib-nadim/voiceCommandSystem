
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path

import noisereduce as nr
import scipy.signal 

def clearNoise(arr):
	reduced_noise = nr.reduce_noise(y=arr, sr=16000)
	return reduced_noise  * 0.90  # to avoid clipping


def max_normalized_cross_correlation(a, b):
    """
    Compute the maximum normalized cross-correlation and corresponding lag.

    The normalized cross-correlation is a value between -1 and 1, where:
    - 1 indicates a perfect match at a certain lag.
    - -1 indicates a perfect inverse match.
    - 0 indicates no linear relationship.
    """
    encoder = VoiceEncoder()
    emb_a = encoder.embed_utterance(a)
    emb_b = encoder.embed_utterance(b)
    a = emb_a.flatten()
    b = emb_b.flatten()
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    max_corr = similarity
    return max_corr, 0
