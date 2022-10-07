import numpy as np

from scipy import signal


def lowpass_filter(data, lowcut=10e3, fs=100e3, order=4):
    """
    Apply low-pass filter using scipy's signal functions.
    """
    b, a = signal.butter(order, lowcut, fs=fs, btype='lowpass')
    data_filt = signal.lfilter(b, a, data)

    return data_filt


def remove_artefacts(data):
    """
    Remove 15 kHz noise in data and replace with mean of recording.
    """
    channel_mean = np.mean(data)
    limit = 3 * np.std(data)

    data[data > limit] = channel_mean
    data[data < -limit] = channel_mean

    return data