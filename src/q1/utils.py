from scipy.signal import savgol_filter
from scipy.signal import medfilt
import numpy as np


def normalise(data, method, window_length, **kwargs):
    if window_length % 2 == 0:
        raise ValueError('window_length must be an odd integer')

    if method == 'savgol':
        filter = savgol_filter(
            data['flux'], window_length=window_length, polyorder=kwargs['polyorder']
        )
    elif method == 'median':
        filter = medfilt(data['flux'], kernel_size=window_length)
    else:
        raise ValueError(f'Normalization method {method} not recognized')

    data['flux'] = data['flux'] / filter
    data['flux_error'] = data['flux_error'] / filter

    return data


def identify_outliers(data, threshold=3.0):
    # Calculate the median and standard deviation of the data
    median = np.median(data)
    std = np.std(data)

    # Identify outliers based on the threshold
    outliers = np.abs(data - median) > threshold * std

    return outliers
