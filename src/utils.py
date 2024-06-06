from scipy.signal import savgol_filter, medfilt
import numpy as np
import pickle
import os


def normalise(data, method, window_length, **kwargs):
    if window_length % 2 == 0:
        raise ValueError('window_length must be an odd integer')

    if method == 'savgol':
        filter = savgol_filter(
            data, window_length=window_length, polyorder=kwargs['polyorder']
        )
    elif method == 'median':
        filter = medfilt(data, kernel_size=window_length)
    else:
        raise ValueError(f'Normalization method {method} not recognized')

    normalized_data = data / filter

    return normalized_data


def identify_outliers(data, threshold=3.0):
    # Calculate the median and standard deviation of the data
    median = np.median(data)
    std = np.std(data)

    # Identify outliers based on the threshold
    outliers = np.abs(data - median) > threshold * std

    return outliers


def save_pickle(obj: object, path: str) -> None:
    """
    Save a Python object to a pickle file.

    Parameters
    ----------
    obj : object
        The Python object to save.
    path : str
        The path to save the object to.

    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> object:
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    path : str
        The path to the pickle file.

    Returns
    -------
    object
        The Python object loaded from the pickle file.

    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_dir_if_required(script_filepath: str, dir_name: str) -> str:
    cwd = os.path.dirname(os.path.realpath(script_filepath))
    dir_to_make = os.path.join(cwd, dir_name)

    if not os.path.exists(dir_to_make):
        os.makedirs(dir_to_make)

    return dir_to_make
