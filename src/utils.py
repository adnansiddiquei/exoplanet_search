from scipy.signal import savgol_filter, medfilt
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


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


def plot_folded_lightcurve(tls_results, plot_model=True):
    fig, ax = plt.subplots()

    plt.scatter(
        (tls_results.folded_phase * tls_results.period) - (tls_results.period / 2),
        tls_results.folded_y,
        color='blue',
        s=1,
        alpha=0.5,
        zorder=2,
    )

    if plot_model:
        plt.plot(
            (tls_results.model_folded_phase * tls_results.period)
            - (tls_results.period / 2),
            tls_results.model_folded_model,
            color='red',
        )

    plt.xlabel('Phase')
    plt.ylabel('Relative flux')

    return fig, ax


def calc_transit_duration(tls_results):
    tmp = (
        tls_results.model_folded_phase[tls_results.model_folded_model < 1]
        * tls_results.period
    )
    return tmp[-1] - tmp[0]


def plot_periodogram(tls_results):
    fig, ax = plt.subplots(figsize=(10, 5))

    periods = tls_results.periods  # Array of tested periods
    power = tls_results.power  # Corresponding power for each period
    mask = periods < (tls_results.period * 4.2)

    # Plot the periodogram
    plt.plot(periods[mask], power[mask], 'k-')
    plt.xlabel('Period (days)')
    plt.ylabel('Power')

    plt.axvline(tls_results.period, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(tls_results.period * 2, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(tls_results.period * 3, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(tls_results.period * 4, color='red', linestyle='dashed', linewidth=1)

    return fig, ax
