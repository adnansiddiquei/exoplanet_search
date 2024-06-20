import matplotlib.pyplot as plt

from src.utils import (
    create_dir_if_required,
    save_dill,
    identify_outliers,
    plot_folded_lightcurve,
    normalise,
    plot_periodogram,
)
from transitleastsquares import transitleastsquares, transit_mask
import pandas as pd
import os
import numpy as np


if __name__ == '__main__':
    out_dir = create_dir_if_required(__file__, 'out')
    cwd = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(cwd, '../../data')
    data_file = os.path.join(data_dir, 'ex1_tess_lc.txt')

    tess_data = pd.read_csv(
        data_file,
        sep=' ',
        header=None,
        names=['time', 'flux', 'flag', 'flux_error'],
        skiprows=4,
    )

    # remove outliers
    outliers = identify_outliers(tess_data['flux'], threshold=3.0)
    tess_data = tess_data[~outliers].reset_index(drop=True)

    # compute the strongest period, and save it down into output folder
    if not os.path.exists(os.path.join(out_dir, 'tls_results_0.pkl')):
        print('Computing tls_results_0.pkl...')
        model = transitleastsquares(
            tess_data['time'], tess_data['flux'], dy=tess_data['flux_error']
        )
        tls_results_0 = model.power()
        save_dill(tls_results_0, os.path.join(out_dir, 'tls_results_0.pkl'))
        print('Computed tls_results_0.pkl. Saved to out folder.')
    else:  # unless we have already computed it, then load it up
        tls_results_0 = pd.read_pickle(os.path.join(out_dir, 'tls_results_0.pkl'))
        print('Loaded tls_results_0.pkl')

    # plot the folded lightcurve, this is all solar activity
    fig, ax = plot_folded_lightcurve(tls_results_0, plot_model=False)
    fig.savefig(os.path.join(out_dir, 'tls_results_0.png'), bbox_inches='tight')

    # the strongest period is solar activity, we need to normalise it out
    time_diff = tess_data['time'].diff()[1]
    optimal_window_length = round((tls_results_0.period / time_diff) / 2)
    optimal_window_length += 1 if optimal_window_length % 2 == 0 else 0  # ensure odd

    tess_data = normalise(
        tess_data, window_length=optimal_window_length, method='savgol', polyorder=2
    )

    for i in range(2):
        # compute the strongest period, and save it down into output folder
        out_file = os.path.join(out_dir, f'tls_results_{i+1}.pkl')
        if not os.path.exists(out_file):
            print(f'Computing tls_results_{i+1}.pkl...')
            # First do a run through to find the best period with TLS
            model = transitleastsquares(
                tess_data['time'], tess_data['flux'], dy=tess_data['flux_error']
            )
            tls_results_initial = model.power(period_max=25)
            save_dill(
                tls_results_initial,
                os.path.join(out_dir, f'tls_results_{i+1}_initial.pkl'),
            )

            # Due to a bug, we now need to remove the entire gap in the middle of the data, but we need to remove the
            # gap as a multiple of the period
            data = tess_data.copy()
            last_in_group1 = tess_data[tess_data['time'] < 1500].index[-1]
            first_in_group2 = tess_data[tess_data['time'] > 1500].index[0]
            time_diff = (
                tess_data.loc[first_in_group2, 'time']
                - tess_data.loc[last_in_group1, 'time']
            )
            time_to_delete = (
                time_diff // tls_results_initial.period
            ) * tls_results_initial.period
            data['time'] = np.where(
                data['time'] > 1500, data['time'] - time_to_delete, data['time']
            )

            # now we can run TLS again, on the data without the gap
            model = transitleastsquares(
                data['time'], data['flux'], dy=data['flux_error']
            )
            tls_results = model.power(
                period_max=tls_results_initial.period + 0.4,
                period_min=tls_results_initial.period - 0.4,
            )

            save_dill(tls_results, out_file)
            print(f'Computed tls_results_{i+1}.pkl. Saved to out folder.')
        else:  # unless we have already computed it, then load it up
            tls_results = pd.read_pickle(out_file)
            tls_results_initial = pd.read_pickle(
                os.path.join(out_dir, f'tls_results_{i+1}_initial.pkl')
            )
            print(f'Loaded tls_results_{i+1}.pkl')

        # now we plot all the results
        duration = tls_results.duration

        # plot and save the periodogram
        plt.cla()
        fig, ax = plot_periodogram(tls_results_initial)
        fig.savefig(
            os.path.join(out_dir, f'tls_results_{i+1}_periodogram.png'),
            bbox_inches='tight',
        )

        plt.cla()
        fig, ax = plot_folded_lightcurve(tls_results, plot_model=True)
        plt.xlim(-duration * 3, duration * 3)
        ylims = ax.get_ylim()
        plt.text(
            -duration * 2.9,
            ylims[0] + (ylims[1] - ylims[0]) * 0.95,
            rf'Period: {tls_results.period:.5f}',
        )
        plt.text(
            -duration * 2.9,
            ylims[0] + (ylims[1] - ylims[0]) * 0.90,
            rf'Depth: {tls_results.depth:.5f}',
        )
        fig.savefig(
            os.path.join(out_dir, f'tls_results_{i+1}.png'), bbox_inches='tight'
        )

        # we now delete the in-transit data so the next iteration can find the second planet
        intransit = transit_mask(
            tess_data['time'],
            tls_results_initial.period,
            3.5 * tls_results_initial.duration,
            tls_results_initial.T0,
        )

        tess_data = tess_data[~intransit].reset_index(drop=True)
