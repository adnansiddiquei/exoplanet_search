import matplotlib.pyplot as plt

from src.utils import (
    create_dir_if_required,
    save_pickle,
    identify_outliers,
    plot_folded_lightcurve,
    normalise,
    calc_transit_duration,
    plot_periodogram,
)
from transitleastsquares import transitleastsquares, transit_mask
import pandas as pd
import os


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
        save_pickle(tls_results_0, os.path.join(out_dir, 'tls_results_0.pkl'))
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

    for i in range(3):
        # compute the strongest period, and save it down into output folder
        out_file = os.path.join(out_dir, f'tls_results_{i+1}.pkl')
        if not os.path.exists(out_file):
            print(f'Computing tls_results_{i+1}.pkl...')
            model = transitleastsquares(
                tess_data['time'], tess_data['flux'], dy=tess_data['flux_error']
            )
            tls_results = model.power()
            save_pickle(tls_results, out_file)
            print(f'Computed tls_results_{i+1}.pkl. Saved to out folder.')
        else:  # unless we have already computed it, then load it up
            tls_results = pd.read_pickle(out_file)
            print(f'Loaded tls_results_{i+1}.pkl')

        duration = calc_transit_duration(tls_results)

        # plot and save the periodogram
        plt.cla()
        fig, ax = plot_periodogram(tls_results)
        fig.savefig(
            os.path.join(out_dir, f'tls_results_{i+1}_periodogram.png'),
            bbox_inches='tight',
        )

        # plot the folded lightcurve, this is all solar activity
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

        intransit = transit_mask(
            tess_data['time'], tls_results.period, 3.5 * duration, tls_results.T0
        )
        tess_data = tess_data[~intransit].reset_index(drop=True)
