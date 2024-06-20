import numpy as np
import matplotlib.pyplot as plt
from .utils import invert_transform, invert_scale, fold


def plot_1planet_model(
    time, residuals, rv_err, orbital_model, samples, log_prob_samples
):
    # extract the best fit params
    best_fit_params = samples[np.argmax(log_prob_samples)]
    params = {
        key: value for key, value in zip(orbital_model.params_keys, best_fit_params)
    }

    # extract the estiamtes for the period and amplitude
    periods = samples[:, 1]
    periods = periods[(periods > 2.5) & (periods < 7.5)]
    p_mean, p_std = periods.mean(), periods.std()
    amp_mean, amp_std = np.mean(samples[:, 0]), np.std(samples[:, 0])

    # start plotting
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 0.5]}
    )

    folded_time = fold(time, params['P_0'], params['phase_offset_0'])
    x_preds = np.linspace(folded_time.min(), folded_time.max(), 1000)

    axes[0].errorbar(
        folded_time / folded_time.max(),
        residuals,
        yerr=rv_err,
        fmt='kx',
        label='RV (Stellar Noise Removed)',
    )

    axes[0].plot(
        x_preds / x_preds.max(),
        orbital_model._sin_wave(x_preds, params),
        label='Best fit model',
    )

    axes[0].margins(x=0.03)
    axes[1].set_xlabel('Phase')
    axes[0].set_ylabel('RV ($m s^{-1}$)')
    axes[0].axhline(0, color='red', linestyle='--')
    y_min, y_max = axes[0].get_ylim()
    axes[0].text(
        0.75,
        y_max - 0.05 * (y_max - y_min),
        rf'Period = {p_mean:.2f} $\pm$ {p_std:.2f} days'
        + '\n'
        + rf'Amplitude = {amp_mean:.2e} $\pm$ {amp_std:.1e} $ms^{{-1}}$',
        color='black',
        fontsize=14,
        ha='right',
        va='top',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    )

    residuals_2 = residuals - orbital_model._sin_wave(folded_time, params)
    axes[1].errsrbar(
        folded_time / folded_time.max(), residuals_2, yerr=rv_err, fmt='kx'
    )
    axes[1].margins(x=0.03)
    axes[1].set_ylabel('Residuals')
    axes[1].axhline(0, color='red', linestyle='--')

    axes[0].legend()
    return fig, axes


def triple_plot(kernels, time, y, y_err, time_scaler, y_scalers):
    t_linspace = np.linspace(time.min(), time.max(), 1400)

    posteriors = []

    for i, kernel in enumerate(kernels):
        kernel.compute_kernel(time, time, y_err[:, i]).compute_loglikelihood(y[:, i])
        mu_post, cov_post = kernel.compute_posterior(t_linspace)
        posteriors.append((mu_post, cov_post))

    # Plot the results
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [1, 0.5, 1, 1]}
    )

    titles = ['(a) RV', '(c) FWHM', '(d) BIS']
    y_label = 'RV ($m s^{-1}$)'
    x_label = 'Time (BJD)'

    for i, ((mu_post, cov_post), axes_index) in enumerate(zip(posteriors, [0, 2, 3])):
        std_post = np.sqrt(np.diag(cov_post))

        axes[axes_index].errorbar(
            invert_transform(time, time_scaler),
            invert_transform(y[:, i], y_scalers[i]),
            yerr=invert_scale(y_err[:, i], y_scalers[i]),
            fmt='kx',
            label=f'{titles[i]} training data',
        )

        axes[axes_index].plot(
            invert_transform(t_linspace, time_scaler),
            invert_transform(mu_post, y_scalers[i]),
            'b',
            label='GP Mean',
        )

        axes[axes_index].fill_between(
            invert_transform(t_linspace, time_scaler),
            invert_transform(mu_post - 2 * std_post, y_scalers[i]),
            invert_transform(mu_post + 2 * std_post, y_scalers[i]),
            color='blue',
            alpha=0.2,
            label='Confidence interval (±2 std)',
        )

        axes[axes_index].margins(x=0.01)
        axes[axes_index].set_xlabel(x_label)
        axes[axes_index].set_ylabel(y_label)
        axes[axes_index].legend()
        axes[axes_index].set_title(titles[i])

    # Compute and plot the residuals
    kernels[0].compute_kernel(time, time, y_err[:, 0]).compute_loglikelihood(y[:, 0])
    mu_post, cov_post = kernels[0].compute_posterior(time)
    residuals = y[:, 0] - mu_post

    axes[1].errorbar(
        invert_transform(time, time_scaler),
        invert_scale(residuals, y_scalers[0]),
        yerr=invert_scale(y_err[:, 0], y_scalers[0]),
        fmt='kx',
        label='Residuals',
    )

    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel('Residuals ($m s^{-1}$)')
    axes[1].set_title('(b) Residuals of RV Fit')
    axes[1].legend()

    plt.tight_layout()
    return fig
