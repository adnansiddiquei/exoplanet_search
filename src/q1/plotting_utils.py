import matplotlib.pyplot as plt


def plot_lightcurves(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))

    left = data[data['time'] < 1500]
    right = data[data['time'] > 1500]

    [axes[i].margins(x=0.05) for i in range(2)]
    [axes[i].set_xlabel('Time (BJD)') for i in range(2)]
    axes[0].set_ylabel('Flux')

    axes[0].plot(left['time'], left['flux'], 'o', markersize=1)
    axes[1].plot(right['time'], right['flux'], 'o', markersize=1)

    axes[0].set_ylim(*axes[1].get_ylim())

    return fig, axes


def plot_periodogram(tls_results, xlim, vertical_lines=10):
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.plot(tls_results.periods, tls_results.power)
    plt.xlim(xlim)

    for i in range(1, vertical_lines + 1):
        plt.axvline(tls_results.period * i, color='red', linestyle='dashed', lw=1)

    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    return fig, ax


def plot_folded_lightcurve(tls_results, plot_model=True, text=None, xlim=None):
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

    if xlim is not None:
        plt.xlim(xlim)

    if text is not None:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()

        x_text = xlims[0] + (xlims[1] - xlims[0]) * 0.05
        y_text = ylims[1] - (ylims[1] - ylims[0]) * 0.05

        plt.text(
            x_text,
            y_text,
            text,
            color='black',
            fontsize=10,
            ha='left',
            va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
        )

    plt.xlabel('Phase')
    plt.ylabel('Relative flux')

    return fig, ax
