import matplotlib.pyplot as plt
import numpy as np
from .utils import (
    StellarAndPlanetGP,
    MultiQuasiPeriodicKernel,
    SinusoidalModel,
    invert_scale,
    invert_transform,
)
from .plotting_utils import (
    triple_plot,
    plot_1planet_model,
    lombscargle_periodogram,
    data_plot,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import create_dir_if_required, save_dill, load_dill
import os
import corner


def main():
    # -----------------------------------------------------------------------------------------------
    # LOAD THE DATA
    # -----------------------------------------------------------------------------------------------
    out_dir = create_dir_if_required(__file__, 'out')
    cwd = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(cwd, '../../data')
    data_file = os.path.join(data_dir, 'ex2_RVs.txt')

    # load the data
    data = pd.read_csv(
        data_file,
        sep=',',
        header=None,
        names=[
            'time',
            'radial_velocity',
            'radial_velocity_uncertainty',
            'FWHM_CCF',
            'FWHM_CCF_uncertainty',
            'BIS',
            'BIS_uncertainty',
            'instrument',
        ],
        skiprows=8,
    )

    # -----------------------------------------------------------------------------------------------
    # PREPROCESS THE DATA - Standardise
    # -----------------------------------------------------------------------------------------------
    time_scaler = StandardScaler()
    rv_scaler = StandardScaler()
    fwhm_scaler = StandardScaler()
    bis_scaler = StandardScaler()
    y_scalers = [rv_scaler, fwhm_scaler, bis_scaler]

    time = data['time'].values
    rv = data['radial_velocity'].values
    rv_err = data['radial_velocity_uncertainty'].values
    fwhm = data['FWHM_CCF'].values
    fwhm_err = data['FWHM_CCF_uncertainty'].values
    bis = data['BIS'].values
    bis_err = data['BIS_uncertainty'].values

    time = time_scaler.fit_transform(time.reshape(-1, 1)).ravel()
    rv = rv_scaler.fit_transform(rv.reshape(-1, 1)).ravel()
    rv_err = (rv_err / rv_scaler.scale_).ravel()
    fwhm = fwhm_scaler.fit_transform(fwhm.reshape(-1, 1)).ravel()
    fwhm_err = (fwhm_err / fwhm_scaler.scale_).ravel()
    bis = bis_scaler.fit_transform(bis.reshape(-1, 1)).ravel()
    bis_err = (bis_err / bis_scaler.scale_).ravel()

    y = np.vstack([rv, fwhm, bis]).T
    y_err = np.vstack([rv_err, fwhm_err, bis_err]).T

    # set a seed for reproducibility
    np.random.seed(42)

    # Do the initial plot of the data
    fig, ax = data_plot(data)
    plt.savefig(f'{out_dir}/data_plot.png', bbox_inches='tight')

    # -----------------------------------------------------------------------------------------------
    # LOMBSCARGLE PERIODOGRAM
    # -----------------------------------------------------------------------------------------------

    fig, ax = lombscargle_periodogram(data['time'], data['radial_velocity'])
    plt.savefig(f'{out_dir}/lombscargle_periodogram.png', bbox_inches='tight')

    # -----------------------------------------------------------------------------------------------
    # MODEL STELLAR NOISE - with a Quasi-Periodic GP model, jointly fitting the RV data and the stellar indicators
    # -----------------------------------------------------------------------------------------------

    # Create a QuasiPeriodicKernel GP model which we will fit to the rv data and stellar activity indicators
    # simultaneously
    gp = StellarAndPlanetGP(time, y, y_err, num_planets=0)

    # this is where we will save the sampler model
    out_file_stellar = f'{out_dir}/stellar_noise_mcmc_sampler.dill'

    if not os.path.exists(out_file_stellar):
        np.random.seed(42)
        stellar_noise_mcmc_sampler = gp.run_mcmc(
            uniform_priors={
                'stellar_amp': (1e-4, 3 * (rv.max() - rv.min())),
                'fwhm_amp': (1e-4, 3 * (fwhm.max() - fwhm.min())),
                'bis_amp': (1e-4, 3 * (bis.max() - bis.min())),
                'stellar_lsp': (1e-4, 10 * (time.max() - time.min())),
                'stellar_lse': (1e-4, 3 * (time.max() - time.min())),
            },
            gaussian_priors={'stellar_P': ((31, 10) / time_scaler.scale_)},
            log_uniform_priors={
                'jitter': (1e-10, 1e-6),
            },
            nwalkers=250,
            niterations=2000,
        )

        save_dill(stellar_noise_mcmc_sampler, out_file_stellar)
    else:
        stellar_noise_mcmc_sampler = load_dill(out_file_stellar)

    # extract the samples from the MCMC sampler
    burn_in = 100
    thin = 20

    # the samples extracted are in the normalised space
    normalised_samples = stellar_noise_mcmc_sampler.get_chain(
        discard=burn_in, thin=thin, flat=True
    )
    log_prob_samples = stellar_noise_mcmc_sampler.get_log_prob(
        discard=burn_in, thin=thin, flat=True
    )

    # we transform the samples back to the original space
    samples = normalised_samples.copy()
    samples[:, 0] = invert_transform(samples[:, 0], rv_scaler)
    samples[:, 1] = invert_transform(samples[:, 1], fwhm_scaler)
    samples[:, 2] = invert_transform(samples[:, 2], bis_scaler)
    samples[:, 3] = invert_scale(samples[:, 3], time_scaler)
    samples[:, 4] = invert_scale(samples[:, 4], time_scaler)
    samples[:, 5] = invert_scale(samples[:, 5], time_scaler)
    samples[:, 6] = np.log(samples[:, 6])

    threshold = np.percentile(log_prob_samples, 10)
    filtered_samples = samples[log_prob_samples >= threshold]

    # Output a corner plot
    labels = gp.params_keys.copy()
    labels[0] += r' $(\theta_{1})$'
    labels[1] += r' $(\theta_{1})$'
    labels[2] += r' $(\theta_{1})$'
    labels[3] += r' $(\theta_{3}$)'
    labels[4] += r' $(\theta_{4})$'
    labels[5] += r' $(\theta_{2})$'

    figure = corner.corner(filtered_samples, labels=labels, truths=None)
    plt.savefig(f'{out_dir}/stellar_noise_corner_plot.png', bbox_inches='tight')

    # Let's also output the GP fit with the highest likelihood parameters
    best_fit_params = normalised_samples[np.argmax(log_prob_samples)]
    params = {key: value for key, value in zip(gp.params_keys, best_fit_params)}

    rv_fit = MultiQuasiPeriodicKernel(
        [params['stellar_amp']],
        [params['stellar_P']],
        [params['stellar_lsp']],
        params['stellar_lse'],
    )

    fwhm_fit = MultiQuasiPeriodicKernel(
        [params['fwhm_amp']],
        [params['stellar_P']],
        [params['stellar_lsp']],
        params['stellar_lse'],
    )

    bis_fit = MultiQuasiPeriodicKernel(
        [params['bis_amp']],
        [params['stellar_P']],
        [params['stellar_lsp']],
        params['stellar_lse'],
    )

    fig = triple_plot(
        [rv_fit, fwhm_fit, bis_fit], time, y, y_err, time_scaler, y_scalers
    )

    plt.savefig(f'{out_dir}/stellar_noise_fit.png', bbox_inches='tight')
    plt.clf()

    # -----------------------------------------------------------------------------------------------
    # FIT 1 PLANET MODEL - with a sinusoidal model, fitting the residuals of the RV data - stellar noise
    # -----------------------------------------------------------------------------------------------

    # Now lets try to fit a planet to the residuals
    rv_fit.compute_kernel(time, time, rv_err).compute_loglikelihood(rv)
    mu_post, cov_post = rv_fit.compute_posterior(time)
    normalised_residuals = rv - mu_post
    real_residuals = invert_scale(normalised_residuals, rv_scaler)
    data['residuals'] = real_residuals
    data.to_csv(f'{out_dir}/ex2_RVs.csv', index=False)

    # We start by just trying to fit a simple sinusoidal model
    orbital_model = SinusoidalModel(
        data['time'], real_residuals, data['radial_velocity_uncertainty'], num_planets=1
    )

    ignorant_prior_amplitude = (0, 1.5 * (real_residuals.max() - real_residuals.min()))
    ignorant_prior_P = (0, (data['time'].max() - data['time'].min()))
    ignorant_prior_phase_offset = (0, 2 * np.pi)

    # First we scan the entire parameter space, with very ignorant, large, uniform priors
    if not os.path.exists(f'{out_dir}/1planet_model_1.dill'):
        np.random.seed(42)
        orbital_model_sampler = orbital_model.run_mcmc(
            uniform_priors={
                'amplitude_0': ignorant_prior_amplitude,
                'P_0': ignorant_prior_P,
                'phase_offset_0': ignorant_prior_phase_offset,
            },
            nwalkers=100,
            niterations=1000,
        )

        save_dill(orbital_model_sampler, f'{out_dir}/1planet_model_1.dill')
    else:
        orbital_model_sampler = load_dill(f'{out_dir}/1planet_model_1.dill')

    # Extract the samples from the MCMC sampler
    samples = orbital_model_sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    log_prob_samples = orbital_model_sampler.get_log_prob(
        discard=burn_in, thin=thin, flat=True
    )
    planet_period = samples[:, 1]

    # Extract the 10% most likely samples
    threshold = np.percentile(log_prob_samples, 95)
    filtered_planet_periods = planet_period[log_prob_samples >= threshold]
    filtered_log_prob_samples = log_prob_samples[log_prob_samples >= threshold]

    # plot the likelihood vs. period as a visualisation
    fig, ax = plt.subplots()
    plt.plot(planet_period, log_prob_samples, 'x', label='All samples')
    plt.plot(
        filtered_planet_periods,
        filtered_log_prob_samples,
        'x',
        label='99th percentile',
    )
    plt.xlabel('Period')
    plt.ylabel('Log Likelihood')
    plt.legend()
    plt.ylim(-1500, -1000)
    plt.savefig(
        f'{out_dir}/1planet_model_period_likelihood_1.png',
        bbox_inches='tight',
    )

    # this will print "95th percentile range: 1.47 - 222.87"
    print(
        f'95th percentile range: {filtered_planet_periods.min():.2f} - {filtered_planet_periods.max():.2f} which '
        f'consists {len(filtered_planet_periods)} samples out of {len(planet_period)} samples.'
    )

    # Now based on the above first scan, we can refine the priors for the period to a smaller range
    # this model will yield a much better estimate of the parameters
    if not os.path.exists(f'{out_dir}/1planet_model_2.dill'):
        np.random.seed(42)
        orbital_model_sampler = orbital_model.run_mcmc(
            uniform_priors={
                'amplitude_0': ignorant_prior_amplitude,
                'P_0': (
                    0,
                    filtered_planet_periods.max() * 1.1,
                ),  # so we now know the period is between 1.47 and 222.87
                'phase_offset_0': ignorant_prior_phase_offset,
            },
            nwalkers=250,
            niterations=2000,
        )
        save_dill(orbital_model_sampler, f'{out_dir}/1planet_model_2.dill')
    else:
        orbital_model_sampler = load_dill(f'{out_dir}/1planet_model_2.dill')

    # Extract the samples from the MCMC sampler
    samples = orbital_model_sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    log_prob_samples = orbital_model_sampler.get_log_prob(
        discard=burn_in, thin=thin, flat=True
    )

    # plot the corner plot
    labels = orbital_model.params_keys
    figure = corner.corner(samples, labels=labels, truths=None)
    plt.savefig(f'{out_dir}/1planet_model_corner_plot.png', bbox_inches='tight')

    # plot the fit
    fig, ax = plot_1planet_model(
        data['time'],
        real_residuals,
        data['radial_velocity_uncertainty'],
        orbital_model,
        samples,
        log_prob_samples,
    )
    plt.savefig(f'{out_dir}/1planet_model_fit.png', bbox_inches='tight')

    # -----------------------------------------------------------------------------------------------
    # FIT 2 PLANET MODEL - with the sum of 2 sins
    # -----------------------------------------------------------------------------------------------
    # print('Fitting 2 planet model...')
    # orbital_model_2 = SinusoidalModel(
    #     data['time'], real_residuals, data['radial_velocity_uncertainty'], num_planets=2
    # )
    #
    # if not os.path.exists(f'{out_dir}/2planet_model_1.dill'):
    #     print('Running MCMC on 2 planet model (run 1)...')
    #     np.random.seed(42)
    #     orbital_model_sampler_2 = orbital_model_2.run_mcmc(
    #         uniform_priors={
    #             'amplitude_0': ignorant_prior_amplitude,
    #             'P_0': ignorant_prior_P,
    #             'phase_offset_0': ignorant_prior_phase_offset,
    #             'amplitude_1': ignorant_prior_amplitude,
    #             'P_1': ignorant_prior_P,
    #             'phase_offset_1': ignorant_prior_phase_offset,
    #         },
    #         nwalkers=250,
    #         niterations=2000,
    #     )
    #
    #     save_dill(orbital_model_sampler_2, f'{out_dir}/2planet_model_1.dill')
    # else:
    #     orbital_model_sampler_2 = load_dill(f'{out_dir}/2planet_model_1.dill')
    #
    # if not os.path.exists(f'{out_dir}/2planet_model_2.dill'):
    #     print('Running MCMC on 2 planet model (run 2)...')
    #     np.random.seed(42)
    #     orbital_model_sampler_2 = orbital_model_2.run_mcmc(
    #         uniform_priors={
    #             'amplitude_0': ignorant_prior_amplitude,
    #             'P_0': (0, 800),
    #             'phase_offset_0': ignorant_prior_phase_offset,
    #             'amplitude_1': ignorant_prior_amplitude,
    #             'P_1': ignorant_prior_P,
    #             'phase_offset_1': ignorant_prior_phase_offset,
    #         },
    #         nwalkers=250,
    #         niterations=2000,
    #     )
    #
    #     save_dill(orbital_model_sampler_2, f'{out_dir}/2planet_model_2.dill')
    # else:
    #     orbital_model_sampler_2 = load_dill(f'{out_dir}/2planet_model_2.dill')
    #
    # if not os.path.exists(f'{out_dir}/2planet_model_3.dill'):
    #     print('Running MCMC on 2 planet model (run 3)...')
    #     np.random.seed(42)
    #     orbital_model_sampler_2 = orbital_model_2.run_mcmc(
    #         uniform_priors={
    #             'amplitude_0': ignorant_prior_amplitude,
    #             'P_0': (0, 100),
    #             'phase_offset_0': ignorant_prior_phase_offset,
    #             'amplitude_1': ignorant_prior_amplitude,
    #             'P_1': ignorant_prior_P,
    #             'phase_offset_1': ignorant_prior_phase_offset,
    #         },
    #         nwalkers=250,
    #         niterations=2000,
    #     )
    #
    #     save_dill(orbital_model_sampler_2, f'{out_dir}/2planet_model_3.dill')
    # else:
    #     orbital_model_sampler_2 = load_dill(f'{out_dir}/2planet_model_3.dill')


if __name__ == '__main__':
    main()
