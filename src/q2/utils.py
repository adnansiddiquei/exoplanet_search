import numpy as np
import scipy.linalg as spl
from scipy.optimize import minimize
import emcee
from functools import partial


def invert_transform(sample, scaler):
    return scaler.inverse_transform(sample.reshape(-1, 1)).ravel()


def invert_scale(sample, scaler):
    return sample * scaler.scale_


def fold(time, period, phase_offset):
    return np.mod((time - phase_offset), period)


class SinusoidalModel:
    def __init__(self, time, y, y_err, num_planets=1):
        assert num_planets > 0 and isinstance(num_planets, int)

        self.time = time
        self.y = y
        self.y_err = y_err
        self.num_planets = num_planets

        self.params_keys = (
            [f'amplitude_{i}' for i in range(self.num_planets)]
            + [f'P_{i}' for i in range(self.num_planets)]
            + [f'phase_offset_{i}' for i in range(self.num_planets)]
        )

        self.model_params = None

    def _sin_wave(self, x, theta):
        y = np.zeros_like(x)

        for i in range(self.num_planets):
            amplitude = theta[f'amplitude_{i}']
            P = theta[f'P_{i}']
            phase_offset = theta[f'phase_offset_{i}']

            y += amplitude * np.sin((2 * np.pi * x / P) + phase_offset)

        return y

    def run_mcmc(
        self,
        uniform_priors=None,
        gaussian_priors=None,
        log_uniform_priors=None,
        nwalkers=50,
        niterations=1000,
    ):
        uniform_priors = {} if uniform_priors is None else uniform_priors
        gaussian_priors = {} if gaussian_priors is None else gaussian_priors
        log_uniform_priors = {} if log_uniform_priors is None else log_uniform_priors

        all_priors = {**uniform_priors, **gaussian_priors, **log_uniform_priors}.keys()
        assert set(all_priors).issubset(set(self.params_keys))
        assert len(all_priors) == len(self.params_keys)

        prior_sampler = {}

        for key, (mu, sigma) in gaussian_priors.items():
            prior_sampler[key] = partial(np.random.normal, mu, sigma)

        for key, (lower_bound, upper_bound) in uniform_priors.items():
            prior_sampler[key] = partial(np.random.uniform, lower_bound, upper_bound)

        for key, (lower_bound, upper_bound) in log_uniform_priors.items():
            prior_sampler[key] = partial(
                lambda lb, ub: np.exp(np.random.uniform(np.log(lb), np.log(ub))),
                lower_bound,
                upper_bound,
            )

        # re-order the prior_sampler so that it matches the order of the self.params_keys created in __init__
        prior_sampler = {key: prior_sampler[key] for key in self.params_keys}

        def log_likelihood(theta):
            theta = {key: value for key, value in zip(self.params_keys, theta)}

            model = self._sin_wave(self.time, theta)

            return -0.5 * np.sum(
                (self.y - model) ** 2 / self.y_err**2
                + np.log(2 * np.pi * self.y_err**2)
            )

        def log_prior(theta):
            theta = {key: value for key, value in zip(self.params_keys, theta)}

            prior = 0

            # implement prior for each parameter
            # guassian priors
            for key in gaussian_priors.keys():
                prior += (
                    -0.5
                    * ((theta[key] - gaussian_priors[key][0]) / gaussian_priors[key][1])
                    ** 2
                )

            # uniform priors
            for key in uniform_priors.keys():
                lower_bound, upper_bound = uniform_priors[key]

                if not (lower_bound < theta[key] < upper_bound):
                    return -np.inf

            # log uniform priors
            for key in log_uniform_priors.keys():
                lower_bound, upper_bound = log_uniform_priors[key]

                if not (lower_bound < theta[key] < upper_bound):
                    return -np.inf

                prior += -np.log(theta[key])

            return prior

        def log_posterior(theta):
            lp = log_prior(theta)

            if not np.isfinite(lp):
                return -np.inf

            return log_likelihood(theta) + lp

        # generate starting points by sampling from the prior distributions
        samples_from_prior = np.array(
            [[func() for func in list(prior_sampler.values())] for _ in range(nwalkers)]
        )

        ndim = len(self.params_keys)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

        sampler.run_mcmc(samples_from_prior, niterations, progress=True)

        return sampler


class PeriodicKernel:
    def __init__(self, amplitude, P, length_scale_periodic):
        self.amplitude = amplitude
        self.length_scale_periodic = length_scale_periodic
        self.P = P

        self.kernel = None
        self.loglikelihood = None
        self.x1 = None
        self.x2 = None
        self.y = None

    def compute_kernel(self, x1, x2, y_err=None, jitter=1e-10):
        self.x1 = x1
        self.x2 = x2

        tau = np.subtract.outer(x1, x2)  # pairwise differences

        self.kernel = (self.amplitude**2) * np.exp(
            -2 * ((np.sin(np.pi * tau / self.P) ** 2) / (self.length_scale_periodic**2))
        )

        if y_err is not None:
            self.kernel += np.diag(y_err)

        np.fill_diagonal(self.kernel, self.kernel.diagonal() + jitter)

        return self

    def compute_loglikelihood(self, y):
        self.y = y

        factor, flag = spl.cho_factor(self.kernel)
        lodget = 2 * np.sum(np.log(np.diag(factor)))
        gof = np.dot(y, spl.cho_solve((factor, flag), y))
        self.loglikelihood = -0.5 * (gof + lodget + len(y) * np.log(2 * np.pi))

        return self

    def clone(self):
        return PeriodicKernel(self.amplitude, self.P, self.length_scale_periodic)

    def compute_posterior(self, x_test):
        assert (
            self.kernel is not None
        ), 'Kernel has not been computed. Please run compute_kernel first.'
        assert (
            self.loglikelihood is not None
        ), 'Loglikelihood has not been computed. Please run compute_loglikelihood first.'

        temp_kernel = self.clone()

        K_test_train = temp_kernel.compute_kernel(
            x_test, self.x1
        ).kernel  # Kernel between test and train points
        K_test_test = temp_kernel.compute_kernel(
            x_test, x_test
        ).kernel  # Kernel for the test points

        factor, flag = spl.cho_factor(
            self.kernel
        )  # Cholesky decomposition of the training kernel matrix
        K_inv_y = spl.cho_solve(
            (factor, flag), self.y
        )  # Solve self.kernel^-1 * y_train

        mu_post = np.dot(K_test_train, K_inv_y)  # Compute the posterior mean

        v = spl.cho_solve(
            (factor, flag), K_test_train.T
        )  # Solve self.kernel^-1 * K_test_train.T
        cov_post = K_test_test - np.dot(
            K_test_train, v
        )  # Compute the posterior covariance

        return mu_post, cov_post

    def fit(self, x, y, y_err, bounds=None, jitter=1e-10):
        self.compute_kernel(x, x, y_err, jitter).compute_loglikelihood(y)

        if bounds is None:
            bounds = [[(1e-4, None), (1e-4, None), (1e-4, None)]]

        def objective(theta):
            return (
                -PeriodicKernel(*theta)
                .compute_kernel(x, x, y_err, jitter)
                .compute_loglikelihood(y)
                .loglikelihood
            )

        result = minimize(
            objective,
            np.array([self.amplitude, self.P, self.length_scale_periodic]),
            method='L-BFGS-B',
            bounds=bounds,
        )

        return result


class MultiQuasiPeriodicKernel:
    def __init__(self, amplitudes, Ps, length_scale_periodics, length_scale_exp):
        assert len(amplitudes) == len(Ps) == len(length_scale_periodics)
        assert isinstance(length_scale_exp, float)

        self.amplitudes = amplitudes
        self.Ps = Ps
        self.length_scale_periodics = length_scale_periodics
        self.length_scale_exp = length_scale_exp
        self.periodics = len(amplitudes)

        self.kernel = None
        self.loglikelihood = None
        self.x1 = None
        self.x2 = None
        self.y = None

    def compute_kernel(self, x1, x2, y_err=None, jitter=1e-10):
        self.x1 = x1
        self.x2 = x2

        tau = np.subtract.outer(x1, x2)  # pairwise differences

        self.kernel = (self.amplitudes[0] ** 2) * np.exp(
            -(
                ((tau / self.length_scale_exp) ** 2) / 2
                + (np.sin(np.pi * tau / self.Ps[0]) ** 2)
                / (self.length_scale_periodics[0] ** 2)
            )
        )

        for i in range(1, self.periodics):
            additional_kernel = (self.amplitudes[i] ** 2) * np.exp(
                -(
                    (np.sin(np.pi * tau / self.Ps[i]) ** 2)
                    / (self.length_scale_periodics[i] ** 2)
                )
            )
            self.kernel += additional_kernel

        if y_err is not None:
            self.kernel += np.diag(y_err)

        np.fill_diagonal(self.kernel, self.kernel.diagonal() + jitter)

        return self

    def compute_loglikelihood(self, y):
        self.y = y

        factor, flag = spl.cho_factor(self.kernel)
        lodget = 2 * np.sum(np.log(np.diag(factor)))
        gof = np.dot(y, spl.cho_solve((factor, flag), y))
        self.loglikelihood = -0.5 * (gof + lodget + len(y) * np.log(2 * np.pi))

        return self

    def clone(self):
        return MultiQuasiPeriodicKernel(
            self.amplitudes, self.Ps, self.length_scale_periodics, self.length_scale_exp
        )

    def compute_posterior(self, x_test):
        assert (
            self.kernel is not None
        ), 'Kernel has not been computed. Please run compute_kernel first.'
        assert self.loglikelihood is not None, (
            'Loglikelihood has not been computed. Please run compute_loglikelihood '
            'first.'
        )

        temp_kernel = self.clone()

        K_test_train = temp_kernel.compute_kernel(
            x_test, self.x1
        ).kernel  # Kernel between test and train points
        K_test_test = temp_kernel.compute_kernel(
            x_test, x_test
        ).kernel  # Kernel for the test points

        factor, flag = spl.cho_factor(
            self.kernel
        )  # Cholesky decomposition of the training kernel matrix
        K_inv_y = spl.cho_solve(
            (factor, flag), self.y
        )  # Solve self.kernel^-1 * y_train

        mu_post = np.dot(K_test_train, K_inv_y)  # Compute the posterior mean

        v = spl.cho_solve(
            (factor, flag), K_test_train.T
        )  # Solve self.kernel^-1 * K_test_train.T
        cov_post = K_test_test - np.dot(
            K_test_train, v
        )  # Compute the posterior covariance

        return mu_post, cov_post

    def fit(self, x, y, y_err, bounds=None, jitter=1e-10):
        self.compute_kernel(x, x, y_err, jitter).compute_loglikelihood(y)

        if bounds is None:
            bounds = [[(1e-4, None) for _ in range(3 * self.periodics + 1)]]

        def objective(theta):
            amplitudes = theta[0 : self.periodics]
            Ps = theta[self.periodics : self.periodics + 2]
            length_scale_periodics = theta[self.periodics + 2 : self.periodics + 4]
            length_scale_exp = theta[-1]

            return (
                -MultiQuasiPeriodicKernel(
                    amplitudes, Ps, length_scale_periodics, length_scale_exp
                )
                .compute_kernel(x, x, y_err, jitter)
                .compute_loglikelihood(y)
                .loglikelihood
            )

        result = minimize(
            objective,
            np.array(
                [
                    *self.amplitudes,
                    *self.Ps,
                    *self.length_scale_periodics,
                    self.length_scale_exp,
                ]
            ),
            method='L-BFGS-B',
            bounds=bounds,
        )

        return result


class StellarAndPlanetGP:
    def __init__(self, x, y, y_err=None, num_planets=1):
        self.x = x
        self.y = y if len(y.shape) > 1 else y.reshape(-1, 1)

        if y_err is None:
            y_err = np.zeros_like(y)

        self.y_err = y_err if len(y_err.shape) > 1 else y_err.reshape(-1, 1)

        self.y_dim = y.shape[1] if len(y.shape) > 1 else 1

        self.num_planets = num_planets
        self.params_keys = [
            'stellar_amp',
            'fwhm_amp',
            'bis_amp',
            'stellar_P',
            'stellar_lsp',
            'stellar_lse',
            *[f'p{i + 1}_amp' for i in range(self.num_planets)],
            *[f'p{i + 1}_P' for i in range(self.num_planets)],
            *[f'p{i + 1}_lsp' for i in range(self.num_planets)],
            'jitter',
        ]

        self.kernels = None
        self.model_params = {}

    def _generate_kernels(self, params=None):
        if params is None:
            params = self.params

        assert set(params.keys()).issubset(set(self.params_keys))

        stellar_amp = params['stellar_amp']
        fwhm_amp = params['fwhm_amp']
        bis_amp = params['bis_amp']

        stellar_P = params['stellar_P']
        stellar_lsp = params['stellar_lsp']
        stellar_lse = params['stellar_lse']

        planet_amps = [params[f'p{i + 1}_amp'] for i in range(self.num_planets)]
        planet_Ps = [params[f'p{i + 1}_P'] for i in range(self.num_planets)]
        planet_lsps = [params[f'p{i + 1}_lsp'] for i in range(self.num_planets)]

        kernels = [
            MultiQuasiPeriodicKernel(
                [stellar_amp, *planet_amps],
                [stellar_P, *planet_Ps],
                [stellar_lsp, *planet_lsps],
                stellar_lse,
            ),
            MultiQuasiPeriodicKernel(
                [fwhm_amp], [stellar_P], [stellar_lsp], stellar_lse
            ),
            MultiQuasiPeriodicKernel(
                [bis_amp], [stellar_P], [stellar_lsp], stellar_lse
            ),
        ]

        return kernels

    def fit(self, initial_theta=None, bounds=None, jitter=1e-10):
        if initial_theta is None:
            initial_theta = {key: np.random.normal(1, 0.01) for key in self.params_keys}
        else:
            initial_theta = {
                key: initial_theta[key]
                if key in initial_theta.keys()
                else np.random.normal(1, 0.01)
                for key in self.params_keys
            }

        if bounds is None:
            bounds = {key: (1e-4, None) for key in initial_theta.keys()}
        else:
            bounds = {
                key: bounds[key] if key in bounds.keys() else (1e-4, None)
                for key in initial_theta.keys()
            }

        assert set(initial_theta.keys()).issubset(set(self.params_keys))

        def objective(theta):
            theta = {key: value for key, value in zip(initial_theta.keys(), theta)}

            loglikelihood = 0

            self.kernels = self._generate_kernels(theta)

            for i, kernel in enumerate(self.kernels):
                loglikelihood += (
                    kernel.compute_kernel(self.x, self.x, self.y_err[:, i], jitter)
                    .compute_loglikelihood(self.y[:, i])
                    .loglikelihood
                )

            return -loglikelihood

        result = minimize(
            objective,
            np.array([*initial_theta.values()]),
            method='L-BFGS-B',
            bounds=bounds.values(),
        )

        self.model_params = {
            key: value for key, value in zip(self.params_keys, result.x)
        }
        return result

    def run_mcmc(
        self,
        uniform_priors=None,
        gaussian_priors=None,
        log_uniform_priors=None,
        nwalkers=50,
        niterations=100,
    ):
        uniform_priors = {} if uniform_priors is None else uniform_priors
        gaussian_priors = {} if gaussian_priors is None else gaussian_priors
        log_uniform_priors = {} if log_uniform_priors is None else log_uniform_priors

        all_priors = {**uniform_priors, **gaussian_priors, **log_uniform_priors}.keys()
        assert set(all_priors).issubset(set(self.params_keys))
        assert len(all_priors) == len(self.params_keys)

        prior_sampler = {}

        for key, (mu, sigma) in gaussian_priors.items():
            prior_sampler[key] = partial(np.random.normal, mu, sigma)

        for key, (lower_bound, upper_bound) in uniform_priors.items():
            prior_sampler[key] = partial(np.random.uniform, lower_bound, upper_bound)

        for key, (lower_bound, upper_bound) in log_uniform_priors.items():
            prior_sampler[key] = partial(
                lambda lb, ub: np.exp(np.random.uniform(np.log(lb), np.log(ub))),
                lower_bound,
                upper_bound,
            )

        prior_sampler = {key: prior_sampler[key] for key in self.params_keys}

        def log_likelihood(theta):
            theta = {key: value for key, value in zip(self.params_keys, theta)}

            loglikelihood = 0

            self.kernels = self._generate_kernels(theta)

            for i, kernel in enumerate(self.kernels):
                loglikelihood += (
                    kernel.compute_kernel(
                        self.x, self.x, self.y_err[:, i], theta['jitter']
                    )
                    .compute_loglikelihood(self.y[:, i])
                    .loglikelihood
                )

            return loglikelihood

        def log_prior(theta):
            theta = {key: value for key, value in zip(self.params_keys, theta)}

            prior = 0

            # implement prior for each parameter
            # guassian priors
            for key in gaussian_priors.keys():
                prior += (
                    -0.5
                    * ((theta[key] - gaussian_priors[key][0]) / gaussian_priors[key][1])
                    ** 2
                )

            # uniform priors
            for key in uniform_priors.keys():
                lower_bound, upper_bound = uniform_priors[key]

                if not (lower_bound < theta[key] < upper_bound):
                    return -np.inf

            # log uniform priors
            for key in log_uniform_priors.keys():
                lower_bound, upper_bound = log_uniform_priors[key]

                if not (lower_bound < theta[key] < upper_bound):
                    return -np.inf

                prior += -np.log(theta[key])

            return prior

        def log_posterior(theta):
            lp = log_prior(theta)

            if not np.isfinite(lp):
                return -np.inf

            return log_likelihood(theta) + lp

        # generate starting points by sampling from the prior distributions
        samples_from_prior = np.array(
            [[func() for func in list(prior_sampler.values())] for _ in range(nwalkers)]
        )

        ndim = len(self.params_keys)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
        sampler.run_mcmc(samples_from_prior, niterations, progress=True)

        return sampler