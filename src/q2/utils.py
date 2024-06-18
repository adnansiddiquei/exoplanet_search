import numpy as np
import scipy.linalg as spl
from scipy.optimize import minimize
import emcee


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


class QuasiPeriodicKernel:
    def __init__(self, amplitude, P, length_scale_periodic, length_scale_exp):
        self.amplitude = amplitude
        self.length_scale_exp = length_scale_exp
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
            -(
                ((tau / self.length_scale_exp) ** 2) / 2
                + (np.sin(np.pi * tau / self.P) ** 2) / (self.length_scale_periodic**2)
            )
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
        return QuasiPeriodicKernel(
            self.amplitude, self.P, self.length_scale_periodic, self.length_scale_exp
        )

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
            bounds = [[(1e-4, None), (1e-4, None), (1e-4, None), (1e-4, None)]]

        def objective(theta):
            return (
                -QuasiPeriodicKernel(*theta)
                .compute_kernel(x, x, y_err, jitter)
                .compute_loglikelihood(y)
                .loglikelihood
            )

        result = minimize(
            objective,
            np.array(
                [
                    self.amplitude,
                    self.P,
                    self.length_scale_periodic,
                    self.length_scale_exp,
                ]
            ),
            method='L-BFGS-B',
            bounds=bounds,
        )

        return result


class StellarActivityGP:
    def __init__(self, x, y, y_err=None):
        self.x = x
        self.y = y if len(y.shape) > 1 else y.reshape(-1, 1)

        if y_err is None:
            y_err = np.zeros_like(y)

        self.y_err = y_err if len(y_err.shape) > 1 else y_err.reshape(-1, 1)

        self.y_dim = y.shape[1] if len(y.shape) > 1 else 1

    def fit(
        self,
        amplitudes: list,
        P,
        length_scale_periodic,
        length_scale_exp,
        bounds=None,
        jitter=1e-10,
    ):
        assert (
            len(amplitudes) == self.y_dim
        ), 'Number of amplitudes must match the number of dimensions in y'

        if bounds is None:
            bounds = [(1e-4, None) for _ in range(self.y_dim)] + [
                (1e-4, None),
                (1e-4, None),
                (1e-4, None),
            ]

        def objective(theta):
            loglikelihood = 0

            amplitudes = theta[:-3]
            P = theta[-3]
            length_scale_periodic = theta[-2]
            length_scale_exp = theta[-1]

            kernels = [
                QuasiPeriodicKernel(
                    amplitude, P, length_scale_periodic, length_scale_exp
                )
                for amplitude in amplitudes
            ]

            for i, kernel in enumerate(kernels):
                loglikelihood += (
                    kernel.compute_kernel(self.x, self.x, self.y_err[:, i], jitter)
                    .compute_loglikelihood(self.y[:, i])
                    .loglikelihood
                )

            return -loglikelihood

        result = minimize(
            objective,
            np.array([*amplitudes, P, length_scale_periodic, length_scale_exp]),
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
        ]

        self.kernels = None
        self.model_params = {}

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

            stellar_amp = theta['stellar_amp']
            fwhm_amp = theta['fwhm_amp']
            bis_amp = theta['bis_amp']

            stellar_P = theta['stellar_P']
            stellar_lsp = theta['stellar_lsp']
            stellar_lse = theta['stellar_lse']

            planet_amps = [theta[f'p{i + 1}_amp'] for i in range(self.num_planets)]
            planet_Ps = [theta[f'p{i + 1}_P'] for i in range(self.num_planets)]
            planet_lsps = [theta[f'p{i + 1}_lsp'] for i in range(self.num_planets)]

            loglikelihood = 0

            self.kernels = [
                MultiQuasiPeriodicKernel(
                    [stellar_amp, *planet_amps],
                    [stellar_P, *planet_Ps],
                    [stellar_lsp, *planet_lsps],
                    stellar_lse,
                ),
                QuasiPeriodicKernel(fwhm_amp, stellar_P, stellar_lsp, stellar_lse),
                QuasiPeriodicKernel(bis_amp, stellar_P, stellar_lsp, stellar_lse),
            ]

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
        self, initial_theta, bounds=None, jitter=1e-10, nwalkers=50, niterations=100
    ):
        assert set(initial_theta.keys()).issubset(set(self.params_keys))

        for key in initial_theta.keys():
            initial_theta[key] = initial_theta[key] + np.random.normal(0, 0.01)

        if bounds is None:
            bounds = {key: (1e-4, None) for key in initial_theta.keys()}
        else:
            bounds = {
                key: bounds[key] if key in bounds.keys() else (1e-4, None)
                for key in initial_theta.keys()
            }

        def log_likelihood(theta):
            theta = {key: value for key, value in zip(initial_theta.keys(), theta)}

            stellar_amp = theta['stellar_amp']
            fwhm_amp = theta['fwhm_amp']
            bis_amp = theta['bis_amp']

            stellar_P = theta['stellar_P']
            stellar_lsp = theta['stellar_lsp']
            stellar_lse = theta['stellar_lse']

            planet_amps = [theta[f'p{i + 1}_amp'] for i in range(self.num_planets)]
            planet_Ps = [theta[f'p{i + 1}_P'] for i in range(self.num_planets)]
            planet_lsps = [theta[f'p{i + 1}_lsp'] for i in range(self.num_planets)]

            loglikelihood = 0

            self.kernels = [
                MultiQuasiPeriodicKernel(
                    [stellar_amp, *planet_amps],
                    [stellar_P, *planet_Ps],
                    [stellar_lsp, *planet_lsps],
                    stellar_lse,
                ),
                QuasiPeriodicKernel(fwhm_amp, stellar_P, stellar_lsp, stellar_lse),
                QuasiPeriodicKernel(bis_amp, stellar_P, stellar_lsp, stellar_lse),
            ]

            for i, kernel in enumerate(self.kernels):
                loglikelihood += (
                    kernel.compute_kernel(self.x, self.x, self.y_err[:, i], jitter)
                    .compute_loglikelihood(self.y[:, i])
                    .loglikelihood
                )

            return loglikelihood

        def log_prior(theta):
            for key, value in zip(initial_theta.keys(), theta):
                lower_bound, upper_bound = bounds[key]

                if lower_bound is not None and value < lower_bound:
                    return -np.inf

                if upper_bound is not None and value > upper_bound:
                    return -np.inf

            return 0

        def log_posterior(theta):
            lp = log_prior(theta)

            if not np.isfinite(lp):
                return -np.inf

            return log_likelihood(theta) + lp

        ndim = len(initial_theta)

        initial_guesses = np.array(
            list(initial_theta.values())
        ) + 0.1 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
        sampler.run_mcmc(initial_guesses, niterations, progress=True)

        return sampler
