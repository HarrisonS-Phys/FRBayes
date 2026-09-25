"""
likelihoods.py - NumPyro likelihoods for the Fourier-domain FRB pulse model.

CONVENTIONS (must match model.py and the paper):
1. FOURIER TRANSFORM: data and templates use X_c = dt * sum_b x_b exp(-2 pi i f_c t_b),
   i.e. dt * rfft(x).
2. NOISE: `sigma_noise` is the standard deviation of the real (equivalently, imaginary) part
   of each Fourier bin. For white noise with per-channel time-domain std varsig_a, this is
   sqrt(N dt^2 varsig_a^2 / 2) for every bin retained below.
3. DC AND NYQUIST BINS ARE DISCARDED in the Fourier-domain likelihoods:
   - DC (f = 0) is set by each channel's baseline level, which the model does not include;
     discarding it makes the fit insensitive to per-channel baseline offsets.
   - Nyquist (f = 1 / (2 dt), present only when N is even) is purely real for real data and
     receives equal aliased contributions from +f and -f, so the continuous-FT template does
     not describe it.
   These are 2 of ~N/2 bins per channel and carry negligible information for resolved pulses.
   Every retained bin has independent real and imaginary parts of equal variance sigma^2,
   so the noise-weighted inner product is
       <x|y> = sum_{a,c} Re(conj(x_ac) y_ac) / sigma_ac^2,   summed over retained bins c.

PRECISION: float32 throughout; see the note in model.py on relative times and residual DM.
"""

from typing import Callable, Dict, Tuple
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from jax.scipy.stats import norm
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import OrderedTransform, biject_to


_EPS = jnp.finfo(jnp.float32).eps


def _validate_fourier_inputs(data_fd, sigma_noise, n_time):
    # Static shape validation (valid during JAX tracing)
    expected_nfourier = n_time // 2 + 1
    if data_fd.shape[-1] != expected_nfourier:
        raise ValueError(
            f"data_fd has {data_fd.shape[-1]} Fourier bins, "
            f"but n_time={n_time} implies {expected_nfourier}."
        )


# -------------------------------------------------------------------
# 1. Priors
# -------------------------------------------------------------------

def sample_ordered_arrival_times(name, distribution, n_pulses):
    """Sample n_pulses ordered arrival times from a target distribution.

    Enforces temporal ordering t_0 < t_1 < ... < t_{P-1} while correctly
    accounting for the order statistics permutation factor ln(P!) and 
    support bijector Jacobians without JAX tracer boolean errors.
    """
    if n_pulses == 1:
        return numpyro.sample(name, distribution.expand([1]))

    # Bijector mapping R -> distribution support
    to_support = dist.transforms.biject_to(distribution.support)
    sign = getattr(to_support, "sign", 1.0)

    # Sample base unconstrained variables
    z = numpyro.sample(
        f"{name}_base",
        dist.Normal(0.0, 1.0).expand([n_pulses])
    )

    # Order in unconstrained space
    ordered = jnp.sort(z, axis=-1)

    # Pure jnp condition to avoid TracerBoolConversionError
    ordered_for_support = jnp.where(sign < 0, jnp.flip(ordered, axis=-1), ordered)

    # Map back to target distribution support
    t = to_support(ordered_for_support)

    # Correct log-density for order statistics: sum(ln p(t_i)) + sum(ln |J|) + ln(P!)
    log_prob_base = jnp.sum(distribution.log_prob(t))
    log_det_J = jnp.sum(to_support.log_abs_det_jacobian(ordered_for_support, t))
    log_factorial_p = jnp.sum(jnp.log(jnp.arange(1, n_pulses + 1, dtype=jnp.float32)))

    numpyro.factor(
        f"{name}_order_factor",
        log_prob_base + log_det_J + log_factorial_p
    )

    return numpyro.deterministic(name, t)

def sample_pulse_parameters(
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    time_param: str = "mean_arrival_time",
) -> Dict[str, jnp.ndarray]:
    """
    Samples every pulse parameter from the prior dictionary into stacked arrays of shape
    (n_pulses,). Arrival times are ordered across pulses; all other parameters are i.i.d.
    """
    if n_pulses > 1 and time_param not in priors:
        raise ValueError(
            f"'{time_param}' must be in priors to time-order the pulses; "
            f"without ordering there are n_pulses! equivalent posterior modes."
        )

    params_stacked = {}
    for param_name, distribution in priors.items():
        if param_name == time_param:
            params_stacked[param_name] = sample_ordered_arrival_times(param_name, distribution, n_pulses)
        else:
            # Same prior for every pulse, sampled as one batched site
            params_stacked[param_name] = numpyro.sample(param_name, distribution.expand([n_pulses]))

    return params_stacked


def overlap_log_prior(
    R: jnp.ndarray,
    K: float = 100.0,
    R0: float = 0.9,
    w: float = 0.002,
) -> jnp.ndarray:
    """
    Softplus overlap prior (Eq. softplus_overlap_prior of the paper):

        ln p = -K / ln[1 + exp((1 - R0) / w)] * sum_{j<k} ln[1 + exp((R_jk - R0) / w)]

    Approximately flat for R_jk < R0 and rising with slope approaching K / (1 - R0) above it;
    w sets the sharpness of the transition, and K is the penalty at total overlap R_jk = 1.
    """
    # Each pair once, since R is symmetric
    iu = jnp.triu_indices(R.shape[0], k=1)

    # jax.nn.softplus(x) = ln(1 + e^x), evaluated without overflow
    normalization = jax.nn.softplus((1.0 - R0) / w)
    return -(K / normalization) * jnp.sum(jax.nn.softplus((R[iu] - R0) / w))


# -------------------------------------------------------------------
# 2. Templates, matched filter, and Gram matrix
# -------------------------------------------------------------------

def _retained_fourier_bins(n_time: int) -> slice:
    """Fourier bins used by the likelihoods: all except DC (c = 0) and Nyquist (c = N/2, even N)."""
    return slice(1, (n_time + 1) // 2)


def pulse_templates(
    params_stacked: Dict[str, jnp.ndarray],
    single_pulse_fn: Callable[[Dict[str, jnp.ndarray]], jnp.ndarray],
) -> jnp.ndarray:
    """Evaluates all pulse templates in parallel; shape (n_pulses, nfreq, nfourier)."""
    return jax.vmap(single_pulse_fn)(params_stacked)


def matched_filter_and_gram(
    templates: jnp.ndarray,
    data_fd: jnp.ndarray,
    inv_var: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    rho_j = <d|h_j>    unnormalized matched filter, shape (n_pulses,)
    G_jk  = <h_j|h_k>  Gram matrix of template overlaps, shape (n_pulses, n_pulses)

    All inputs are restricted to the retained Fourier bins; inv_var = 1 / sigma_noise^2.
    """
    weighted = templates * inv_var[None]
    rho = jnp.real(jnp.einsum("jac,ac->j", weighted, jnp.conj(data_fd)))
    gram = jnp.real(jnp.einsum("jac,kac->jk", jnp.conj(templates), weighted))

    # Symmetrize against rounding
    return rho, 0.5 * (gram + gram.T)


def _snr_and_overlaps(
    rho: jnp.ndarray,
    gram: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    SNR_opt,j = sqrt(G_jj),   SNR_mf,j = rho_j / SNR_opt,j,   R_jk = G_jk / (SNR_opt,j SNR_opt,k).
    """
    # Floor guard using float32 machine epsilon against division by zero for degenerate templates
    snr_opt = jnp.sqrt(jnp.maximum(jnp.diag(gram), _EPS))
    snr_mf = rho / snr_opt

    # Compute correlation matrix R_jk and symmetrize without hard-replacing the diagonal
    R = gram / (snr_opt[:, None] * snr_opt[None, :])
    R = 0.5 * (R + R.T)

    return snr_opt, snr_mf, R


# -------------------------------------------------------------------
# 3. Fluence marginalization (zero-mean Gaussian fluence prior)
# -------------------------------------------------------------------

def _gaussian_fluence_prior_cholesky(
    R: jnp.ndarray,
    s_opt: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cholesky factor of [R + diag(s_opt^-2) + jitter*I], the matrix defining both the
    marginalized likelihood and the conditional fluence distribution under the zero-mean
    Gaussian fluence prior with s_(j) = s_opt,(j) / SNR_opt,(j). Includes 1e-6 diagonal jitter
    to maintain positive-definiteness when R_jk -> 1 in float32.

    Returns (cholesky factor, s_opt broadcast to shape (n_pulses,)).
    """
    s_opt_broadcast = jnp.broadcast_to(jnp.asarray(s_opt, dtype=jnp.float32), R.shape[:1])
    jitter = 1e-6 * jnp.eye(R.shape[0], dtype=jnp.float32)
    A_matrix = R + jnp.diag(s_opt_broadcast ** -2) + jitter
    return jnp.linalg.cholesky(A_matrix), s_opt_broadcast


def marginalized_log_likelihood(
    rho: jnp.ndarray,
    gram: jnp.ndarray,
    s_opt: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fluence-marginalized log-likelihood, up to a parameter-independent constant:

        ln p = 1/2 sum_{j,k} SNR_mf,j [R + diag(s_opt^-2)]^-1_jk SNR_mf,k
               - 1/2 ln det[1 + diag(s_opt^2) R]

    Returns (log_likelihood, R, snr_opt, snr_mf).
    """
    snr_opt, snr_mf, R = _snr_and_overlaps(rho, gram)
    chol, s_opt_broadcast = _gaussian_fluence_prior_cholesky(R, s_opt)

    # Quadratic term via a triangular solve rather than an explicit inverse
    v = jax.scipy.linalg.solve_triangular(chol, snr_mf, lower=True)
    quadratic = 0.5 * jnp.sum(v ** 2)

    # ln det[1 + diag(s_opt^2) R] = sum_j ln s_opt,j^2 + ln det[R + diag(s_opt^-2)]
    log_det = 2.0 * jnp.sum(jnp.log(s_opt_broadcast)) + 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

    return quadratic - 0.5 * log_det, R, snr_opt, snr_mf


def draw_fluences_fullnormal_gaussmarg(
    gram: jnp.ndarray,
    rho: jnp.ndarray,
    s_opt: float,
    key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Draws fluences for the marginalization performed in `fourier_marginalized_likelihood`.

    The conditional fluence distribution depends on the fluence prior that was marginalized
    over; this sampler is specific to the zero-mean Gaussian prior with
    s_(j) = s_opt,(j) / SNR_opt,(j). In SNR units its conditional is the multivariate normal

        SNR_opt,j A_j  ~  N( [R + diag(s_opt^-2)]^-1 SNR_mf,  [R + diag(s_opt^-2)]^-1 ).

    One draw is made per posterior sample, and draws with any negative fluence are flagged.

    gram: (n_samples, n_pulses, n_pulses), the 'gram' deterministic site
    rho:  (n_samples, n_pulses),           the 'rho_mf' deterministic site

    Returns (fluences, accept) with shapes (n_samples, n_pulses) and (n_samples,).
    Apply `accept` to ALL posterior sites; the retained samples are exact draws from the
    posterior under a positivity-constrained fluence prior.
    """
    def single_draw(g, r, k):
        snr_opt, snr_mf, R = _snr_and_overlaps(r, g)
        chol, _ = _gaussian_fluence_prior_cholesky(R, s_opt)

        # Conditional mean, in SNR units
        mean = jax.scipy.linalg.cho_solve((chol, True), snr_mf)

        # Correlated draw with covariance [R + diag(s_opt^-2)]^-1
        z = jax.random.normal(k, snr_mf.shape, dtype=snr_mf.dtype)
        draw = mean + jax.scipy.linalg.solve_triangular(chol, z, lower=True, trans=1)

        # Convert from SNR units to fluence
        return draw / snr_opt

    keys = jax.random.split(key, rho.shape[0])
    fluences = jax.vmap(single_draw)(gram, rho, keys)

    # Reject any draw with a negative fluence
    accept = jnp.all(fluences > 0, axis=-1)
    return fluences, accept

def draw_fluences_halfnormal_gaussmarg(
    snr_opt: jnp.ndarray,
    snr_mf: jnp.ndarray,
    s_opt: float,
    key: jax.Array,
) -> jnp.ndarray:
    """
    Draws fluences for the marginalization performed in `FD_marg_halfnormal_Gaussianlikelihood`.

    Under the half-normal prior with s_(j) = s_opt,(j) / SNR_opt,(j), and with the pulses
    assumed non-overlapping, the components are conditionally independent and each fluence
    follows a normal distribution truncated to A > 0:

        A_(j) ~ N( mu_(j), sigma_A,(j)^2 ) restricted to A_(j) > 0,
        mu_(j)      = [s_opt,(j)^2 / (1 + s_opt,(j)^2)] SNR_mf,(j) / SNR_opt,(j),
        sigma_A,(j) = [s_opt,(j) / sqrt(1 + s_opt,(j)^2)] / SNR_opt,(j),

    so that mu_(j) / sigma_A,(j) = z_(j). Positivity is built into the prior, so unlike the
    full-normal marginalization every draw is valid and no rejection step is required.

    snr_opt: (n_samples, n_pulses), the 'snr_opt' deterministic site
    snr_mf:  (n_samples, n_pulses), the 'snr_mf' deterministic site

    Returns fluences of shape (n_samples, n_pulses).
    """
    s = jnp.asarray(s_opt, dtype=snr_mf.dtype)
    z = s * snr_mf / jnp.sqrt(1.0 + s ** 2)

    # Conditional standard deviation of the fluence
    sigma_A = (s / jnp.sqrt(1.0 + s ** 2)) / snr_opt

    # Standardized truncated-normal draw: y ~ N(z, 1) restricted to y > 0, by inverse CDF.
    # Written as z - ndtri(v Phi(z)) rather than ndtri(Phi(-z) + u[1 - Phi(-z)]) so that
    # no precision is lost when Phi(-z) approaches 1 (strongly negative z).
    v = jax.random.uniform(key, snr_mf.shape, dtype=snr_mf.dtype,
                           minval=jnp.finfo(snr_mf.dtype).tiny, maxval=1.0)
    y = z - jax.scipy.special.ndtri(v * jax.scipy.stats.norm.cdf(z))

    # Convert from SNR units to fluence
    return sigma_A * y


# -------------------------------------------------------------------
# 4. NumPyro models
# -------------------------------------------------------------------

def FD_marg_fullnormal_Gaussianlikelihood(
    data_fd: jnp.ndarray,
    sigma_noise: jnp.ndarray,
    n_time: int,
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    single_pulse_fn: Callable,
    s_opt: float,
    overlap_K: float = 100.0,
    overlap_R0: float = 0.9,
    overlap_w: float = 0.002,
    time_param: str = "mean_arrival_time",
):
    """
    Fourier-domain Gaussian likelihood with the pulse fluences marginalized analytically using a zero-mean Gaussian prior.

    data_fd:     complex (nfreq, nfourier), in the convention of the module docstring
    sigma_noise: std of the real/imaginary part of each bin, shape (nfreq, 1) or (nfreq, nfourier)
    n_time:      number of time samples N (identifies the Nyquist bin)
    s_opt:       characteristic SNR scale of the fluence prior (scalar or per-pulse)

    Fluences must NOT be in priors: templates are evaluated at unit fluence.
    Recover fluences with `draw_fluences_fullnormal_gaussmarg` from the 'gram' and 'rho_mf' sites.
    """
    if "A" in priors:
        raise ValueError("'A' is marginalized analytically and must not be in priors.")

    _validate_fourier_inputs(data_fd, sigma_noise, n_time)

    # Sample shape parameters and evaluate the unit-fluence templates
    params_stacked = sample_pulse_parameters(n_pulses, priors, time_param)
    templates = pulse_templates(params_stacked, single_pulse_fn)

    # Restrict to the retained bins (DC and Nyquist discarded; see module docstring)
    keep = _retained_fourier_bins(n_time)
    sigma = jnp.broadcast_to(jnp.asarray(sigma_noise, dtype=jnp.float32), data_fd.shape)[:, keep]
    rho, gram = matched_filter_and_gram(templates[..., keep], data_fd[:, keep], 1.0 / sigma ** 2)

    # Analytic marginalization over the fluences
    log_like, R, snr_opt, snr_mf = marginalized_log_likelihood(rho, gram, s_opt)

    # Stored per sample so fluences can be drawn afterwards without re-evaluating templates
    numpyro.deterministic("gram", gram)
    numpyro.deterministic("rho_mf", rho)
    numpyro.deterministic("snr_opt", snr_opt)
    numpyro.deterministic("snr_mf", snr_mf)

    if n_pulses > 1:
        # Overlaps recorded for diagnostics, and penalized near total overlap
        numpyro.deterministic("overlap", R[jnp.triu_indices(n_pulses, k=1)])
        numpyro.factor("overlap_prior", overlap_log_prior(R, overlap_K, overlap_R0, overlap_w))

    numpyro.factor("marginalized_log_likelihood", log_like)

def FD_marg_halfnormal_Gaussianlikelihood(
    data_fd: jnp.ndarray,
    sigma_noise: jnp.ndarray,
    n_time: int,
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    single_pulse_fn: Callable,
    s_opt: float,
    overlap_K: float = 100.0,
    overlap_R0: float = 0.9,
    overlap_w: float = 0.002,
    time_param: str = "mean_arrival_time",
):
    """
    Fourier-domain likelihood with the fluences marginalized under a half-normal prior,
    assuming the pulse components do not overlap (G_jk = SNR_opt,j^2 delta_jk):

        ln p = sum_j { z_j^2 / 2 + ln[1 + erf(z_j / sqrt(2))] - 1/2 ln(1 + s_opt,j^2) },
        z_j  = s_opt,j SNR_mf,j / sqrt(1 + s_opt,j^2).

    Unlike the full-normal marginalization this is NOT symmetric under SNR_mf -> -SNR_mf:
    the erf term suppresses negative-fluence solutions, so anti-correlated placements are
    not spurious maxima. The non-overlap approximation is exact only for disjoint components,
    so the overlap prior below is what keeps it valid; check the recorded 'overlap' afterwards.

    Fluences must NOT be in priors: templates are evaluated at unit fluence. The conditional
    fluence distribution here is a normal TRUNCATED to A > 0 (independent across components),
    not the multivariate normal of the full-normal marginalization.
    """
    if "A" in priors:
        raise ValueError("'A' is marginalized analytically and must not be in priors.")

    # Sample shape parameters and evaluate the unit-fluence templates
    params_stacked = sample_pulse_parameters(n_pulses, priors, time_param)
    templates = pulse_templates(params_stacked, single_pulse_fn)

    # Restrict to the retained bins (DC and Nyquist discarded; see module docstring)
    keep = _retained_fourier_bins(n_time)
    sigma = jnp.broadcast_to(jnp.asarray(sigma_noise, dtype=jnp.float32), data_fd.shape)[:, keep]
    rho, gram = matched_filter_and_gram(templates[..., keep], data_fd[:, keep], 1.0 / sigma ** 2)

    # Only the diagonal of the Gram matrix enters the likelihood: this is where the
    # non-overlap approximation is applied. The off-diagonals are kept for the prior below.
    snr_opt, snr_mf, R = _snr_and_overlaps(rho, gram)

    s = jnp.broadcast_to(jnp.asarray(s_opt, dtype=gram.dtype), (n_pulses,))
    z = s * snr_mf / jnp.sqrt(1.0 + s ** 2)

    # ln[1 + erf(z/sqrt2)] = ln2 + ln Phi(z), evaluated without overflowing exp(z^2/2)
    log_like = jnp.sum(
        0.5 * z ** 2 + jnp.log(2.0) + norm.logcdf(z) - 0.5 * jnp.log1p(s ** 2)
    )

    # Stored per sample for fluence recovery and diagnostics
    numpyro.deterministic("gram", gram)
    numpyro.deterministic("rho_mf", rho)
    numpyro.deterministic("snr_opt", snr_opt)
    numpyro.deterministic("snr_mf", snr_mf)
    numpyro.deterministic("z", z)

    if n_pulses > 1:
        # Overlaps recorded for diagnostics, and penalized to keep the non-overlap approximation valid
        numpyro.deterministic("overlap", R[jnp.triu_indices(n_pulses, k=1)])
        numpyro.factor("overlap_prior", overlap_log_prior(R, overlap_K, overlap_R0, overlap_w))

    numpyro.factor("marginalized_log_likelihood", log_like)


def FD_gaussian_likelihood(
    data_fd: jnp.ndarray,
    sigma_noise: jnp.ndarray,
    n_time: int,
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    single_pulse_fn: Callable,
    time_param: str = "mean_arrival_time",
):
    """
    Fourier-domain Gaussian likelihood with fluences sampled directly (not marginalized),
    ln p = -1/2 <d - M | d - M>. Useful for validating the marginalized model.

    'A' must be in priors, otherwise every fluence would silently default to 1.
    """
    if "A" not in priors:
        raise ValueError("'A' must be in priors; otherwise fluences default to 1.")

    _validate_fourier_inputs(data_fd, sigma_noise, n_time)

    # Sample all parameters and sum the pulses (fluences included)
    params_stacked = sample_pulse_parameters(n_pulses, priors, time_param)
    model_fd = jnp.sum(pulse_templates(params_stacked, single_pulse_fn), axis=0)

    # Restrict to the retained bins (DC and Nyquist discarded; see module docstring)
    keep = _retained_fourier_bins(n_time)
    residual = (data_fd - model_fd)[:, keep]
    sigma = jnp.broadcast_to(jnp.asarray(sigma_noise, dtype=jnp.float32), data_fd.shape)[:, keep]

    # Real and imaginary parts are independent with variance sigma^2
    numpyro.factor("log_likelihood", -0.5 * jnp.sum(jnp.abs(residual) ** 2 / sigma ** 2))


def TD_gaussian_likelihood(
    data_td: jnp.ndarray,
    sigma_noise_td: jnp.ndarray,
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    single_pulse_fn: Callable,
    dt: float,
    time_param: str = "mean_arrival_time",
):
    """
    Time-domain Gaussian likelihood on the inverse transform of the Fourier-domain model.
    Validates the Fourier convention and Gaussian likelihood when DC/Nyquist handling is matched.

    data_td:        real (nfreq, ntime); must be baseline-subtracted, since this likelihood
                    uses every Fourier bin of the model, including DC
    sigma_noise_td: per-channel time-domain noise std, shape (nfreq, 1)

    The model is periodic over the segment (see the wraparound discussion in the paper).
    'A' must be in priors.
    """
    if "A" not in priors:
        raise ValueError("'A' must be in priors; otherwise fluences default to 1.")

    # Sample all parameters and sum the pulses (fluences included)
    params_stacked = sample_pulse_parameters(n_pulses, priors, time_param)
    model_fd = jnp.sum(pulse_templates(params_stacked, single_pulse_fn), axis=0)

    # Inverse of X_c = dt * rfft(x) is x = irfft(X, n=N) / dt
    model_td = jnp.fft.irfft(model_fd, n=data_td.shape[-1], axis=-1) / jnp.float32(dt)

    numpyro.sample("obs", dist.Normal(model_td, sigma_noise_td), obs=data_td)
