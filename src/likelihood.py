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
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import OrderedTransform, biject_to


_TINY = jnp.finfo(jnp.float32).tiny


# -------------------------------------------------------------------
# 1. Priors
# -------------------------------------------------------------------

def sample_ordered_arrival_times(
    name: str,
    distribution: dist.Distribution,
    n_pulses: int,
) -> jnp.ndarray:
    to_support = biject_to(distribution.support)

    # Bijectors in NumPyro define .sign (+1 for increasing, -1 for decreasing)
    if getattr(to_support, "sign", 1) != 1:
        raise ValueError(
            f"Cannot time-order prior '{name}': support bijection {to_support!r} "
            f"is monotonically decreasing and would invert arrival times to t_1 > t_2."
        )

    # Unconstrained latent vector
    z = numpyro.sample(
        f"{name}_unconstrained",
        dist.ImproperUniform(constraints.real_vector, (), event_shape=(n_pulses,)),
    )

    to_ordered = OrderedTransform()
    ordered = to_ordered(z)
    t = to_support(ordered)

    log_jacobian = (
        to_ordered.log_abs_det_jacobian(z, ordered)
        + jnp.sum(to_support.log_abs_det_jacobian(ordered, t))
    )
    numpyro.factor(f"{name}_prior", jnp.sum(distribution.log_prob(t)) + log_jacobian)

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

    Constant for R_jk < R0 and rising with slope approaching K / (1 - R0) above it;
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
    eye = jnp.eye(rho.shape[0], dtype=gram.dtype)

    # Floor guards against division by zero for a degenerate (all-zero) template
    snr_opt = jnp.sqrt(jnp.maximum(jnp.diag(gram), _TINY))
    snr_mf = rho / snr_opt

    R = gram / (snr_opt[:, None] * snr_opt[None, :])
    # Diagonal is exactly 1 by definition
    R = R * (1.0 - eye) + eye
    return snr_opt, snr_mf, R


# -------------------------------------------------------------------
# 3. Fluence marginalization (zero-mean Gaussian fluence prior)
# -------------------------------------------------------------------

def _gaussian_fluence_prior_cholesky(
    R: jnp.ndarray,
    s_opt: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cholesky factor of [R + diag(s_opt^-2)], the matrix defining both the marginalized
    likelihood and the conditional fluence distribution under the zero-mean Gaussian fluence
    prior with s_(j) = s_opt,(j) / SNR_opt,(j). Shared by both so they cannot fall out of sync.

    Returns (cholesky factor, s_opt broadcast to shape (n_pulses,)).
    """
    s_opt = jnp.broadcast_to(jnp.asarray(s_opt, dtype=R.dtype), R.shape[:1])
    return jnp.linalg.cholesky(R + jnp.diag(s_opt ** -2)), s_opt


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
    chol, s_opt = _gaussian_fluence_prior_cholesky(R, s_opt)

    # Quadratic term via a triangular solve rather than an explicit inverse
    v = jax.scipy.linalg.solve_triangular(chol, snr_mf, lower=True)
    quadratic = 0.5 * jnp.sum(v ** 2)

    # ln det[1 + diag(s_opt^2) R] = sum_j ln s_opt,j^2 + ln det[R + diag(s_opt^-2)]
    log_det = 2.0 * jnp.sum(jnp.log(s_opt)) + 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))

    return quadratic - 0.5 * log_det, R, snr_opt, snr_mf


def draw_fluences_gaussian_marginalized(
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


# -------------------------------------------------------------------
# 4. NumPyro models
# -------------------------------------------------------------------

def fourier_marginalized_likelihood(
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
    Fourier-domain likelihood with the pulse fluences marginalized analytically.

    data_fd:     complex (nfreq, nfourier), in the convention of the module docstring
    sigma_noise: std of the real/imaginary part of each bin, shape (nfreq, 1) or (nfreq, nfourier)
    n_time:      number of time samples N (identifies the Nyquist bin)
    s_opt:       characteristic SNR scale of the fluence prior (scalar or per-pulse)

    Fluences must NOT be in priors: templates are evaluated at unit fluence.
    Recover fluences with `draw_fluences_gaussian_marginalized` from the 'gram' and 'rho_mf' sites.
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


def fourier_gaussian_likelihood(
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

    # Sample all parameters and sum the pulses (fluences included)
    params_stacked = sample_pulse_parameters(n_pulses, priors, time_param)
    model_fd = jnp.sum(pulse_templates(params_stacked, single_pulse_fn), axis=0)

    # Restrict to the retained bins (DC and Nyquist discarded; see module docstring)
    keep = _retained_fourier_bins(n_time)
    residual = (data_fd - model_fd)[:, keep]
    sigma = jnp.broadcast_to(jnp.asarray(sigma_noise, dtype=jnp.float32), data_fd.shape)[:, keep]

    # Real and imaginary parts are independent with variance sigma^2
    numpyro.factor("log_likelihood", -0.5 * jnp.sum(jnp.abs(residual) ** 2 / sigma ** 2))


def time_domain_gaussian_likelihood(
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
