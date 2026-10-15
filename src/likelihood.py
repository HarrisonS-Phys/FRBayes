"""
likelihoods.py - NumPyro likelihoods for the Fourier-domain FRB pulse model.

CONVENTIONS (must match model.py and the paper):
1. FOURIER TRANSFORM: data and templates use X_c = dt * sum_b x_b exp(-2 pi i f_c t_b),
   i.e. dt * rfft(x).
2. NOISE: `sigma_noise_fd` is the standard deviation of the real (equivalently, imaginary) part
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
import numpy as np
from jax.scipy.special import ndtri
from jax.scipy.stats import norm
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import OrderedTransform, biject_to
import warnings


_EPS = jnp.finfo(jnp.float32).eps  # floor for SNR_opt in _snr_and_overlaps, guarding division by a degenerate (zero-power) template

# z below this is where norm.cdf(z) loses precision and then underflows to 0 in float32,
# which sends ndtri to -inf in the truncated-normal draw. See draw_fluences_halfnormal_nooverlap_gaussmarg.
_Z_UNDERFLOW_WARN = -10.0

def _validate_fourier_inputs(data_fd, sigma_noise_fd):
    """
    Checks that sigma_noise_fd broadcasts against data_fd. The number of Fourier bins is whatever
    data.py selected, so there is nothing to validate it against here; the model must be built
    with the matching `fourier_freqs`, which the einsum shapes will catch if it is not.
    """
    sigma_noise_fd = jnp.asarray(sigma_noise_fd)
    if sigma_noise_fd.ndim != 2:
        raise ValueError(
            f"sigma_noise_fd must be 2D, shape (nfreq, 1) or (nfreq, nfourier); got {sigma_noise_fd.shape}."
        )
    if sigma_noise_fd.shape[0] != data_fd.shape[0]:
        raise ValueError(
            f"sigma_noise_fd has {sigma_noise_fd.shape[0]} channels but data_fd has {data_fd.shape[0]}."
        )
    if sigma_noise_fd.shape[1] not in (1, data_fd.shape[-1]):
        raise ValueError(
            f"sigma_noise_fd has {sigma_noise_fd.shape[1]} Fourier bins; expected 1 (broadcast) "
            f"or {data_fd.shape[-1]} to match data_fd."
        )

# -------------------------------------------------------------------
# 1. Priors
# -------------------------------------------------------------------

def sample_ordered_arrival_times(name, distribution, n_pulses):
    """Sample n_pulses ordered arrival times from a target distribution.

    Enforces temporal ordering t_0 < t_1 < ... < t_{P-1}, with the prior density of the ordered
    times equal to prod_j p(t_j) on the ordered region (i.e. the order statistics of n_pulses
    i.i.d. draws), and no other relative-time prior imposed.

    The sampler explores an unconstrained vector z with a standard normal sample site. That
    site contributes its own log-density to the joint, which is NOT part of the intended prior,
    so it is subtracted in the factor below; leaving it in place multiplies the prior on t by
    prod_j N(z_j; 0, 1), which pulls the arrival times toward the middle of the support. The
    ln(P!) term is a constant and is kept only so the factor matches the order-statistics
    density as written.
    """
    if n_pulses == 1:
        return numpyro.sample(name, distribution.expand([1]))

    # Ordering by sorting is only equivalent to an ordered prior if the map to the support is
    # monotonic, which `sign` reports; multivariate supports have no such notion
    if distribution.event_shape != ():
        raise ValueError(
            f"the arrival-time prior must be univariate; got event_shape {distribution.event_shape}."
        )
    to_support = dist.transforms.biject_to(distribution.support)
    try:
        sign = to_support.sign
    except (AttributeError, NotImplementedError):
        raise ValueError(
            f"support transform {type(to_support).__name__} of the arrival-time prior does not "
            f"define `sign`, so it cannot be verified as monotonic and the ordering may not hold."
        )

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

    # Correct log-density for order statistics: sum(ln p(t_i)) + sum(ln |J|) + ln(P!),
    # minus the log-density of the base sample site so it is not double counted
    log_prob_base = jnp.sum(distribution.log_prob(t))
    log_det_J = jnp.sum(to_support.log_abs_det_jacobian(ordered_for_support, t))
    log_factorial_p = jnp.sum(jnp.log(jnp.arange(1, n_pulses + 1, dtype=jnp.float32)))
    log_prob_z_base = jnp.sum(dist.Normal(0.0, 1.0).log_prob(z))

    numpyro.factor(
        f"{name}_order_factor",
        log_prob_base + log_det_J + log_factorial_p - log_prob_z_base
    )

    return numpyro.deterministic(name, t)

def sample_contained_widths(
    n_pulses: int,
    extent_max: float,
    sigma_min: float,
    bot_over_ref: float,
    n_sigma: float = 3.0,
    n_tau: float = 5.0,
    scattering_index: float = -4.0,
    flat_in_sigma_tau: bool = True,
):
    """
    Samples the intrinsic width and scattering timescale subject to the pulse fitting inside the
    segment, which is a single joint condition on the convolved pulse rather than two separate
    bounds:

        n_sigma * sigma + n_tau * tau_bot <= extent_max

    with tau_bot the scattering timescale at the BOTTOM of the band, where it is longest. Giving
    sigma and tau independent bounds instead would allow a draw at both maxima, which is twice as
    wide as intended and wraps around the segment; the Fourier-domain model is periodic, so a
    pulse that does not fit reappears at the start of the segment.

    Enforced by sampling sigma, then giving the scattering whatever of the budget is left:
    a box in (sigma, u), so the sampler sees no diagonal boundary. tau -> 0 is reachable at u = 0,
    which is what an unscattered burst needs.

    extent_max:       the share of the segment allotted to the pulse, xi * W_seg. The rest is for
                      the dispersion sweep, so it should be consistent with how D_max was set.
    sigma_min:        narrowest resolvable width, 1 / (2 pi f_max) from the Fourier truncation:
                      a narrower template has not rolled off within the retained bins, so the
                      data do not constrain it and the sampler would wander.
    bot_over_ref:     nu_bot / nu_r, converting tau at the bottom of the band to tau_ref
    n_sigma, n_tau:   how many widths and e-folds must fit. 3 leaves 0.1% of the Gaussian and 5
                      leaves 0.7% of the exponential outside the window.
    flat_in_sigma_tau: the construction is flat in the SAMPLED coordinates (sigma, tau_frac).
                      That is not flat in the PHYSICAL ones (sigma, tau), because tau_frac is a
                      fraction of a budget that shrinks as sigma grows: a wide pulse has less of
                      the segment left for its tail, so its tau is squeezed into a shorter range
                      and its density there is correspondingly higher. True by default, which
                      adds a Jacobian factor cancelling that, giving a prior flat in (sigma, tau)
                      over the allowed region. Set False to leave the prior flat in (sigma,
                      tau_frac) instead, i.e. to treat the fraction rather than the timescale as
                      the quantity with the uniform prior.

    Returns (sigma, tau_ref), both shape (n_pulses,). 'tau_ref' and 'tau_bot' are recorded as
    deterministic sites; 'sigma' and 'tau_frac' are the sampled ones.
    """
    sigma = numpyro.sample(
        "sigma", dist.Uniform(sigma_min, extent_max / n_sigma).expand([n_pulses])
    )
    tau_frac = numpyro.sample("tau_frac", dist.Uniform(0.0, 1.0).expand([n_pulses]))

    # Whatever of the containment budget the width has not used
    budget = extent_max - n_sigma * sigma
    tau_bot = tau_frac * budget / n_tau

    numpyro.deterministic("tau_bot", tau_bot)
    tau_ref = numpyro.deterministic(
        "tau_ref", tau_bot * jnp.float32(bot_over_ref) ** (-jnp.float32(scattering_index))
    )

    if flat_in_sigma_tau:
        # d(tau) / d(tau_frac) = budget / n_tau, so adding its log cancels the varying width of
        # tau's conditional range and leaves the prior flat in (sigma, tau)
        numpyro.factor("width_containment_jacobian", jnp.sum(jnp.log(budget / n_tau)))

    return sigma, tau_ref

def wraparound_penalty(ratio, ratio_min=0.9, K=100.0, w=0.01):
    """Soft penalty: ~0 for ratio > ratio_min, rising to K nats at ratio = 0, sharpness w."""
    return -(K / jax.nn.softplus(ratio_min / w)) * jax.nn.softplus((ratio_min - ratio) / w)

def _wraparound_ratio_value(params_stacked, single_pulse_fn, amplitudes, inv_noise_var):
    """
    Pure computation, no numpyro sites: R = sqrt(P(f1) / P(f1/2)) of the full summed model,
    P the noise-weighted power, f1 = 1 / W_seg. Used both inside the models and post-hoc.

    amplitudes: (n_pulses,) multipliers on top of the templates (ones if the templates
                already carry the fluence). Only their ratios matter.
    """
    lowfreq_diagnostic = getattr(single_pulse_fn, "lowfreq_diagnostic", None)
    if lowfreq_diagnostic is None:
        raise ValueError("single_pulse_fn has no .lowfreq_diagnostic closure; build it with "
                         "setup_model(..., _wraparoundFDdiagnostic=True).")

    models = jax.vmap(lowfreq_diagnostic)(params_stacked)                 # (P, nfreq, 2)
    model = jnp.sum(amplitudes[:, None, None] * models, axis=0)           # (nfreq, 2)
    power = jnp.sum(jnp.abs(model) ** 2 * jnp.asarray(inv_noise_var)[:, :1], axis=0)  # [P(f1/2), P(f1)]
    return jnp.sqrt(power[1] / jnp.maximum(power[0], jnp.finfo(jnp.float32).tiny))


def wraparound_ratio(params_stacked, single_pulse_fn, amplitudes, inv_noise_var,
                     containment_FD=None, site_name="wraparound_ratio"):
    """In-model version: records the ratio under `site_name`, and penalizes it if
    containment_FD is given."""
    ratio = numpyro.deterministic(
        site_name,
        _wraparound_ratio_value(params_stacked, single_pulse_fn, amplitudes, inv_noise_var),
    )
    if containment_FD is not None:
        numpyro.factor("wraparound_penalty", wraparound_penalty(ratio, **containment_FD))
    return ratio

def sample_pulse_parameters(
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    time_param: str = "mean_arrival_time",
    containment: dict = None,          # WIDTH CONTAINMENT SUPPORT: remove with the block below
    spectral_coeff_map: dict = None,   # ORTHOGONAL SED SUPPORT: remove with the block below
) -> Dict[str, jnp.ndarray]:
    """
    Samples every pulse parameter from the prior dictionary into stacked arrays of shape
    (n_pulses,). Arrival times are ordered across pulses; all other parameters are i.i.d.

    spectral_coeff_map: if the orthogonal spectral basis is in use, the conversion coefficients
                        from `orthogonal_spectral_basis`; gamma and beta are then recorded as
                        deterministic sites alongside the sampled c1 and c2. Defaults to None,
                        in which case nothing in that block runs.
    containment:        if given, sigma and tau_ref are drawn jointly so the pulse fits inside
                        the segment (see `sample_contained_widths`) rather than from independent
                        priors; 'sigma' and 'tau_ref' must then be absent from `priors`. Passed
                        through as keyword arguments, so it can also override n_sigma, n_tau and
                        flat_in_sigma_tau.

    TO REMOVE ORTHOGONAL SED SUPPORT: delete the `spectral_coeff_map` argument above, its
    delimited block below, and the argument and its pass-through in each likelihood in Section 4.
    The same applies to `containment`. Nothing else depends on either.
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

    # ------------------------- BEGIN ORTHOGONAL SED SUPPORT -------------------------
    # c1 and c2 are coefficients in a basis built from this burst's own channel weights, so they
    # are not comparable between bursts; gamma and beta are defined identically for every burst
    # and are what should be reported. Recording them here keeps the conversion in one place.
    if spectral_coeff_map is not None:
        missing = {"c1", "c2"} - params_stacked.keys()
        if missing:
            raise ValueError(
                f"spectral_coeff_map was given but {sorted(missing)} not in priors; it applies "
                f"only when sampling the orthogonal spectral basis."
            )
        c1, c2 = params_stacked["c1"], params_stacked["c2"]
        numpyro.deterministic(
            "gamma", spectral_coeff_map["gamma_c1"] * c1 + spectral_coeff_map["gamma_c2"] * c2
        )
        numpyro.deterministic("beta", spectral_coeff_map["beta_c2"] * c2)
    # -------------------------- END ORTHOGONAL SED SUPPORT --------------------------

    # ------------------------ BEGIN WIDTH CONTAINMENT SUPPORT ------------------------
    # sigma and tau_ref are coupled by the requirement that the pulse fit inside the segment, so
    # they cannot be expressed as independent entries in `priors` and are drawn here instead.
    # `containment["method"]` selects how containment is imposed; the remaining entries are the
    # arguments of that method, which differ between them.
    if containment is not None:
        clash = {"sigma", "tau_ref"} & priors.keys()
        if clash:
            raise ValueError(
                f"{sorted(clash)} must not be in priors when `containment` is given; the widths "
                f"are drawn jointly by the containment method."
            )

        methods = {
            "sample_contained_widths": sample_contained_widths,        # n_sigma widths + n_tau e-folds fit in the segment
        }
        settings = dict(containment)
        method = settings.pop("method", "sample_contained_widths")
        if method not in methods:
            raise ValueError(f"unknown containment method '{method}'; expected one of {sorted(methods)}.")

        params_stacked["sigma"], params_stacked["tau_ref"] = methods[method](n_pulses, **settings)
    # ------------------------- END WIDTH CONTAINMENT SUPPORT -------------------------

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

def pulse_templates(
    params_stacked: Dict[str, jnp.ndarray],
    single_pulse_fn: Callable[[Dict[str, jnp.ndarray]], jnp.ndarray],
) -> jnp.ndarray:
    """Evaluates all pulse templates in parallel; shape (n_pulses, nfreq, nfourier)."""
    return jax.vmap(single_pulse_fn)(params_stacked)


def matched_filter_and_gram(
    templates: jnp.ndarray,
    data_fd: jnp.ndarray,
    inv_noise_var: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    rho_j = <d|h_j>    unnormalized matched filter, shape (n_pulses,)
    G_jk  = <h_j|h_k>  Gram matrix of template overlaps, shape (n_pulses, n_pulses)

    All inputs are restricted to the retained Fourier bins; inv_noise_var = 1 / sigma_noise_fd^2.
    """
    weighted = templates * inv_noise_var[None]
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
# 3. Fluence marginalization
#
# These are pure functions of arrays: given the matched filter and the Gram matrix they return
# a marginalized log-likelihood, or draw fluences, with no NumPyro sample sites involved.
# They are kept separate from the NumPyro models of Section 4 for two reasons. First, the
# algebra can then be unit-tested directly against numerical integration over the fluences,
# without running a sampler. Second, each marginalized likelihood shares its linear algebra
# with its paired fluence sampler, so the conditional distribution used to recover fluences
# cannot drift out of step with the prior that was marginalized over.
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


def fullnormal_gaussmarg_log_likelihood(
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

def halfnormal_nooverlap_gaussmarg_log_likelihood(
    rho: jnp.ndarray,
    gram: jnp.ndarray,
    s_opt: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fluence-marginalized log-likelihood under a half-normal fluence prior, assuming the pulse
    components do not overlap (G_jk = SNR_opt,j^2 delta_jk), up to a parameter-independent
    constant:

        ln p = sum_j { z_j^2 / 2 + ln[1 + erf(z_j / sqrt(2))] - 1/2 ln(1 + s_opt,j^2) },
        z_j  = s_opt,j SNR_mf,j / sqrt(1 + s_opt,j^2).

    Only the diagonal of `gram` enters: that is where the non-overlap approximation is applied.
    The full matrix is still taken as an argument so the caller can use the off-diagonals for
    the overlap prior and for diagnostics.

    Unlike the full-normal case this is NOT symmetric under SNR_mf -> -SNR_mf: the erf term
    suppresses negative-fluence solutions, so anti-correlated template placements are not
    spurious likelihood maxima.

    Returns (log_likelihood, R, snr_opt, snr_mf).
    """
    snr_opt, snr_mf, R = _snr_and_overlaps(rho, gram)
    s_opt_broadcast = jnp.broadcast_to(jnp.asarray(s_opt, dtype=jnp.float32), rho.shape)

    z = s_opt_broadcast * snr_mf / jnp.sqrt(1.0 + s_opt_broadcast ** 2)

    # ln[1 + erf(z/sqrt2)] = ln2 + ln Phi(z), evaluated without overflowing exp(z^2/2)
    log_like = jnp.sum(
        0.5 * z ** 2 + jnp.log(2.0) + norm.logcdf(z) - 0.5 * jnp.log1p(s_opt_broadcast ** 2)
    )

    return log_like, R, snr_opt, snr_mf


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

def draw_fluences_halfnormal_nooverlap_gaussmarg(
    snr_opt: jnp.ndarray,
    snr_mf: jnp.ndarray,
    s_opt: float,
    key: jax.Array,
) -> jnp.ndarray:
    """
    Draws fluences for the marginalization performed in `FD_marg_halfnormal_nooverlap_gaussianlikelihood`.

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
    # norm.cdf(z) loses precision and then underflows to 0 in float32 for z below about -13,
    # which sends ndtri to -inf and the draw to +inf. A component there has a fitted fluence
    # more than 10 sigma below zero, i.e. a pathological fit rather than a physical one, so this
    # warns rather than silently clipping. The check needs concrete values, so it is skipped
    # when the function is traced (under jit); post-processing use is the intended case.
    try:
        z_min = float(jnp.min(z))
        if z_min < _Z_UNDERFLOW_WARN:
            warnings.warn(
                f"minimum z = {z_min:.1f} is below {_Z_UNDERFLOW_WARN}: the truncated-normal "
                f"draw may underflow to a non-finite fluence. Such components have strongly "
                f"negative fitted fluences and should be inspected rather than trusted.",
                RuntimeWarning,
            )
    except TypeError:
        pass

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

def wraparound_ratio_posthoc(samples, fluences, single_pulse_fn, inv_noise_var,
                             param_names=("mean_arrival_time", "sigma", "DM", "tau_ref", "c1", "c2")):
    """
    Exact wraparound ratio for every posterior draw, from fluences drawn after sampling
    (draw_fluences_halfnormal_nooverlap_gaussmarg or draw_fluences_fullnormal_gaussmarg).

    samples:     dict of posterior draws, each (n_draws, n_pulses), as from mcmc.get_samples()
    fluences:    (n_draws, n_pulses)
    param_names: the parameters single_pulse_fn reads; add 'delta' if it is sampled

    Returns (n_draws,).
    """
    params = {k: samples[k] for k in param_names}
    return jax.vmap(
        lambda p, a: _wraparound_ratio_value(p, single_pulse_fn, a, inv_noise_var)
    )(params, fluences)

# -------------------------------------------------------------------
# 4. NumPyro models
#
# "Model" here is NumPyro's sense of the word: a function that declares the prior sample sites
# and contributes a log-likelihood factor, thereby specifying the full posterior. It is NOT a
# model of the burst. The physical pulse models -- the templates T and F of model.py -- enter
# only through `single_pulse_fn`, and each function below samples the pulse parameters,
# evaluates those templates, and applies one of the likelihoods of Sections 2 and 3.
# -------------------------------------------------------------------

def FD_marg_fullnormal_gaussianlikelihood(
    data_fd: jnp.ndarray,
    sigma_noise_fd: jnp.ndarray,
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    single_pulse_fn: Callable,
    s_opt: float,
    overlap_K: float = 100.0,
    overlap_R0: float = 0.9,
    overlap_w: float = 0.002,
    time_param: str = "mean_arrival_time",
    containment: dict = None, #support for contained pulse priors within data segment, remove with pass-through below if deleting
    containment_FD: dict = None,
    spectral_coeff_map: dict = None,   # ORTHOGONAL SED SUPPORT: remove with the pass-through below if deleting orthog support
):
    """
    Fourier-domain Gaussian likelihood with the pulse fluences marginalized analytically using a zero-mean Gaussian prior.

    data_fd:     complex (nfreq, nfourier), in the convention of the module docstring
    sigma_noise_fd: std of the real/imaginary part of each bin, shape (nfreq, 1) or (nfreq, nfourier)
    s_opt:       characteristic SNR scale of the fluence prior (scalar or per-pulse)
    spectral_coeff_map: if the orthogonal spectral basis is in use, the conversion coefficients
                        from `orthogonal_spectral_basis`; gamma and beta are then recorded as
                        deterministic sites alongside the sampled c1 and c2.
    containment:    if given, sigma and tau_ref are drawn jointly so the pulse fits inside the
                    segment (see `sample_contained_widths`) rather than from independent priors;
                    'sigma' and 'tau_ref' must then be absent from `priors`.
    containment_FD: if given, dict(ratio_min, K, w), the wraparound penalty is applied during
                    sampling, with the ratio evaluated at the conditional-mean fluences and
                    recorded as 'wraparound_ratio_realtime_approx'. If None, nothing
                    wraparound-related is computed during sampling; compute the exact ratio
                    afterwards with `wraparound_ratio_posthoc` from the drawn fluences.

    Fluences must NOT be in priors: templates are evaluated at unit fluence.
    Recover fluences with `draw_fluences_fullnormal_gaussmarg` from the 'gram' and 'rho_mf' sites.
    """
    if "A" in priors:
        raise ValueError("'A' is marginalized analytically and must not be in priors.")

    _validate_fourier_inputs(data_fd, sigma_noise_fd)

    # Sample shape parameters and evaluate the unit-fluence templates
    #params_stacked = sample_pulse_parameters(n_pulses, priors, time_param)
    #params_stacked = sample_pulse_parameters(n_pulses, priors, time_param, spectral_coeff_map)
    params_stacked = sample_pulse_parameters(
        n_pulses, priors, time_param,
        spectral_coeff_map=spectral_coeff_map, containment=containment,
    )
    templates = pulse_templates(params_stacked, single_pulse_fn)

    # Restrict to the retained bins (DC and Nyquist discarded; see module docstring)
    sigma = jnp.broadcast_to(jnp.asarray(sigma_noise_fd, dtype=jnp.float32), data_fd.shape)
    inv_noise_var = 1.0 / sigma ** 2
    rho, gram = matched_filter_and_gram(templates, data_fd, inv_noise_var)

    # Analytic marginalization over the fluences
    log_like, R, snr_opt, snr_mf = fullnormal_gaussmarg_log_likelihood(rho, gram, s_opt)

    # Wraparound penalty, only if requested; otherwise the exact ratio is computed post-hoc
    if containment_FD is not None:
        # The fluences are integrated out, so the wraparound ratio, which depends on their
        # relative sizes, has no per-draw amplitude to use. We plug in the conditional mean
        # E[A | theta, d], which here is closed-form and nearly free: the conditional is the
        # Gaussian N([R + diag(s_opt^-2)]^-1 SNR_mf, ...) in SNR units, so its mean is one
        # Cholesky solve on an n_pulses x n_pulses matrix, divided by SNR_opt to give fluence.
        # Exact for a single pulse, where the amplitude cancels from the ratio. For several
        # pulses it is a plug-in approximation, and because the conditional mean depends on the
        # data, the penalty is then a data-dependent soft constraint rather than a strict prior.
        # The exact ratio is computed after sampling from the drawn fluences.
        chol, _ = _gaussian_fluence_prior_cholesky(R, s_opt)
        A_approx = jax.scipy.linalg.cho_solve((chol, True), snr_mf) / snr_opt
        wraparound_ratio(params_stacked, single_pulse_fn, A_approx, inv_noise_var,
                         containment_FD, site_name="wraparound_ratio_realtime_approx")

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

def FD_marg_halfnormal_nooverlap_gaussianlikelihood(
    data_fd: jnp.ndarray,
    sigma_noise_fd: jnp.ndarray,
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    single_pulse_fn: Callable,
    s_opt: float,
    apply_overlap_prior: bool = False,
    overlap_K: float = 100.0,
    overlap_R0: float = 0.9,
    overlap_w: float = 0.002,
    time_param: str = "mean_arrival_time",
    containment: dict = None, #support for contained pulse priors within data segment, remove with pass-through below if deleting
    containment_FD: dict = None,
    spectral_coeff_map: dict = None,   # ORTHOGONAL SED SUPPORT: remove with the pass-through below if deleting orthog support
):
    """
    Fourier-domain Gaussian likelihood with the pulse fluences marginalized analytically using a
    half-normal prior, assuming the pulse components do not overlap.

    data_fd:     complex (nfreq, nfourier), in the convention of the module docstring
    sigma_noise_fd: std of the real/imaginary part of each bin, shape (nfreq, 1) or (nfreq, nfourier)
    s_opt:       characteristic SNR scale of the fluence prior (scalar or per-pulse)
    spectral_coeff_map: if the orthogonal spectral basis is in use, the conversion coefficients
                        from `orthogonal_spectral_basis`; gamma and beta are then recorded as
                        deterministic sites alongside the sampled c1 and c2.
    containment:    if given, sigma and tau_ref are drawn jointly so the pulse fits inside the
                    segment (see `sample_contained_widths`) rather than from independent priors;
                    'sigma' and 'tau_ref' must then be absent from `priors`.
    containment_FD: if given, dict(ratio_min, K, w), the wraparound penalty is applied during
                    sampling, with the ratio evaluated at the conditional-mean fluences and
                    recorded as 'wraparound_ratio_realtime_approx'. If None, nothing
                    wraparound-related is computed during sampling; compute the exact ratio
                    afterwards with `wraparound_ratio_posthoc` from the drawn fluences.

    Fluences must NOT be in priors: templates are evaluated at unit fluence.
    Recover fluences with `draw_fluences_halfnormal_nooverlap_gaussmarg` from the 'snr_opt' and
    'snr_mf' sites; the conditional there is a truncated normal, not the multivariate normal of
    the full-normal marginalization.

    OVERLAP PRIOR (n_pulses > 1 only, and OFF by default):
    The likelihood discards the off-diagonal elements of the Gram matrix, so it cannot itself
    notice two components landing on the same pulse -- it would read that as two independent
    detections and double-count the evidence. The overlap prior of `overlap_log_prior` is
    therefore not a refinement of the likelihood but the thing that enforces its validity
    condition. Whether it is applied is controlled by `apply_overlap_prior`, kept separate from
    `overlap_K` so that switching the prior off is distinct from choosing a zero-nat penalty,
    which is itself a valid setting. It defaults to False because at the usual settings the
    prior is flat until R_jk approaches 1, so it only catches total collapse while the
    non-overlap approximation is already inaccurate by R_jk ~ 0.3; if it is switched on,
    overlap_R0 should be chosen low enough to actually bound the approximation error rather
    than left at 0.9.

    The normalized overlaps R_jk are recorded as 'overlap' whether or not the prior is applied,
    since the Gram matrix is computed regardless and they are the post-hoc check on whether the
    non-overlap assumption held. Inspect them before trusting a multi-pulse fit.
    """
    if "A" in priors:
        raise ValueError("'A' is marginalized analytically and must not be in priors.")

    _validate_fourier_inputs(data_fd, sigma_noise_fd)

    # Sample shape parameters and evaluate the unit-fluence templates
    #params_stacked = sample_pulse_parameters(n_pulses, priors, time_param)
    #params_stacked = sample_pulse_parameters(n_pulses, priors, time_param, spectral_coeff_map)
    params_stacked = sample_pulse_parameters(
        n_pulses, priors, time_param,
        spectral_coeff_map=spectral_coeff_map, containment=containment,
    )
    templates = pulse_templates(params_stacked, single_pulse_fn)

    # Restrict to the retained bins (DC and Nyquist discarded; see module docstring)
    sigma = jnp.broadcast_to(jnp.asarray(sigma_noise_fd, dtype=jnp.float32), data_fd.shape)
    inv_noise_var = 1.0 / sigma ** 2
    rho, gram = matched_filter_and_gram(templates, data_fd, inv_noise_var)

    # Analytic marginalization over the fluences; only the diagonal of the Gram matrix enters
    log_like, R, snr_opt, snr_mf = halfnormal_nooverlap_gaussmarg_log_likelihood(rho, gram, s_opt)

    # Wraparound penalty, only if requested; otherwise the exact ratio is computed post-hoc
    if containment_FD is not None:
        # The fluences are integrated out, so the wraparound ratio, which depends on their
        # relative sizes, has no per-draw amplitude to use. We plug in the conditional mean
        # E[A | theta, d], which here is closed-form and nearly free: under the half-normal prior
        # and non-overlap, each A_j is an independent normal truncated to A > 0, with location
        # z_j sigma_A,j, so its mean is sigma_A,j [z_j + phi(z_j) / Phi(z_j)]. The ratio
        # phi / Phi is evaluated in log space so it stays finite for negative z.
        # Exact for a single pulse, where the amplitude cancels from the ratio. For several
        # pulses it is a plug-in approximation, and because the conditional mean depends on the
        # data, the penalty is then a data-dependent soft constraint rather than a strict prior.
        # The exact ratio is computed after sampling from the drawn fluences.
        s = jnp.broadcast_to(jnp.asarray(s_opt, dtype=snr_mf.dtype), (n_pulses,))
        z = s * snr_mf / jnp.sqrt(1.0 + s ** 2)
        sigma_A = (s / jnp.sqrt(1.0 + s ** 2)) / snr_opt
        A_approx = sigma_A * (z + jnp.exp(norm.logpdf(z) - norm.logcdf(z)))
        wraparound_ratio(params_stacked, single_pulse_fn, A_approx, inv_noise_var,
                         containment_FD, site_name="wraparound_ratio_realtime_approx")

    # Stored per sample so fluences can be drawn afterwards without re-evaluating templates.
    # z_j is deliberately not stored: it is a fixed function of 'snr_mf' and the chosen s_opt,
    # z_j = s_opt,j SNR_mf,j / sqrt(1 + s_opt,j^2), so storing it would duplicate information and
    # leave the two marginalized likelihoods with different sets of recorded sites.
    numpyro.deterministic("gram", gram)
    numpyro.deterministic("rho_mf", rho)
    numpyro.deterministic("snr_opt", snr_opt)
    numpyro.deterministic("snr_mf", snr_mf)

    if n_pulses > 1:
        # Recorded regardless, as the check on the non-overlap assumption
        numpyro.deterministic("overlap", R[jnp.triu_indices(n_pulses, k=1)])

        # A Python bool, so this resolves at trace time: with the prior off, nothing is added
        # to the compiled graph
        if apply_overlap_prior:
            numpyro.factor("overlap_prior", overlap_log_prior(R, overlap_K, overlap_R0, overlap_w))

    numpyro.factor("marginalized_log_likelihood", log_like)

def FD_gaussianlikelihood(
    data_fd: jnp.ndarray,
    sigma_noise_fd: jnp.ndarray,
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    single_pulse_fn: Callable,
    time_param: str = "mean_arrival_time",
    containment: dict = None, #support for contained pulse priors within data segment, remove with pass-through below if deleting
    containment_FD: dict = None,
    spectral_coeff_map: dict = None,
):
    """
    Fourier-domain Gaussian likelihood with fluences sampled directly (not marginalized),
    ln p = -1/2 <d - M | d - M>.

    The amplitude is given EITHER as 'A', a fluence, OR as 'A_snr', the amplitude in units of
    the template's optimal SNR, with the fluence then derived as A = A_snr / SNR_opt. Exactly
    one of the two must be in priors; which was used is printed at trace time, and is recorded
    by the priors dictionary itself, which should be stored alongside the samples. Either way
    the fluence is available afterwards as the 'A' site, so downstream code need not branch.

    'A_snr' makes the prior a conditional one, p(A | theta), whose scale follows the template's
    detectability: a prior of fixed fluence scale instead penalizes templates in proportion to
    how detectable they are, biasing recovered widths and scattering timescales toward the more
    detectable, narrower templates. It is also scale-free, so the same number works for every
    burst where fluence bounds depend on how the data are normalized, and with
    dist.HalfNormal(s_opt) it reproduces exactly the prior that
    FD_marg_halfnormal_nooverlap_Gaussianlikelihood integrates over, making the two comparable.

    The cost is sampling efficiency: SNR_opt depends on the shape parameters, so A_snr couples
    to them through a ratio and the posterior geometry is worse. 'A' avoids that coupling at the
    price of a shape-dependent prior. Which is preferable depends on whether the comparison with
    the marginalized likelihood, or the sampling, matters more for the run at hand.

    'snr_opt' is recorded only in the 'A_snr' case, where it is computed; with 'A' the templates
    already carry the fluence and the optimal SNR is not formed.
    """
    amplitude_in_snr = "A_snr" in priors
    if amplitude_in_snr == ("A" in priors):
        raise ValueError("exactly one of 'A' (a fluence) and 'A_snr' (SNR units) must be in priors.")

    # Fires at trace time, so once per compilation rather than once per likelihood evaluation
    print(f"sampling the amplitude as {'A_snr (SNR units)' if amplitude_in_snr else 'A (fluence)'}")

    _validate_fourier_inputs(data_fd, sigma_noise_fd)

    params_stacked = sample_pulse_parameters(
        n_pulses, priors, time_param,
        spectral_coeff_map=spectral_coeff_map, containment=containment,
    )
    templates = pulse_templates(params_stacked, single_pulse_fn)

    inv_noise_var = 1.0 / jnp.broadcast_to(jnp.asarray(sigma_noise_fd, dtype=jnp.float32), data_fd.shape) ** 2

    if amplitude_in_snr:
        # Templates carry unit fluence, so their power gives SNR_opt
        snr_opt = jnp.sqrt(jnp.maximum(jnp.sum(jnp.abs(templates) ** 2 * inv_noise_var[None], axis=(1, 2)), _EPS))
        numpyro.deterministic("snr_opt", snr_opt)

        A = numpyro.deterministic("A", params_stacked["A_snr"] / snr_opt)
        model_fd = jnp.sum(A[:, None, None] * templates, axis=0)
        wraparound_amplitudes = A
    else:
        # Templates (and the low-frequency diagnostic) already carry the fluence
        model_fd = jnp.sum(templates, axis=0)
        wraparound_amplitudes = jnp.ones(n_pulses)

    # Recorded every run; penalized only if containment_FD is given
    wraparound_ratio(params_stacked, single_pulse_fn, wraparound_amplitudes,
                     inv_noise_var, containment_FD)

    residual = data_fd - model_fd
    numpyro.factor("log_likelihood", -0.5 * jnp.sum(jnp.abs(residual) ** 2 * inv_noise_var))

def TD_gaussianlikelihood(
    data_td: jnp.ndarray,
    sigma_noise_td: jnp.ndarray,
    n_pulses: int,
    priors: Dict[str, dist.Distribution],
    single_pulse_fn: Callable,
    dt: float,
    time_param: str = "mean_arrival_time",
    containment: dict = None, #support for contained pulse priors within data segment, remove with pass-through below if deleting
    spectral_coeff_map: dict = None,   # ORTHOGONAL SED SUPPORT: remove with the pass-through below if deleting orthog support
):
    """
    Time-domain Gaussian likelihood on the inverse transform of the Fourier-domain model.
    Validates the Fourier convention and Gaussian likelihood when DC/Nyquist handling is matched.

    data_td:        real (nfreq, ntime); must be baseline-subtracted, since this likelihood
                    uses every Fourier bin of the model, including DC
    sigma_noise_td: per-channel time-domain noise std, shape (nfreq, 1)
    spectral_coeff_map: if the orthogonal spectral basis is in use, the conversion coefficients
                        from `orthogonal_spectral_basis`; gamma and beta are then recorded as
                        deterministic sites alongside the sampled c1 and c2.
    containment:    if given, sigma and tau_ref are drawn jointly so the pulse fits inside the
                    segment (see `sample_contained_widths`) rather than from independent priors;
                    'sigma' and 'tau_ref' must then be absent from `priors`.
    The model is periodic over the segment (see the wraparound discussion in the paper).
    'A' must be in priors.
    """
    if "A" not in priors:
        raise ValueError("'A' must be in priors; otherwise fluences default to 1.")

    # Sample all parameters and sum the pulses (fluences included)
    #params_stacked = sample_pulse_parameters(n_pulses, priors, time_param)
    #params_stacked = sample_pulse_parameters(n_pulses, priors, time_param, spectral_coeff_map)
    params_stacked = sample_pulse_parameters(
        n_pulses, priors, time_param,
        spectral_coeff_map=spectral_coeff_map, containment=containment,
    )
    model_fd = jnp.sum(pulse_templates(params_stacked, single_pulse_fn), axis=0)

    # Inverse of X_c = dt * rfft(x) is x = irfft(X, n=N) / dt
    model_td = jnp.fft.irfft(model_fd, n=data_td.shape[-1], axis=-1) / jnp.float32(dt)

    numpyro.sample("obs", dist.Normal(model_td, sigma_noise_td), obs=data_td)
