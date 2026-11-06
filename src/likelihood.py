"""
likelihood.py - The NumPyro model of a burst: priors on the pulse parameters, the repulsion prior
between pulses, and the Fourier-domain Gaussian likelihood.

The data are each channel's Fourier transform X = dt * rfft(x) on the bins data.py kept (it drops
DC and the upper band, Nyquist included), and sigma_noise_fd is the standard deviation of the real
(equally, the imaginary) part of each bin. Real and imaginary parts are independent with equal
variance, so the log-likelihood of a model M is -1/2 sum |X - M|^2 / sigma^2 over channels and
retained bins. The pulse templates come from model.py (single_pulse_fn) and are evaluated on exactly
those bins. Everything is float32; see the precision note in model.py.
"""

from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


_EPS = jnp.finfo(jnp.float32).eps  # floor on a template's power, so a zero template cannot divide by zero


def _validate_fourier_inputs(data_fd, sigma_noise_fd):
    """
    Raises ValueError unless sigma_noise_fd has one row per channel of data_fd and either one column
    (the same noise in every bin of a channel) or one column per Fourier bin.

    data_fd:        complex data on the retained Fourier bins, shape (n_chan, n_fourier)
    sigma_noise_fd: noise standard deviation of each bin, shape (n_chan, 1) or (n_chan, n_fourier)
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
# 1. Pulse parameters and templates
# -------------------------------------------------------------------

def sample_pulse_parameters(
    n_pulses: int,
    shared_params: tuple,
    priors: Dict[str, dist.Distribution],
    time_param: str = "mean_arrival_time",
) -> Dict[str, jnp.ndarray]:
    """
    Samples every parameter in `priors` and returns each as an array of shape (n_pulses,). A
    parameter in shared_params is sampled once, as the site '<name>_shared', and recorded repeated
    for every pulse under its own name; every other parameter is sampled independently for each
    pulse from the same prior. The pulses are not ordered in time, so their labels are made
    consistent after sampling by postprocess.order_pulses.

    n_pulses:      number of pulses in the model
    shared_params: names of the parameters sampled once for the whole burst, e.g. ("DM", "tau_ref")
    priors:        one NumPyro distribution per parameter name, as built by run.set_priors
    time_param:    name of the arrival-time parameter, which cannot be shared
    """
    unknown = set(shared_params) - priors.keys()
    if unknown:
        raise ValueError(f"shared_params {sorted(unknown)} are not in priors.")
    if time_param in shared_params:
        raise ValueError("arrival times cannot be shared; they are what distinguish the pulses.")

    params_stacked = {}
    for param_name, distribution in priors.items():
        if param_name in shared_params:
            # One value for the whole burst, recorded repeated per pulse so downstream code need not branch
            value = numpyro.sample(f"{param_name}_shared", distribution)
            params_stacked[param_name] = numpyro.deterministic(
                param_name, jnp.broadcast_to(value, (n_pulses,))
            )
        else:
            # The same prior for every pulse, sampled as one batched site
            params_stacked[param_name] = numpyro.sample(param_name, distribution.expand([n_pulses]))

    return params_stacked


def pulse_templates(
    params_stacked: Dict[str, jnp.ndarray],
    single_pulse_fn: Callable[[Dict[str, jnp.ndarray]], jnp.ndarray],
) -> jnp.ndarray:
    """
    Evaluates the template of every pulse at once, shape (n_pulses, n_chan, n_fourier).

    params_stacked:  parameter arrays of shape (n_pulses,), from sample_pulse_parameters
    single_pulse_fn: the model closure from model.setup_model; it reads only the parameters it needs
    """
    return jax.vmap(single_pulse_fn)(params_stacked)


def log_sed_peaks(params_stacked, single_pulse_fn) -> jnp.ndarray:
    """
    ln max_a F_a, the log of each pulse's SED at its brightest channel, shape (n_pulses,). Without
    'A', model.py divides each template by this peak so the SED cannot overflow float32, and an
    amplitude B on such a template is the fluence B exp(-log_sed_peak).

    params_stacked:  parameter arrays of shape (n_pulses,), from sample_pulse_parameters
    single_pulse_fn: the model closure from model.setup_model, whose .log_sed_peak gives ln max_a F_a
    """
    return jax.vmap(single_pulse_fn.log_sed_peak)(params_stacked)


def fluence_for_normalized_log_sed_models(amplitude, log_sed_peak):
    """
    Fluence from an amplitude on a peak-normalized template, amplitude * exp(-log_sed_peak),
    computed in log space so neither factor overflows float32 on its own. The sign is carried
    separately because the log needs a positive argument.

    amplitude:    amplitude on the peak-normalized template (A_snr / snr_opt in FD_gaussianlikelihood)
    log_sed_peak: ln max_a F_a of the same pulse, from log_sed_peaks
    """
    return jnp.sign(amplitude) * jnp.exp(jnp.log(jnp.abs(amplitude)) - log_sed_peak)


# -------------------------------------------------------------------
# 2. Repulsion prior between pulses
# -------------------------------------------------------------------

def repulsive_toa_prior(t, sigma, tau, snr, x_min=1.0, w=0.05, K=1000.0, snr_on=3.0, w_snr=0.33,
                        c_ratio=0.0, max_log_ratio=4.0, tau_factor=None, tau_scale=1.0):
    """
    A soft wall that keeps every pair of ACTIVE pulses apart, so that two pulses cannot overlap to
    absorb model misfit together, while an idle (near-zero SNR) pulse can still pass through an
    active one. A pulse's scattered profile is a Gaussian convolved with an exponential tail, with
    mean t + tau and standard deviation sqrt(sigma^2 + (tau_scale tau)^2); a pair's separation x is
    the difference of their means over the sum of their standard deviations. The wall costs K nats
    at x = 0, falls towards zero at the required separation x_min_jk, and is negligible beyond it;
    each pulse's share is gated by on = sigmoid((snr - snr_on) / w_snr), ~1 when active and ~0 when
    idle. It depends only on the parameters, never on the data, so it is a prior; it is computed in
    the likelihood only because a pulse's SNR comes from its template.

        x_min_jk = x_min (1 + c_ratio min(|ln snr_j - ln snr_k|, max_log_ratio))
        wall_jk  = softplus((x_min_jk - x_jk) / w) / softplus(x_min_jk / w)
        ln p     = -K sum_{j<k} on_j on_k wall_jk

    t:             arrival times at the reference frequency (s), shape (n_pulses,)
    sigma:         intrinsic Gaussian widths (s), shape (n_pulses,)
    tau:           scattering times at the reference frequency, tau_ref (s), shape (n_pulses,)
    snr:           each pulse's optimal SNR at its current amplitude, shape (n_pulses,)
    x_min:         required separation for pulses of similar brightness; 1 is where two equal Gaussians are just resolved
    w:             softness of the wall in units of x; with the defaults the cost is ~1 nat by x = 1.2 x_min_jk
    K:             cost at full overlap (nats); it must exceed the log-likelihood an overlap could gain
    snr_on:        SNR at which a pulse counts as active
    w_snr:         width of that transition; with the defaults an idle pulse (snr ~ 0) has on ~ 1e-4
    c_ratio:       growth of the required separation with the pair's log brightness ratio; 0 (default) uses x_min for every pair
    max_log_ratio: cap on the log brightness ratio, so a pulse switching off (snr -> 0) cannot send x_min_jk to infinity
    tau_factor:    per pulse, the SNR^2-weighted mean of (nu / nu_ref)^delta over channels; each pair uses the larger of its two, None uses tau_ref
    tau_scale:     weight on the scattering tail in the widths; 1 gives the plain standard deviation

    Returns (log prior, x), x of shape (n_pulses (n_pulses - 1) / 2,) in the pair order of
    jnp.triu_indices(n_pulses, k=1).
    """
    j, k = jnp.triu_indices(t.shape[0], k=1)
    # Tail length in the channels of whichever pulse of the pair sits lower in the band (1 = tau_ref)
    f = 1.0 if tau_factor is None else jnp.maximum(tau_factor[j], tau_factor[k])
    tau_j, tau_k = tau[j] * f, tau[k] * f
    # Only the tail is weighted by tau_scale; the Gaussian part keeps the resolution criterion
    width_j = jnp.sqrt(sigma[j] ** 2 + (tau_scale * tau_j) ** 2)
    width_k = jnp.sqrt(sigma[k] ** 2 + (tau_scale * tau_k) ** 2)
    x = jnp.abs((t[j] + tau_j) - (t[k] + tau_k)) / (width_j + width_k)

    # The floor keeps the log finite for a pulse at zero amplitude; the cap keeps the wall bounded there
    s = jnp.maximum(snr, 1e-3)
    log_ratio = jnp.minimum(jnp.abs(jnp.log(s[j]) - jnp.log(s[k])), max_log_ratio)
    x_min_jk = x_min * (1.0 + c_ratio * log_ratio)

    wall = jax.nn.softplus((x_min_jk - x) / w) / jax.nn.softplus(x_min_jk / w)   # 1 at x = 0, ~0 past x_min_jk
    on = jax.nn.sigmoid((snr - snr_on) / w_snr)                                    # activity gate of each pulse
    return -K * jnp.sum(on[j] * on[k] * wall), x


# -------------------------------------------------------------------
# 3. The NumPyro model
# -------------------------------------------------------------------

def FD_gaussianlikelihood(
    data_fd: jnp.ndarray,
    sigma_noise_fd: jnp.ndarray,
    n_pulses: int,
    shared_params: tuple,
    priors: Dict[str, dist.Distribution],
    single_pulse_fn: Callable,
    time_param: str = "mean_arrival_time",
    repulsion: dict | None = {},
):
    """
    The NumPyro model of a burst: samples the pulse parameters from `priors`, sums the pulse
    templates into a model M on the retained Fourier bins, and adds the log-likelihood
    -1/2 sum |X - M|^2 / sigma^2 (factor site 'log_likelihood') and, with more than one pulse, the
    repulsion prior (factor site 'repulsive_toa_prior', with the pair separations recorded as
    'toa_separation'). The amplitude is either 'A', a fluence that single_pulse_fn applies to the
    template itself, or 'A_snr', the pulse's optimal SNR, which scales the peak-normalized template
    by A_snr / snr_opt and records the fluence as the deterministic site 'A' (with 'snr_opt' and
    'log_sed_peak'). An amplitude in SNR units makes its prior scale with each template's
    detectability, so the prior does not favour narrow, easily detected templates, and one prior
    suits every burst.

    data_fd:         complex data on the retained Fourier bins, shape (n_chan, n_fourier), from data.prepare_burst
    sigma_noise_fd:  noise std of the real/imaginary part of each bin, shape (n_chan, 1) or (n_chan, n_fourier)
    n_pulses:        number of pulses in the model
    shared_params:   names of the parameters sampled once for the whole burst, e.g. ("DM", "tau_ref")
    priors:          one prior per parameter, as built by run.set_priors, with exactly one of 'A' and 'A_snr'
    single_pulse_fn: the model closure from model.setup_model
    time_param:      name of the arrival-time parameter
    repulsion:       keyword arguments for repulsive_toa_prior, plus band_tau (default True: measure each pulse's tail in its own channels); {} uses the defaults, None turns the repulsion off
    """
    amplitude_in_snr = "A_snr" in priors
    if amplitude_in_snr == ("A" in priors):
        raise ValueError("exactly one of 'A' (a fluence) and 'A_snr' (SNR units) must be in priors.")

    _validate_fourier_inputs(data_fd, sigma_noise_fd)

    params_stacked = sample_pulse_parameters(n_pulses, shared_params, priors, time_param)
    templates = pulse_templates(params_stacked, single_pulse_fn)        # (n_pulses, n_chan, n_fourier)

    # 1 / sigma^2 on every bin, whether the noise was given per channel or per bin
    inv_noise_var = 1.0 / jnp.broadcast_to(jnp.asarray(sigma_noise_fd, dtype=jnp.float32), data_fd.shape) ** 2

    if amplitude_in_snr:
        # Without 'A', model.py returns each template with its SED peak scaled to 1; its noise-weighted
        # norm is then the SNR per unit amplitude
        snr_opt = jnp.sqrt(jnp.maximum(jnp.sum(jnp.abs(templates) ** 2 * inv_noise_var[None], axis=(1, 2)), _EPS))
        numpyro.deterministic("snr_opt", snr_opt)

        A_peak_channel = params_stacked["A_snr"] / snr_opt                 # amplitude on the peak-normalized template
        model_fd = jnp.sum(A_peak_channel[:, None, None] * templates, axis=0)
        log_peak = numpyro.deterministic("log_sed_peak", log_sed_peaks(params_stacked, single_pulse_fn))
        numpyro.deterministic("A", fluence_for_normalized_log_sed_models(A_peak_channel, log_peak))
        pulse_snr = params_stacked["A_snr"]                                 # each pulse's SNR, for the repulsion
    else:
        # With 'A', model.py has already multiplied each template by its fluence
        model_fd = jnp.sum(templates, axis=0)
        pulse_snr = jnp.sqrt(jnp.sum(jnp.abs(templates) ** 2 * inv_noise_var[None], axis=(1, 2)))

    # Repulsion between active pulses; see repulsive_toa_prior
    if repulsion is not None and n_pulses > 1:
        rep = dict(repulsion)
        tau_factor = None
        if rep.pop("band_tau", True):
            # Each pulse's SNR^2 share per channel, and its mean scattering factor (nu / nu_ref)^delta there
            power_chan = jnp.sum(jnp.abs(templates) ** 2 * inv_noise_var[None], axis=-1)       # (n_pulses, n_chan)
            w_chan = power_chan / jnp.sum(power_chan, axis=-1, keepdims=True)
            nu_over_ref = jnp.asarray(single_pulse_fn.grid.normalized_freqs_channels).reshape(-1)
            # the sampled scattering index if there is one, else the model's fixed value
            delta = jnp.reshape(params_stacked.get("delta", single_pulse_fn.constants.delta), (-1, 1))
            tau_factor = jnp.sum(w_chan * nu_over_ref[None, :] ** delta, axis=-1)                # (n_pulses,)
        log_prior, separation = repulsive_toa_prior(
            params_stacked[time_param], params_stacked["sigma"], params_stacked["tau_ref"],
            pulse_snr, tau_factor=tau_factor, **rep
        )
        numpyro.deterministic("toa_separation", separation)
        numpyro.factor("repulsive_toa_prior", log_prior)

    residual = data_fd - model_fd
    numpyro.factor("log_likelihood", -0.5 * jnp.sum(jnp.abs(residual) ** 2 * inv_noise_var))
