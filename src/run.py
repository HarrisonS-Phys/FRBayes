"""
run.py - Priors, NUTS sampling and the maximum-likelihood pulse search for one burst.

Choose the device first, before anything imports jax (devices.use_device). A fit is then

    priors = set_priors(burst, single_pulse_fn, s_opt=100.0)
    mcmc   = run_fit(burst, single_pulse_fn, priors, n_pulses=n_pulses)
    idata  = results.save_run(out_path, mcmc, burst, priors, single_pulse_fn=single_pulse_fn,
                              dm_dedispersion=..., dedispersion_ref_freq=...)

and maxL_iterative_pulse_fit lets the data decide the number of pulses with maximum-likelihood
fits (maxL_fit), adding one at a time while the residual holds leftover signal that a new pulse
explains; run_fit(maxL_start=...) then samples the kept fit's posterior with every chain started
at its best point. The amplitude prior is set once, in set_priors, as
'A_snr' ~ HalfNormal(s_opt) in units of the template's optimal SNR, unless a fluence prior A=... is
given instead.
"""

import inspect
import time
from functools import partial
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import optax
from jax.flatten_util import ravel_pytree
from numpyro import handlers
from numpyro.distributions.transforms import biject_to
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import constrain_fn, initialize_model

from likelihood import FD_gaussianlikelihood, repulsive_toa_prior
from priors import dispersion_prior_bounds, spectral_index_bound, width_prior_bounds
from results import describe_priors

AMPLITUDE_PARAMS = {"A", "A_snr"}           # fluence, or amplitude in SNR units; exactly one is sampled
DELTA_INIT = -4.0                           # starting value of a sampled scattering index, also used for the starting wall clearance


# -------------------------------------------------------------------
# 1. Priors
# -------------------------------------------------------------------

def set_priors(
    burst,
    single_pulse_fn: Callable,
    toa: float = None,
    s_opt: float = None,
    gamma_range=None,
    beta_range=(-200.0, 10.0),
    verbose: bool = True,
    **overrides,
) -> Dict[str, dist.Distribution]:
    """
    Builds the priors dict of a fit from the data window, then applies `overrides`. Every pulse gets
    the same priors, so the dict does not depend on the number of pulses. Defaults:
    mean_arrival_time ~ Uniform over the window; DM ~ Uniform(-D_max, D_max) from
    priors.dispersion_prior_bounds; sigma ~ LogUniform and tau_ref ~ Uniform from
    priors.width_prior_bounds at `toa` (log-uniform in width since its scale is unknown, flat in
    tau_ref so that no scattering, tau_ref = 0, is inside the prior); beta ~ Uniform over beta_range
    and gamma ~ Uniform(-gamma_max, gamma_max) with gamma_max from priors.spectral_index_bound (every
    allowed spectral width can peak anywhere in the band), or over gamma_range if given; and
    A_snr ~ HalfNormal(s_opt) if s_opt is given. Raises if a value
    is not a NumPyro distribution or if both 'A' and 'A_snr' are set.

    burst:           the ConditionedData; its window, time resolution and band set the default bounds
    single_pulse_fn: the model closure from model.setup_model, for the DM bounds and the scattering index
    toa:             arrival time the width bounds are computed for (s, relative to the segment start); None uses the middle of the window, which gives the widest sigma bound
    s_opt:           scale of the A_snr prior (SNR units); every fit needs it unless an A=... or A_snr=... override is given
    gamma_range:     (low, high) of the spectral index prior; None sets it from the band (priors.spectral_index_bound)
    beta_range:      (low, high) of the spectral running prior
    verbose:         print the width bounds and every prior
    overrides:       name=distribution replaces a default or adds a parameter (e.g. A=..., delta=...); name=None removes a default
    """
    W_seg = burst.n_time * burst.dt
    if toa is None:
        toa = 0.5 * W_seg

    # The tau_ref cap depends on the scattering index, which an override may make a sampled 'delta'
    width = width_prior_bounds(burst, single_pulse_fn, toa, delta_prior=overrides.get("delta"), verbose=verbose)
    dm = dispersion_prior_bounds(burst, single_pulse_fn)

    # gamma's bound comes from the lowest beta allowed, an override's if beta is overridden
    beta_prior = overrides.get("beta") or dist.Uniform(float(beta_range[0]), float(beta_range[1]))
    if gamma_range is None:
        gamma_max = spectral_index_bound(burst, float(beta_prior.low))
        gamma_range = (-gamma_max, gamma_max)

    priors = {
        "mean_arrival_time": dist.Uniform(0.0, float(burst.times[-1])),   # (s, relative to segment start)
        "DM": dist.Uniform(dm["DM_min"], dm["DM_max"]),                    # residual DM (pc cm^-3)
        "sigma": dist.LogUniform(width["sigma_min"], width["sigma_max"]),  # intrinsic width (s), per pulse
        "tau_ref": dist.Uniform(width["tau_ref_min"], width["tau_ref_max"]),  # scattering time at nu_r (s)
        "gamma": dist.Uniform(float(gamma_range[0]), float(gamma_range[1])),  # spectral index
        "beta": beta_prior,                                                   # spectral running
    }
    if s_opt is not None:
        priors["A_snr"] = dist.HalfNormal(float(s_opt))

    for name, value in overrides.items():
        if value is None:
            priors.pop(name, None)
        else:
            priors[name] = value

    not_distributions = {k: type(v).__name__ for k, v in priors.items()
                         if not isinstance(v, dist.Distribution)}
    if not_distributions:
        raise TypeError(f"priors must be NumPyro distributions; got {not_distributions}.")
    if AMPLITUDE_PARAMS <= priors.keys():
        raise ValueError(
            "both 'A' (a fluence) and 'A_snr' (SNR units) are set; s_opt sets 'A_snr', so pass "
            "either s_opt or an A=... override, not both."
        )

    if verbose:
        _print_priors(priors)

    return priors


def _print_priors(priors, indent=""):
    """
    Prints every prior, one per line, as results.describe_priors describes it.

    priors: the priors dict, from set_priors
    indent: text put before each line
    """
    for name, description in describe_priors(priors).items():
        print(f"{indent}{name.removeprefix('prior.')}: {description}")


# -------------------------------------------------------------------
# 2. Starting points
# -------------------------------------------------------------------

def _interior(distribution, value, margin=1e-3):
    """
    `value` moved just inside the bounds of a bounded prior; values of unbounded priors pass through.

    distribution: the prior the value must lie inside
    value:        the starting value to move
    margin:       how far inside the bounds to stay, as a fraction of the prior's range
    """
    low, high = getattr(distribution, "low", None), getattr(distribution, "high", None)
    if low is None or high is None:
        return value
    low, high = float(low), float(high)
    pad = margin * (high - low)
    return min(max(value, low + pad), high - pad)


def _starting_values(burst, priors: Dict[str, dist.Distribution], n_pulses: int):
    """
    Starting values for every parameter except the arrival times, the same in every chain, as arrays
    of shape (n_pulses,) (and scalars under '<name>_shared' for shared parameters). Pulses start broad
    (sigma at 0.9 of its upper bound), so a pulse overlaps nearby emission wherever it starts and
    feels a gradient towards it; with no residual DM, a flat spectrum, little scattering (0.5 dt),
    A_snr = 5, large enough that the likelihood pulls on the pulse's position, and a sampled delta at
    DELTA_INIT. Parameters without a value here (e.g. 'A') are left to NumPyro's default. Every value
    is moved just inside its prior's bounds.

    burst:    the ConditionedData, whose time resolution sets the starting scattering
    priors:   the priors the model samples; only parameters present get a starting value
    n_pulses: number of pulses in the fit
    """
    targets = {"DM": 0.0, "gamma": 0.0, "beta": 0.0, "tau_ref": 0.5 * burst.dt, "A_snr": 5.0, "delta": DELTA_INIT}
    if getattr(priors.get("sigma"), "high", None) is not None:
        targets["sigma"] = 0.9 * float(priors["sigma"].high)

    values = {
        name: jnp.full((n_pulses,), _interior(priors[name], value), dtype=jnp.float32)
        for name, value in targets.items() if name in priors
    }
    # Shared parameters are sampled once, as '<name>_shared'; the extra entries are ignored when nothing is shared
    for name in list(values):
        values[f"{name}_shared"] = values[name][0]
    return values


def _announce_delta_init(priors, warm=False):
    """
    Prints, when the scattering index is sampled, that every pulse starts at DELTA_INIT and what is
    still to be checked about that choice, with a warning about warm starts if warm.

    priors: the priors of the fit
    warm:   True if the fit is warm-started from a previous fit
    """
    if "delta" not in priors:
        return
    print(f"delta is sampled: every pulse starts at delta = {DELTA_INIT:g}, which also sets the starting "
          f"distance from the repulsion wall. TO-DO: check that this start does not make sampling "
          f"difficult when the best fit has delta far from {DELTA_INIT:g}.")
    if warm:
        print(f"warning: warm starts restart every pulse's delta at {DELTA_INIT:g} instead of keeping the "
              f"previous fit's value. tau_ref is restarted (and profiled in the maximum-likelihood search) "
              f"because an inflated scattering tail can cover a missing pulse and trap the fit; delta "
              f"reshapes that tail across the band, so it may do the same, but it is not profiled. "
              f"TO-DO: confirm that restarting delta is a safe choice.")


def init_to_value_random_times(site=None, values={}, time_param="mean_arrival_time", time_prior=None):
    """
    NumPyro initialization strategy: init_to_value for every site except the arrival times, which each
    chain draws from their prior with the random key NumPyro gives that chain, so every chain starts
    from a different arrangement of pulses.

    site:       the sample site NumPyro asks a starting value for; None returns the strategy itself, as NumPyro requires
    values:     starting values for every other site, as for init_to_value
    time_param: name of the arrival-time site
    time_prior: the arrival-time prior the times are drawn from
    """
    if site is None:
        return partial(init_to_value_random_times, values=values, time_param=time_param,
                       time_prior=time_prior)
    if site["type"] == "sample" and site["name"] == time_param:
        key = site["kwargs"].get("rng_key")
        if key is None:
            key = jax.random.PRNGKey(0)
        return time_prior.sample(key, site["fn"].shape())
    return init_to_value(site, values=values)


def default_init(
    burst,
    priors: Dict[str, dist.Distribution],
    n_pulses: int = 1,
    time_param: str = "mean_arrival_time",
):
    """
    The default starting point of the sampler: every parameter from _starting_values, and arrival
    times drawn at random from their prior for each chain (init_to_value_random_times).

    burst:      the ConditionedData of the burst
    priors:     the priors the model samples
    n_pulses:   number of pulses in the fit
    time_param: name of the arrival-time site
    """
    values = _starting_values(burst, priors, n_pulses)
    if time_param not in priors:
        return init_to_value(values=values)
    return init_to_value_random_times(values=values, time_param=time_param, time_prior=priors[time_param])


def wall_clearance(burst, single_pulse_fn, priors, values, n_pulses, likelihood, likelihood_kwargs):
    """
    How far apart two starting pulses must be to clear the repulsion wall of
    likelihood.repulsive_toa_prior. Returns (x, tail): pulses j and k clear it when
    |t_j - t_k| >= x (width_j + width_k), with width = sqrt(sigma^2 + tail^2). tail is the starting
    tau_ref times tau_scale times the largest band factor (nu / nu_ref)^delta over the channels (1
    without band_tau), using the model's scattering index or, if delta is sampled, its starting
    value DELTA_INIT. x = x_min + 10 w, where the soft
    wall's cost has fallen to ~0.002 nats (it is still ~35 nats at x_min itself); x = 0 without
    repulsion or with one pulse. All pulses start with the same tau, so the shift of their centres
    by the tail cancels between them.

    burst:             the ConditionedData, for the channels and nu_ref
    single_pulse_fn:   the model closure, whose constants give the fixed scattering index
    priors:            the priors of the fit, to tell whether 'delta' is sampled
    values:            the starting values (from _starting_values or _previous_fit_values), for the starting tau_ref
    n_pulses:          number of pulses in the fit
    likelihood:        the likelihood function, whose `repulsion` default applies if likelihood_kwargs has none
    likelihood_kwargs: the keyword arguments the likelihood is given, e.g. repulsion={...}
    """
    param = inspect.signature(likelihood).parameters.get("repulsion")
    rep = likelihood_kwargs.get("repulsion", param.default if param is not None else None)
    if rep is None or n_pulses == 1:
        return 0.0, 0.0
    # repulsive_toa_prior's own defaults, overridden by what the fit passes
    rep = {**{name: p.default for name, p in inspect.signature(repulsive_toa_prior).parameters.items()
              if p.default is not inspect.Parameter.empty}, **rep}
    f = 1.0
    if rep.get("band_tau", True):
        nu = np.asarray(burst.freq_channels, dtype=float).reshape(-1) / float(burst.freq_channels_ref)
        delta = DELTA_INIT if "delta" in priors else float(single_pulse_fn.constants.delta)
        f = float(np.max(nu ** delta))
    if "tau_ref_shared" in values:
        tau0 = float(values["tau_ref_shared"])
    else:
        tau0 = float(values["tau_ref"][0]) if "tau_ref" in values else 0.0
    return float(rep["x_min"]) + 10.0 * float(rep["w"]), float(rep["tau_scale"]) * tau0 * f


def _previous_fit_values(params, defaults, priors, n_pulses):
    """
    Starting values for a WARM start: the previous fit's n_pulses - 1 pulses as they were, plus one
    new pulse, last, with the default starting values (or the median of the previous pulses for a
    parameter without a default). tau_ref and a sampled delta are not inherited but start from their
    defaults: a previous fit missing a component often stretched its scattering tail across it, and
    tau_ref is profiled in maxL_fit anyway (see _announce_delta_init for delta). Every value is moved
    just inside its prior's bounds.

    params:   the previous fit's best point, name -> value: per-pulse values of shape (n_pulses - 1,), shared ones ('<name>_shared') scalars; sites the model does not sample are ignored
    defaults: _starting_values for n_pulses pulses, which give the new pulse its values
    priors:   the priors of the new fit
    n_pulses: number of pulses in the new fit
    """
    reset = ("tau_ref", "delta")
    values = dict(defaults)
    for name, v in params.items():
        if name.removesuffix("_shared") in reset:
            continue                                        # starts from its default instead
        prior = priors.get(name.removesuffix("_shared"))
        if prior is None:
            continue                                        # deterministic, or not in this fit
        v = np.asarray(v, dtype=float)
        inside = np.vectorize(lambda u: _interior(prior, float(u)))
        if name.endswith("_shared") and v.ndim == 0:
            values[name] = jnp.float32(inside(v))
        elif v.shape == (n_pulses - 1,):
            new = float(np.asarray(defaults[name])[-1]) if name in defaults else float(np.median(v))
            values[name] = jnp.asarray(np.append(inside(v), new), dtype=jnp.float32)
        else:
            raise ValueError(f"the previous fit's '{name}' has shape {v.shape}; expected "
                             f"({n_pulses - 1},), one value per pulse of the previous fit.")
    return values


def box_times(burst, M_FD, result, n_pulses, n_starts, n_warm, time_prior, sigma_range, seed,
              clearance=(0.0, 0.0), fixed=None):
    """
    Starting arrival times and widths for a warm start, each of shape (n_starts, n_pulses), from
    the residual boxes of the previous fit. The first n_warm starts keep the `fixed` pulses
    (the previous fit's) and put each new pulse in its own significant box of `result`, at a time
    drawn inside the box with probability proportional to the whitened residual power there; every
    pulse of the other starts gets a time drawn uniformly from the prior. Every pulse not fixed gets
    a width drawn log-uniformly over sigma_range. Pulses are placed one at a time, each only at
    times clear of the walls of those already placed (wall_clearance); a pulse with no clear time in
    its box gets a random clear time, and a start with no clear time anywhere is reported as starting
    inside a wall. Returns (times, sigmas, bands): bands, shape (n_starts, n_pulses, 2), is the
    frequency range (MHz, outer channel edges) of the box each pulse was placed in, NaN for a pulse
    not placed in a box or placed in one spanning the whole band.

    burst:         the ConditionedData of the burst
    M_FD:          the previous fit's model on the retained Fourier bins
    result:        postprocess.excess_power_search of the previous fit's residual
    n_pulses:      number of pulses in the new fit
    n_starts:      number of optimizer starts
    n_warm:        how many of them start warm
    time_prior:    the arrival-time prior (bounded)
    sigma_range:   (low, high) the new widths are drawn log-uniformly between
    seed:          integer the draws are made from
    clearance:     (x, tail) from wall_clearance; tail may be an array with one value per start
    fixed:         (times, sigmas) of the pulses warm starts keep, each of shape (m,), m < n_pulses
    """
    x, tail = clearance
    rng = np.random.default_rng(seed)
    n, dt = int(burst.n_time), float(burst.dt)
    t = np.arange(n) * dt
    lo, hi = float(time_prior.low), float(time_prior.high)
    pad = 1e-3 * (hi - lo)                                   # keep off the prior's edges
    in_prior = (t >= lo + pad) & (t <= hi - pad)
    Z = (np.asarray(burst.data_fd) - np.asarray(M_FD)) / np.asarray(burst.sigma_noise_fd)
    k = np.rint(np.asarray(burst.fourier_freqs, dtype=float).reshape(-1) * n * dt).astype(int)
    nu = np.asarray(burst.freq_channels, dtype=float).reshape(-1)
    half_chan = 0.5 * float(np.median(np.abs(np.diff(nu)))) if nu.size > 1 else 0.0   # half a channel width (MHz)

    # Residual power |r(t)|^2 inside each significant box: the whitened residual on the box's
    # channels and Fourier band, transformed to time and summed over those channels
    boxes, box_bands = [], []
    for tile in result["tiles"]:
        if not tile["significant"]:
            continue
        (c0, c1), n_b = tile["channel_index"], tile["resolution"]
        j0, j1 = tile["band_index"][0] * n_b, tile["band_index"][1] * n_b
        full = np.zeros((c1 - c0, n), dtype=complex)
        full[:, k[j0:j1]] = Z[c0:c1, j0:j1]
        power = np.sum(np.abs(np.fft.ifft(full, axis=-1)) ** 2, axis=0)
        t_lo, t_hi = tile["t_range"]
        inside = (t >= t_lo) & (t <= t_hi)
        boxes.append(np.where(inside & in_prior, power, 0.0))
        # The box's frequency range out to its outer channel edges; None if it spans the whole band
        box_bands.append(None if (c0, c1) == (0, nu.size) else
                         (float(nu[c0:c1].min()) - half_chan, float(nu[c0:c1].max()) + half_chan))

    log_lo, log_hi = np.log(sigma_range[0]), np.log(sigma_range[1])
    sigmas = np.exp(rng.uniform(log_lo, log_hi, size=(n_starts, n_pulses)))     # log-uniform widths
    times = np.empty((n_starts, n_pulses))
    bands = np.full((n_starts, n_pulses, 2), np.nan)                            # box band of each placed pulse
    m = 0
    if fixed is not None:                     # warm starts keep the previous fit's pulses as they were
        m = len(fixed[0])
        times[:n_warm, :m], sigmas[:n_warm, :m] = fixed[0], fixed[1]
    tail = np.asarray(tail, dtype=float)
    widths = np.sqrt(sigmas ** 2 + (tail.reshape(-1, 1) if tail.ndim else tail) ** 2)

    in_wall = []
    for c in range(n_starts):
        warm = c < n_warm
        chosen = rng.permutation(len(boxes))[:n_pulses - m] if warm else []
        for j in range(m if warm else 0, n_pulses):
            # Times clear of the walls of every pulse already placed in this start
            free = in_prior & np.all([np.abs(t - times[c, i]) >= x * (widths[c, j] + widths[c, i])
                                      for i in range(j)], axis=0)
            jj = j - (m if warm else 0)                      # index among the pulses being placed
            p = boxes[chosen[jj]] * free if jj < len(chosen) else free.astype(float)
            if jj < len(chosen) and p.sum() > 0 and box_bands[chosen[jj]] is not None:
                bands[c, j] = box_bands[chosen[jj]]          # placed in a box narrower than the band
            if p.sum() <= 0:                                 # box full: start this pulse at random
                p = free.astype(float)
            if p.sum() <= 0:                                 # no free time left anywhere
                p = in_prior.astype(float)
                in_wall.append(c)
            times[c, j] = rng.choice(t, p=p / p.sum())

    in_wall = sorted(set(in_wall))
    if in_wall:
        print(f"box_times: {len(in_wall)} of {n_starts} starts could not keep every pulse out of the "
              f"walls (widths too broad for {n_pulses} pulses in this segment): {in_wall}")
    return times, sigmas, bands


def _box_spectrum(burst, priors, nu_lo, nu_hi):
    """
    Spectral index and running (gamma, beta) of a spectrum peaked on the band [nu_lo, nu_hi]. The
    running power law ln F = gamma x + beta x^2, x = ln(nu / nu_ref), is a Gaussian in x for beta < 0,
    peaked at x_p = -gamma / (2 beta) with full width at half maximum 2 sqrt(ln 2 / -beta); matching
    the peak to the band's centre in x and the width to its width Delta = ln(nu_hi / nu_lo) gives
    beta = -4 ln 2 / Delta^2 and gamma = -2 beta x_p. beta is moved inside its prior first, then gamma
    is computed and moved inside its own, which shifts the peak if gamma's bound is reached. Returns
    (gamma, beta, moved), moved True if either prior moved the matched value.

    burst:  the ConditionedData, for nu_ref
    priors: the priors of the fit, with 'gamma' and 'beta'
    nu_lo:  lower edge of the box's band (MHz)
    nu_hi:  upper edge of the box's band (MHz)
    """
    x_p = 0.5 * (np.log(nu_lo) + np.log(nu_hi)) - np.log(float(burst.freq_channels_ref))   # band centre in x
    beta_match = -4.0 * np.log(2.0) / np.log(nu_hi / nu_lo) ** 2
    beta = _interior(priors["beta"], beta_match)
    gamma_match = -2.0 * beta * x_p
    gamma = _interior(priors["gamma"], gamma_match)
    return gamma, beta, bool(beta != beta_match or gamma != gamma_match)


def _split_values(warm_values, priors, n_pulses, k, d):
    """
    Starting values for a SPLIT start: the warm values with old pulse k replaced by two halves, one in
    its own slot and one in the new pulse's (the last): times t_k - d and t_k + d, each with width
    sigma_k / 2 and amplitude A_snr_k / sqrt(2) (or fluence A_k / 2), so their total SNR (or fluence)
    is pulse k's, and pulse k's other values. Every value is moved just inside its prior's bounds.

    warm_values: starting values from _previous_fit_values (the previous fit's pulses, then the new pulse)
    priors:      the priors of the new fit
    n_pulses:    number of pulses in the new fit
    k:           index of the old pulse to split
    d:           half the separation of the two halves (s)
    """
    new = n_pulses - 1
    values = {}
    for name, v in warm_values.items():
        v = np.array(v, dtype=float)
        if v.shape == (n_pulses,):
            v[new] = v[k]                                   # the second half starts as a copy of pulse k
        values[name] = v
    t_k = values["mean_arrival_time"][k]
    values["mean_arrival_time"][[k, new]] = [t_k - d, t_k + d]
    values["sigma"][[k, new]] = 0.5 * values["sigma"][k]
    if "A_snr" in values:
        values["A_snr"][[k, new]] = values["A_snr"][k] / np.sqrt(2.0)
    if "A" in values:
        values["A"][[k, new]] = 0.5 * values["A"][k]
    for name, v in values.items():
        if name in priors and v.shape == (n_pulses,):
            values[name] = np.array([_interior(priors[name], float(u)) for u in v])
    return {name: jnp.asarray(v, dtype=jnp.float32) for name, v in values.items()}


def warm_start_points(burst, single_pulse_fn, priors, likelihood, model_kwargs, likelihood_kwargs, prev_params,
                      prev_M, search, n_pulses, n_starts, n_warm, seed, split_starts=True, box_spectrum=True):
    """
    Starting points for a maximum-likelihood fit (maxL_fit) with one more pulse than a previous fit.
    The first n_warm start from the previous fit
    (_previous_fit_values) with the new pulse in a significant residual box (box_times); with
    box_spectrum, a new pulse placed in a box narrower than the band starts with a spectrum peaked
    on the box's band (_box_spectrum) instead of a flat one, so it starts on a narrowband feature's
    power rather than spread across the band. With split_starts, the last warm starts instead split
    one old pulse in two (_split_values), widest old pulse first, one per old pulse up to half the
    warm starts, which gives a start that can separate two close pulses the previous fit covered
    with one wide pulse. The rest start from scratch (_starting_values) at random times and widths.
    Pulses start clear of each other's repulsion wall where the window has room (wall_clearance).
    Returns (z, init_times, init_sigma): the starting points on the unconstrained scale the optimizer
    works on, a dict of arrays with leading axis n_starts, and the starting arrival times
    and widths, each (n_starts, n_pulses).

    burst:             the ConditionedData of the burst
    single_pulse_fn:   the model closure from model.setup_model
    priors:            the priors of the new fit (set_priors)
    likelihood:        the NumPyro model
    model_kwargs:      the likelihood's arguments for the new fit, to map starting values to the unconstrained scale
    likelihood_kwargs: the keyword arguments the likelihood is given (e.g. repulsion={...}), for wall_clearance
    prev_params:       the previous fit's best point, name -> value, n_pulses - 1 pulses
    prev_M:            the previous fit's model on the retained Fourier bins
    search:            postprocess.excess_power_search of the previous fit's residual
    n_pulses:          number of pulses in the new fit
    n_starts:          number of optimizer starts
    n_warm:            how many of them start from the previous fit
    seed:              integer the random placements are drawn from
    split_starts:      turn some warm starts into split starts
    box_spectrum:      start a box-placed new pulse with a spectrum peaked on its box's band (needs per-pulse 'gamma' and 'beta')
    """
    time_param = "mean_arrival_time"
    time_prior, sigma_prior = priors[time_param], priors.get("sigma")
    if getattr(sigma_prior, "high", None) is None:
        raise ValueError("warm starts draw a width per pulse: they need a bounded 'sigma' prior.")
    values = _starting_values(burst, priors, n_pulses)                                    # random starts
    warm_values = _previous_fit_values(prev_params, values, priors, n_pulses)              # warm starts

    def unconstrained(vals):
        """The starting values on the unconstrained scale the optimizer works on."""
        return initialize_model(jax.random.PRNGKey(seed), likelihood, init_strategy=init_to_value(values=vals),
                                model_kwargs=model_kwargs)[0].z
    z_random, z_warm = unconstrained(values), unconstrained(warm_values)

    sigma_range = (_interior(sigma_prior, float(sigma_prior.low)), _interior(sigma_prior, float(sigma_prior.high)))
    x, tail = wall_clearance(burst, single_pulse_fn, priors, values, n_pulses, likelihood, likelihood_kwargs)
    _, tail_warm = wall_clearance(burst, single_pulse_fn, priors, warm_values, n_pulses, likelihood, likelihood_kwargs)
    kept_times = np.array([_interior(time_prior, float(v)) for v in np.asarray(prev_params[time_param]).reshape(-1)])
    init_times, init_sigma, init_bands = box_times(
        burst, prev_M, search, n_pulses, n_starts, n_warm, time_prior, sigma_range, seed,
        clearance=(x, np.where(np.arange(n_starts) < n_warm, tail_warm, tail)),
        fixed=(kept_times, np.asarray(warm_values["sigma"])[: n_pulses - 1]))

    # Split starts: the last warm starts, each splitting one old pulse, widest first
    splits = {}
    if split_starts:
        n_split = min(n_pulses - 1, n_warm // 2)
        widest = np.argsort(-np.asarray(warm_values["sigma"])[: n_pulses - 1])[:n_split]
        splits = {n_warm - n_split + i: int(k) for i, k in enumerate(widest)}

    # Box-matched spectra apply only when the spectrum is sampled per pulse
    box_spectrum = box_spectrum and "gamma" in z_warm and "beta" in z_warm
    new, n_matched, n_moved = n_pulses - 1, 0, 0

    # Each start's own point: its warm, split or random values, with its own arrival times and widths
    per_start = []
    for c in range(n_starts):
        if c in splits:
            k = splits[c]
            half_width = np.sqrt((0.5 * float(np.asarray(warm_values["sigma"])[k])) ** 2 + float(tail_warm) ** 2)
            vals = _split_values(warm_values, priors, n_pulses, k, d=max(x, 1.0) * half_width)
            init_times[c], init_sigma[c] = np.asarray(vals[time_param]), np.asarray(vals["sigma"])
            z = dict(unconstrained(vals))
        else:
            z = dict(z_warm if c < n_warm else z_random)
            if box_spectrum and c < n_warm and np.all(np.isfinite(init_bands[c, new])):
                gamma, beta, moved = _box_spectrum(burst, priors, *init_bands[c, new])
                for name, value in (("gamma", gamma), ("beta", beta)):
                    z[name] = z[name].at[new].set(biject_to(priors[name].support).inv(jnp.float32(value)))
                n_matched, n_moved = n_matched + 1, n_moved + int(moved)
        z[time_param] = biject_to(time_prior.support).inv(jnp.asarray(init_times[c], dtype=jnp.float32))
        z["sigma"] = biject_to(sigma_prior.support).inv(jnp.asarray(init_sigma[c], dtype=jnp.float32))
        per_start.append(z)
    if splits:
        print(f"split starts: {len(splits)} warm start(s) each split one of the previous fit's pulses in two "
              f"(pulses {sorted(splits.values())})")
    if n_matched:
        print(f"box-matched spectra: {n_matched} warm start(s) start the new pulse with a spectrum peaked on its "
              f"box's band" + (f"; in {n_moved} of them a gamma or beta prior bound limited the match"
                               if n_moved else ""))
    return {key: jnp.stack([z[key] for z in per_start]) for key in per_start[0]}, init_times, init_sigma


# -------------------------------------------------------------------
# 3. Sampling
# -------------------------------------------------------------------

def run_fit(
    burst,
    single_pulse_fn: Callable,
    priors: Dict[str, dist.Distribution],
    likelihood: Callable = FD_gaussianlikelihood,
    n_pulses: int = 1,
    num_warmup: int = 500,
    num_samples: int = 200,
    num_chains: int = 1,
    seed: int = None,
    init_strategy=None,
    dense_mass: bool = True,
    max_tree_depth: int | tuple[int, int] = (10, 8),
    chain_method: str = "auto",
    progress_bar: bool = True,
    shared_params: tuple = ("DM", "tau_ref"),
    extra_fields=("num_steps", "accept_prob", "diverging", "potential_energy"),
    target_accept_prob: float = 0.8,
    maxL_start: dict = None,
    **likelihood_kwargs,
) -> MCMC:
    """
    Runs NUTS on one burst and returns the finished MCMC object, with the run's settings attached as
    mcmc.seed, mcmc.priors, mcmc.n_pulses, mcmc.likelihood_kwargs (the likelihood's defaults, what was
    passed, and shared_params). Prints the likelihood, how the amplitude is parameterized, and the
    seed. Raises unless exactly one of 'A' and 'A_snr' is in priors.

    burst:              the ConditionedData of the burst
    single_pulse_fn:    the model closure from model.setup_model
    priors:             the priors from set_priors
    likelihood:         the NumPyro model, likelihood.FD_gaussianlikelihood by default
    n_pulses:           number of pulses
    num_warmup:         warmup steps per chain, in which the step size and mass matrix adapt; keep >= 500 with dense_mass, whose matrix comes from the last warmup window only
    num_samples:        draws kept per chain after warmup
    num_chains:         number of chains; on the CPU up to as many as devices.use_device made run in parallel, on a GPU all run as one batch
    seed:               integer the run's random key is made from; None draws one, which is printed and stored as mcmc.seed
    init_strategy:      a NumPyro initialization strategy; None uses default_init (broad pulses at random times), unless maxL_start is given
    dense_mass:         adapt a full covariance mass matrix (True) rather than a diagonal one
    max_tree_depth:     largest NUTS tree depth d (at most 2^d - 1 leapfrog steps per draw), or a pair (warmup, sampling)
    chain_method:       "auto" matches the device ("parallel" on the CPU, "vectorized" on a GPU), or any NumPyro chain_method
    progress_bar:       show NumPyro's progress bar
    shared_params:      parameters sampled once for the whole burst (as '<name>_shared'); () keeps every parameter per pulse
    extra_fields:       per-draw sampler quantities kept beside the samples, read with mcmc.get_extra_fields()
    target_accept_prob: mean acceptance probability the step size is tuned for; higher gives smaller steps and fewer divergences
    maxL_start:         a maximum-likelihood fit to start every chain at its best point: maxL_fit's result, or maxL_iterative_pulse_fit's (its kept 'fit' is used); n_pulses is taken from it, and priors should be its 'priors'
    likelihood_kwargs:  passed straight to the likelihood, e.g. repulsion={...}
    """
    if maxL_start is not None:
        if init_strategy is not None:
            raise ValueError("maxL_start sets every chain's starting point: pass it without init_strategy.")
        best = maxL_start.get("fit", maxL_start)
        n_pulses = int(np.size(best["map"]["mean_arrival_time"]))
        init_strategy = init_to_value(values={k: jnp.asarray(v) for k, v in best["map"].items() if k in best["z"]})

    unknown = set(shared_params) - priors.keys()
    if unknown:
        raise ValueError(f"shared_params {sorted(unknown)} are not in priors.")

    filled_in = {"data_fd", "sigma_noise_fd", "n_pulses", "priors", "single_pulse_fn"}
    clash = filled_in & likelihood_kwargs.keys()
    if clash:
        raise ValueError(f"{sorted(clash)} are filled in by run_fit and cannot be passed as likelihood_kwargs.")

    if len(AMPLITUDE_PARAMS & priors.keys()) != 1:
        raise ValueError(
            "the likelihood samples the amplitude, so exactly one of 'A' and 'A_snr' must be in "
            "priors: pass s_opt to set_priors, or an A=... override."
        )

    if seed is None:
        # JAX has no global random state, so draw a seed and keep it, so the run can be repeated exactly
        seed = int(np.random.default_rng().integers(2**31 - 1))
    amplitude = "fluence in units of SNR as A_snr" if "A_snr" in priors else "fluence as A"
    same_priors = maxL_start is not None and describe_priors(best["priors"]) == describe_priors(priors)
    print(f"run_fit: likelihood {getattr(likelihood, '__name__', likelihood)}, parameterizing {amplitude}, "
          f"seed {seed}; priors" + (" as in the maxL fit" if same_priors else ":"))
    if not same_priors:
        _print_priors(priors, indent="  ")
    if maxL_start is None:
        _announce_delta_init(priors)              # with maxL_start, delta starts at the maxL value instead

    if init_strategy is None:
        init_strategy = default_init(burst, priors, n_pulses)

    if chain_method == "auto":
        # GPU: all chains as one batch on the single device. CPU: one chain per device.
        chain_method = "parallel" if jax.default_backend() == "cpu" else "vectorized"

    kernel = NUTS(likelihood, dense_mass=dense_mass, init_strategy=init_strategy,
                  max_tree_depth=max_tree_depth, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method=chain_method, progress_bar=progress_bar)

    run_kwargs = dict(
        data_fd=burst.data_fd,
        sigma_noise_fd=burst.sigma_noise_fd,
        n_pulses=n_pulses,
        priors=priors,
        single_pulse_fn=single_pulse_fn,
        shared_params=shared_params,
        **likelihood_kwargs,
    )

    mcmc.run(jax.random.PRNGKey(seed), extra_fields=extra_fields, **run_kwargs)

    mcmc.seed = seed                  # the seed actually used; save_run records it
    mcmc.priors = dict(priors)
    mcmc.n_pulses = n_pulses
    # What actually ran: the likelihood's own defaults (e.g. repulsion on), what was passed, and shared_params
    defaults = {name: p.default for name, p in inspect.signature(likelihood).parameters.items()
                if p.default is not inspect.Parameter.empty and name not in filled_in | {"shared_params"}}
    mcmc.likelihood_kwargs = {**defaults, **likelihood_kwargs, "shared_params": shared_params}

    return mcmc


# -------------------------------------------------------------------
# 4. Maximum-likelihood search
# -------------------------------------------------------------------

# the factor sites that are maximized: the fit, and the repulsion prior between active pulses
OBJECTIVE_SITES = ("log_likelihood", "repulsive_toa_prior")
TAU_SITE = "tau_ref_shared"          # the shared scattering time, the site profiled over


def _model_kwargs(burst, single_pulse_fn, priors, n_pulses, shared_params, likelihood_kwargs):
    """
    The likelihood's arguments, as run_fit also builds them.

    burst:             the ConditionedData of the burst
    single_pulse_fn:   the model closure from model.setup_model
    priors:            the priors of the fit
    n_pulses:          number of pulses
    shared_params:     parameters sampled once for the whole burst
    likelihood_kwargs: the other keyword arguments the likelihood is given, e.g. repulsion={...}
    """
    return dict(data_fd=burst.data_fd, sigma_noise_fd=burst.sigma_noise_fd, n_pulses=n_pulses,
                priors=priors, single_pulse_fn=single_pulse_fn, shared_params=shared_params,
                **likelihood_kwargs)


def _starts(burst, single_pulse_fn, priors, likelihood, n_pulses, num_starts, n_warm, model_kwargs,
            likelihood_kwargs, seed, prev=None, search=None, split_starts=True, box_spectrum=True):
    """
    The optimizer's starting points: without prev, every start is random (default_init); with it,
    warm_start_points. Returns (z, init_times,
    init_sigma) as warm_start_points does, z a dict of unconstrained arrays with leading axis
    num_starts (init_times and init_sigma are None for random starts).

    burst:             the ConditionedData of the burst
    single_pulse_fn:   the model closure from model.setup_model
    priors:            the priors of the fit
    likelihood:        the NumPyro model
    n_pulses:          number of pulses
    num_starts:        number of starting points
    n_warm:            how many of them start from prev (warm starts)
    model_kwargs:      the likelihood's arguments (_model_kwargs)
    likelihood_kwargs: the keyword arguments the likelihood is given, for the wall clearance
    seed:              integer the random starts are drawn from
    prev:              the previous fit (maxL_fit's result, n_pulses - 1 pulses), or None
    search:            postprocess.excess_power_search of prev's residual
    split_starts:      let some warm starts split one old pulse in two
    box_spectrum:      start a box-placed new pulse with a spectrum peaked on its box's band
    """
    if prev is None:
        keys = jax.random.split(jax.random.PRNGKey(seed), num_starts)
        info = initialize_model(keys, likelihood, init_strategy=default_init(burst, priors, n_pulses),
                                model_kwargs=model_kwargs)[0]
        return info.z, None, None
    return warm_start_points(burst, single_pulse_fn, priors, likelihood, model_kwargs, likelihood_kwargs,
                             prev["map"], prev["M"], search, n_pulses, num_starts, n_warm, seed,
                             split_starts=split_starts, box_spectrum=box_spectrum)


def _adam(f, n_steps, lr, lr_decay):
    """
    Optax's Adam on f for n_steps steps, the step size decaying geometrically from lr to
    lr * lr_decay; a non-finite gradient component is replaced by 0 so one bad step cannot poison
    the run. Returns a function x -> (x_end, f(x_end)).

    f:        objective to minimize, a function of the flat parameter vector
    n_steps:  number of Adam steps
    lr:       starting step size, on the unconstrained scale
    lr_decay: final step size as a fraction of lr
    """
    opt = optax.adam(optax.exponential_decay(init_value=lr, transition_steps=max(n_steps - 1, 1),
                                             decay_rate=lr_decay))
    grad_f = jax.grad(f)

    def run(x):
        def step(carry, _):
            x, state = carry
            g = grad_f(x)
            updates, state = opt.update(jnp.where(jnp.isfinite(g), g, 0.0), state, x)
            return (optax.apply_updates(x, updates), state), None
        (x, _), _ = jax.lax.scan(step, (x, opt.init(x)), None, length=n_steps)
        return x, f(x)
    return run


def maxL_fit(burst, single_pulse_fn, priors, n_pulses, likelihood=FD_gaussianlikelihood, num_starts=64,
             n_warm=None, prev=None, search=None, split_starts=True, box_spectrum=True,
             shared_params=("DM", "tau_ref"), n_adam=1000, lr=0.05, lr_decay=0.01, tau_profile=8,
             n_polish=500, lr_polish=0.01, tau_floor_dt_scale=0.1, tau_zero_dt_scale=1e-3, tau_parallel=False,
             seed=0, verbose=True, **likelihood_kwargs):
    """
    Maximum-likelihood fit of n_pulses pulses. From each of num_starts starting points (warm and
    random starts, _starts), it maximizes the log-likelihood plus
    the repulsion prior on the unconstrained scale; the other priors act only as bounds, through the
    transforms NUTS samples in, with their densities and Jacobians left out, so the maximum does not
    depend on the parameterization. The shared scattering time is profiled: each start is run once
    for every value of tau_ref on a grid (one effectively-zero value, then at least tau_profile
    values log-spaced from tau_floor_dt_scale * dt to the prior's upper bound, more if neighbouring
    values are further apart than the polish can move tau_ref), first with tau_ref held at that value
    (n_adam Adam steps) and then with it released (n_polish steps), and the best run over all values
    and starts is kept. Holding tau_ref first stops the optimizer from stretching the scattering tail
    across pulses that have not settled yet, and the best held ln L at each value is the profile
    likelihood in tau_ref. Returns a dict: 'logL' (best log-likelihood + repulsion), 'logL_starts'
    (each start's best over the grid), 'ends' (each start's best end point, sampled sites,
    constrained), 'map' (every site at the best point), 'post' (the same with a leading draw axis
    of 1, for plotting.model_from_draw), 'M' (the best model), 'z' (the best unconstrained point),
    'priors', 'init_times', 'init_sigma', 'seconds', 'tau_grid', 'profile' (best ln L with tau_ref
    held at each grid value), 'logL_runs' (n_tau, num_starts) and 'tau_end_runs' (tau_ref at the end
    of each run, after release).

    burst:              the ConditionedData of the burst
    single_pulse_fn:    the model closure from model.setup_model
    priors:             the priors from set_priors
    n_pulses:           number of pulses
    likelihood:         the NumPyro model, likelihood.FD_gaussianlikelihood by default
    num_starts:         number of starting points, all optimized at once
    n_warm:             how many start from prev (warm starts); None for half
    prev:               the previous fit (this function's result, n_pulses - 1 pulses) for warm starts, or None for random starts only
    search:             postprocess.excess_power_search of prev's residual
    split_starts:       let some warm starts split one old pulse in two (warm_start_points)
    box_spectrum:       start a box-placed new pulse with a spectrum peaked on its box's band (warm_start_points)
    shared_params:      parameters sampled once for the whole burst; must include 'tau_ref', which is profiled
    n_adam:             Adam steps with tau_ref held at each grid value
    lr:                 starting step size of those steps, on the unconstrained scale
    lr_decay:           final step size of each Adam run as a fraction of its starting one
    tau_profile:        fewest log-spaced tau_ref values; raised when the range would leave gaps the polish cannot cross
    n_polish:           Adam steps after tau_ref is released
    lr_polish:          starting step size of the polish
    tau_floor_dt_scale: lowest log-spaced tau_ref value, in units of the sample time dt
    tau_zero_dt_scale:  the effectively-zero tau_ref value added below the log grid, in units of dt; None leaves it out
    tau_parallel:       run all tau_ref values at once (memory x number of values) instead of one after another
    seed:               integer the random starts are drawn from
    verbose:            print the likelihood (not for a fit warm-started from prev, a later step of a search that printed it) and the best fit, with its tau_ref
    likelihood_kwargs:  passed straight to the likelihood, e.g. repulsion={...}
    """
    from plotting import model_from_draw

    t_start = time.time()
    n_warm = num_starts // 2 if n_warm is None else n_warm
    model_kwargs = _model_kwargs(burst, single_pulse_fn, priors, n_pulses, shared_params, likelihood_kwargs)
    if verbose:
        if prev is None:                          # a warm-started fit is a later step of a search that said it already
            amplitude = "fluence in units of SNR as A_snr" if "A_snr" in priors else "fluence as A"
            print(f"maxL_fit: likelihood {getattr(likelihood, '__name__', likelihood)}, parameterizing {amplitude}")
        _announce_delta_init(priors, warm=prev is not None)
    z0, init_times, init_sigma = _starts(burst, single_pulse_fn, priors, likelihood, n_pulses, num_starts, n_warm,
                                         model_kwargs, likelihood_kwargs, seed, prev, search, split_starts,
                                         box_spectrum)

    def trace_at(z):
        params = constrain_fn(likelihood, (), model_kwargs, z)
        return handlers.trace(handlers.substitute(handlers.seed(likelihood, 0), data=params)).get_trace(**model_kwargs)

    def neg_log_l(z):
        tr = trace_at(z)
        return -sum(jnp.sum(tr[k]["fn"].log_prob(tr[k]["value"])) for k in OBJECTIVE_SITES if k in tr)

    z_one = jax.tree_util.tree_map(lambda v: v[0], z0)
    if TAU_SITE not in z_one:
        raise ValueError("maxL_fit profiles tau_ref, so it must be shared between pulses (shared_params containing 'tau_ref').")
    _, unravel = ravel_pytree(z_one)
    f = lambda x: neg_log_l(unravel(x))
    x0 = jax.vmap(lambda z: ravel_pytree(z)[0])(z0)                                  # (num_starts, d)

    # The tau_ref grid
    tau_prior = priors["tau_ref"]
    dt = float(burst.dt)
    # a margin of 1e-6 of the prior's range keeps the grid inside the bounds (finite transformed
    # coordinates) without clipping small values; the default 1e-3 would put the floor near 1 us
    tau_lo = _interior(tau_prior, float(tau_prior.low), margin=1e-6)
    tau_hi = _interior(tau_prior, float(tau_prior.high))
    # Neighbouring grid values no further apart than the polish can move tau_ref. Adam moves a
    # parameter by at most about its step size per step, so the polish travels at most the sum of
    # its step sizes on the unconstrained scale, where tau_ref well below its cap is ~ln tau_ref
    tau_lo_log = max(tau_floor_dt_scale * dt, tau_lo)
    reach = lr_polish * n_polish * ((1.0 - lr_decay) / np.log(1.0 / lr_decay) if lr_decay != 1.0 else 1.0)
    n_log = int(tau_profile)
    if reach > 0:
        n_log = max(n_log, int(np.ceil(np.log(tau_hi / tau_lo_log) / reach)) + 1)
    tau_grid = np.geomspace(tau_lo_log, tau_hi, n_log)
    if tau_zero_dt_scale is not None:
        tau_grid = np.concatenate([[max(tau_zero_dt_scale * dt, tau_lo)], tau_grid])
    tau_u = biject_to(tau_prior.support).inv(jnp.asarray(tau_grid, dtype=jnp.float32))

    def run_at_tau(tu):
        """Every start with tau_ref held at tu, then released: end points, objective, objective while held."""
        pin = lambda z: {**z, TAU_SITE: tu}
        f_pin = lambda x: neg_log_l(pin(unravel(x)))
        adam_pinned, adam_free = _adam(f_pin, n_adam, lr, lr_decay), _adam(f, n_polish, lr_polish, lr_decay)

        def one(x):
            x, f_pinned = adam_pinned(x)
            x = ravel_pytree(pin(unravel(x)))[0]               # start the release at the grid value
            x, fx = adam_free(x)
            return x, fx, f_pinned
        return jax.vmap(one)(x0)

    runner = jax.jit(jax.vmap(run_at_tau) if tau_parallel else (lambda tu: jax.lax.map(run_at_tau, tu)))
    x_runs, f_runs, f_pinned = runner(tau_u)                     # (n_tau, num_starts, ...)
    f_runs = np.where(np.isfinite(np.asarray(f_runs)), np.asarray(f_runs), np.inf)
    profile = -np.min(np.where(np.isfinite(np.asarray(f_pinned)), np.asarray(f_pinned), np.inf), axis=1)
    best_tau = np.argmin(f_runs, axis=0)                         # each start's best tau value
    x_end = jnp.asarray(np.asarray(x_runs)[best_tau, np.arange(num_starts)])
    f_end = f_runs[best_tau, np.arange(num_starts)]

    # where every start ended: its sampled parameters, constrained (for diagnosing the optimizer)
    ends = jax.vmap(lambda x: constrain_fn(likelihood, (), model_kwargs, unravel(x)))(x_end)
    ends = {k: np.asarray(v) for k, v in ends.items()}
    best = int(np.argmin(f_end))
    z_best = unravel(x_end[best])
    tr = trace_at(z_best)
    values = {k: np.asarray(s["value"]) for k, s in tr.items()
              if s["type"] in ("sample", "deterministic") and not s.get("is_observed", False)}
    post = {k: v[None] for k, v in values.items()}
    M, _ = model_from_draw(single_pulse_fn, post, 0)
    logL = -float(f_end[best])
    seconds = time.time() - t_start
    tau_end = jax.vmap(jax.vmap(lambda x: constrain_fn(likelihood, (), model_kwargs, unravel(x))[TAU_SITE]))(x_runs)

    out = dict(logL=logL, logL_starts=-f_end, ends=ends, map=values, post=post, M=np.asarray(M), z=z_best,
               priors=priors, init_times=init_times, init_sigma=init_sigma, seconds=seconds, tau_grid=tau_grid,
               profile=profile, logL_runs=-f_runs, tau_end_runs=np.asarray(tau_end))

    if verbose:
        near = int(np.sum(-f_end >= logL - 1.0))
        kind = "warm" if prev is not None and best < n_warm else "random"
        print(f"maxL_fit: {n_pulses} pulse(s), {num_starts} starts ({n_warm if prev is not None else 0} warm)"
              f" x {len(tau_grid)} tau values: best ln L + repulsion = {logL:.1f} at tau_ref = "
              f"{float(values[TAU_SITE]) * 1e6:.3g} \u00b5s (start {best}, {kind}); "
              f"{near} start(s) within 1 nat; {seconds:.0f} s")
    return out


def maxL_iterative_pulse_fit(burst, single_pulse_fn, likelihood=FD_gaussianlikelihood, snr_min=8.0, n_max=5,
                             num_starts=64, n_warm=None, split_starts=True, box_spectrum=True, s_opt=100.0,
                             seed=None, epsilon=0.2, p_false_alarm=0.01, plot_steps=False, verbose=True,
                             prior_kwargs=None, shared_params=("DM", "tau_ref"), n_adam=1000, lr=0.05,
                             lr_decay=0.01, tau_profile=8, n_polish=500, lr_polish=0.01, tau_floor_dt_scale=0.1,
                             tau_zero_dt_scale=1e-3, tau_parallel=False, **likelihood_kwargs):
    """
    Finds the number of pulses with maximum-likelihood fits (maxL_fit). Fits 1, 2, ...
    pulses; each step searches the previous fit's residual (postprocess.excess_power_search) and
    stops if nothing is significant, otherwise fits one more pulse and accepts it only if the whole
    fit's log-likelihood + repulsion rises by at least snr_min^2 / 2 + k (k the parameters a new
    pulse adds, one nat each as in AIC) and at least one significant box's own gain reaches the same
    threshold (postprocess.boxes_explained); a rejected pulse ends the search. To sample the kept
    fit's posterior, pass the result to run_fit as maxL_start. Returns a dict: 'n_pulses', 'fit'
    (the kept maxL_fit result), 'fits' ({n: maxL_fit result}), 'steps' (each with 'search', and
    when fitted 'check', 'd_logL' and 'accept'), 'stop', 'snr_min', 'logL_min' (the threshold),
    'min_channels' and 'seed'.

    burst:              the ConditionedData of the burst
    single_pulse_fn:    the model closure from model.setup_model
    likelihood:         the NumPyro model, likelihood.FD_gaussianlikelihood by default
    snr_min:            SNR of the weakest pulse to add; sets the gain threshold and the box search's minimum SNR
    n_max:              largest number of pulses to try
    num_starts:         starting points per fit
    n_warm:             how many start from the previous fit; None for half
    split_starts:       let some warm starts split one old pulse in two (warm_start_points)
    box_spectrum:       start a box-placed new pulse with a spectrum peaked on its box's band (warm_start_points)
    s_opt:              scale of the A_snr prior (set_priors)
    seed:               integer the random starts are drawn from; None draws one, which is printed and returned
    epsilon:            model error the residual search tolerates, as a fraction of the model's amplitude
    p_false_alarm:      chance the residual search finds a significant box in pure noise (Bonferroni-corrected)
    plot_steps:         plot each step's check (plotting.plot_boxes_explained)
    verbose:            print each step
    prior_kwargs:       passed to set_priors
    shared_params:      parameters sampled once for the whole burst; must include 'tau_ref'
    n_adam, lr, lr_decay, tau_profile, n_polish, lr_polish, tau_floor_dt_scale, tau_zero_dt_scale, tau_parallel: passed to maxL_fit
    likelihood_kwargs:  passed straight to the likelihood, e.g. repulsion={...}
    """
    from postprocess import excess_power_search, boxes_explained, min_box_channels
    from plotting import plot_boxes_explained

    prior_kwargs = dict(prior_kwargs or {})
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31 - 1))
    n_warm = num_starts // 2 if n_warm is None else n_warm
    priors1 = set_priors(burst, single_pulse_fn, s_opt=s_opt, verbose=False, **prior_kwargs)
    k_new = sum(1 for name in priors1 if name not in shared_params)        # parameters a new pulse adds
    logL_min = snr_min ** 2 / 2.0 + k_new
    min_channels = min_box_channels(burst, priors1)
    if verbose:
        print(f"maxL_iterative_pulse_fit: snr_min {snr_min:g}, accept a pulse if ln L rises by >= {logL_min:g} "
              f"(snr_min^2 / 2 + {k_new} parameters), minimum box width {min_channels} channels, "
              f"up to {n_max} pulses, seed {seed}, tau profile over at least {tau_profile} values from "
              f"{tau_floor_dt_scale:g} dt" + (f" plus {tau_zero_dt_scale:g} dt" if tau_zero_dt_scale is not None else "")
              + "; priors:")
        _print_priors(priors1, indent="  ")

    def fit(n, prev=None, search=None):
        """The maximum-likelihood fit of n pulses, warm-started from prev when given."""
        priors = set_priors(burst, single_pulse_fn, s_opt=s_opt, verbose=False, **prior_kwargs)
        return maxL_fit(burst, single_pulse_fn, priors, n, likelihood=likelihood, num_starts=num_starts,
                        n_warm=n_warm, prev=prev, search=search, split_starts=split_starts,
                        box_spectrum=box_spectrum, shared_params=shared_params, n_adam=n_adam, lr=lr,
                        lr_decay=lr_decay, tau_profile=tau_profile, n_polish=n_polish, lr_polish=lr_polish,
                        tau_floor_dt_scale=tau_floor_dt_scale, tau_zero_dt_scale=tau_zero_dt_scale,
                        tau_parallel=tau_parallel, seed=seed, verbose=verbose, **likelihood_kwargs)

    fits, steps = {1: fit(1)}, []
    n_final, stop = 1, f"reached n_max = {n_max}"
    for n in range(2, n_max + 1):
        ep = excess_power_search(burst, fits[n - 1]["M"], epsilon=epsilon, min_snr=snr_min,
                                 p_false_alarm=p_false_alarm, min_channels=min_channels, verbose=verbose)
        if not ep["add_pulse"]:
            steps.append(dict(n=n, search=ep))
            stop = "no significant leftover signal"
            break
        fits[n] = fit(n, prev=fits[n - 1], search=ep)
        check = boxes_explained(burst, fits[n - 1]["M"], fits[n]["M"], ep, min_gain=logL_min, verbose=verbose)
        if plot_steps:
            plot_boxes_explained(burst, fits[n - 1]["M"], fits[n]["M"], check)
        d_logL = fits[n]["logL"] - fits[n - 1]["logL"]
        accept = bool(check["any_removed"] and d_logL >= logL_min)
        steps.append(dict(n=n, search=ep, check=check, d_logL=d_logL, accept=accept))
        if verbose:
            print(f"{n - 1} -> {n} pulses: delta ln L = {d_logL:.1f} (need {logL_min:g}), explains leftover "
                  f"signal: {check['any_removed']} -> {'accept' if accept else 'reject'}")
        if not accept:
            reasons = ([] if check["any_removed"] else ["it explains no significant leftover signal"]) + \
                      ([] if d_logL >= logL_min else [f"delta ln L {d_logL:.1f} < {logL_min:g}"])
            stop = f"{n}-pulse fit rejected: " + " and ".join(reasons)
            break
        n_final = n

    if verbose:
        print(f"final: {n_final} pulses ({stop}); optimizer time {sum(f['seconds'] for f in fits.values()):.0f} s")
    return dict(n_pulses=n_final, fit=fits[n_final], fits=fits, steps=steps, stop=stop, snr_min=snr_min,
                logL_min=logL_min, min_channels=min_channels, seed=seed)
