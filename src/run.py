"""
run.py - Priors, NUTS sampling and the Bayesian iterative pulse search for one burst.

Choose the device first, before anything imports jax (devices.use_device). A fit is then

    priors = set_priors(burst, single_pulse_fn, s_opt=100.0)
    mcmc   = run_fit(burst, single_pulse_fn, priors, n_pulses=n_pulses)
    idata  = results.save_run(out_path, mcmc, burst, priors, data_path, single_pulse_fn=single_pulse_fn,
                              dm_dedispersion=..., dedispersion_ref_freq=...)

and iterative_pulse_fit lets the data decide the number of pulses, adding one at a time while the
residual holds leftover signal that a new pulse explains (maxL_itfit.maxL_iterative_pulse_fit does
the same with maximum-likelihood fits). The amplitude prior is set once, in set_priors, as
'A_snr' ~ HalfNormal(s_opt) in units of the template's optimal SNR, unless a fluence prior A=... is
given instead.
"""

import inspect
from functools import partial
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model

from likelihood import FD_gaussianlikelihood, repulsive_toa_prior
from priors import dispersion_prior_bounds, width_prior_bounds
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
    gamma_range=(-10.0, 200.0),
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
    tau_ref so that no scattering, tau_ref = 0, is inside the prior); gamma and beta ~ Uniform over
    gamma_range and beta_range; and A_snr ~ HalfNormal(s_opt) if s_opt is given. Raises if a value
    is not a NumPyro distribution or if both 'A' and 'A_snr' are set.

    burst:           the ConditionedData; its window, time resolution and band set the default bounds
    single_pulse_fn: the model closure from model.setup_model, for the DM bounds and the scattering index
    toa:             arrival time the width bounds are computed for (s, relative to the segment start); None uses the middle of the window, which gives the widest sigma bound
    s_opt:           scale of the A_snr prior (SNR units); every fit needs it unless an A=... or A_snr=... override is given
    gamma_range:     (low, high) of the spectral index prior
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

    priors = {
        "mean_arrival_time": dist.Uniform(0.0, float(burst.times[-1])),   # (s, relative to segment start)
        "DM": dist.Uniform(dm["DM_min"], dm["DM_max"]),                    # residual DM (pc cm^-3)
        "sigma": dist.LogUniform(width["sigma_min"], width["sigma_max"]),  # intrinsic width (s), per pulse
        "tau_ref": dist.Uniform(width["tau_ref_min"], width["tau_ref_max"]),  # scattering time at nu_r (s)
        "gamma": dist.Uniform(float(gamma_range[0]), float(gamma_range[1])),  # spectral index
        "beta": dist.Uniform(float(beta_range[0]), float(beta_range[1])),     # spectral running
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
        for name, description in describe_priors(priors).items():
            print(f"{name.removeprefix('prior.')}: {description}")

    return priors


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


def _previous_fit_values(params, defaults, priors, n_pulses, reset=("tau_ref",)):
    """
    Starting values for a WARM start: the previous fit's n_pulses - 1 pulses as they were, plus one
    new pulse, last, with the default starting values (or the median of the previous pulses for a
    parameter without a default). A sampled delta is always restarted at DELTA_INIT rather than
    inherited (see _announce_delta_init). Every value is moved just inside its prior's bounds.

    params:   one draw of the previous fit, name -> value: per-pulse values of shape (n_pulses - 1,), shared ones ('<name>_shared') scalars; sites the model does not sample are ignored
    defaults: _starting_values for n_pulses pulses, which give the new pulse its values
    priors:   the priors of the new fit
    n_pulses: number of pulses in the new fit
    reset:    parameters not inherited, which start from their defaults instead; by default tau_ref, since a previous fit missing a component often stretched its scattering tail across it (delta is always added)
    """
    reset = tuple(reset) + ("delta",)
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
            raise ValueError(f"warm_start params['{name}'] has shape {v.shape}; expected "
                             f"({n_pulses - 1},), one value per pulse of the previous fit.")
    return values


def box_times(burst, M_FD, result, n_pulses, num_chains, n_warm_chains, time_prior, sigma_range, seed,
              clearance=(0.0, 0.0), fixed=None):
    """
    Starting arrival times and widths for a warm start, each of shape (num_chains, n_pulses), from
    the residual boxes of the previous fit. The first n_warm_chains chains keep the `fixed` pulses
    (the previous fit's) and put each new pulse in its own significant box of `result`, at a time
    drawn inside the box with probability proportional to the whitened residual power there; every
    pulse of the other chains gets a time drawn uniformly from the prior. Every pulse not fixed gets
    a width drawn log-uniformly over sigma_range. Pulses are placed one at a time, each only at
    times clear of the walls of those already placed (wall_clearance); a pulse with no clear time in
    its box gets a random clear time, and a chain with no clear time anywhere is reported as starting
    inside a wall.

    burst:         the ConditionedData of the burst
    M_FD:          the previous fit's model on the retained Fourier bins
    result:        postprocess.excess_power_search of the previous fit's residual
    n_pulses:      number of pulses in the new fit
    num_chains:    number of chains (or optimizer starts)
    n_warm_chains: how many of them start warm
    time_prior:    the arrival-time prior (bounded)
    sigma_range:   (low, high) the new widths are drawn log-uniformly between
    seed:          integer the draws are made from
    clearance:     (x, tail) from wall_clearance; tail may be an array with one value per chain
    fixed:         (times, sigmas) of the pulses warm chains keep, each of shape (m,), m < n_pulses
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

    # Residual power |r(t)|^2 inside each significant box: the whitened residual on the box's
    # channels and Fourier band, transformed to time and summed over those channels
    boxes = []
    for tile in result["tiles"]:
        if not tile["significant"]:
            continue
        (c0, c1), n_b = tile["channel_index"], tile["resolution"]
        j0, j1 = tile["band_index"][0] * n_b, tile["band_index"][1] * n_b
        full = np.zeros((c1 - c0, n), dtype=complex)
        full[:, k[j0:j1]] = Z[c0:c1, j0:j1]
        power = np.sum(np.abs(np.fft.ifft(full, axis=-1)) ** 2, axis=0)
        spans = [tile["t_range"]] + ([tile["t_wrap"]] if tile["t_wrap"] else [])
        inside = np.any([(t >= a) & (t <= b) for a, b in spans], axis=0)
        boxes.append(np.where(inside & in_prior, power, 0.0))

    log_lo, log_hi = np.log(sigma_range[0]), np.log(sigma_range[1])
    sigmas = np.exp(rng.uniform(log_lo, log_hi, size=(num_chains, n_pulses)))   # log-uniform widths
    times = np.empty((num_chains, n_pulses))
    m = 0
    if fixed is not None:                     # warm chains keep the previous fit's pulses as they were
        m = len(fixed[0])
        times[:n_warm_chains, :m], sigmas[:n_warm_chains, :m] = fixed[0], fixed[1]
    tail = np.asarray(tail, dtype=float)
    widths = np.sqrt(sigmas ** 2 + (tail.reshape(-1, 1) if tail.ndim else tail) ** 2)

    in_wall = []
    for c in range(num_chains):
        warm = c < n_warm_chains
        chosen = rng.permutation(len(boxes))[:n_pulses - m] if warm else []
        for j in range(m if warm else 0, n_pulses):
            # Times clear of the walls of every pulse already placed in this chain
            free = in_prior & np.all([np.abs(t - times[c, i]) >= x * (widths[c, j] + widths[c, i])
                                      for i in range(j)], axis=0)
            jj = j - (m if warm else 0)                      # index among the pulses being placed
            p = boxes[chosen[jj]] * free if jj < len(chosen) else free.astype(float)
            if p.sum() <= 0:                                 # box full: start this pulse at random
                p = free.astype(float)
            if p.sum() <= 0:                                 # no free time left anywhere
                p = in_prior.astype(float)
                in_wall.append(c)
            times[c, j] = rng.choice(t, p=p / p.sum())

    in_wall = sorted(set(in_wall))
    if in_wall:
        print(f"box_times: {len(in_wall)} of {num_chains} chains could not keep every pulse out of the "
              f"walls (widths too broad for {n_pulses} pulses in this segment): {in_wall}")
    return times, sigmas


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
                      prev_M, search, n_pulses, n_starts, n_warm, seed, reset=("tau_ref",), split_starts=True):
    """
    Starting points for a fit with one more pulse than a previous fit, shared by NUTS (run_fit) and
    the maximum-likelihood search (maxL_fit). The first n_warm start from the previous fit
    (_previous_fit_values) with the new pulse in a flagged residual box (box_times); with
    split_starts, the last of those instead split one old pulse in two (_split_values), widest old
    pulse first, one per old pulse up to half the warm starts, which gives a start that can separate
    two close pulses the previous fit covered with one wide pulse. The rest start from scratch
    (_starting_values) at random times and widths. Pulses start clear of each other's repulsion wall
    where the window has room (wall_clearance). Returns (z, init_times, init_sigma): the starting
    points on the unconstrained scale NUTS and the optimizers work on, a dict of arrays with leading
    axis n_starts, and the starting arrival times and widths, each (n_starts, n_pulses).

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
    n_starts:          number of starts (chains or optimizer starts)
    n_warm:            how many of them start from the previous fit
    seed:              integer the random placements are drawn from
    reset:             parameters the warm starts restart from their defaults instead of inheriting (delta always is)
    split_starts:      turn some warm starts into split starts
    """
    time_param = "mean_arrival_time"
    time_prior, sigma_prior = priors[time_param], priors.get("sigma")
    if getattr(sigma_prior, "high", None) is None:
        raise ValueError("warm starts draw a width per pulse: they need a bounded 'sigma' prior.")
    values = _starting_values(burst, priors, n_pulses)                                    # random starts
    warm_values = _previous_fit_values(prev_params, values, priors, n_pulses, reset=reset)  # warm starts

    def unconstrained(vals):
        """The starting values on the unconstrained scale NUTS and the optimizers work on."""
        return initialize_model(jax.random.PRNGKey(seed), likelihood, init_strategy=init_to_value(values=vals),
                                model_kwargs=model_kwargs)[0].z
    z_random, z_warm = unconstrained(values), unconstrained(warm_values)

    sigma_range = (_interior(sigma_prior, float(sigma_prior.low)), _interior(sigma_prior, float(sigma_prior.high)))
    x, tail = wall_clearance(burst, single_pulse_fn, priors, values, n_pulses, likelihood, likelihood_kwargs)
    _, tail_warm = wall_clearance(burst, single_pulse_fn, priors, warm_values, n_pulses, likelihood, likelihood_kwargs)
    kept_times = np.array([_interior(time_prior, float(v)) for v in np.asarray(prev_params[time_param]).reshape(-1)])
    init_times, init_sigma = box_times(
        burst, prev_M, search, n_pulses, n_starts, n_warm, time_prior, sigma_range, seed,
        clearance=(x, np.where(np.arange(n_starts) < n_warm, tail_warm, tail)),
        fixed=(kept_times, np.asarray(warm_values["sigma"])[: n_pulses - 1]))

    # Split starts: the last warm starts, each splitting one old pulse, widest first
    splits = {}
    if split_starts:
        n_split = min(n_pulses - 1, n_warm // 2)
        widest = np.argsort(-np.asarray(warm_values["sigma"])[: n_pulses - 1])[:n_split]
        splits = {n_warm - n_split + i: int(k) for i, k in enumerate(widest)}

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
        z[time_param] = biject_to(time_prior.support).inv(jnp.asarray(init_times[c], dtype=jnp.float32))
        z["sigma"] = biject_to(sigma_prior.support).inv(jnp.asarray(init_sigma[c], dtype=jnp.float32))
        per_start.append(z)
    if splits:
        print(f"split starts: {len(splits)} warm start(s) each split one of the previous fit's pulses in two "
              f"(pulses {sorted(splits.values())})")
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
    warm_start: dict = None,
    n_warm_chains: int = None,
    warm_start_tau: bool = False,
    split_starts: bool = True,
    **likelihood_kwargs,
) -> MCMC:
    """
    Runs NUTS on one burst and returns the finished MCMC object, with the run's settings attached as
    mcmc.seed, mcmc.priors, mcmc.n_pulses, mcmc.likelihood_kwargs (the likelihood's defaults, what was
    passed, and shared_params), mcmc.init_times and mcmc.init_sigma (warm starts only). Prints the
    likelihood, how the amplitude is parameterized, and the seed. Raises unless exactly one of 'A'
    and 'A_snr' is in priors.

    burst:              the ConditionedData of the burst
    single_pulse_fn:    the model closure from model.setup_model
    priors:             the priors from set_priors
    likelihood:         the NumPyro model, likelihood.FD_gaussianlikelihood by default
    n_pulses:           number of pulses
    num_warmup:         warmup steps per chain, in which the step size and mass matrix adapt; keep >= 500 with dense_mass, whose matrix comes from the last warmup window only
    num_samples:        draws kept per chain after warmup
    num_chains:         number of chains; on the CPU up to as many as devices.use_device made run in parallel, on a GPU all run as one batch
    seed:               integer the run's random key is made from; None draws one, which is printed and stored as mcmc.seed
    init_strategy:      a NumPyro initialization strategy; None uses default_init (broad pulses at random times)
    dense_mass:         adapt a full covariance mass matrix (True) rather than a diagonal one
    max_tree_depth:     largest NUTS tree depth d (at most 2^d - 1 leapfrog steps per draw), or a pair (warmup, sampling)
    chain_method:       "auto" matches the device ("parallel" on the CPU, "vectorized" on a GPU), or any NumPyro chain_method
    progress_bar:       show NumPyro's progress bar
    shared_params:      parameters sampled once for the whole burst (as '<name>_shared'); () keeps every parameter per pulse
    extra_fields:       per-draw sampler quantities kept beside the samples, read with mcmc.get_extra_fields()
    target_accept_prob: mean acceptance probability the step size is tuned for; higher gives smaller steps and fewer divergences
    warm_start:         dict(params=..., result=..., M_FD=...): the previous fit's maximum a posteriori draw (n_pulses - 1 pulses), its model, and an excess_power_search of its residual; see warm_start_points for how chains are started from it
    n_warm_chains:      how many chains start warm; None for all of them
    warm_start_tau:     let warm chains inherit the previous fit's tau_ref instead of restarting it at 0.5 dt
    split_starts:       let some warm chains start with one of the previous fit's pulses split in two (warm_start_points)
    likelihood_kwargs:  passed straight to the likelihood, e.g. repulsion={...}
    """
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
    print(f"run_fit: likelihood {getattr(likelihood, '__name__', likelihood)}, parameterizing {amplitude}, "
          f"seed {seed}")
    _announce_delta_init(priors, warm=warm_start is not None)

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

    # WARM START: warm chains from the previous fit plus one new pulse (or a split pulse), the rest from scratch
    init_params, init_times, init_sigma = None, None, None
    if warm_start is not None:
        if "sigma" in shared_params:
            raise ValueError("warm_start draws a width per pulse, so 'sigma' cannot be shared.")
        n_warm = num_chains if n_warm_chains is None else n_warm_chains
        init_params, init_times, init_sigma = warm_start_points(
            burst, single_pulse_fn, priors, likelihood, run_kwargs, likelihood_kwargs, warm_start["params"],
            warm_start["M_FD"], warm_start["result"], n_pulses, num_chains, n_warm, seed,
            reset=() if warm_start_tau else ("tau_ref",), split_starts=split_starts)
        if num_chains == 1:
            init_params = {key: v[0] for key, v in init_params.items()}

    mcmc.run(jax.random.PRNGKey(seed), extra_fields=extra_fields, init_params=init_params, **run_kwargs)

    mcmc.init_times = init_times      # warm-started runs only (None otherwise)
    mcmc.init_sigma = init_sigma
    mcmc.seed = seed                  # the seed actually used; save_run records it
    mcmc.priors = dict(priors)
    mcmc.n_pulses = n_pulses
    # What actually ran: the likelihood's own defaults (e.g. repulsion on), what was passed, and shared_params
    defaults = {name: p.default for name, p in inspect.signature(likelihood).parameters.items()
                if p.default is not inspect.Parameter.empty and name not in filled_in | {"shared_params"}}
    mcmc.likelihood_kwargs = {**defaults, **likelihood_kwargs, "shared_params": shared_params}

    return mcmc


# -------------------------------------------------------------------
# 4. Finding the number of pulses
# -------------------------------------------------------------------

def iterative_pulse_fit(
    burst,
    single_pulse_fn: Callable,
    likelihood: Callable = FD_gaussianlikelihood,
    snr_min: float = 8.0,
    n_max: int = 5,
    num_chains: int = 64,
    n_warm_chains: int = None,
    warm_start_tau: bool = False,
    s_opt: float = 100.0,
    seed: int = None,
    epsilon: float = 0.2,
    p_false_alarm: float = 0.01,
    plot_steps: bool = False,
    verbose: bool = True,
    prior_kwargs: dict = None,
    **run_kwargs,
) -> dict:
    """
    Fits 1, 2, ... pulses with NUTS, adding one at a time while the data call for it. Each step from
    n - 1 to n pulses runs postprocess.excess_power_search on the (n - 1)-pulse fit's residual and
    stops if nothing is significant; otherwise fits n pulses (run_fit, half the chains warm-started
    from the previous fit's maximum a posteriori draw plus a new pulse in a flagged box or one old
    pulse split in two; see warm_start_points), and accepts
    the n-pulse fit only if it explains at least one flagged box (postprocess.boxes_explained) AND
    its ELPD rises by at least snr_min^2 / 2, the log-likelihood gain of a pulse of SNR snr_min.
    snr_min also sets the box significance and the model change that counts as explaining a box, so
    all three tests speak the same language. Each fit's chains are combined by
    postprocess.combine_chains, and its maximum a posteriori draw gives the model whose residual the
    next step searches.

    burst:          the ConditionedData of the burst
    single_pulse_fn: the model closure from model.setup_model
    likelihood:     the NumPyro model, likelihood.FD_gaussianlikelihood by default
    snr_min:        the detection SNR above
    n_max:          largest number of pulses tried
    num_chains:     chains per fit
    n_warm_chains:  chains warm-started from the previous fit; None for half of num_chains
    warm_start_tau: let warm chains inherit the previous fit's tau_ref (run_fit)
    s_opt:          scale of the A_snr prior (set_priors)
    seed:           one seed for every fit; None draws one
    epsilon:        model error tolerated by excess_power_search, as a fraction of the model's amplitude
    p_false_alarm:  false-alarm probability of the whole excess-power search
    plot_steps:     draw plotting.plot_boxes_explained for every step tried
    verbose:        print each step's priors, chain grouping, search and decision
    prior_kwargs:   further arguments to set_priors (e.g. overrides)
    run_kwargs:     further arguments to run_fit and the likelihood (e.g. num_warmup, repulsion={})

    Returns a dict: 'n_pulses' (kept), 'fit' (fits[n_pulses]), 'fits' ({n: dict with 'mcmc', 'priors',
    'post' (combined draws), 'info' (from combine_chains), 'map' (maximum a posteriori draw), 'M' (its
    model) and 'elpd'}), 'steps' (each with 'search', and when an n-pulse fit was run 'check',
    'd_elpd' and 'accept'), 'stop' (why the search stopped), 'snr_min', 'elpd_min', 'min_channels', 'seed'.
    """
    from postprocess import (pointwise_log_lik, combine_chains, _loo_by_observation,
                             excess_power_search, boxes_explained, min_box_channels)
    from plotting import model_from_draw, plot_boxes_explained

    prior_kwargs = dict(prior_kwargs or {})
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31 - 1))
    if n_warm_chains is None:
        n_warm_chains = num_chains // 2
    elpd_min = snr_min ** 2 / 2.0

    # Narrowest band one pulse can occupy under the spectral prior: the fewest channels a box may span
    min_channels = min_box_channels(burst, set_priors(burst, single_pulse_fn, s_opt=s_opt,
                                                      verbose=False, **prior_kwargs))
    if verbose:
        print(f"iterative_pulse_fit: snr_min {snr_min:g} (delta ELPD threshold {elpd_min:g}), "
              f"minimum box width {min_channels} channels, up to {n_max} pulses")

    def fit(n, ep=None, prev=None):
        """Fits n pulses, warm-started from `prev` with the search `ep` of its residual if given."""
        priors = set_priors(burst, single_pulse_fn, s_opt=s_opt, verbose=verbose, **prior_kwargs)
        warm = {} if ep is None else dict(warm_start=dict(params=prev["map"], result=ep, M_FD=prev["M"]),
                                          n_warm_chains=n_warm_chains, warm_start_tau=warm_start_tau)
        mcmc = run_fit(burst, single_pulse_fn, priors, likelihood, n_pulses=n, num_chains=num_chains,
                       seed=seed, **warm, **run_kwargs)
        s = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
        U = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["potential_energy"])
        ll = pointwise_log_lik(burst, single_pulse_fn, s, thin=1)
        post, info = combine_chains(s, ll, U, verbose=verbose)
        ll_post = ll[info["chain"], info["draw"]]
        best = int(np.argmax(-U[info["chain"], info["draw"]]))        # maximum a posteriori draw
        M, _ = model_from_draw(single_pulse_fn, post, best)
        return dict(mcmc=mcmc, priors=priors, post=post, info=info, M=np.asarray(M),
                    map={k: np.asarray(v)[best] for k, v in post.items()},
                    elpd=float(_loo_by_observation(ll_post)[0].sum()))

    fits, steps = {1: fit(1)}, []
    n_final, stop = 1, f"reached n_max = {n_max}"
    for n in range(2, n_max + 1):
        ep = excess_power_search(burst, fits[n - 1]["M"], epsilon=epsilon, min_snr=snr_min,
                                 p_false_alarm=p_false_alarm, min_channels=min_channels, verbose=verbose)
        if not ep["add_pulse"]:
            steps.append(dict(n=n, search=ep))
            stop = "no significant leftover signal"
            break
        fits[n] = fit(n, ep, fits[n - 1])
        check = boxes_explained(burst, fits[n - 1]["M"], fits[n]["M"], ep, min_change_snr=snr_min,
                                verbose=verbose)
        if plot_steps:
            plot_boxes_explained(burst, fits[n - 1]["M"], fits[n]["M"], check)
        d_elpd = fits[n]["elpd"] - fits[n - 1]["elpd"]
        accept = bool(check["any_removed"] and d_elpd >= elpd_min)
        steps.append(dict(n=n, search=ep, check=check, d_elpd=d_elpd, accept=accept))
        if verbose:
            print(f"{n - 1} -> {n} pulses: delta ELPD = {d_elpd:.1f} (need {elpd_min:.1f}), explains "
                  f"leftover signal: {check['any_removed']} -> {'accept' if accept else 'reject'}")
        if not accept:
            reasons = ([] if check["any_removed"] else ["it explains no flagged leftover signal"]) + \
                      ([] if d_elpd >= elpd_min else [f"delta ELPD {d_elpd:.1f} < {elpd_min:g}"])
            stop = f"{n}-pulse fit rejected: " + " and ".join(reasons)
            break
        n_final = n

    if verbose:
        print(f"final: {n_final} pulses ({stop})")
    return dict(n_pulses=n_final, fit=fits[n_final], fits=fits, steps=steps, stop=stop,
                snr_min=snr_min, elpd_min=elpd_min, min_channels=min_channels, seed=seed)
