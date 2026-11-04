"""
run.py - Priors, sampling, and post-processing for a fit of one burst.

First choose the device, before anything imports jax (see devices.py):

    from devices import use_device
    use_device("auto")          # or "gpu" / "cpu"

A fit is then three calls, followed by saving with results.save_run:

    priors = set_priors(burst, single_pulse_fn, n_pulses, s_opt=100.0)
    mcmc   = run_fit(burst, single_pulse_fn, priors, FD_gaussianlikelihood, n_pulses=n_pulses)
    extra  = posthoc_samples(mcmc, burst, single_pulse_fn, priors)
    idata  = save_run(out_path, mcmc, burst, priors, data_path,
                      single_pulse_fn=single_pulse_fn, extra_samples=extra)

To let the data decide how many pulses there are, iterative_pulse_fit fits 1, 2, ... pulses,
adding one at a time while the residual holds leftover signal that a new pulse explains:

    result = iterative_pulse_fit(burst, single_pulse_fn, snr_min=8.0, n_max=5, seed=2)
    result["n_pulses"], result["fit"]["mcmc"]

THE AMPLITUDE PRIOR IS SET ONCE, in set_priors, as 'A_snr' ~ HalfNormal(s_opt), in units of the
template's optimal SNR. The unmarginalized likelihood samples it directly. The marginalized
likelihoods integrate over that same prior analytically, so run_fit removes 'A_snr' from the
priors they receive and passes its scale on as their `s_opt` argument instead. The priors
dictionary therefore always records the amplitude prior that was actually used, and fits with
the different likelihoods are directly comparable.

POST-PROCESSING differs by likelihood. The unmarginalized likelihood samples the fluence and
records the exact wraparound ratio during sampling, so it needs nothing afterwards. The
marginalized likelihoods integrate the fluence out, so posthoc_samples draws it from its exact
conditional and computes the exact wraparound ratio from those draws.
"""

from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model
from functools import partial

from likelihood import (
    FD_gaussianlikelihood,
    FD_marg_fullnormal_gaussianlikelihood,
    FD_marg_halfnormal_nooverlap_gaussianlikelihood,
    draw_fluences_fullnormal_gaussmarg,
    draw_fluences_halfnormal_nooverlap_gaussmarg,
    wraparound_ratio_posthoc,
    fluence_for_normalized_log_sed_models,
    repulsive_toa_prior,
)
from priors import dispersion_prior_bounds, spectral_coeff_bounds, width_prior_bounds
from results import describe_priors
import inspect

# Likelihoods run_fit supports, by how they treat the fluence
MARGINALIZED = (FD_marg_halfnormal_nooverlap_gaussianlikelihood, FD_marg_fullnormal_gaussianlikelihood)
UNMARGINALIZED = (FD_gaussianlikelihood,)

AMPLITUDE_PARAMS = {"A", "A_snr"}           # fluence, or amplitude in SNR units
ORTHOGONAL_SED_PARAMS = {"c1", "c2"}        # coefficients of the orthogonal spectral basis
PHYSICAL_SED_PARAMS = {"gamma", "beta"}     # spectral index and running


# -------------------------------------------------------------------
# 1. Priors
# -------------------------------------------------------------------

def set_priors(
    burst,
    single_pulse_fn: Callable,
    n_pulses: int,
    toa: float = None,
    s_opt: float = None,
    gamma_range=(-10.0, 200.0),
    beta_range=(-200.0, 10.0),
    verbose: bool = True,
    shrink_sigma_npulses: bool = False,
    **overrides,
) -> Dict[str, dist.Distribution]:
    """
    Builds the priors dictionary for a fit, with defaults derived from the data segment, and
    checks it for conflicts.

    Defaults:
        mean_arrival_time  Uniform over the segment
        DM                 Uniform(-D_max, D_max), from dispersion_prior_bounds
        sigma              LogUniform, from width_prior_bounds at `toa`: equal weight per decade
                           of width, since the width's scale is not known in advance. The lower
                           bound is the resolution floor, where unresolved pulses collect. The upper
                           bound is divided by n_pulses if shrink_sigma_npulses.
        tau_ref            Uniform, from width_prior_bounds at `toa`. Flat rather than log-uniform,
                           so that no scattering (tau_ref -> 0) stays inside the prior.
        c1, c2             Uniform, from spectral_coeff_bounds mapping the (gamma, beta) box
        A_snr              HalfNormal(s_opt), only if s_opt is given

    burst:       the ConditionedData of the burst; its segment length, time resolution, band
                 and spectral basis set the default bounds.
    single_pulse_fn:
                 the model closure from model.setup_model. Used for the DM bounds, and to check
                 that the spectral parameters match its SED.
    n_pulses:    number of pulses the fit will use; must be the same n_pulses passed to run_fit.
    shrink_sigma_npulses:
                 give each pulse 1/n_pulses of the single-pulse upper bound on sigma. Default
                 False: every pulse gets the single-pulse bound, so fits with different numbers of
                 pulses share the same per-pulse prior and are nested.
    toa:         arrival time the width bounds are computed for (s, relative to the segment
                 start). Defaults to the middle of the segment, which gives the widest bound on
                 sigma. The bounds assume the pulse sits there: a draw near the edge of the
                 segment with a width near its bound can still wrap around, which the
                 wraparound ratio will show.
    s_opt:       scale of the amplitude prior, in SNR units. Every likelihood needs it, except an
                 unmarginalized fit given its own amplitude prior as an override (A=... or
                 A_snr=...).
    gamma_range, beta_range:
                 the catalogue-wide box in (gamma, beta) that the c1, c2 bounds are mapped from.
                 Ignored if gamma and beta are given as overrides.
    verbose:     print each prior once the dictionary is built.
    overrides:   name=distribution replaces a default, or adds a parameter (A=..., delta=...).
                 name=None removes a default, e.g. sigma=None, tau_ref=None when the widths are
                 drawn jointly by `containment` instead.

    Raises if: any value is not a NumPyro distribution; both 'A' and 'A_snr' are set; the
    orthogonal (c1, c2) and physical (gamma, beta) spectral parameters are mixed or only one
    of a pair is given; the spectral parameters do not match the SED of `single_pulse_fn`; or
    n_pulses is so large that the upper bound on sigma falls below its lower bound.
    """
    W_seg = burst.n_time * burst.dt
    if toa is None:
        toa = 0.5 * W_seg

    width = width_prior_bounds(burst, toa=toa)
    dm = dispersion_prior_bounds(burst, single_pulse_fn)

    # Each of n pulses gets 1/n of the single-pulse upper bound on sigma
    sigma_max = width["sigma_max"] / (n_pulses if shrink_sigma_npulses else 1)
    if sigma_max <= width["sigma_min"]:
        raise ValueError(f"sigma upper bound / n_pulses = {sigma_max:.3g} s is below the lower bound "
                         f"{width['sigma_min']:.3g} s: too many pulses for this segment.")

    priors = {
        "mean_arrival_time": dist.Uniform(0.0, float(burst.times[-1])),   # (s, relative to segment start)
        "DM": dist.Uniform(dm["DM_min"], dm["DM_max"]),                    # residual DM (pc cm^-3)
        "sigma": dist.LogUniform(width["sigma_min"], sigma_max),           # intrinsic width (s), per pulse; scale-free
        "tau_ref": dist.Uniform(width["tau_ref_min"], width["tau_ref_max"]),  # scattering at nu_r (s)
    }

    # Orthogonal-basis spectral coefficients by default, unless gamma or beta are given directly.
    # c1 and c2 are specific to this burst's basis; gamma and beta are recorded as deterministic
    # sites during sampling, and are what should be reported.
    if not (PHYSICAL_SED_PARAMS & overrides.keys()):
        spectral = spectral_coeff_bounds(burst.spectral_coeff_map, list(gamma_range), list(beta_range))
        priors["c1"] = dist.Uniform(spectral["c1_min"], spectral["c1_max"])
        priors["c2"] = dist.Uniform(spectral["c2_min"], spectral["c2_max"])

    if s_opt is not None:
        priors["A_snr"] = dist.HalfNormal(float(s_opt))

    for name, value in overrides.items():
        if value is None:
            priors.pop(name, None)
        else:
            priors[name] = value

    # ---------------------------- conflict checks ----------------------------
    not_distributions = {k: type(v).__name__ for k, v in priors.items()
                         if not isinstance(v, dist.Distribution)}
    if not_distributions:
        raise TypeError(f"priors must be NumPyro distributions; got {not_distributions}.")

    if AMPLITUDE_PARAMS <= priors.keys():
        raise ValueError(
            "both 'A' (a fluence) and 'A_snr' (SNR units) are set; s_opt sets 'A_snr', so pass "
            "either s_opt or an A=... override, not both."
        )

    orthogonal = ORTHOGONAL_SED_PARAMS & priors.keys()
    physical = PHYSICAL_SED_PARAMS & priors.keys()
    if orthogonal and physical:
        raise ValueError(
            f"{sorted(orthogonal)} and {sorted(physical)} both set; the spectrum is either "
            f"(c1, c2) in the orthogonal basis or (gamma, beta), not both."
        )
    if orthogonal and orthogonal != ORTHOGONAL_SED_PARAMS:
        raise ValueError(f"only {sorted(orthogonal)} set; c1 and c2 must be given together.")
    if physical and physical != PHYSICAL_SED_PARAMS:
        raise ValueError(f"only {sorted(physical)} set; gamma and beta must be given together.")

    # model.py stores the SED function's name itself, as a string
    sed_name = getattr(single_pulse_fn, "sed_fn", None)
    if sed_name == "running_power_law_log_sed" and orthogonal:
        raise ValueError("the model uses running_power_law_log_sed, which reads gamma and beta, but "
                         "the priors set c1 and c2.")
    if sed_name == "orthogonal_poly_log_sed" and physical:
        raise ValueError("the model uses the orthogonal SED basis, which reads c1 and c2, but "
                         "the priors set gamma and beta.")

    if verbose:
        for name, description in describe_priors(priors).items():
            print(f"{name.removeprefix('prior.')}: {description}")

    return priors


def _marginalized_s_opt(priors: Dict[str, dist.Distribution]) -> float:
    """
    s_opt for a marginalized likelihood, read from the 'A_snr' prior.

    priors: the priors dictionary of the fit, as returned by set_priors.

    Both marginalized likelihoods correspond to a HalfNormal(s_opt) prior on the amplitude in SNR
    units: the half-normal one integrates over it directly, and the full-normal one integrates
    over the zero-mean normal and becomes that prior once draws with a negative fluence are
    rejected. So that is the only amplitude prior they accept.
    """
    if "A" in priors:
        raise ValueError(
            "'A' (a fluence) cannot be used with a marginalized likelihood, which integrates out "
            "the amplitude in SNR units. Pass s_opt to set_priors instead."
        )
    if "A_snr" not in priors:
        raise ValueError("marginalized likelihoods need the amplitude prior: pass s_opt to set_priors.")
    if not isinstance(priors["A_snr"], dist.HalfNormal):
        raise ValueError(
            f"marginalized likelihoods integrate over a HalfNormal amplitude prior; got "
            f"{type(priors['A_snr']).__name__} for 'A_snr'."
        )
    return float(priors["A_snr"].scale)


# -------------------------------------------------------------------
# 2. Sampling
# -------------------------------------------------------------------

# ---- Starting point of the sampler ----
#
# default_init builds it in two parts:
#   every parameter except the arrival times:  _starting_values, the same in every chain
#   the arrival times:                         random_init_times=True  (DEFAULT) each chain draws
#                                                  its own from the prior (init_to_value_random_times)
#                                              random_init_times=False (ALTERNATIVE) the same
#                                                  evenly spaced times in every chain
#                                                  (_evenly_spaced_times)

def _interior(distribution, value, margin=1e-3):
    """
    Moves `value` just inside the bounds of a bounded distribution; unbounded ones pass through.

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
    Starting values for every parameter EXCEPT the arrival times, the same in every chain.
    Returns (values, sigma_init): `values` maps site names to arrays of shape (n_pulses,), and
    sigma_init is the starting width (None if the sigma prior has no upper bound).

    burst:    the ConditionedData of the burst; its time resolution sets the starting scattering
    priors:   the priors the model samples; only parameters present in it get a starting value
    n_pulses: number of pulses in the fit

    See default_init for why each value is chosen.
    """
    targets = {"DM": 0.0, "c1": 0.0, "c2": 0.0, "gamma": 0.0, "beta": 0.0,
               "tau_ref": 0.5 * burst.dt, "A_snr": 5.0}

    sigma_init = None
    if getattr(priors.get("sigma"), "high", None) is not None:
        sigma_init = 0.9 * float(priors["sigma"].high)
        targets["sigma"] = sigma_init

    values = {
        name: jnp.full((n_pulses,), _interior(priors[name], value), dtype=jnp.float32)
        for name, value in targets.items() if name in priors
    }

    # Shared parameters are sampled once, as '<name>_shared'; start them at the same value
    # (harmless when nothing is shared, since such sites then don't exist)
    for name in list(values):
        values[f"{name}_shared"] = values[name][0]

    return values, sigma_init


def init_to_value_random_times(site=None, values={}, time_param="mean_arrival_time", time_prior=None):
    """
    DEFAULT arrival-time start. init_to_value for every site except the arrival times, which each
    chain draws at random from their prior, using the random key NumPyro gives that chain's
    initialization, so every chain starts from a different arrangement of pulses.

    Unordered times: the arrival-time site itself is drawn from the prior.
    Ordered times: the sampled site is the unconstrained base vector z of
    sample_ordered_arrival_times, whose own prior is a standard normal; drawing z from that would
    bunch the times in the middle of the window. Instead the times are drawn from the arrival-time
    prior and z = (map to support)^-1 (t), which sample_ordered_arrival_times sorts and maps back
    to exactly those times. Both cases are recognized from the site name, so this strategy does
    not need to be told which one the likelihood uses.

    site:       the sample site NumPyro asks a starting value for. None returns the strategy
                itself, in the partial form NumPyro requires.
    values:     starting values for every other site, as for init_to_value
    time_param: name of the arrival-time site
    time_prior: the arrival-time prior the times are drawn from
    """
    if site is None:
        return partial(init_to_value_random_times, values=values, time_param=time_param,
                       time_prior=time_prior)
    if site["type"] == "sample" and site["name"] in (time_param, f"{time_param}_base"):
        key = site["kwargs"].get("rng_key")
        if key is None:
            key = jax.random.PRNGKey(0)
        times = time_prior.sample(key, site["fn"].shape())
        if site["name"] == time_param:
            return times
        return biject_to(time_prior.support).inv(times)
    return init_to_value(site, values=values)


def _evenly_spaced_times(burst, time_prior, n_pulses: int, sigma_init):
    """
    ALTERNATIVE arrival-time start: the same times for every chain, evenly spaced about the middle
    of the segment, 2 sigma_init apart,

        t_j = W_seg / 2 + 2 sigma_init (j - (n_pulses - 1) / 2),

    or W_seg / (2 n_pulses) apart if sigma has no upper bound (sigma_init None). Each time is
    moved just inside the prior bounds.

    burst:      the ConditionedData of the burst; its length sets the middle of the segment
    time_prior: the arrival-time prior
    n_pulses:   number of pulses in the fit
    sigma_init: the starting width, from _starting_values
    """
    W_seg = burst.n_time * burst.dt
    offsets = np.arange(n_pulses) - 0.5 * (n_pulses - 1)
    spacing = 2.0 * sigma_init if sigma_init is not None else W_seg / (2.0 * n_pulses)
    times = [_interior(time_prior, 0.5 * W_seg + spacing * k) for k in offsets]
    return jnp.asarray(times, dtype=jnp.float32)


def default_init(
    burst,
    priors: Dict[str, dist.Distribution],
    n_pulses: int = 1,
    time_param: str = "mean_arrival_time",
    ordered_times: bool = False,  # Strict time ordering prior on pulses
    random_init_times: bool = True,   # each chain draws its own arrival times from the prior
):
    """
    Starting point of the sampler: broad pulses with no residual dispersion, a flat spectrum,
    little scattering and a modest amplitude, at arrival times drawn at random for each chain
    (default) or evenly spaced.

    Every parameter except the arrival times (_starting_values), the same in every chain:
      Starting broad matters because the priors span the whole segment: a narrow starting pulse
      that misses the burst has no overlap with it, so the likelihood has no gradient pointing
      toward it. A broad pulse overlaps nearby features wherever it starts, and warmup narrows it.
      Widths: sigma = 0.9 * (upper bound of the sigma prior) for every pulse. With priors from
      set_priors(..., n_pulses=n) that bound is already the single-pulse bound divided by n.
      A flat spectrum (c1 = c2 = 0, or gamma = beta = 0) lets a pulse respond to emission anywhere
      in the band. A_snr starts at 5 because the likelihood's pull on a pulse's arrival time is
      proportional to its amplitude: a pulse starting near zero feels no pull toward any feature;
      at 5 it does, and the likelihood then decides whether it grows or switches off.

    Arrival times:
      random_init_times=True (DEFAULT): every chain draws its own times from the arrival-time
        prior (init_to_value_random_times), so different chains start from different
        arrangements of pulses and together explore more of the possible solutions. Intended for
        running many chains and keeping the best.
      random_init_times=False (ALTERNATIVE): the same times for every chain, evenly spaced about
        the middle of the segment, 2 sigma apart (_evenly_spaced_times). Chains then differ only
        through the sampler's own randomness.

    burst:             the ConditionedData of the burst
    priors:            the priors the model samples. run_fit passes them with 'A_snr' removed for
                       a marginalized likelihood. Only parameters present get a starting value;
                       the rest use NumPyro's default.
    n_pulses:          number of pulses in the fit
    time_param:        name of the arrival-time site
    ordered_times:     must match the likelihood's ordered_times, since ordered times are sampled
                       through a different site ('<time_param>_base'). Only the evenly spaced start
                       uses it; the random start recognizes the site by its name.
    random_init_times: True for the random start (default), False for the evenly spaced one.

    Values are moved just inside their prior bounds if they would fall outside them.
    """
    values, sigma_init = _starting_values(burst, priors, n_pulses)

    if time_param not in priors:
        return init_to_value(values=values)

    time_prior = priors[time_param]

    # DEFAULT: each chain draws its own arrival times from the prior
    if random_init_times:
        return init_to_value_random_times(values=values, time_param=time_param,
                                          time_prior=time_prior)

    # ALTERNATIVE: the same evenly spaced arrival times in every chain
    times = _evenly_spaced_times(burst, time_prior, n_pulses, sigma_init)
    if n_pulses == 1 or not ordered_times:
        values[time_param] = times
    else:
        # With several pulses the sampled site is the unconstrained base vector z of
        # sample_ordered_arrival_times, which sorts z and maps it into the prior's support.
        # Setting z = (that map)^-1 (t) therefore starts the arrival times at exactly t.
        to_support = biject_to(time_prior.support)
        values[f"{time_param}_base"] = jnp.asarray(to_support.inv(times), dtype=jnp.float32)

    return init_to_value(values=values)


def wall_clearance(burst, values, n_pulses, likelihood, likelihood_kwargs):
    """
    How far apart two starting pulses must be to clear the repulsion wall. Returns (x, tail): pulses
    j and k clear it when |t_j - t_k| >= x (width_j + width_k), width = sqrt(sigma^2 + tail^2). x is 0
    if the fit has no repulsion.

    width is as in repulsive_toa_prior, with tail = tau_scale tau f at the starting tau, f the largest
    band tau factor (nu / nu_ref)^-4 over the channels (1 without band_tau). All pulses start with
    the same tau, so the shift of the centres by tau f is the same for both pulses of a pair and
    cancels. x = x_min + 10 w: at x_min itself the soft wall still costs K ln 2 / softplus(x_min / w)
    (~35 nats with the defaults), 10 w further on ~0.002 nats.
    """
    param = inspect.signature(likelihood).parameters.get("repulsion")
    rep = likelihood_kwargs.get("repulsion", param.default if param is not None else None)
    if rep is None or n_pulses == 1:
        return 0.0, 0.0
    rep = {**{name: p.default for name, p in inspect.signature(repulsive_toa_prior).parameters.items()
              if p.default is not inspect.Parameter.empty}, **rep}
    f = 1.0
    if rep.get("band_tau", True):
        nu = np.asarray(burst.freq_channels, dtype=float).reshape(-1) / float(burst.freq_channels_ref)
        f = float(np.max(nu ** -4.0))
    if "tau_ref_shared" in values:
        tau0 = float(values["tau_ref_shared"])
    else:
        tau0 = float(values["tau_ref"][0]) if "tau_ref" in values else 0.0
    return float(rep["x_min"]) + 10.0 * float(rep["w"]), float(rep["tau_scale"]) * tau0 * f


def _previous_fit_values(params, defaults, priors, n_pulses, reset=("tau_ref",)):
    """
    Starting values for a WARM chain: the previous fit's n_pulses - 1 pulses as they were, and one
    new pulse, last, with the default starting values.

    params:   one draw of the previous fit, name -> value: per-pulse values of shape (n_pulses - 1,),
              shared ones ('<name>_shared') scalars. Sites the model does not sample are ignored.
    defaults: _starting_values for n_pulses pulses; the new pulse takes its values from these (or
              the median of the previous pulses for a parameter without a default).
    priors:   the priors of the new fit; every value is moved just inside their bounds (e.g. a width
              beyond a sigma bound that shrinks with n_pulses).
    reset:    parameters NOT inherited, which start from their defaults instead (shared or not).
              By default tau_ref: a component the previous fit was missing is often covered by an
              inflated scattering tail (a shared tau stretched across it), and inheriting that tail
              leaves the new pulse little to gain where the missing component is.
    """
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
    WARM-START arrival times and widths, each of shape (num_chains, n_pulses), from the residual boxes
    of the previous fit.

    The first n_warm_chains chains (warm) keep the `fixed` pulses -- the previous fit's, as they
    were -- and put each NEW pulse in its own significant box of `result` (an excess_power_search of
    the previous fit's residual; a random choice of boxes if there are more boxes than new pulses),
    at a time drawn inside the box with probability proportional to |r(t)|^2: r is the whitened
    residual on the box's channels and Fourier band, transformed to time and summed over those
    channels. Every pulse of the remaining chains (random) gets a time drawn uniformly from the
    arrival-time prior. Every pulse not fixed gets a width drawn log-uniformly over sigma_range (the
    sigma prior).

    Pulses are placed one at a time, and each is drawn only from times that clear the walls of the
    pulses already placed (wall_clearance), so no draw is ever redone. A box with no such time left
    gives its pulse a random time instead. If no time is left anywhere (widths too broad for the
    segment), the pulse is drawn from the whole prior and the chain is reported as starting in a
    wall.

    burst:         the ConditionedData of the burst
    M_FD:          the previous fit's model on the retained Fourier bins
    result:        excess_power_search(burst, M_FD)
    time_prior:    the arrival-time prior (bounded)
    sigma_range:   (low, high) the widths are drawn log-uniformly between
    seed:          integer the draws are made from
    clearance:     (x, tail) from wall_clearance; tail may also be an array, one per chain
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

    boxes = []
    for tile in result["tiles"]:
        if not tile["significant"]:
            continue
        (c0, c1), n_b = tile["channel_index"], tile["resolution"]
        j0, j1 = tile["band_index"][0] * n_b, tile["band_index"][1] * n_b
        full = np.zeros((c1 - c0, n), dtype=complex)
        full[:, k[j0:j1]] = Z[c0:c1, j0:j1]
        power = np.sum(np.abs(np.fft.ifft(full, axis=-1)) ** 2, axis=0)          # |r(t)|^2
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


def run_fit(
    burst,
    single_pulse_fn: Callable,
    priors: Dict[str, dist.Distribution],
    likelihood: Callable,
    n_pulses: int = 1,
    num_warmup: int = 500,
    num_samples: int = 200,
    num_chains: int = 1,
    seed: int = None,
    init_strategy=None,
    dense_mass: bool = True,
    max_tree_depth: int | tuple[int, int] = (10,8),  # int, or (warmup, sampling) e.g. (10, 8)
    chain_method: str = "auto",
    progress_bar: bool = True,
    shared_params: tuple = ("DM", "tau_ref"),
    extra_fields=("num_steps", "accept_prob", "diverging", "potential_energy"),
    target_accept_prob: float = 0.8,
    ordered_times: bool = False, # Strict time ordering prior on pulses
    random_init_times: bool = True, # Randomize intialized toas for sampler
    warm_start: dict = None,        # dict(result=excess_power_search result, M_FD=previous model)
    n_warm_chains: int = None,      # chains warm-started from the boxes; None = all
    warm_start_tau: bool = False,   # warm chains inherit the previous fit's tau_ref (default: start afresh)
    **likelihood_kwargs,
) -> MCMC:
    """
    Runs NUTS on one burst and returns the finished MCMC object.

    burst:             the ConditionedData of the burst: the data, noise and spectral basis
    single_pulse_fn:   the model closure from model.setup_model
    priors:            as returned by set_priors. For a marginalized likelihood, 'A_snr' is
                       removed before sampling and its scale passed as `s_opt`; the dictionary
                       passed in is not modified, so it can be saved as-is.
    likelihood:        the likelihood function itself, one of MARGINALIZED or UNMARGINALIZED
    n_pulses:          number of pulses; must be the same n_pulses passed to set_priors
    num_warmup:        warmup steps per chain, in which the step size and mass matrix are adapted
                       and then discarded. With dense_mass, keep at least 500: the final mass
                       matrix is estimated from the last warmup window only, which is 200 draws
                       at 500 and 100 at 300.
    num_samples:       draws kept per chain after warmup
    num_chains:        number of chains. On the CPU, up to as many as devices.use_device created
                       run in parallel; on a GPU all run as one batch.
    seed:              integer the run's random key is made from; None draws one at random.
                       Either way the seed used is printed and stored as mcmc.seed, which
                       save_run records, so any run can be repeated exactly.
    init_strategy:     a NumPyro initialization strategy. None uses default_init, with the
                       ordered_times and random_init_times given here: by default, broad pulses
                       at arrival times drawn at random for each chain.
    dense_mass:        True adapts a full covariance mass matrix, which follows correlations
                       between parameters (e.g. arrival time, width and scattering); False a
                       diagonal one.
    max_tree_depth:    largest NUTS tree depth d, i.e. at most 2^d - 1 leapfrog steps per draw.
                       An int applies to warmup and sampling; a pair (warmup, sampling), e.g.
                       (10, 8), sets them separately.
    chain_method:      "auto" matches the device chosen with devices.use_device: on a GPU all
                       chains run as one batch ("vectorized"); on the CPU one chain per device
                       ("parallel"), and NumPyro runs chains one after another, with a warning,
                       if num_chains exceeds the devices. Any NumPyro chain_method ("parallel",
                       "vectorized", "sequential") can also be given directly.
    progress_bar:      show NumPyro's progress bar
    shared_params:     parameters sampled once and used by every pulse (sampled as
                       '<name>_shared', recorded per pulse under the usual name). () keeps every
                       parameter per-pulse.
    extra_fields:      per-draw sampler quantities stored beside the samples, read with
                       mcmc.get_extra_fields(): leapfrog steps (tree depth), acceptance
                       probability, divergences, and potential energy (for comparing chains).
    target_accept_prob: mean acceptance probability the step size is tuned for. Higher gives
                       smaller steps and fewer divergences, at the cost of longer trajectories.
    ordered_times:     True samples the arrival times in increasing order
                       (sample_ordered_arrival_times); False samples them independently, and
                       pulses are relabeled after sampling.
    random_init_times: passed to default_init: True (default) draws each chain's starting arrival
                       times at random, False starts every chain from the same evenly spaced
                       times. Ignored if init_strategy is given.
    warm_start:        dict(params=..., result=..., M_FD=...): the previous fit's maximum a
                       posteriori draw (name -> value, one value per pulse or one for a shared
                       parameter; it has n_pulses - 1 pulses), that draw's model, and an
                       excess_power_search of its residual. The first n_warm_chains chains (warm)
                       start from that draw, every parameter as it was except tau_ref, which starts
                       afresh at 0.5 dt unless warm_start_tau, plus one new pulse in a significant
                       box chosen at random, at a time drawn from the residual power there, with a
                       log-uniform width and the default starting values otherwise.
                       The other chains (random) start every pulse from scratch: times uniform,
                       widths log-uniform over the sigma prior, everything else as in default_init,
                       so a bad previous fit cannot trap every chain. No pulse starts inside another's
                       wall where the segment has room (box_times). The starting times and widths
                       are stored as mcmc.init_times and mcmc.init_sigma.
    n_warm_chains:     how many chains start warm; None (default) all of them.
    warm_start_tau:    True: warm chains also inherit the previous fit's tau_ref. False (default):
                       tau_ref starts afresh at 0.5 dt, because a component the previous fit was
                       missing is often covered by an inflated scattering tail (a shared tau
                       stretched across it), and inheriting that tail leaves the new pulse little to
                       gain where the missing component is.
    likelihood_kwargs: passed straight to the likelihood, e.g. containment=, containment_FD=,
                       apply_overlap_prior=. Arguments run_fit fills in itself (the data, priors,
                       model, spectral_coeff_map and s_opt) cannot be given here.

    Raises if the amplitude prior does not match the likelihood: an unmarginalized fit needs
    exactly one of 'A' and 'A_snr', and a marginalized fit needs 'A_snr' ~ HalfNormal and no 'A'.
    """
    unknown = set(shared_params) - priors.keys()
    if unknown:
        raise ValueError(f"shared_params {sorted(unknown)} are not in priors.")

    filled_in = {"data_fd", "sigma_noise_fd", "n_pulses", "priors", "single_pulse_fn",
                 "spectral_coeff_map", "s_opt"}
    clash = filled_in & likelihood_kwargs.keys()
    if clash:
        raise ValueError(f"{sorted(clash)} are filled in by run_fit and cannot be passed as likelihood_kwargs.")

    model_priors = dict(priors)
    model_kwargs = {}

    if likelihood in MARGINALIZED:
        model_kwargs["s_opt"] = _marginalized_s_opt(priors)
        del model_priors["A_snr"]
    elif likelihood in UNMARGINALIZED:
        if len(AMPLITUDE_PARAMS & priors.keys()) != 1:
            raise ValueError(
                "an unmarginalized likelihood samples the amplitude, so exactly one of 'A' and "
                "'A_snr' must be in priors: pass s_opt to set_priors, or an A=... override."
            )
    else:
        supported = [f.__name__ for f in MARGINALIZED + UNMARGINALIZED]
        raise ValueError(f"run_fit supports {supported}; got {getattr(likelihood, '__name__', likelihood)}.")

    if seed is None:
        # JAX has no global random state: every run needs a key made from some integer. Draw one
        # at random, and keep it so the run can be reproduced exactly.
        seed = int(np.random.default_rng().integers(2**31 - 1))
    print(f"run_fit seed: {seed}")

    if init_strategy is None:
        init_strategy = default_init(burst, model_priors, n_pulses, ordered_times=ordered_times,
                                     random_init_times=random_init_times)

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
        priors=model_priors,
        single_pulse_fn=single_pulse_fn,
        # Only meaningful when sampling the orthogonal basis; with gamma and beta it must be None
        spectral_coeff_map=burst.spectral_coeff_map if "c1" in model_priors else None,
        shared_params=shared_params,
        ordered_times=ordered_times,
        **model_kwargs,
        **likelihood_kwargs,
    )

    # WARM START: warm chains from the previous fit plus one new pulse, random chains from scratch
    init_params, init_times, init_sigma = None, None, None
    if warm_start is not None:
        time_param = "mean_arrival_time"
        time_prior, sigma_prior = model_priors[time_param], model_priors.get("sigma")
        if getattr(sigma_prior, "high", None) is None or "sigma" in shared_params:
            raise ValueError("warm_start draws a width per pulse: it needs a bounded 'sigma' prior, not shared.")
        n_warm = num_chains if n_warm_chains is None else n_warm_chains
        values, _ = _starting_values(burst, model_priors, n_pulses)                          # random chains
        warm_values = _previous_fit_values(warm_start["params"], values, model_priors, n_pulses,
                                           reset=() if warm_start_tau else ("tau_ref",))     # warm chains

        def unconstrained(vals):
            return initialize_model(jax.random.PRNGKey(seed), likelihood, init_strategy=init_to_value(values=vals),
                                    model_kwargs=run_kwargs)[0].z         # unconstrained, as NUTS samples
        z_random, z_warm = unconstrained(values), unconstrained(warm_values)

        sigma_range = (_interior(sigma_prior, float(sigma_prior.low)), _interior(sigma_prior, float(sigma_prior.high)))
        x, tail = wall_clearance(burst, values, n_pulses, likelihood, likelihood_kwargs)
        _, tail_warm = wall_clearance(burst, warm_values, n_pulses, likelihood, likelihood_kwargs)
        kept_times = np.array([_interior(time_prior, float(v))
                               for v in np.asarray(warm_start["params"][time_param]).reshape(-1)])
        init_times, init_sigma = box_times(
            burst, warm_start["M_FD"], warm_start["result"], n_pulses, num_chains, n_warm, time_prior,
            sigma_range, seed, clearance=(x, np.where(np.arange(num_chains) < n_warm, tail_warm, tail)),
            fixed=(kept_times, np.asarray(warm_values["sigma"])[: n_pulses - 1]))

        # Each chain's own start. With ordered times the pulses are put in time order, every
        # per-pulse parameter moving with its pulse; otherwise the order is left as it is.
        name = f"{time_param}_base" if ordered_times and n_pulses > 1 else time_param
        per_chain = []
        for c in range(num_chains):
            z = dict(z_warm if c < n_warm else z_random)
            order = np.argsort(init_times[c]) if name != time_param else np.arange(n_pulses)
            init_times[c], init_sigma[c] = init_times[c][order], init_sigma[c][order]
            z = {key: (v[order] if jnp.shape(v) == (n_pulses,) else v) for key, v in z.items()}
            z[name] = biject_to(time_prior.support).inv(jnp.asarray(init_times[c], dtype=jnp.float32))
            z["sigma"] = biject_to(sigma_prior.support).inv(jnp.asarray(init_sigma[c], dtype=jnp.float32))
            per_chain.append(z)
        init_params = {key: jnp.stack([z[key] for z in per_chain]) for key in per_chain[0]}
        if num_chains == 1:
            init_params = {key: v[0] for key, v in init_params.items()}

    mcmc.run(jax.random.PRNGKey(seed), extra_fields=extra_fields, init_params=init_params, **run_kwargs)

    mcmc.init_times = init_times      # warm-started runs only (None otherwise)
    mcmc.init_sigma = init_sigma
    mcmc.seed = seed                  # the seed actually used; save_run records it
    mcmc.priors = dict(priors)                          # as passed in (a marginalized fit keeps A_snr here)
    mcmc.n_pulses = n_pulses
    # Record what actually ran: the likelihood's own defaults (e.g. repulsion is on unless turned
    # off), what was passed, and run_fit's settings the likelihood receives
    passed_by_run_fit = filled_in | {"shared_params", "ordered_times"}
    defaults = {name: p.default for name, p in inspect.signature(likelihood).parameters.items()
                if p.default is not inspect.Parameter.empty and name not in passed_by_run_fit}
    mcmc.likelihood_kwargs = {**defaults, **likelihood_kwargs,
                              "shared_params": shared_params, "ordered_times": ordered_times}

    return mcmc


# ---- Finding the number of pulses ----

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
    Fits 1, 2, ... pulses, adding one at a time while the data call for it, and returns every fit
    and the number of pulses kept.

    Each step from n - 1 to n pulses:
      1. excess_power_search of the (n - 1)-pulse fit's residual. If nothing is significant, stop.
      2. Fit n pulses (run_fit): n_warm_chains chains start from the (n - 1)-pulse fit's maximum a
         posteriori draw plus one new pulse in a flagged residual box, the rest from scratch.
      3. Accept the n-pulse fit only if it explains at least one flagged box (boxes_explained) AND
         its ELPD exceeds the previous fit's by at least snr_min^2 / 2. Otherwise stop.
    Each fit pools chains that sampled the same mode and weights distinct modes by their peak
    posterior (combine_chains); its maximum a posteriori draw gives the model whose residual the
    next step searches.

    ONE DETECTION THRESHOLD. snr_min sets all three thresholds, so they speak the same language: a
    flagged box must hold leftover signal of at least this SNR (beyond noise and the tolerated
    model error), a new pulse must change the model by at least this SNR inside a box for it to
    count as explained, and the ELPD must rise by at least snr_min^2 / 2, the log-likelihood gain of
    fitting a pulse of SNR snr_min. The default 8 is the usual single-detector threshold of
    gravitational-wave searches.

    burst, single_pulse_fn: as for set_priors and run_fit
    likelihood:     the unmarginalized likelihood (FD_gaussianlikelihood); the residual model is
                    built from the sampled fluences, which the marginalized likelihoods do not sample
    snr_min:        the detection SNR above
    n_max:          largest number of pulses tried
    num_chains:     chains per fit
    n_warm_chains:  chains warm-started from the previous fit (None: half of num_chains)
    warm_start_tau: passed to run_fit: True lets warm chains inherit the previous fit's tau_ref
    s_opt:          scale of the A_snr prior (set_priors)
    seed:           one seed for every fit; None draws one, printed by run_fit
    epsilon, p_false_alarm: passed to excess_power_search (tolerated model error, false-alarm
                    probability of the whole search)
    plot_steps:     draw plotting.plot_boxes_explained for every step tried
    verbose:        print each step's priors, chain grouping, search and decision
    prior_kwargs:   further arguments to set_priors (e.g. shrink_sigma_npulses, overrides)
    run_kwargs:     further arguments to run_fit and the likelihood (e.g. num_warmup, repulsion={})

    Returns a dict:
      'n_pulses'   the number of pulses kept
      'fit'        the kept fit, as fits[n_pulses]
      'fits'       {n: fit} for every number of pulses fitted, each a dict with 'mcmc', 'priors',
                   'post' (pooled posterior draws), 'info' (from combine_chains), 'map' (maximum a
                   posteriori draw), 'M' (its model) and 'elpd'
      'steps'      one dict per step tried: n, 'search' (excess_power_search result), and, when an
                   n-pulse fit was run, 'check' (boxes_explained result), 'd_elpd' and 'accept'
      'stop'       why the search stopped
      'snr_min', 'elpd_min', 'min_channels', 'seed'
    """
    from postprocess import (pointwise_log_lik, combine_chains, _loo_by_observation,
                             excess_power_search, boxes_explained, min_box_channels)
    from plotting import model_from_draw, plot_boxes_explained

    if likelihood not in UNMARGINALIZED:
        raise ValueError("iterative_pulse_fit needs an unmarginalized likelihood, whose samples carry the fluence.")
    prior_kwargs = dict(prior_kwargs or {})
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31 - 1))
    if n_warm_chains is None:
        n_warm_chains = num_chains // 2
    elpd_min = snr_min ** 2 / 2.0

    # narrowest band one pulse can occupy under the spectral prior: the fewest channels a box may span
    min_channels = min_box_channels(burst, set_priors(burst, single_pulse_fn, 1, s_opt=s_opt,
                                                      verbose=False, **prior_kwargs))
    if verbose:
        print(f"iterative_pulse_fit: snr_min {snr_min:g} (delta ELPD threshold {elpd_min:g}), "
              f"minimum box width {min_channels} channels, up to {n_max} pulses")

    def fit(n, ep=None, prev=None):
        priors = set_priors(burst, single_pulse_fn, n, s_opt=s_opt, verbose=verbose, **prior_kwargs)
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


# -------------------------------------------------------------------
# 3. Post-processing
# -------------------------------------------------------------------

def posthoc_samples(
    mcmc: MCMC,
    burst,
    single_pulse_fn: Callable,
    priors: Dict[str, dist.Distribution],
    seed: int = None,
) -> Dict[str, np.ndarray]:
    """
    Posterior quantities that cannot be recorded during sampling, shaped (n_chains, n_draws, ...)
    for results.save_run(..., extra_samples=...).

    Unmarginalized likelihood: returns {}. The fluence is sampled and the wraparound ratio is
    recorded exactly during sampling.

    Marginalized likelihoods: returns
        'A'                 fluences drawn from their exact conditional, one per posterior draw
        'wraparound_ratio'  the exact wraparound ratio of each draw, from those fluences
    and, for the full-normal likelihood only,
        'A_accept'          1 where every fluence of the draw is positive, 0 otherwise. Draws with
                            A_accept == 0 must be discarded, from EVERY posterior variable: the
                            retained draws are exact samples under the positive fluence prior.
                            They are flagged rather than removed so the (chain, draw) shape holds.

    mcmc:            the finished MCMC object from run_fit
    burst:           the ConditionedData the fit used; its noise levels weight the wraparound ratio
    single_pulse_fn: the model closure the fit used
    priors:          the priors passed to run_fit, still containing 'A_snr', whose scale is s_opt
    seed:            integer the random key of the fluence draws is made from. Must be given for a
                     marginalized likelihood: jax.random.PRNGKey(None) raises.

    The likelihood is read from mcmc itself, and s_opt from the 'A_snr' prior, so neither can
    differ from what the fit used.
    """
    likelihood = mcmc.sampler.model
    if likelihood in UNMARGINALIZED:
        return {}
    if likelihood not in MARGINALIZED:
        raise ValueError(f"unrecognized likelihood {getattr(likelihood, '__name__', likelihood)}.")

    s_opt = _marginalized_s_opt(priors)
    key = jax.random.PRNGKey(seed)

    # The fluence samplers and the wraparound ratio work on a flat draw axis
    by_chain = mcmc.get_samples(group_by_chain=True)
    n_chains, n_draws = by_chain["snr_opt"].shape[:2]
    flat = {k: v.reshape((n_chains * n_draws,) + v.shape[2:]) for k, v in by_chain.items()}
    n_pulses = flat["snr_opt"].shape[-1]

    accept = None
    if likelihood is FD_marg_halfnormal_nooverlap_gaussianlikelihood:
        peak_channel_fluences = draw_fluences_halfnormal_nooverlap_gaussmarg(flat["snr_opt"], flat["snr_mf"], s_opt, key)
    else:
        peak_channel_fluences, accept = draw_fluences_fullnormal_gaussmarg(flat["gram"], flat["rho_mf"], s_opt, key)

    # Every per-pulse site is passed to the model, which reads only the parameters it needs.
    # Amplitude sites are excluded, since the model would apply them on top of `fluences`.
    per_pulse = tuple(
        k for k, v in flat.items()
        if v.shape == (n_chains * n_draws, n_pulses) and k not in AMPLITUDE_PARAMS
    )
    ratio = wraparound_ratio_posthoc(
        flat, peak_channel_fluences, single_pulse_fn, 1.0 / burst.sigma_noise_fd ** 2, param_names=per_pulse
    )

    fluences = fluence_for_normalized_log_sed_models(peak_channel_fluences, flat["log_sed_peak"])
    extra = {"A": fluences, "wraparound_ratio": ratio}
    if accept is not None:
        extra["A_accept"] = accept.astype(jnp.int8)

    return {k: np.asarray(v).reshape((n_chains, n_draws) + v.shape[1:]) for k, v in extra.items()}
