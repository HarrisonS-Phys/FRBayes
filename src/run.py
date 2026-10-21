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
from functools import partial

from likelihood import (
    FD_gaussianlikelihood,
    FD_marg_fullnormal_gaussianlikelihood,
    FD_marg_halfnormal_nooverlap_gaussianlikelihood,
    draw_fluences_fullnormal_gaussmarg,
    draw_fluences_halfnormal_nooverlap_gaussmarg,
    wraparound_ratio_posthoc,
    fluence_for_normalized_log_sed_models,
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
    **overrides,
) -> Dict[str, dist.Distribution]:
    """
    Builds the priors dictionary for a fit, with defaults derived from the data segment, and
    checks it for conflicts.

    Defaults:
        mean_arrival_time  Uniform over the segment
        DM                 Uniform(-D_max, D_max), from dispersion_prior_bounds
        sigma, tau_ref     Uniform, from width_prior_bounds at `toa`; the upper bound on sigma
                           is divided by n_pulses
        c1, c2             Uniform, from spectral_coeff_bounds mapping the (gamma, beta) box
        A_snr              HalfNormal(s_opt), only if s_opt is given

    burst:       the ConditionedData of the burst; its segment length, time resolution, band
                 and spectral basis set the default bounds.
    single_pulse_fn:
                 the model closure from model.setup_model. Used for the DM bounds, and to check
                 that the spectral parameters match its SED.
    n_pulses:    number of pulses the fit will use; must be the same n_pulses passed to run_fit.
                 Each pulse gets 1/n_pulses of the single-pulse upper bound on sigma.
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
    sigma_max = width["sigma_max"] / n_pulses
    if sigma_max <= width["sigma_min"]:
        raise ValueError(f"sigma upper bound / n_pulses = {sigma_max:.3g} s is below the lower bound "
                         f"{width['sigma_min']:.3g} s: too many pulses for this segment.")

    priors = {
        "mean_arrival_time": dist.Uniform(0.0, float(burst.times[-1])),   # (s, relative to segment start)
        "DM": dist.Uniform(dm["DM_min"], dm["DM_max"]),                    # residual DM (pc cm^-3)
        "sigma": dist.Uniform(width["sigma_min"], sigma_max),              # intrinsic width (s), per pulse
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
               "tau_ref": 5.0 * burst.dt, "A_snr": 5.0}

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

    mcmc.run(
        jax.random.PRNGKey(seed),
        data_fd=burst.data_fd,
        sigma_noise_fd=burst.sigma_noise_fd,
        n_pulses=n_pulses,
        priors=model_priors,
        single_pulse_fn=single_pulse_fn,
        # Only meaningful when sampling the orthogonal basis; with gamma and beta it must be None
        spectral_coeff_map=burst.spectral_coeff_map if "c1" in model_priors else None,
        extra_fields=extra_fields,
        shared_params=shared_params,
        ordered_times=ordered_times,
        **model_kwargs,
        **likelihood_kwargs,
    )

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
