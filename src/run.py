"""
run.py - Priors, sampling, and post-processing for a fit of one burst.

A fit is three calls, followed by saving with results.save_run:

    priors = set_priors(burst, single_pulse_fn, s_opt=100.0)
    mcmc   = run_fit(burst, single_pulse_fn, priors, FD_gaussianlikelihood)
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
        sigma, tau_ref     Uniform, from width_prior_bounds at `toa`
        c1, c2             Uniform, from spectral_coeff_bounds mapping the (gamma, beta) box
        A_snr              HalfNormal(s_opt), only if s_opt is given

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
    overrides:   name=distribution replaces a default, or adds a parameter (A=..., delta=...).
                 name=None removes a default, e.g. sigma=None, tau_ref=None when the widths are
                 drawn jointly by `containment` instead.

    Raises if: any value is not a NumPyro distribution; both 'A' and 'A_snr' are set; the
    orthogonal (c1, c2) and physical (gamma, beta) spectral parameters are mixed or only one
    of a pair is given; or the spectral parameters do not match the SED of `single_pulse_fn`.
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

def _interior(distribution, value, margin=1e-3):
    """Moves `value` just inside the bounds of a bounded distribution; unbounded ones pass through."""
    low, high = getattr(distribution, "low", None), getattr(distribution, "high", None)
    if low is None or high is None:
        return value
    low, high = float(low), float(high)
    pad = margin * (high - low)
    return min(max(value, low + pad), high - pad)


def default_init(
    burst, 
    priors: Dict[str, dist.Distribution], 
    n_pulses: int = 1,
    time_param: str = "mean_arrival_time",
    ordered_times: bool = True, # Strict time ordering prior on pulses
):
    """
    Starts the sampler from a broad pulse in the middle of the segment, with no residual
    dispersion, a flat spectrum and little scattering, split evenly among the pulses.

    Starting broad matters because the priors span the whole segment: a narrow starting pulse
    that misses the burst has no overlap with it, so the likelihood has no gradient pointing
    toward it. A pulse covering most of the segment overlaps the burst wherever it is, and
    warmup narrows it down.

    One pulse: width sigma_1 = 0.9 * (upper bound of the sigma prior), centred on the middle
    of the segment.

    n pulses: the same region is shared out. Each pulse has width sigma_1 / n, and the centres
    are evenly spaced about the middle of the segment, 2 sigma_1 / n apart:

        t_j = W_seg / 2 + (2 sigma_1 / n) (j - (n - 1) / 2),    j = 0, ..., n - 1

    so together they tile the central +-sigma_1 of the single-pulse start, and neighbouring
    pulses are two of their own widths apart, which keeps their initial overlap modest. Every
    other parameter starts at the same value for all pulses; the arrival times are what
    distinguish them, and they are distinct and ordered, as the ordering prior requires.

    Only parameters present in `priors` are set; the rest use NumPyro's default. Values are
    moved just inside their prior bounds if they would fall outside them, e.g. a width
    sigma_1 / n below the resolvability floor.
    """
    W_seg = burst.n_time * burst.dt
    targets = {"DM": 0.0, "c1": 0.0, "c2": 0.0, "gamma": 0.0, "beta": 0.0, "tau_ref": 5.0 * burst.dt}

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

    # ---- arrival times ----
    if time_param in priors:
        time_prior = priors[time_param]
        offsets = np.arange(n_pulses) - 0.5 * (n_pulses - 1)
        spacing = 2.0 * sigma_init if sigma_init is not None else W_seg / (2.0 * n_pulses)
        times = [_interior(time_prior, 0.5 * W_seg + spacing * k) for k in offsets]

        if n_pulses == 1 or not ordered_times:
            values[time_param] = jnp.asarray(times, dtype=jnp.float32)
        else:
            # With several pulses the sampled site is the unconstrained base vector z of
            # sample_ordered_arrival_times, which sorts z and maps it into the prior's support.
            # Setting z = (that map)^-1 (t) therefore starts the arrival times at exactly t.
            to_support = biject_to(time_prior.support)
            values[f"{time_param}_base"] = jnp.asarray(
                to_support.inv(jnp.asarray(times, dtype=jnp.float32)), dtype=jnp.float32
            )

    return init_to_value(values=values)


def run_fit(
    burst,
    single_pulse_fn: Callable,
    priors: Dict[str, dist.Distribution],
    likelihood: Callable,
    n_pulses: int = 1,
    num_warmup: int = 1000,
    num_samples: int = 200,
    num_chains: int = 1,
    seed: int = 0,
    init_strategy=None,
    dense_mass: bool = True,
    max_tree_depth: int | tuple[int, int] = 10,    # or (warmup, sampling), e.g. (10, 8), # int, or (warmup, sampling) e.g. (10, 8)
    chain_method: str = "parallel",
    progress_bar: bool = True,
    shared_params: tuple = ("DM", "tau_ref"),
    extra_fields=("num_steps", "accept_prob", "diverging", "potential_energy"),
    target_accept_prob: float = 0.8,
    ordered_times: bool = True, # Strict time ordering prior on pulses
    **likelihood_kwargs,
) -> MCMC:
    """
    Runs NUTS on one burst and returns the finished MCMC object.

    likelihood:        the likelihood function itself, one of MARGINALIZED or UNMARGINALIZED
    priors:            as returned by set_priors. For a marginalized likelihood, 'A_snr' is
                       removed before sampling and its scale passed as `s_opt`; the dictionary
                       passed in is not modified, so it can be saved as-is.
    init_strategy:     None uses default_init: a broad pulse in the middle of the segment, or for
                       several pulses that region shared out evenly among them
    shared_params: parameters sampled once and used by every pulse (sampled as '<name>_shared',
                   recorded per pulse under the usual name). () keeps every parameter per-pulse.
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
        init_strategy = default_init(burst, model_priors, n_pulses, ordered_times)

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
    mcmc.likelihood_kwargs = dict(likelihood_kwargs)    # e.g. repulsion, shared_params, ordered_times

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
