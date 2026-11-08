"""
maxL_itfit.py - TEST of iterative pulse fitting with maximum-likelihood fits at every step.

The same loop as run.iterative_pulse_fit -- residual search, warm starts from the previous fit
plus one new pulse in a flagged box, box-explained check -- but each step is a batch of
optimizers instead of NUTS:

    maximized:  log-likelihood + repulsion prior, nothing else. The priors only bound the
                parameters, through the same transforms NUTS samples in; their densities and the
                transform Jacobians are left out, so the maximum does not depend on the
                parameterization.
    starts:     the same as run_fit's chains (run.warm_start_points): n_warm warm starts (the
                previous fit's best point, tau_ref and delta reset, plus a new pulse in a flagged box,
                or with split_starts one old pulse split in two) and the rest random. With one
                pulse, every start is random (default_init).
    optimizer:  Adam on the unconstrained parameters (n_adam steps, step size decaying
                geometrically from lr to lr * lr_decay).
    tau profile (tau_profile = n_tau > 0): every start is run once for each value of the shared
                scattering time on a grid: one effectively-zero value, tau_zero * dt (default
                1e-3 dt, where the scattering factor is indistinguishable from 1; None to omit),
                then at least n_tau values log-spaced from tau_floor * dt (default 0.1 dt) to the
                tau_ref prior's upper bound, more if needed so that neighbouring values are no
                further apart than the polish can move tau_ref. tau is measurable well below dt at high SNR (through the
                phase of the scattering transfer function and its nu^-4 frequency dependence), so
                the log grid starts below one bin; 0 itself cannot be used, since it is -infinity
                both on a log grid and in the transformed coordinates of the Uniform(0, tau_max)
                prior. Each run first optimizes everything else with tau_ref held at its grid
                value (n_adam Adam steps), then releases tau_ref and refines jointly (n_polish
                Adam steps starting at step size lr_polish). The best run over all values and
                starts is kept. Holding tau fixed first stops the optimizer from inflating the
                scattering tail before the pulses have settled, the trap seen on 22310619_BB.
                The best pinned ln L at each grid value is the profile likelihood in tau,
                returned as 'profile'.
    accept:     the n-pulse fit explains a flagged box (boxes_explained) AND
                    ln L_n - ln L_{n-1} >= snr_min^2 / 2 + k,
                k the number of parameters a new pulse adds (sampled per pulse; 5 for
                mean_arrival_time, sigma, A_snr, gamma, beta with DM and tau_ref shared).

Nothing in run.py is changed. Optionally (final_nuts=True) the kept fit is followed by one NUTS
run (run.run_fit) started from its best point.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree
from numpyro import handlers
from numpyro.distributions.transforms import biject_to
from numpyro.infer import init_to_value
from numpyro.infer.util import constrain_fn, initialize_model

from likelihood import FD_gaussianlikelihood
from run import set_priors, run_fit, default_init, warm_start_points, _interior, _announce_delta_init

# the factor sites that are maximized: the fit, and the repulsion prior between active pulses
OBJECTIVE_SITES = ("log_likelihood", "repulsive_toa_prior")
TAU_SITE = "tau_ref_shared"          # the shared scattering time, the site profiled over


def _model_kwargs(burst, single_pulse_fn, priors, n_pulses, shared_params, likelihood_kwargs):
    """The likelihood's arguments, as run_fit builds them for an unmarginalized fit."""
    return dict(data_fd=burst.data_fd, sigma_noise_fd=burst.sigma_noise_fd, n_pulses=n_pulses,
                priors=priors, single_pulse_fn=single_pulse_fn, shared_params=shared_params,
                **likelihood_kwargs)


def _starts(burst, single_pulse_fn, priors, likelihood, n_pulses, num_starts, n_warm, model_kwargs,
            likelihood_kwargs, seed, prev=None, search=None, split_starts=True):
    """
    Unconstrained starting points, a dict of arrays with leading axis num_starts: the same starts
    run_fit would give its chains (run.warm_start_points). Without `prev`, every start is random
    (default_init with random arrival times).
    """
    if prev is None:
        keys = jax.random.split(jax.random.PRNGKey(seed), num_starts)
        info = initialize_model(keys, likelihood, init_strategy=default_init(burst, priors, n_pulses),
                                model_kwargs=model_kwargs)[0]
        return info.z, None, None
    return warm_start_points(burst, single_pulse_fn, priors, likelihood, model_kwargs, likelihood_kwargs,
                             prev["map"], prev["M"], search, n_pulses, num_starts, n_warm, seed,
                             split_starts=split_starts)


def _adam(f, n_steps, lr, lr_decay):
    """
    Optax's Adam on f from x for n_steps steps, the step size decaying geometrically from lr to
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
             n_warm=None, prev=None, search=None, split_starts=True, shared_params=("DM", "tau_ref"),
             n_adam=2000, lr=0.05, lr_decay=0.01, tau_profile=0, n_polish=500, lr_polish=0.01,
             tau_floor=0.1, tau_zero=1e-3, tau_parallel=False, seed=0, verbose=True, **likelihood_kwargs):
    """
    Maximum-likelihood fit of n_pulses pulses from num_starts starting points (see the module
    docstring). prev: the previous fit (with 'map' and 'M') and search: the excess_power_search
    of its residual, for warm starts; both None for random starts only.

    split_starts: let some warm starts split one old pulse in two (run.warm_start_points)
    tau_profile:  0 (default) optimizes everything at once from each start. n_tau > 0 runs every
                  start at each pinned value of tau_ref first (module docstring), with n_polish
                  Adam steps at lr_polish after releasing it; n_tau is the fewest log-spaced values,
                  raised for a burst whose range would otherwise leave gaps the polish cannot cross.
    tau_floor:    lower end of the log-spaced tau grid, in units of dt (default 0.1)
    tau_zero:     the effectively-zero tau value added below the log grid, in units of dt
                  (default 1e-3); None leaves it out
    tau_parallel: run the tau values all at once (memory x n_tau) instead of one after another

    Returns a dict: 'logL' (best log-likelihood + repulsion), 'logL_starts' (each start's best over
    the tau values), 'ends' (each start's best end point, sampled sites, constrained), 'map'
    (every site at the best point, sampled and deterministic), 'post' (the same with a leading
    draw axis of 1, for plotting.model_from_draw), 'M' (the best model), 'z' (best unconstrained
    point), 'priors', 'init_times', 'init_sigma', 'seconds', and with a tau profile 'tau_grid',
    'profile' (best ln L with tau_ref held at each grid value), 'logL_runs' (n_tau, num_starts)
    and 'tau_end_runs' (tau_ref at the end of each run, after release).
    """
    from plotting import model_from_draw

    t_start = time.time()
    n_warm = num_starts // 2 if n_warm is None else n_warm
    model_kwargs = _model_kwargs(burst, single_pulse_fn, priors, n_pulses, shared_params, likelihood_kwargs)
    if verbose:
        amplitude = "fluence in units of SNR as A_snr" if "A_snr" in priors else "fluence as A"
        print(f"maxL_fit: likelihood {getattr(likelihood, '__name__', likelihood)}, parameterizing {amplitude}")
        _announce_delta_init(priors, warm=prev is not None)
    z0, init_times, init_sigma = _starts(burst, single_pulse_fn, priors, likelihood, n_pulses, num_starts, n_warm,
                                         model_kwargs, likelihood_kwargs, seed, prev, search, split_starts)

    def trace_at(z):
        params = constrain_fn(likelihood, (), model_kwargs, z)
        return handlers.trace(handlers.substitute(handlers.seed(likelihood, 0), data=params)).get_trace(**model_kwargs)

    def neg_log_l(z):
        tr = trace_at(z)
        return -sum(jnp.sum(tr[k]["fn"].log_prob(tr[k]["value"])) for k in OBJECTIVE_SITES if k in tr)

    z_one = jax.tree_util.tree_map(lambda v: v[0], z0)
    _, unravel = ravel_pytree(z_one)
    f = lambda x: neg_log_l(unravel(x))
    x0 = jax.vmap(lambda z: ravel_pytree(z)[0])(z0)                                  # (num_starts, d)

    if tau_profile:
        if TAU_SITE not in z_one:
            raise ValueError("tau_profile needs tau_ref shared between pulses (shared_params containing 'tau_ref').")
        tau_prior = priors["tau_ref"]
        dt = float(burst.dt)
        # a margin of 1e-6 of the prior's range keeps the grid inside the bounds (finite transformed
        # coordinates) without clipping small values; the default 1e-3 would put the floor near 1 us
        tau_lo = _interior(tau_prior, float(tau_prior.low), margin=1e-6)
        tau_hi = _interior(tau_prior, float(tau_prior.high))
        # Neighbouring grid values no further apart than the polish can move tau_ref. Adam moves a
        # parameter by at most about its step size per step, so the polish travels at most the sum of
        # its step sizes on the unconstrained scale, where tau_ref well below its cap is ~ln tau_ref
        tau_lo_log = max(tau_floor * dt, tau_lo)
        reach = lr_polish * n_polish * ((1.0 - lr_decay) / np.log(1.0 / lr_decay) if lr_decay != 1.0 else 1.0)
        n_log = int(tau_profile)
        if reach > 0:
            n_log = max(n_log, int(np.ceil(np.log(tau_hi / tau_lo_log) / reach)) + 1)
        tau_grid = np.geomspace(tau_lo_log, tau_hi, n_log)
        if tau_zero is not None:
            tau_grid = np.concatenate([[max(tau_zero * dt, tau_lo)], tau_grid])
        tau_u = biject_to(tau_prior.support).inv(jnp.asarray(tau_grid, dtype=jnp.float32))

        def run_at_tau(tu):
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
    else:
        adam = _adam(f, n_adam, lr, lr_decay)
        x_end, f_end = jax.jit(jax.vmap(adam))(x0)
        f_end = np.where(np.isfinite(np.asarray(f_end)), np.asarray(f_end), np.inf)

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

    out = dict(logL=logL, logL_starts=-f_end, ends=ends, map=values, post=post, M=np.asarray(M), z=z_best,
               priors=priors, init_times=init_times, init_sigma=init_sigma, seconds=seconds)
    if tau_profile:
        tau_end = jax.vmap(jax.vmap(lambda x: constrain_fn(likelihood, (), model_kwargs, unravel(x))[TAU_SITE]))(x_runs)
        out.update(tau_grid=tau_grid, profile=profile, logL_runs=-f_runs, tau_end_runs=np.asarray(tau_end))

    if verbose:
        near = int(np.sum(-f_end >= logL - 1.0))
        kind = "warm" if prev is not None and best < n_warm else "random"
        print(f"maxL_fit: {n_pulses} pulse(s), {num_starts} starts ({n_warm if prev is not None else 0} warm)"
              + (f" x {len(tau_grid)} tau values" if tau_profile else "")
              + f": best ln L + repulsion = {logL:.1f} (start {best}, {kind}); {near} start(s) within 1 nat; {seconds:.0f} s")
        if tau_profile:
            print("  tau profile (tau_ref held fixed):   " + "  ".join(
                f"{t * 1e6:.3g}us:{p:.1f}" for t, p in zip(tau_grid, profile)))
            print(f"  best run: tau_ref {float(values[TAU_SITE]) * 1e6:.1f} us after release")
    return out


def log_l_at(burst, single_pulse_fn, priors, n_pulses, params, likelihood=FD_gaussianlikelihood,
             shared_params=("DM", "tau_ref"), **likelihood_kwargs):
    """
    The quantity maxL_fit maximizes, log-likelihood + repulsion prior, at one point.

    params: the sampled sites at that point, constrained (e.g. a NUTS fit's 'map' draw, or
            maxL_fit's): 'mean_arrival_time', 'sigma', 'A_snr', 'gamma', 'beta' per pulse and
            'DM_shared', 'tau_ref_shared'. Other keys are ignored.
    """
    model_kwargs = _model_kwargs(burst, single_pulse_fn, priors, n_pulses, shared_params, likelihood_kwargs)
    proto = handlers.trace(handlers.seed(likelihood, 0)).get_trace(**model_kwargs)
    sampled = {k: jnp.asarray(params[k]) for k, s in proto.items()
               if s["type"] == "sample" and not s.get("is_observed", False)}
    tr = handlers.trace(handlers.substitute(handlers.seed(likelihood, 0), data=sampled)).get_trace(**model_kwargs)
    return float(sum(jnp.sum(tr[k]["fn"].log_prob(tr[k]["value"])) for k in OBJECTIVE_SITES if k in tr))


def maxL_iterative_pulse_fit(burst, single_pulse_fn, likelihood=FD_gaussianlikelihood, snr_min=8.0, n_max=5,
                             num_starts=64, n_warm=None, split_starts=True, s_opt=100.0, seed=None,
                             epsilon=0.2, p_false_alarm=0.01, plot_steps=False, verbose=True, prior_kwargs=None,
                             shared_params=("DM", "tau_ref"), final_nuts=False, nuts_kwargs=None,
                             n_adam=2000, lr=0.05, lr_decay=0.01, tau_profile=0, n_polish=500,
                             lr_polish=0.01, tau_floor=0.1, tau_zero=1e-3, tau_parallel=False, **likelihood_kwargs):
    """
    run.iterative_pulse_fit with maximum-likelihood fits at every step (module docstring).

    split_starts, tau_profile, n_polish, lr_polish, tau_floor, tau_zero, tau_parallel: passed to maxL_fit
                 (tau_profile = 0: no tau profile)
    final_nuts:  also run NUTS (run.run_fit) on the kept number of pulses, every chain started at
                 the best point, with nuts_kwargs (e.g. num_chains, num_warmup) passed to run_fit
    likelihood_kwargs: passed to the likelihood (e.g. repulsion={})

    Returns a dict like iterative_pulse_fit's: 'n_pulses', 'fit', 'fits' ({n: maxL_fit result}),
    'steps' (each with 'search', and when fitted 'check', 'd_logL', 'accept'), 'stop', 'snr_min',
    'logL_min' (the threshold on the gain, per added pulse), 'min_channels', 'seed', and 'mcmc'
    (final_nuts only).
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
              f"up to {n_max} pulses, seed {seed}"
              + (f", tau profile over at least {tau_profile} values from {tau_floor:g} dt"
                 + (f" plus {tau_zero:g} dt" if tau_zero is not None else "") if tau_profile else ""))

    def fit(n, prev=None, search=None):
        priors = set_priors(burst, single_pulse_fn, s_opt=s_opt, verbose=False, **prior_kwargs)
        return maxL_fit(burst, single_pulse_fn, priors, n, likelihood=likelihood, num_starts=num_starts,
                        n_warm=n_warm, prev=prev, search=search, split_starts=split_starts,
                        shared_params=shared_params, n_adam=n_adam, lr=lr, lr_decay=lr_decay,
                        tau_profile=tau_profile, n_polish=n_polish, lr_polish=lr_polish, tau_floor=tau_floor,
                        tau_zero=tau_zero, tau_parallel=tau_parallel,
                        seed=seed, verbose=verbose, **likelihood_kwargs)

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
        check = boxes_explained(burst, fits[n - 1]["M"], fits[n]["M"], ep, min_change_snr=snr_min, verbose=verbose)
        if plot_steps:
            plot_boxes_explained(burst, fits[n - 1]["M"], fits[n]["M"], check)
        d_logL = fits[n]["logL"] - fits[n - 1]["logL"]
        accept = bool(check["any_removed"] and d_logL >= logL_min)
        steps.append(dict(n=n, search=ep, check=check, d_logL=d_logL, accept=accept))
        if verbose:
            print(f"{n - 1} -> {n} pulses: delta ln L = {d_logL:.1f} (need {logL_min:g}), explains leftover "
                  f"signal: {check['any_removed']} -> {'accept' if accept else 'reject'}")
        if not accept:
            reasons = ([] if check["any_removed"] else ["it explains no flagged leftover signal"]) + \
                      ([] if d_logL >= logL_min else [f"delta ln L {d_logL:.1f} < {logL_min:g}"])
            stop = f"{n}-pulse fit rejected: " + " and ".join(reasons)
            break
        n_final = n

    if verbose:
        print(f"final: {n_final} pulses ({stop}); optimizer time {sum(f['seconds'] for f in fits.values()):.0f} s")
    out = dict(n_pulses=n_final, fit=fits[n_final], fits=fits, steps=steps, stop=stop, snr_min=snr_min,
               logL_min=logL_min, min_channels=min_channels, seed=seed)

    if final_nuts:
        kept = fits[n_final]
        sampled = {k: jnp.asarray(v) for k, v in kept["map"].items() if k in kept["z"]}
        out["mcmc"] = run_fit(burst, single_pulse_fn, kept["priors"], likelihood, n_pulses=n_final, seed=seed,
                              init_strategy=init_to_value(values=sampled), shared_params=shared_params,
                              **dict(nuts_kwargs or {}), **likelihood_kwargs)
    return out
