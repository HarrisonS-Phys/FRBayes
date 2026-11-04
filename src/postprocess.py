"""
postprocess.py - Tools applied to posterior samples after a fit.

order_pulses       relabels the pulses of every draw consistently, so that per-pulse summaries,
                   R-hat, projections and plots refer to the same pulse in every draw and chain.
pointwise_log_lik  the log-likelihood of every Fourier bin (or channel) for every draw, the input
                   to leave-one-out cross-validation.
stack_chains       weights chains that never mixed by how well they predict left-out data
                   (Bayesian stacking, Yao, Vehtari & Gelman 2022), and draws a combined posterior.
combine_chains     pools chains that sampled the same mode (group_chains) and weights distinct modes.

STACKING NON-MIXING CHAINS. With several pulses the posterior is multimodal and chains started in
different places rarely cross between modes, so their proportions in the pooled samples say
nothing about how much each mode matters. Stacking replaces "keep or discard each chain" with
weights: each chain k gets leave-one-out predictive densities p(y_i | y_-i, chain k) for every
observation i (by Pareto-smoothed importance sampling within that chain), and the weights w
maximize sum_i log sum_k w_k p(y_i | y_-i, chain k). Chains stuck in poorly fitting modes get
weights near zero; chains in the same mode share theirs. The weights reflect predictive
performance, not posterior mass, and like any predictive criterion they cannot tell a component
from structure that the model only absorbs because it is misspecified: stacking chooses among
chains at a fixed number of pulses, while the number itself is decided separately.

    samples = mcmc.get_samples(group_by_chain=True)
    log_lik = pointwise_log_lik(burst, fit_fn, samples, thin=10)
    U = mcmc.get_extra_fields(group_by_chain=True)["potential_energy"]
    posterior, info = combine_chains(samples, log_lik, U)               # pool within modes, weight between

combine_chains first groups chains into modes (group_chains: pairwise rank-normalized R-hat), pools
the chains of each mode, and weights distinct modes by the height of their posterior peak (or, on
request, by stacking, whose weights favour modes that predict some observations well even when
they fit the data far worse, and which among chains of one mode are an arbitrary tie-break). stacking_weights and stack_chains remain
available for weighting chains individually.

LABEL SWITCHING. The likelihood is a sum over pulses and, without a strict time-ordering prior, the
pulses are exchangeable: swapping two pulses' parameters leaves the posterior unchanged. Different
chains (and occasionally a single chain) therefore use different labels for the same pulse, so
"pulse 2" means different things in different draws. Any per-pulse statistic -- a posterior
median, R-hat, the pulse to drop in a projection -- is meaningless until the labels are made
consistent. order_pulses does this by sorting the pulses of each draw by one parameter (arrival
time by default) and applying the same permutation to EVERY per-pulse quantity, so each pulse keeps
all of its own parameters. The model is a sum over pulses, so it is unchanged.

    samples = order_pulses(mcmc.get_samples(group_by_chain=True))   # or mcmc.get_samples()

Sorting cannot merge different solutions: chains that found different sets of pulses still
disagree after ordering, which is multimodality rather than label switching.
"""

import numpy as np


# Sites recorded once per PAIR of pulses, in the pair order of np.triu_indices(n_pulses, k=1)
PAIR_SITES = ("toa_separation", "overlap")

# Sites recorded as an (n_pulses, n_pulses) matrix
MATRIX_SITES = ("gram",)


def order_pulses(samples, by="mean_arrival_time", descending=False, active_snr=3.0,
                 pair_sites=PAIR_SITES, matrix_sites=MATRIX_SITES, exclude=(), return_perm=False,
                 verbose=True):
    """
    Relabels the pulses of every draw by sorting them on one parameter, and applies the same
    permutation to every per-pulse, per-pair and pulse-by-pulse site.

    samples:     dict of posterior arrays, from mcmc.get_samples() (shape (draws, ...)) or
                 mcmc.get_samples(group_by_chain=True) (shape (chains, draws, ...)); jax or numpy
    by:          the per-pulse site to sort on. Default the arrival time, which for well-separated
                 pulses gives each pulse the same label in every draw and chain.
    descending:  sort from largest to smallest (e.g. by="A_snr", descending=True: brightest first)
    active_snr:  if given and 'A_snr' is in samples, pulses with A_snr below it in a draw are put
                 after the active ones (each group sorted on `by`). An idle pulse wanders freely,
                 and without this it can pass an active pulse in time and shift that pulse's label
                 in some draws. None sorts all pulses on `by` alone.
    pair_sites:  names of sites recorded per pair of pulses, in np.triu_indices(n, k=1) order;
                 their pairs are re-indexed to the new labels. Named explicitly because with three
                 pulses a per-pair site has the same length as a per-pulse one.
    matrix_sites: names of sites recorded as (n_pulses, n_pulses) matrices; both axes are permuted.
                 Named explicitly because with two pulses a per-pulse site with a trailing axis of
                 length 2 would look like one.
    exclude:     names of sites to leave untouched, for a site whose axis merely happens to have
                 the length n_pulses
    return_perm: also return the permutation, shape (..., n_pulses): new pulse i of a draw is old
                 pulse perm[..., i]
    verbose:     print which sites were permuted and which were left alone

    A site is treated as:
        per pulse        its axis after the sample axes has length n_pulses (shared parameters,
                         stored broadcast to every pulse, are permuted too, which leaves them unchanged)
        per pair         its name is in pair_sites and that axis has length n_pulses (n_pulses - 1) / 2
        pulse-by-pulse   its name is in matrix_sites and its next two axes are (n_pulses, n_pulses),
                         e.g. the Gram matrix; both are permuted
    and is left unchanged otherwise (scalars per draw such as '<name>_shared', or anything not
    sampled per draw).

    Returns a new dict (and the permutation if return_perm), leaving `samples` unchanged.
    """
    samples = {k: np.asarray(v) for k, v in samples.items()}
    if by not in samples:
        raise KeyError(f"order_pulses: '{by}' is not in samples.")
    key = samples[by].astype(float)
    n_pulses = key.shape[-1]
    lead = key.ndim - 1                                # the sample axes: (draws,) or (chains, draws)
    if n_pulses < 2:
        return (samples, np.zeros(key.shape, int)) if return_perm else samples

    key = -key if descending else key
    if active_snr is not None:
        if "A_snr" in samples and samples["A_snr"].shape == key.shape:
            idle = samples["A_snr"] < active_snr
            # Idle pulses after the active ones: shift their keys past every active key
            key = np.where(idle, key + (np.nanmax(key) - np.nanmin(key) + 1.0), key)
        elif verbose:
            print("order_pulses: no per-pulse 'A_snr' in samples, so active pulses are not put first.")
    perm = np.argsort(key, axis=-1, kind="stable")    # new pulse i is old pulse perm[..., i]

    n_pairs = n_pulses * (n_pulses - 1) // 2
    I, J = np.triu_indices(n_pulses, k=1)
    pair_of = np.full((n_pulses, n_pulses), -1)
    pair_of[I, J] = np.arange(n_pairs)                 # old pair index of old pulses (a, b), a < b
    pa, pb = perm[..., I], perm[..., J]                # old pulses behind each new pair
    old_pair = pair_of[np.minimum(pa, pb), np.maximum(pa, pb)]

    def expand(index, x, axis):
        """Broadcast an index over the sample axes to the shape take_along_axis needs for x."""
        shape = index.shape + (1,) * (x.ndim - axis - 1)
        return np.broadcast_to(index.reshape(shape), x.shape[:axis] + index.shape[axis:] + x.shape[axis + 1:])

    out, done = {}, {"per pulse": [], "per pair": [], "pulse-by-pulse": [], "unchanged": []}
    for name, x in samples.items():
        sampled = x.ndim > lead and x.shape[:lead] == key.shape[:lead]
        if name in exclude or not sampled:
            out[name] = x.copy(); done["unchanged"].append(name)
        elif name in pair_sites and x.shape[lead] == n_pairs:
            out[name] = np.take_along_axis(x, expand(old_pair, x, lead), axis=lead)
            done["per pair"].append(name)
        elif name in matrix_sites and x.ndim > lead + 1 and x.shape[lead:lead + 2] == (n_pulses, n_pulses):
            rows = np.take_along_axis(x, expand(perm, x, lead), axis=lead)
            cols = perm.reshape(perm.shape[:lead] + (1, n_pulses) + (1,) * (x.ndim - lead - 2))
            out[name] = np.take_along_axis(rows, np.broadcast_to(cols, rows.shape), axis=lead + 1)
            done["pulse-by-pulse"].append(name)
        elif x.shape[lead] == n_pulses:
            out[name] = np.take_along_axis(x, expand(perm, x, lead), axis=lead)
            done["per pulse"].append(name)
        else:
            out[name] = x.copy(); done["unchanged"].append(name)

    if verbose:
        order = f"{by} ({'descending' if descending else 'ascending'})"
        if active_snr is not None and "A_snr" in samples:
            order += f", active pulses (A_snr >= {active_snr:g}) first"
        print(f"order_pulses: {n_pulses} pulses ordered by {order}.")
        for kind, names in done.items():
            if names:
                print(f"  {kind}: {', '.join(names)}")
    return (out, perm) if return_perm else out


# -------------------------------------------------------------------
# Pointwise log-likelihood and stacking of non-mixing chains
# -------------------------------------------------------------------

# Parameters single_pulse_fn may read (as in plotting.MODEL_PARAMS); 'A' is the fluence, which
# single_pulse_fn applies itself, recorded as a deterministic by the A_snr likelihood
MODEL_PARAMS = ("mean_arrival_time", "sigma", "DM", "tau_ref", "c1", "c2", "gamma", "beta", "delta", "A")


def pointwise_log_lik(burst, single_pulse_fn, samples, thin=1, chunk=64, by="bin"):
    """
    The log-likelihood of every observation for every (thinned) draw of every chain.

    burst:           the ConditionedData the fit used
    single_pulse_fn: the model closure the fit used (padded or not)
    samples:         posterior samples grouped by chain, mcmc.get_samples(group_by_chain=True):
                     per-pulse sites shaped (chains, draws, n_pulses). Must contain 'A', the
                     fluence, which the A_snr likelihood records.
    thin:            keep every thin-th draw. A few hundred draws per chain are plenty, and the
                     output has (chains x draws x observations) entries.
    chunk:           draws evaluated at once; lower it if memory runs out
    by:              "bin" (default): one observation per channel and retained Fourier bin, each
                     independent under the noise model. "channel": one per channel, summed over
                     its bins (far fewer observations, but each is more influential, so the
                     importance sampling in stacking_weights is less reliable).

    Each complex bin has independent real and imaginary parts of standard deviation sigma, so
        log p(X | M) = -|X - M|^2 / (2 sigma^2) - log(2 pi sigma^2),
    the per-bin version of the likelihood the fit used, with its normalization.

    Returns an array of shape (chains, kept draws, observations).
    """
    import jax
    import jax.numpy as jnp

    if "A" not in samples:
        raise KeyError("samples has no 'A' (the fluence). Use the samples of an A_snr fit, which "
                       "record it, or add the fluences drawn by posthoc_samples.")
    s = {k: np.asarray(v) for k, v in samples.items()}
    if s["mean_arrival_time"].ndim != 3:
        raise ValueError("samples must be grouped by chain: mcmc.get_samples(group_by_chain=True).")
    n_chains, n_draws, n_pulses = s["mean_arrival_time"].shape
    keep = np.arange(0, n_draws, thin)

    def per_pulse(v):
        v = v[:, keep]
        if v.ndim == 2:                                    # a scalar per draw: give it to every pulse
            v = np.repeat(v[..., None], n_pulses, axis=-1)
        return v.reshape(n_chains * keep.size, n_pulses).astype(np.float32)

    params = {k: per_pulse(s[k]) for k in MODEL_PARAMS if k in s}

    X = jnp.asarray(burst.data_fd)
    var = jnp.broadcast_to(jnp.asarray(burst.sigma_noise_fd, dtype=jnp.float32), X.shape) ** 2
    const = -jnp.log(2.0 * jnp.pi * var)

    @jax.jit
    def batch(p):
        M = jax.vmap(lambda q: jnp.sum(jax.vmap(single_pulse_fn)(q), axis=0))(p)   # (B, n_chan, n_fourier)
        ll = -0.5 * jnp.abs(X[None] - M) ** 2 / var[None] + const[None]
        return ll.reshape(ll.shape[0], -1) if by == "bin" else jnp.sum(ll, axis=-1)

    if by not in ("bin", "channel"):
        raise ValueError(f"by must be 'bin' or 'channel'; got {by!r}.")
    n_total = n_chains * keep.size
    out = []
    for start in range(0, n_total, chunk):
        stop = min(start + chunk, n_total)
        p = {k: jnp.asarray(v[start:stop]) for k, v in params.items()}
        if stop - start < chunk:                          # pad the last chunk so batch compiles once
            p = {k: jnp.concatenate([v, jnp.repeat(v[-1:], chunk - (stop - start), axis=0)]) for k, v in p.items()}
        out.append(np.asarray(batch(p))[: stop - start])
    return np.concatenate(out).reshape(n_chains, keep.size, -1)


def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def _psis_log_weights(log_lik_chain, r_eff):
    """
    Normalized leave-one-out log importance weights within one chain, shape (observations, draws),
    and the Pareto k-hat of each observation. The LOO importance ratio of draw s for observation i
    is 1 / p(y_i | theta_s), so the log ratios are -log_lik. Uses ArviZ's PSIS when available; otherwise, and for
    any observation where PSIS fails numerically, truncated importance sampling (Ionides 2008),
    with k-hat returned as NaN when PSIS was unavailable and as inf where it failed.
    """
    raw = -log_lik_chain.T                                                       # (obs, draws)
    n_obs, n = raw.shape
    cap = _logsumexp(raw, axis=-1) - np.log(n) + 0.5 * np.log(n)                 # mean weight x sqrt(n)
    tis = np.minimum(raw, cap[:, None])
    try:
        from arviz_stats.base import array_stats
    except ImportError:
        return tis - _logsumexp(tis, axis=-1)[:, None], np.full(n_obs, np.nan)

    # Observations whose largest ratios are (nearly) all equal -- e.g. bins where every draw's model
    # is ~0, so the log-likelihood differs between draws only by float32 rounding -- have bounded
    # weights: truncated importance sampling is exact enough for them, and PSIS cannot fit a Pareto
    # tail to them. They get k-hat 0. Below 1e-4 nats the spread cannot matter to the weights.
    lw, khat = tis.copy(), np.zeros(n_obs)
    n_tail = int(np.ceil(min(0.2 * n, 3.0 * np.sqrt(n / r_eff))))
    usable = np.ptp(np.sort(raw, axis=-1)[:, -n_tail:], axis=-1) > 1e-4
    if usable.any():
        try:
            with np.errstate(all="ignore"):
                # arviz_stats' psislw takes the LOG-LIKELIHOOD and negates it itself to form the
                # LOO log ratios -log p(y_i | theta_s); passing -log_lik would invert the weights
                lw_u, k_u = array_stats.psislw(log_lik_chain[:, usable], r_eff=r_eff, axis=0)
            lw_u, k_u = np.array(lw_u, dtype=float), np.array(k_u, dtype=float)
            bad = ~(np.all(np.isfinite(lw_u), axis=-1) & np.isfinite(k_u))
            lw_u[bad], k_u[bad] = tis[usable][bad], np.inf      # PSIS failed here: flagged
        except ValueError:                                       # PSIS failed for the whole chain
            lw_u, k_u = tis[usable], np.full(usable.sum(), np.inf)
        lw[usable], khat[usable] = lw_u, k_u
    return lw - _logsumexp(lw, axis=-1)[:, None], khat


def _loo_by_observation(log_lik_draws, r_eff=1.0):
    """
    PSIS leave-one-out log predictive density of every observation, from one set of draws, and
    the fraction of observations whose Pareto k-hat exceeds 0.7 (NaN when PSIS is unavailable).

    log_lik_draws: shape (draws, observations), e.g. one chain of pointwise_log_lik, or several
                   chains of the same mode concatenated along the draws
    """
    lw, khat = _psis_log_weights(log_lik_draws, r_eff)
    E = _logsumexp(lw + log_lik_draws.T, axis=-1)
    frac_high = np.nan if np.all(np.isnan(khat)) else float(np.mean(khat > 0.7))
    return E, frac_high


def _stacking_em(E, max_iter=20000, tol=1e-10):
    """
    Stacking weights maximizing sum_i log sum_k w_k exp(E[k, i]), by EM for the weights of a
    mixture with fixed components: the responsibilities of each component for each observation,
    averaged over observations. E: shape (components, observations).
    """
    w = np.full(E.shape[0], 1.0 / E.shape[0])
    for _ in range(max_iter):
        a = np.log(np.maximum(w, 1e-300))[:, None] + E
        r = np.exp(a - _logsumexp(a, axis=0)[None, :])
        w_new = r.mean(axis=1)
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    return w


def stacking_weights(log_lik, method="stacking", r_eff=1.0, n_bootstrap=1000, max_iter=20000,
                     tol=1e-10, seed=0, verbose=True):
    """
    Stacking weights for chains that did not mix (Yao, Vehtari & Gelman 2022).

    log_lik: pointwise log-likelihood, shape (chains, draws, observations), from pointwise_log_lik
    method:  "stacking" (default; Yao, Vehtari & Gelman 2022) or "pseudo-bma+" (see below)
    r_eff:   relative efficiency of the draws for PSIS (ESS / draws); 1 is fine for thinned chains
    n_bootstrap: Bayesian-bootstrap replicates for "pseudo-bma+"
    max_iter, tol: of the EM iteration for the stacking weights
    seed:    for the bootstrap

    For each chain k and observation i, the leave-one-out predictive density
        log p(y_i | y_-i, chain k) = log sum_s w_kis p(y_i | theta_ks)
    is estimated by Pareto-smoothed importance sampling over that chain's draws. The weights
    maximize the stacked log predictive density
        sum_i log sum_k w_k p(y_i | y_-i, chain k),    w_k >= 0, sum_k w_k = 1,
    a concave problem, solved here by EM for the weights of a mixture with fixed components.

    Stacking builds the best predictive MIXTURE, so a chain in a clearly worse mode can still get
    a small weight if it predicts some observations better, e.g. by chance noise fluctuations; a
    few percent is typical. When the aim is rather to keep the best mode(s) and drop stuck chains,
    method="pseudo-bma+" gives weights proportional to exp(elpd_loo_k), averaged over Bayesian-
    bootstrap resamplings of the observations so that chains whose elpd_loo is within its
    uncertainty of the best share the weight (Yao et al. 2018, Bayesian Analysis 13, 917). A chain
    tens of nats worse then gets essentially zero weight.

    Prints, chain by chain from the largest weight down: the weight, the chain's own LOO
    predictive density relative to the best chain's, its mean log-likelihood relative to the best,
    and the fraction of observations whose Pareto k-hat exceeds 0.7 (above that the importance
    sampling for that observation is unreliable; a large fraction means that chain's LOO is not
    trustworthy, usually because each observation is too influential, e.g. with by="channel").

    Returns (weights of shape (chains,), table): table is a dict of per-chain arrays 'weight',
    'elpd_loo', 'mean_log_lik', 'frac_khat_high', plus 'elpd_stacked', the stacked total.
    """
    log_lik = np.asarray(log_lik, dtype=np.float64)
    n_chains, _, n_obs = log_lik.shape

    E = np.empty((n_chains, n_obs))                     # log p(y_i | y_-i, chain k)
    frac_high = np.empty(n_chains)
    for k in range(n_chains):
        E[k], frac_high[k] = _loo_by_observation(log_lik[k], r_eff)

    if method == "stacking":
        w = _stacking_em(E, max_iter, tol)
    elif method == "pseudo-bma+":
        # exp(elpd_loo) weights, each from a Bayesian-bootstrap resampling of the observations
        rng = np.random.default_rng(seed)
        alpha = rng.dirichlet(np.ones(n_obs), size=n_bootstrap)          # (bootstrap, obs)
        z = n_obs * alpha @ E.T                                           # (bootstrap, chains)
        w = np.exp(z - _logsumexp(z, axis=1)[:, None]).mean(axis=0)
    else:
        raise ValueError(f"method must be 'stacking' or 'pseudo-bma+'; got {method!r}.")
    w = w / w.sum()

    elpd = E.sum(axis=1)
    mean_ll = log_lik.sum(axis=2).mean(axis=1)
    elpd_stacked = float(np.sum(_logsumexp(np.log(np.maximum(w, 1e-300))[:, None] + E, axis=0)))
    table = dict(weight=w, elpd_loo=elpd, mean_log_lik=mean_ll, frac_khat_high=frac_high,
                 elpd_stacked=elpd_stacked)

    if verbose:
        print(f"{method}: {n_chains} chains, {n_obs} observations; stacked elpd_loo = {elpd_stacked:.1f} "
              f"(best single chain {elpd.max():.1f})")
        for k in np.argsort(-w):
            print(f"  chain {k:3d}: weight {w[k]:.3f}   elpd_loo {elpd[k] - elpd.max():+9.1f}   "
                  f"mean log-lik {mean_ll[k] - mean_ll.max():+9.1f}   k-hat > 0.7: {frac_high[k]:.1%}")
    return w, table


def stack_chains(samples, weights, n_draws=None, seed=0, return_chain=False):
    """
    Draws a combined posterior from chains that did not mix, in proportion to stacking weights.

    samples: posterior samples grouped by chain, mcmc.get_samples(group_by_chain=True)
    weights: one weight per chain, from stacking_weights
    n_draws: number of draws to return; default the total number of draws in samples
    seed:    for the random choice of draws
    return_chain: also return the chain each stacked draw came from, shape (n_draws,)

    Each returned draw is a whole draw of one chain (all of its sites together), the chain chosen
    with probability equal to its weight and the draw uniformly within it. Sites not shaped
    (chains, draws, ...) are left out. Returns a flat dict of arrays shaped (n_draws, ...); run
    order_pulses on it before per-pulse summaries.
    """
    s = {k: np.asarray(v) for k, v in samples.items()}
    ref = s["mean_arrival_time"] if "mean_arrival_time" in s else next(iter(s.values()))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    n_chains, n_per = ref.shape[:2]
    if w.size != n_chains:
        raise ValueError(f"{w.size} weights for {n_chains} chains.")
    n_draws = n_chains * n_per if n_draws is None else int(n_draws)
    rng = np.random.default_rng(seed)
    chain = rng.choice(n_chains, size=n_draws, p=w)
    draw = rng.integers(0, n_per, size=n_draws)
    out = {k: v[chain, draw] for k, v in s.items() if v.ndim >= 2 and v.shape[:2] == (n_chains, n_per)}
    return (out, chain) if return_chain else out


# -------------------------------------------------------------------
# Grouping chains into modes: pool within a mode, stack between modes
# -------------------------------------------------------------------

# Per-pulse sites compared between chains (those present are used; gamma and beta only when the
# spectrum was not sampled as c1, c2, of which they are linear functions)
GROUPING_SITES = ("mean_arrival_time", "sigma", "tau_ref", "DM", "A_snr", "A", "c1", "c2", "delta",
                  "gamma", "beta")


def _split_rank_rhat(x):
    """
    Rank-normalized split R-hat (Vehtari, Gelman, Simpson, Carpenter & Buerkner 2021, Bayesian
    Analysis 16, 667) of x, shape (chains, draws): the larger of the bulk and the folded value.
    Each chain is split in half, all draws are replaced by the normal scores of their ranks, and
    the classic R-hat is taken of those (bulk) and of the scores of |x - median| (folded, which
    catches chains that differ in spread rather than location). A split chain that never moves
    gives infinity. NaN if there are fewer than 8 draws per chain.
    """
    from scipy.special import ndtri
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=float)
    n = x.shape[1] // 2
    if n < 4:
        return np.nan
    y = np.concatenate([x[:, :n], x[:, -n:]], axis=0)                 # split chains
    if np.ptp(y) == 0.0:
        return 1.0                                                     # identical everywhere

    def rhat(u):
        k = u.shape[1]
        W = u.var(axis=1, ddof=1).mean()
        if W <= 0.0:
            return np.inf
        B = k * u.mean(axis=1).var(ddof=1)
        return float(np.sqrt(((k - 1) / k * W + B / k) / W))

    def scores(u):
        r = rankdata(u, method="average").reshape(u.shape)
        return ndtri((r - 0.375) / (u.size + 0.25))

    return max(rhat(scores(y)), rhat(scores(np.abs(y - np.median(y)))))


def group_chains(samples, log_lik=None, rhat_max=1.1, active_snr=3.0, sites=GROUPING_SITES,
                 verbose=True):
    """
    Groups chains that sampled the same mode, by testing every pair of chains for convergence
    against each other.

    samples:    posterior samples grouped by chain, mcmc.get_samples(group_by_chain=True)
    log_lik:    optional, the pointwise log-likelihood of pointwise_log_lik for the same chains; its
                total per draw is compared too
    rhat_max:   two chains are the same mode if every compared quantity has R-hat below this.
                The default 1.1 is looser than the 1.01 used to declare convergence, because
                the question here is coarser: whether two chains sample overlapping posteriors.
                Between two chains, R-hat ~ sqrt(1 + d^2 / 4) for posteriors offset by d of their
                standard deviations, so 1.1 corresponds to d ~ 0.9. In tests with 8 chains of 200
                draws, 1.1 never split chains of one mode at ~100 effective draws per chain and
                did so once in 20 runs at ~35, while always separating chains offset by one
                standard deviation; 1.05 split chains of one mode in nearly every run at ~35.
    active_snr: a pulse whose median A_snr is below this in BOTH chains of a pair is idle in both,
                so only its amplitude is compared: an idle pulse's position and shape wander over
                the prior and say nothing about the mode
    sites:      per-pulse sites compared
    verbose:    print the modes found

    The pulses are first labelled consistently with order_pulses (active pulses first, by arrival
    time). Then, for every pair of chains, the rank-normalized split R-hat (_split_rank_rhat) is
    computed for every per-pulse parameter (once for a parameter shared by all pulses) and for the
    total log-likelihood, and the pair is linked if the largest of them is below rhat_max. The
    modes are the connected groups of linked chains.

    The test fails safe: chains of one mode that have not quite mixed exceed rhat_max and are kept
    apart, to be weighted by stacking like separate modes, while chains linked by it agree closely
    enough that pooling them changes nothing. Its noise grows as the effective draws per chain
    shrink, so with short or sticky chains raise rhat_max a little rather than lower it.

    Returns (modes, rhat): modes is a list of lists of chain indices, best mean log-likelihood
    first when log_lik is given; rhat, shape (chains, chains), holds the largest R-hat of each
    pair (0 on the diagonal).
    """
    s = order_pulses({k: np.asarray(v) for k, v in samples.items()}, active_snr=active_snr, verbose=False)
    if s["mean_arrival_time"].ndim != 3:
        raise ValueError("samples must be grouped by chain: mcmc.get_samples(group_by_chain=True).")
    n_chains, _, n_pulses = s["mean_arrival_time"].shape

    use = [k for k in sites if k in s and s[k].ndim == 3 and s[k].shape[2] == n_pulses]
    if {"c1", "c2"} <= set(use):
        use = [k for k in use if k not in ("gamma", "beta")]
    amplitude = ("A_snr", "A")
    median_snr = np.median(s["A_snr"], axis=1) if "A_snr" in s else None   # (chains, pulses)

    # (name, pulse or None for shared, values of shape (chains, draws))
    quantities = []
    for k in use:
        v = s[k]
        if np.allclose(v, v[..., :1]):
            quantities.append((k, None, v[..., 0]))
        else:
            quantities += [(k, p, v[..., p]) for p in range(n_pulses)]
    if log_lik is not None:
        L = np.asarray(log_lik, dtype=float)
        if L.shape[0] != n_chains:
            raise ValueError(f"log_lik has {L.shape[0]} chains, samples {n_chains}.")
        quantities.append(("log_lik", None, L.sum(axis=2)))

    rhat = np.zeros((n_chains, n_chains))
    for i in range(n_chains):
        for j in range(i + 1, n_chains):
            worst = 0.0
            for name, p, v in quantities:
                if (p is not None and name not in amplitude and median_snr is not None
                        and median_snr[i, p] < active_snr and median_snr[j, p] < active_snr):
                    continue                                   # idle in both: only its amplitude counts
                r = _split_rank_rhat(v[[i, j]])
                worst = max(worst, np.inf if np.isnan(r) else r)
            rhat[i, j] = rhat[j, i] = worst

    # Connected groups of linked chains (union-find)
    parent = list(range(n_chains))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n_chains):
        for j in range(i + 1, n_chains):
            if rhat[i, j] < rhat_max:
                parent[root(i)] = root(j)
    groups = {}
    for c in range(n_chains):
        groups.setdefault(root(c), []).append(c)
    modes = sorted(groups.values(), key=min)
    if log_lik is not None:
        mean_ll = np.asarray(log_lik, dtype=float).sum(axis=2).mean(axis=1)
        modes = sorted(modes, key=lambda m: -np.mean(mean_ll[m]))

    if verbose:
        print(f"group_chains: {n_chains} chains -> {len(modes)} mode(s) (pairs linked where every "
              f"R-hat < {rhat_max:g})")
        for m, chains in enumerate(modes):
            within = max((rhat[a, b] for a in chains for b in chains if a < b), default=0.0)
            print(f"  mode {m}: chains {chains}" + (f", largest R-hat within {within:.3f}" if len(chains) > 1 else ""))
    return modes, rhat


def combine_chains(samples, log_lik, potential_energy=None, mode_weights="peak", rhat_max=1.1,
                   active_snr=3.0, r_eff=1.0, seed=0, verbose=True):
    """
    One posterior from chains that may not have mixed: chains that sampled the same mode are
    pooled with equal weight, and distinct modes are weighted against each other, by the height
    of their posterior peak (default) or by stacking.

    samples:          posterior samples grouped by chain, mcmc.get_samples(group_by_chain=True)
    log_lik:          pointwise_log_lik for the same chains, shape (chains, kept draws,
                      observations). Used to group chains into modes and for the stacking weights.
    potential_energy: shape (chains, draws), mcmc.get_extra_fields(group_by_chain=True)
                      ["potential_energy"]: minus the log posterior (up to a constant). Required
                      for mode_weights="peak".
    mode_weights:     "peak" (default): each mode weighted by its highest posterior density,
                      w_m proportional to exp(-min U over the mode's draws). This is posterior-
                      mass weighting with the modes' widths left out, adequate when modes differ
                      by many nats (their widths would have to differ enormously to matter) and
                      to be replaced by bridge-sampled masses for close modes.
                      "stacking": weights maximizing the leave-one-out predictive density of the
                      mixture (Yao, Vehtari & Gelman 2022). These favour modes that predict some
                      observations better even when they fit the data as a whole far worse, so
                      they are not posterior probabilities of the modes' parameters.
    rhat_max, active_snr: passed to group_chains
    r_eff:            for PSIS, as in stacking_weights
    seed:             for the choice of draws when there is more than one mode
    verbose:          print the modes, their weights and the draws taken from each

    With one mode, every draw of every chain is returned. With several, each mode's chains are
    pooled and draws are taken WITHOUT replacement in proportion to the weights: the largest total
    for which no mode has to repeat a draw. A mode of weight below 1e-3 contributes none. Both
    sets of weights are computed and returned, whichever is used.

    Returns (combined, info): combined is a flat dict of arrays shaped (draws, ...), with pulses
    labelled by order_pulses (active first, by arrival time); info holds 'modes' (lists of chain
    indices), 'weights' (the ones used), 'weights_peak' (None without potential_energy),
    'weights_stacking', 'peak_log_post' and 'mean_log_lik' (per mode, relative to the best),
    'rhat' (from group_chains), and 'chain' and 'draw' (the source of every returned draw, as
    indices into samples; they index log_lik only if it was computed with thin=1).
    """
    if mode_weights not in ("peak", "stacking"):
        raise ValueError(f"mode_weights must be 'peak' or 'stacking'; got {mode_weights!r}.")
    if mode_weights == "peak" and potential_energy is None:
        raise ValueError("mode_weights='peak' needs potential_energy: "
                         "mcmc.get_extra_fields(group_by_chain=True)['potential_energy'].")
    s = order_pulses({k: np.asarray(v) for k, v in samples.items()}, active_snr=active_snr, verbose=False)
    L = np.asarray(log_lik, dtype=float)
    n_chains, n_per = s["mean_arrival_time"].shape[:2]
    modes, rhat = group_chains(s, L, rhat_max=rhat_max, active_snr=active_snr, verbose=verbose)

    mean_ll = np.array([L[m].sum(axis=2).mean() for m in modes])
    mean_ll -= mean_ll.max()

    def normalized(w):
        w = np.where(w < 1e-3, 0.0, w)
        return w / w.sum()

    peak = w_peak = None
    if potential_energy is not None:
        U = np.asarray(potential_energy, dtype=float)
        if U.shape[:2] != (n_chains, n_per):
            raise ValueError(f"potential_energy has shape {U.shape}; expected ({n_chains}, {n_per}).")
        peak = np.array([-U[m].min() for m in modes])                 # highest log posterior of each mode
        peak -= peak.max()
        w_peak = normalized(np.exp(peak))
    if len(modes) == 1:
        w_stack = np.ones(1)
    else:
        E = np.stack([_loo_by_observation(L[m].reshape(-1, L.shape[2]), r_eff)[0] for m in modes])
        w_stack = normalized(_stacking_em(E))
    weights = w_peak if mode_weights == "peak" else w_stack

    rng = np.random.default_rng(seed)
    pools = [np.array([(c, d) for c in m for d in range(n_per)]) for m in modes]
    total = min(len(pool) / w for pool, w in zip(pools, weights) if w > 0)
    counts = [min(int(round(w * total)), len(pool)) for pool, w in zip(pools, weights)]
    take = [np.arange(len(pool)) if k == len(pool) else rng.choice(len(pool), size=k, replace=False)
            for pool, k in zip(pools, counts)]
    src = np.concatenate([pool[t] for pool, t in zip(pools, take)])
    chain, draw = src[:, 0], src[:, 1]

    combined = {k: v[chain, draw] for k, v in s.items() if v.ndim >= 2 and v.shape[:2] == (n_chains, n_per)}
    if verbose:
        print(f"  modes weighted by {'peak posterior' if mode_weights == 'peak' else 'stacking'}:")
        for m, (chains, t) in enumerate(zip(modes, take)):
            height = f"peak log post {peak[m]:+.1f}, " if peak is not None else ""
            other = (f" (stacking would give {w_stack[m]:.3f})" if mode_weights == "peak" else
                     f" (peak would give {w_peak[m]:.3f})" if w_peak is not None else "")
            print(f"  mode {m}: {height}mean log-lik {mean_ll[m]:+.1f}, weight {weights[m]:.3f}{other}, "
                  f"{len(t)} draws from {len(chains)} chain(s)")
        print(f"combine_chains: {len(chain)} draws in all")
    info = dict(modes=modes, weights=weights, weights_peak=w_peak, weights_stacking=w_stack,
                peak_log_post=peak, mean_log_lik=mean_ll, rhat=rhat, chain=chain, draw=draw)
    return combined, info


# -------------------------------------------------------------------
# Residual test, blocking out each pulse's highest-density interval channel by channel
# -------------------------------------------------------------------

def _to_time_domain(burst, X_fd):
    """Fourier-domain arrays on the burst's retained bins back to time (as plotting.to_time_domain)."""
    n, dt = burst.n_time, burst.dt
    k = np.rint(np.asarray(burst.fourier_freqs, dtype=float).reshape(-1) * n * dt).astype(int)
    full = np.zeros(np.shape(X_fd)[:-1] + (n // 2 + 1,), dtype=complex)
    full[..., k] = np.asarray(X_fd)
    return np.fft.irfft(full, n=n, axis=-1) / dt


def draw_params(samples, draw_idx):
    """The per-pulse parameters of one draw of flat samples, as arrays of shape (n_pulses,)."""
    n_pulses = np.shape(samples["mean_arrival_time"])[-1]
    return {k: np.broadcast_to(np.asarray(v[draw_idx]), (n_pulses,)).astype(float)
            for k, v in samples.items() if np.ndim(v) >= 1 and len(v) > draw_idx
            and np.size(v[draw_idx]) in (1, n_pulses)}


def pulse_footprints(burst, single_pulse_fn, params, hdi_mass=0.95, active_snr=3.0, oversample=8):
    """
    The time bins, channel by channel, inside the highest-density interval (HDI) of each active
    pulse, as a boolean mask of shape (n_chan, n_time): the union over pulses.

    burst:           the ConditionedData the fit used
    single_pulse_fn: the model closure the fit used; its constants give nu_ref, k_DM and delta
    params:          one draw's per-pulse parameters, arrays of shape (n_pulses,), e.g.
                     draw_params(stacked, draw_idx) or test_params
    hdi_mass:        probability mass of each pulse's profile, per channel, that the HDI holds.
                     0.95 reaches about 3 tail lengths past the peak (e^-3 = 5%); 0.99 about 4.6.
    active_snr:      pulses with A_snr below this are skipped (if A_snr is in params)
    oversample:      evaluation points per time bin, so pulses narrower than a bin are handled

    In channel c a pulse is a Gaussian (width sigma) arriving at
        t_c = t0 + k_DM DM (nu_c^-2 - nu_ref^-2),
    convolved with the exponential tail of length tau_c = tau_ref (nu_c / nu_ref)^delta: an
    exponentially modified Gaussian. Its mass in each time bin is computed from that closed form
    (renormalized to the part inside the window), and the HDI is the smallest set of bins holding
    hdi_mass of it: for this unimodal profile, one interval running further after the peak than
    before it, and longest at the bottom of the band where tau_c is longest. Intra-channel
    smearing and time binning are not included; they are small next to sigma and tau_c unless
    the pulse is unresolved.
    """
    from scipy.stats import exponnorm

    c = single_pulse_fn.constants
    nu = np.asarray(burst.freq_channels, dtype=float).reshape(-1, 1)          # (n_chan, 1), MHz
    nu_ref, k_dm = float(c.freq_channels_ref), float(c.k_DM)
    dt, n_time = float(burst.dt), int(burst.n_time)
    sub = (np.arange(n_time)[:, None] * dt + (np.arange(oversample)[None, :] + 0.5) * dt / oversample).reshape(1, -1)

    mask = np.zeros((nu.shape[0], n_time), dtype=bool)
    n_pulses = np.shape(params["mean_arrival_time"])[-1]
    for p in range(n_pulses):
        if "A_snr" in params and float(params["A_snr"][p]) < active_snr:
            continue
        sigma = max(float(params["sigma"][p]), 1e-3 * dt)
        tau_ref = float(params["tau_ref"][p]) if "tau_ref" in params else 0.0
        delta = float(params["delta"][p]) if "delta" in params else float(c.delta)
        dm = float(params["DM"][p]) if "DM" in params else 0.0
        t_c = float(params["mean_arrival_time"][p]) + k_dm * dm * (nu ** -2 - nu_ref ** -2)
        tau_c = tau_ref * (nu / nu_ref) ** delta
        K = np.maximum(tau_c / sigma, 1e-6)
        pdf = exponnorm.pdf(sub - t_c, K, scale=sigma)                       # (n_chan, n_time * oversample)
        mass = pdf.reshape(nu.shape[0], n_time, oversample).sum(-1)
        mass /= np.maximum(mass.sum(-1, keepdims=True), np.finfo(float).tiny)
        order = np.argsort(-mass, axis=-1)
        cum = np.cumsum(np.take_along_axis(mass, order, axis=-1), axis=-1)
        n_keep = np.minimum((cum < hdi_mass).sum(-1) + 1, n_time)          # bins until hdi_mass is reached
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(n_time)[None, :].repeat(nu.shape[0], 0), axis=-1)
        mask |= rank < n_keep[:, None]
    return mask


def residual_test(burst, single_pulse_fn, params, M_FD, window=20, hdi_mass=0.95, threshold=10.0,
                  active_snr=3.0, min_weight=0.05, verbose=True):
    """
    Largest residual structure outside the pulses already fitted, with each pulse blocked out
    channel by channel over its highest-density interval, and separately inside those intervals.

    burst, single_pulse_fn: as for the fit
    params:     one draw's per-pulse parameters (draw_params(...) or test_params)
    M_FD:       that draw's model on the burst's Fourier grid, e.g. from plotting.model_from_draw
    window:     length of the sliding window, in time samples (a few pulse widths)
    hdi_mass:   mass of each pulse's per-channel profile blocked out; see pulse_footprints
    threshold:  residual SNR above which, outside the blocked regions, another pulse is worth trying
    active_snr: pulses below this A_snr are not blocked out
    min_weight: times where less than this fraction of the burst's channel weight is left after
                blocking are skipped, since too few channels remain to test

    The residual is whitened channel by channel and combined across channels with the burst's
    SNR^2 weights, using only the unblocked pixels at each time and renormalizing by the weight
    left, so pure noise has the same variance everywhere. It is then scaled to unit variance in
    the off-pulse window (which absorbs any error in the noise level), and in every window of
    `window` samples the excess power over noise gives a residual SNR. Inside the blocked regions
    the same is done with the blocked pixels alone: structure there is the fitted pulses' misfit
    and is reported, never used to add a pulse.

    Returns a dict: snr_free, t_free (start of its window, s), snr_blocked, t_blocked, add_pulse
    (snr_free > threshold), and the mask.
    """
    r = _to_time_domain(burst, np.asarray(burst.data_fd) - np.asarray(M_FD))
    r = r / np.asarray(burst.sigma_noise_td, dtype=float).reshape(-1, 1)
    w = np.asarray(burst.channel_weights, dtype=float).reshape(-1, 1)
    w = w / w.sum()
    mask = pulse_footprints(burst, single_pulse_fn, params, hdi_mass=hdi_mass, active_snr=active_snr)
    n_noise = int(burst.n_noise_bins)

    def profile(use):
        weight = np.sum(w * use, axis=0)
        prof = np.sum(np.sqrt(w) * r * use, axis=0) / np.sqrt(np.maximum(weight, 1e-12))
        return np.where(weight >= min_weight, prof, np.nan)

    free, blocked = profile(~mask), profile(mask)
    scale = np.nanstd(free[:n_noise])

    def worst(prof):
        z = (prof / scale) ** 2 - 1.0
        valid = np.convolve(np.isfinite(z).astype(float), np.ones(window), mode="valid") == window
        excess = np.convolve(np.nan_to_num(z), np.ones(window), mode="valid")
        excess = np.where(valid, excess, -np.inf)
        i = int(np.argmax(excess))
        return (float(np.sqrt(max(excess[i], 0.0))), float(burst.times[i])) if np.isfinite(excess[i]) else (0.0, np.nan)

    snr_free, t_free = worst(free)
    snr_blocked, t_blocked = worst(blocked)
    out = dict(snr_free=snr_free, t_free=t_free, snr_blocked=snr_blocked, t_blocked=t_blocked,
               add_pulse=snr_free > threshold, mask=mask)
    if verbose:
        print(f"outside the pulses' {hdi_mass:.0%} HDIs: residual SNR {snr_free:.1f} in the window "
              f"starting {t_free * 1e3:.2f} ms -> {'try another pulse' if out['add_pulse'] else 'stop'}")
        print(f"inside them: residual SNR {snr_blocked:.1f} at {t_blocked * 1e3:.2f} ms "
              f"(misfit of the fitted pulses; never a new pulse)")
    return out



# -------------------------------------------------------------------
# Excess-power search of the residual (Anderson, Brady, Creighton & Flanagan 2001)
# -------------------------------------------------------------------

def _format_time(box):
    """A box's time extent in ms, with the wrapped part at the end of the segment if it has one."""
    text = f"{box['t_range'][0] * 1e3:.2f}-{box['t_range'][1] * 1e3:.2f} ms"
    if box.get("t_wrap"):
        text += f" + {box['t_wrap'][0] * 1e3:.2f}-{box['t_wrap'][1] * 1e3:.2f} ms (wraps)"
    return text


def _dyadic_ranges(n, min_width=1):
    """Contiguous index ranges [start, stop) of widths min_width, 2 min_width, 4 min_width, ...
    (and n), each width at starts spaced by half the width, so every interval at least min_width
    long is covered by a range at most about twice as long."""
    out = set()
    w = max(1, min(int(min_width), n))
    while w <= n:
        step = max(w // 2, 1)
        out.update((s, s + w) for s in range(0, n - w + 1, step))
        out.add((n - w, n))
        w *= 2
    out.add((0, n))
    r = np.array(sorted(out))
    return r[:, 0], r[:, 1]


def _cells(X, n_b):
    """Complex cells, shape (channels, bands, times): the retained bins of each channel cut into
    bands of n_b consecutive bins, each band inverse-DFT'd (unitary) into n_b time cells; a final
    partial band is left out. The transform is unitary, so sums of products over cells equal the
    same sums over the Fourier bins they come from."""
    n_chan, n_ret = X.shape
    n_bands = n_ret // n_b
    bands = X[:, : n_bands * n_b].reshape(n_chan, n_bands, n_b)
    return np.fft.ifft(bands, axis=-1) * np.sqrt(n_b)


def _cell_powers(X, n_b):
    """Cell powers |z|^2 of _cells."""
    return np.abs(_cells(X, n_b)) ** 2


def min_box_channels(burst, priors):
    """
    The fewest adjacent channels a residual box may span: the narrowest band in which the spectral
    prior lets a single pulse put its power, for a pulse whose spectrum peaks inside the band.
    Pass it to excess_power_search(..., min_channels=...).

    The running power law ln F = gamma x + beta x^2, x = ln(nu / nu_ref), is a Gaussian in x for
    beta < 0, and its power F^2 has standard deviation 1 / (2 sqrt|beta|). The most negative beta
    the prior allows gives the narrowest, with full width at half maximum
        FWHM_x = 2 sqrt(2 ln 2) / (2 sqrt|beta_min|),
    i.e. from nu up to nu exp(FWHM_x), narrowest in MHz at the bottom of the band. Returns the
    fewest channel centres inside such a window, over every start channel whose window fits in the
    band (at least 1). With c1, c2 sampled, beta = beta_c2 c2 (burst.spectral_coeff_map).

    Not covered: spectra so steep that a pulse's power sits in a sliver at the edge of the band.
    Such a sliver still falls inside the boxes of this width at that edge.

    burst:  the ConditionedData of the burst
    priors: the priors of the fit (set_priors); needs a bounded 'beta' or 'c2' prior, else 1
    """
    if "beta" in priors:
        beta_min = float(priors["beta"].low)
    elif "c2" in priors:
        a2 = float(burst.spectral_coeff_map["beta_c2"])
        beta_min = min(a2 * float(priors["c2"].low), a2 * float(priors["c2"].high))
    else:
        return 1
    if beta_min >= 0:
        return 1
    fwhm_x = 2.0 * np.sqrt(2.0 * np.log(2.0)) / (2.0 * np.sqrt(-beta_min))
    nu = np.sort(np.asarray(burst.freq_channels, dtype=float).reshape(-1))
    starts = nu[nu * np.exp(fwhm_x) <= nu[-1]]
    if starts.size == 0:
        return int(nu.size)
    return int(max(1, min(np.sum((nu >= v) & (nu < v * np.exp(fwhm_x))) for v in starts)))


def excess_power_search(burst, M_FD, epsilon=0.2, min_snr=0.0, resolutions=None, p_false_alarm=0.01,
                        x_min=1.0, min_channels=1, top=5, verbose=True):
    """
    Is there signal in the residual that neither noise nor a tolerated error of the fitted model
    explains? An excess-power search (Anderson, Brady, Creighton & Flanagan 2001, Phys. Rev. D 63,
    042003) over boxes in time, Fourier frequency and radio channel, in which each box may hold,
    besides noise, misfit of up to a fraction epsilon of the fitted model in it. Nothing is masked.

    burst:         the ConditionedData the fit used
    M_FD:          the fitted model on the burst's retained Fourier bins (plotting.model_from_draw)
    epsilon:       tolerated model error, as a fraction of the model's amplitude: a box is let off
                   leftover power up to epsilon^2 times the model's power in it. Where the model is
                   bright this is a relative test (leftover must exceed epsilon times the model's
                   SNR there); where the model is absent it is a plain test against noise.
    min_snr:       also require the unexplained SNR of a box to reach this for it to count as
                   significant (0: decide on the p-value alone)
    resolutions:   numbers of Fourier bins per band, n_b, each giving one tiling of time and Fourier
                   frequency with cells of duration W / n_b (W the segment duration). Default: 1, 2,
                   4, ... up to all retained bins.
    p_false_alarm: false-alarm probability of the whole search, after the trials correction
    x_min:        two significant boxes are separate features only if they are this far apart in
                   units of their summed widths (see SEPARATE FEATURES); use the same x_min as the
                   likelihood's repulsion
    min_channels:  fewest adjacent channels a box may span; min_box_channels gives the narrowest
                   band the spectral prior lets one pulse occupy. 1 allows single-channel boxes.
    top:           when nothing is significant, how many of the strongest boxes to return for display
    verbose:       print the outcome

    METHOD. The residual R = data_fd - M_FD and the model M_FD are both whitened by
    sigma_noise_fd, the same noise the likelihood uses. For a resolution n_b, the retained bins are cut into bands of n_b consecutive
    bins and each band is inverse-DFT'd (unitary) into n_b time cells: the band-passed signal in
    time. Each residual cell is a complex N(0, 1) pair under pure noise, so its power is chi^2 with
    2 degrees of freedom, and the model's power in the same cell, P_model, is its SNR^2 there. A
    box of V cells (a range of times, of adjacent bands and of adjacent channels) has residual
    power E. If the box holds noise plus a misfit signal of power L, E is non-central chi^2 with 2V
    degrees of freedom and non-centrality L; the null hypothesis is the largest misfit tolerated,
        L = epsilon^2 * sum_box P_model,
    and the box's significance is p_single = P(E' >= E) for E' drawn from that distribution
    (Patnaik's approximation, c chi^2_nu with c = (2V + 2L)/(2V + L), nu = (2V + L)^2/(2V + 2L),
    exact for L = 0). The best p_single over all boxes is multiplied by the number of boxes
    (Bonferroni, conservative since the boxes overlap). The box's unexplained SNR is
        snr_unexplained = sqrt(max(E - 2V - L, 0)),
    the leftover beyond noise and tolerated misfit; snr_residual = sqrt(max(E - 2V, 0)) and
    snr_model = sqrt(sum_box P_model) are reported alongside.

    No guard band is needed around the pulses: a bright pulse's misfit leaks into neighbouring
    cells through the band edges, but so does the model, so the tolerance leaks with it.

    SEPARATE FEATURES. A box is significant on its own when p_single, times the number of boxes,
    is below p_false_alarm. A large box holding several features is often the MOST significant,
    since it adds up all their leftover power. So features are found in two steps.
    Step 1, which features are separate: the significant boxes are taken SMALLEST first, and a box
    is a new feature only if
      - no feature already kept has its centre inside the box's time span, and
      - its centre c and width w are clear of every kept feature's, |c - c_k| >= x_min (w + w_k),
        the condition the repulsion wall puts on two pulses.
    Channels play no part: every pulse spans the whole band, so boxes at the same time in different
    channels are the same feature. c and w are the mean and sqrt(2) x the standard deviation over
    the box's times of its unexplained power at full time resolution: the residual and model on the
    box's channels and Fourier band, transformed (unitary) to time, giving
        u(t) = sum_c |r_c(t)|^2 - 2 K C / N - epsilon^2 sum_c |m_c(t)|^2,  clipped at 0
    (K bins, C channels, N time samples); sqrt(2) turns the spread of a Gaussian's power into the
    spread of its amplitude.
    Step 2, which box represents each feature: the most significant box whose time span holds
    that feature's centre and no other feature's. A box holding two features' centres represents
    neither, so the features stay separate, while each is represented by the box that captures it
    best -- usually the whole feature rather than its brightest corner, so that boxes_explained
    measures everything a new pulse changes there, and warm starts use all its channels.

    Returns a dict: 'add_pulse' (p < p_false_alarm and some box significant), 'p' (corrected),
    'n_tiles', 'n_significant' (boxes significant on their own), 'tiles' (the separate significant
    features, most significant first, or, if there are none, the `top` strongest boxes for display,
    none overlapping another in time: each with t_range (s), t_wrap (the part at the end of the
    segment, (start, W), for a box including the first time cell, which straddles the segment's
    start and wraps to its end; else None), channel_range (MHz), channel_index, band_index,
    time_index (each (first, last + 1) on the box's own grid), fourier_band (Hz), significant,
    n_cells, snr_unexplained, snr_residual, snr_model, log10_p_single, resolution, and for the
    significant features centre and width (s)), 'epsilon' and 'p_false_alarm'.
    plotting.plot_residual_tiles draws the boxes on the data and residual.
    """
    from scipy.stats import chi2

    sigma = np.broadcast_to(np.asarray(burst.sigma_noise_fd, dtype=float), np.shape(burst.data_fd))
    Z = (np.asarray(burst.data_fd) - np.asarray(M_FD)) / sigma
    Zm = np.asarray(M_FD) / sigma
    n_chan, n_ret = Z.shape
    n_time, dt = int(burst.n_time), float(burst.dt)
    W = n_time * dt
    f = np.asarray(burst.fourier_freqs, dtype=float).reshape(-1)
    nu = np.asarray(burst.freq_channels, dtype=float).reshape(-1)
    if resolutions is None:
        resolutions = [2 ** j for j in range(int(np.log2(n_ret)) + 1)]
        if resolutions[-1] != n_ret:
            resolutions.append(n_ret)
    resolutions = [n_b for n_b in resolutions if n_ret // n_b > 0]

    def plane(X, n_b):
        return _cell_powers(X, n_b)

    def table(P):
        S = np.zeros((P.shape[0] + 1, P.shape[1] + 1, P.shape[2] + 1))
        S[1:, 1:, 1:] = P.cumsum(0).cumsum(1).cumsum(2)              # summed-volume table
        return S

    ranges = {n_b: (_dyadic_ranges(n_chan, min_channels), _dyadic_ranges(n_ret // n_b), _dyadic_ranges(n_b))
              for n_b in resolutions}
    n_tiles = sum(len(c[0]) * len(b[0]) * len(t[0]) for c, b, t in ranges.values())
    log_limit = np.log(p_false_alarm) - np.log(n_tiles)               # significant on its own

    def make_tile(n_b, c0, c1, b0, b1, t0, t1, e, v, lam, pm, logp):
        cell = W / n_b
        lo, hi = (t0 - 0.5) * cell, (t1 - 0.5) * cell
        k0, k1 = b0 * n_b, b1 * n_b
        # Cell m is centred at m W / n_b, so cell 0 straddles the start of the segment and wraps
        # to its end: a box including it also covers [W + lo, W) (the whole segment at n_b = 1)
        return dict(
            t_range=(max(lo, 0.0), min(hi, W)),
            t_wrap=(W + lo, W) if lo < 0 else None,
            channel_range=(float(nu[c0]), float(nu[c1 - 1])),
            channel_index=(int(c0), int(c1)),
            band_index=(int(b0), int(b1)),
            time_index=(int(t0), int(t1)),
            fourier_band=(float(f[k0]), float(f[k1 - 1])),
            n_cells=int(v),
            snr_unexplained=float(np.sqrt(max(e - 2 * v - lam, 0.0))),
            snr_residual=float(np.sqrt(max(e - 2 * v, 0.0))),
            snr_model=float(np.sqrt(pm)),
            log10_p_single=float(logp / np.log(10)), resolution=int(n_b),
            significant=bool(logp < log_limit and e - 2 * v - lam >= min_snr ** 2))

    eps2 = float(epsilon) ** 2
    strongest, significant = [], []
    for n_b in resolutions:
        (cs, ce), (bs, be), (ts, te) = ranges[n_b]
        S, Sm = table(plane(Z, n_b)), table(plane(Zm, n_b))
        c0, c1 = cs[:, None, None], ce[:, None, None]
        b0, b1 = bs[None, :, None], be[None, :, None]
        t0, t1 = ts[None, None, :], te[None, None, :]

        def box(S):
            return (S[c1, b1, t1] - S[c0, b1, t1] - S[c1, b0, t1] - S[c1, b1, t0]
                    + S[c0, b0, t1] + S[c0, b1, t0] + S[c1, b0, t0] - S[c0, b0, t0])

        E, Pm = box(S), np.maximum(box(Sm), 0.0)
        V = np.broadcast_to((c1 - c0) * (b1 - b0) * (t1 - t0), E.shape)
        L = eps2 * Pm
        k = 2.0 * V
        c = (k + 2 * L) / (k + L)
        logp = chi2.logsf(E / c, (k + L) ** 2 / (k + 2 * L))

        def tile_at(i):
            ci, bi, ti = np.unravel_index(i, logp.shape)
            return make_tile(n_b, cs[ci], ce[ci], bs[bi], be[bi], ts[ti], te[ti],
                             E[ci, bi, ti], V[ci, bi, ti], L[ci, bi, ti], Pm[ci, bi, ti], logp[ci, bi, ti])

        n_keep = min(20 * top, logp.size)
        strongest += [tile_at(i) for i in np.argpartition(logp, n_keep - 1, axis=None)[:n_keep]]
        sig = np.flatnonzero((logp < log_limit) & (E - k - L >= min_snr ** 2))
        significant += [tile_at(i) for i in sig]

    strongest.sort(key=lambda tile: tile["log10_p_single"])
    best = strongest[0]
    p = float(min(1.0, 10.0 ** (best["log10_p_single"] + np.log10(n_tiles))))

    def spans(tile):
        return [tile["t_range"]] + ([tile["t_wrap"]] if tile["t_wrap"] else [])

    def in_time(t, tile):
        return any(a <= t <= b for a, b in spans(tile))

    k_full = np.rint(f * n_time * dt).astype(int)
    t_axis = np.arange(n_time) * dt

    def centre_width(tile):
        """Mean and sqrt(2) x spread over the box's times of its unexplained power at full resolution."""
        (c0, c1), n_b = tile["channel_index"], tile["resolution"]
        j0, j1 = tile["band_index"][0] * n_b, tile["band_index"][1] * n_b

        def power(X):
            full = np.zeros((c1 - c0, n_time), dtype=complex)
            full[:, k_full[j0:j1]] = X[c0:c1, j0:j1]
            return np.sum(np.abs(np.fft.ifft(full, axis=-1) * np.sqrt(n_time)) ** 2, axis=0)

        u = np.maximum(power(Z) - 2.0 * (c1 - c0) * (j1 - j0) / n_time - eps2 * power(Zm), 0.0)
        u = u * np.any([(t_axis >= a) & (t_axis <= b) for a, b in spans(tile)], axis=0)
        if u.sum() <= 0:
            a, b = tile["t_range"]
            return 0.5 * (a + b), (b - a) / np.sqrt(12.0)
        mean = np.sum(u * t_axis) / u.sum()
        return float(mean), float(np.sqrt(2.0 * np.sum(u * (t_axis - mean) ** 2) / u.sum()))

    # Step 1, which features are separate: smallest significant boxes first, each kept only if
    # clear of those kept
    found = []
    for tile in sorted(significant, key=lambda tile: (tile["n_cells"], tile["log10_p_single"])):
        if any(in_time(kept["centre"], tile) for kept in found):
            continue
        centre, width = centre_width(tile)
        if all(abs(centre - kept["centre"]) >= x_min * (width + kept["width"]) for kept in found):
            found.append(dict(tile, centre=centre, width=width))

    # Step 2, which box represents each feature: the most significant box whose time span holds
    # that feature's centre and no other feature's (the smallest box itself if none does)
    features = []
    for k, feature in enumerate(found):
        others = [f["centre"] for j, f in enumerate(found) if j != k]
        own = [tile for tile in significant
               if in_time(feature["centre"], tile) and not any(in_time(c, tile) for c in others)]
        best_box = min(own, key=lambda tile: tile["log10_p_single"]) if own else feature
        features.append(dict(best_box, centre=feature["centre"], width=feature["width"]))
    features.sort(key=lambda tile: tile["log10_p_single"])

    if features:
        tiles = features
    else:
        tiles = []                         # nothing significant: the strongest boxes, apart in time
        for tile in strongest:
            if not any(x0 < y1 and y0 < x1 for kept in tiles for x0, x1 in spans(tile) for y0, y1 in spans(kept)):
                tiles.append(tile)
            if len(tiles) == top:
                break

    add = bool(p < p_false_alarm and len(features) > 0)
    out = dict(add_pulse=add, p=p, n_tiles=n_tiles, n_significant=len(significant), tiles=tiles,
               epsilon=float(epsilon), p_false_alarm=float(p_false_alarm))

    if verbose:
        print(f"Residual test (tolerating {epsilon:.0%} model error): {'ADD A PULSE' if add else 'stop'} "
              f"(p = {p:.2g} after {n_tiles} boxes" + (f"; min_snr {min_snr:g}" if min_snr else "") + ")")
        if features:
            print(f"  {len(significant)} boxes significant on their own, making {len(features)} separate feature(s):")
        else:
            print("  nothing significant; strongest box:")
        for t in tiles if features else tiles[:1]:
            where = (f", centre {t['centre'] * 1e3:.2f} ms, width {t['width'] * 1e3:.2f} ms" if "centre" in t else "")
            print(f"    {_format_time(t)}, {t['channel_range'][0]:.0f}-{t['channel_range'][1]:.0f} MHz: "
                  f"unexplained SNR {t['snr_unexplained']:.1f} (residual {t['snr_residual']:.1f}, "
                  f"model {t['snr_model']:.1f}){where}")
    return out


def boxes_explained(burst, M_FD_old, M_FD_new, result, min_efficiency=0.5, min_change_snr=5.0,
                    verbose=True):
    """
    Did adding a pulse fit the leftover signal that called for it? For each significant box of an
    excess_power_search result, checks whether the change in the model between the old and the new
    fit lands on leftover signal in that box (the efficiency below) or on noise.

    burst:          the ConditionedData both fits used
    M_FD_old:       the old fit's model (the one the result was computed for)
    M_FD_new:       the new fit's model, normally with one more pulse
    result:         the dict excess_power_search returned for the OLD fit
    min_efficiency: a box counts as explained when the efficiency reaches this
    min_change_snr: ... and the model changed by at least this SNR within the box; below it the
                    efficiency is mostly noise, and the new fit did not touch the box anyway
    verbose:        print each box

    THE EFFICIENCY. With the whitened residuals r_old = d - M_old and r_new = d - M_new, and the
    model change Delta M = M_new - M_old (so r_new = r_old - Delta M), summed over the cells of box B:

        eta_B = ( sum_B |r_old|^2 - sum_B |r_new|^2 ) / sum_B |Delta M|^2
              = 2 Re sum_B conj(r_old) Delta M / sum_B |Delta M|^2  -  1
              = 2 a_B - 1,

    a_B being the least-squares amplitude of Delta M fitted to the old residual within B. If the
    change matches leftover signal in B, a_B ~ 1 and eta ~ 1: the residual power drops by as much
    power as the change adds. If the change lands on noise, a_B ~ 0 and eta ~ -1: the residual
    power RISES by the power added. It does not depend on what else is in the box: a box holding two
    unfitted pulses still gives eta ~ 1 when the new fit captures one of them. A pulse whose own
    core is misfit by a fraction e of its amplitude gives eta ~ (1 - e^2) / (1 + e^2), 0.83 for
    e = 0.3, so the default 0.5 admits cores misfit by up to ~55%. It checks, box by box, that the
    data want the new component at about the amplitude the whole fit gave it -- the idea of the
    chi^2 consistency test of gravitational-wave searches (Allen 2005, Phys. Rev. D 71, 062001).

    Intended use, in the sequential pulse search: accept the fit with n + 1 pulses only if, besides
    passing the Delta ELPD test, it explains at least one significant box of the n-pulse residual.
    A new pulse that fits a real feature explains that feature's box; one that only absorbs misfit
    the tolerance already excused (e.g. along a scattering tail) leaves the significant boxes
    untouched, and one laid across a feature it cannot fit (e.g. a single bright channel) scores
    low. Requiring SOME significant box, not the most significant, keeps a real pulse that sits
    behind an unfittable feature in the ranking. It cannot catch an absorber on a bright burst's
    tail whose misfit exceeds the tolerance, since that absorber genuinely fits leftover signal.
    plotting.plot_boxes_explained draws the boxes on both residuals and on the model change.

    Returns a dict: 'any_removed', and 'boxes', one entry per significant box of the old result,
    with its t_range, t_wrap, channel_range, channel_index, snr_before (unexplained, as in the
    search), change_snr
    (sqrt(sum_B |Delta M|^2)), efficiency and removed (explained).
    """
    log10_limit = np.log10(result["p_false_alarm"]) - np.log10(result["n_tiles"])
    sigma = np.broadcast_to(np.asarray(burst.sigma_noise_fd, dtype=float), np.shape(burst.data_fd))
    data = np.asarray(burst.data_fd)
    M_old, M_new = np.asarray(M_FD_old), np.asarray(M_FD_new)
    R_old = (data - M_old) / sigma
    D = (M_new - M_old) / sigma

    boxes = []
    for t in result["tiles"]:
        if not t.get("significant", t["log10_p_single"] < log10_limit):
            continue
        n_b = t["resolution"]
        (c0, c1), (b0, b1), (t0, t1) = t["channel_index"], t["band_index"], t["time_index"]
        r = _cells(R_old, n_b)[c0:c1, b0:b1, t0:t1]
        d = _cells(D, n_b)[c0:c1, b0:b1, t0:t1]
        change = float(np.sum(np.abs(d) ** 2))
        eff = (2.0 * float(np.real(np.sum(np.conj(r) * d))) / change - 1.0) if change > 0 else np.nan
        removed = change >= min_change_snr ** 2 and eff >= min_efficiency
        boxes.append(dict(t_range=t["t_range"], channel_range=t["channel_range"],
                          channel_index=t["channel_index"], t_wrap=t.get("t_wrap"),
                          snr_before=t["snr_unexplained"], change_snr=float(np.sqrt(change)),
                          efficiency=float(eff), removed=bool(removed)))

    out = dict(any_removed=any(b["removed"] for b in boxes), boxes=boxes)
    if verbose:
        if not boxes:
            print("boxes_explained: the old result has no significant box")
        for b in boxes:
            why = ("explained" if b["removed"] else
                   f"untouched (model change SNR < {min_change_snr:g})" if b["change_snr"] < min_change_snr
                   else "not explained")
            print(f"box {_format_time(b)}, "
                  f"{b['channel_range'][0]:.0f}-{b['channel_range'][1]:.0f} MHz (SNR {b['snr_before']:.1f}): "
                  f"model change SNR {b['change_snr']:.1f}, efficiency {b['efficiency']:+.2f} -> {why}")
        print(f"  -> {'the new pulse fits leftover signal' if out['any_removed'] else 'the new pulse fits none of the leftover signal'}")
    return out
