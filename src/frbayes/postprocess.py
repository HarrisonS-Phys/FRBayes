"""
postprocess.py - Tools applied to a fit after it has run.

Posterior draws: order_pulses relabels the pulses of every draw consistently (the likelihood is a
sum over unordered pulses, so labels can differ between chains); pointwise_log_lik gives the
log-likelihood of every observation for every draw, the input to leave-one-out cross-validation
(loo, PSIS-LOO), which compares fits of the same data; and combine_chains groups chains that sampled the same mode (group_chains) and returns
the draws of the mode with the highest posterior peak. Residuals: excess_power_search looks for
signal in a fit's residual that neither noise nor a tolerated model error explains, and
boxes_explained checks whether a new pulse fitted that leftover signal; the iterative pulse search
(run.run_maxL_iterative) uses both to decide when to stop.
"""

import numpy as np
from scipy.special import logsumexp


# Sites recorded once per PAIR of pulses, in the pair order of np.triu_indices(n_pulses, k=1)
PAIR_SITES = ("toa_separation",)

# Parameters single_pulse_fn may read (as plotting.MODEL_PARAMS); 'A' is the fluence, which the
# likelihood records with either amplitude prior
MODEL_PARAMS = ("mean_arrival_time", "sigma", "DM", "tau_ref", "gamma", "beta", "delta", "A")

# Per-pulse sites compared between chains when grouping them into modes (those present are used)
GROUPING_SITES = ("mean_arrival_time", "sigma", "tau_ref", "DM", "A_snr", "A", "delta", "gamma", "beta")


# -------------------------------------------------------------------
# 1. Relabelling pulses
# -------------------------------------------------------------------

def order_pulses(samples, by="mean_arrival_time", descending=False, pair_sites=PAIR_SITES, exclude=(),
                 return_perm=False, verbose=True):
    """
    Relabels the pulses of every draw by sorting them on one per-pulse site, applying the same
    permutation to every per-pulse and per-pair site, so each pulse keeps all its own parameters and
    the model, a sum over pulses, is unchanged. A site is per pulse if its first axis after the
    sample axes has length n_pulses, per pair if it is named in pair_sites and that axis has length
    n_pulses (n_pulses - 1) / 2, and is left unchanged otherwise (e.g. '<name>_shared' scalars).
    Sorting cannot merge different solutions: chains that found different sets of pulses still
    disagree afterwards. Returns a new dict, and the permutation (new pulse i of a draw is old pulse
    perm[..., i]) if return_perm.

    samples:     posterior arrays shaped (draws, ...) or (chains, draws, ...), e.g. mcmc.get_samples(group_by_chain=True)
    by:          the per-pulse site to sort on; the arrival time by default
    descending:  sort from largest to smallest (e.g. by="A_snr", descending=True: brightest first)
    pair_sites:  names of per-pair sites, named explicitly because with three pulses a per-pair site has the length of a per-pulse one
    exclude:     names of sites to leave untouched, for an axis that only happens to have length n_pulses
    return_perm: also return the permutation, shape (..., n_pulses)
    verbose:     print the ordering and any sites left unchanged
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
    perm = np.argsort(key, axis=-1, kind="stable")    # new pulse i is old pulse perm[..., i]

    # Which old pair each new pair of pulses was
    n_pairs = n_pulses * (n_pulses - 1) // 2
    I, J = np.triu_indices(n_pulses, k=1)
    pair_of = np.full((n_pulses, n_pulses), -1)
    pair_of[I, J] = np.arange(n_pairs)                 # old pair index of old pulses (a, b), a < b
    pa, pb = perm[..., I], perm[..., J]                # old pulses behind each new pair
    old_pair = pair_of[np.minimum(pa, pb), np.maximum(pa, pb)]

    def expand(index, x, axis):
        """Broadcasts an index over the sample axes to the shape take_along_axis needs for x."""
        shape = index.shape + (1,) * (x.ndim - axis - 1)
        return np.broadcast_to(index.reshape(shape), x.shape[:axis] + index.shape[axis:] + x.shape[axis + 1:])

    out, done = {}, {"per pulse": [], "per pair": [], "unchanged": []}
    for name, x in samples.items():
        sampled = x.ndim > lead and x.shape[:lead] == key.shape[:lead]
        if name in exclude or not sampled:
            out[name] = x.copy(); done["unchanged"].append(name)
        elif name in pair_sites and x.shape[lead] == n_pairs:
            out[name] = np.take_along_axis(x, expand(old_pair, x, lead), axis=lead)
            done["per pair"].append(name)
        elif x.shape[lead] == n_pulses:
            out[name] = np.take_along_axis(x, expand(perm, x, lead), axis=lead)
            done["per pulse"].append(name)
        else:
            out[name] = x.copy(); done["unchanged"].append(name)

    if verbose:
        print(f"order_pulses: {n_pulses} pulses ordered by {by} ({'descending' if descending else 'ascending'})"
              + (f"; left unchanged: {', '.join(done['unchanged'])}" if done["unchanged"] else ""))
    return (out, perm) if return_perm else out


# -------------------------------------------------------------------
# 2. Pointwise log-likelihood and PSIS-LOO
# -------------------------------------------------------------------

def pointwise_log_lik(burst, single_pulse_fn, samples, thin=1, chunk=64):
    """
    The log-likelihood of every observation, one channel at one retained Fourier bin, for every
    (thinned) draw of every chain, shape (chains, kept draws, observations). Each complex bin has
    independent real and imaginary parts of standard deviation sigma, so
    log p(X | M) = -|X - M|^2 / (2 sigma^2) - log(2 pi sigma^2), the per-bin version of the fit's
    likelihood with its normalization.

    burst:           the ConditionedData the fit used
    single_pulse_fn: the model closure the fit used (padded or not)
    samples:         posterior samples grouped by chain, mcmc.get_samples(group_by_chain=True), including 'A'
    thin:            keep every thin-th draw
    chunk:           draws evaluated at once; lower it if memory runs out
    """
    import jax
    import jax.numpy as jnp

    if "A" not in samples:
        raise KeyError("samples has no 'A' (the fluence), which likelihood.FD_gaussianlikelihood records "
                       "with either amplitude prior.")
    s = {k: np.asarray(v) for k, v in samples.items()}
    if s["mean_arrival_time"].ndim != 3:
        raise ValueError("samples must be grouped by chain: mcmc.get_samples(group_by_chain=True).")
    n_chains, n_draws, n_pulses = s["mean_arrival_time"].shape
    keep = np.arange(0, n_draws, thin)

    def per_pulse(v):
        """The kept draws of one site, flattened over chains, as (draws, n_pulses)."""
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
        """Log-likelihood of every observation for a batch of draws, (batch, observations)."""
        M = jax.vmap(lambda q: jnp.sum(jax.vmap(single_pulse_fn)(q), axis=0))(p)   # (B, n_chan, n_fourier)
        ll = -0.5 * jnp.abs(X[None] - M) ** 2 / var[None] + const[None]
        return ll.reshape(ll.shape[0], -1)

    n_total = n_chains * keep.size
    out = []
    for start in range(0, n_total, chunk):
        stop = min(start + chunk, n_total)
        p = {k: jnp.asarray(v[start:stop]) for k, v in params.items()}
        if stop - start < chunk:                          # pad the last chunk so batch compiles once
            p = {k: jnp.concatenate([v, jnp.repeat(v[-1:], chunk - (stop - start), axis=0)]) for k, v in p.items()}
        out.append(np.asarray(batch(p))[: stop - start])
    return np.concatenate(out).reshape(n_chains, keep.size, -1)


def _psis_log_weights(log_lik_draws, r_eff):
    """
    Normalized leave-one-out log importance weights, shape (observations, draws), and the Pareto
    k-hat of each observation, by ArviZ's Pareto-smoothed importance sampling. ArviZ raises for the
    whole set if any observation's largest log-likelihoods are all equal, which happens in bins where
    every draw's model is ~0; such observations have bounded weights, so they get truncated
    importance sampling (Ionides 2008) and k-hat 0 instead. Where PSIS fails numerically, they also
    get truncated importance sampling, with k-hat inf.

    log_lik_draws: shape (draws, observations)
    r_eff:         relative efficiency of the draws (ESS / draws)
    """
    from arviz_stats.base import array_stats

    raw = -log_lik_draws.T                                                    # LOO log ratios, (obs, draws)
    n_obs, n = raw.shape
    cap = logsumexp(raw, axis=-1) - np.log(n) + 0.5 * np.log(n)               # mean weight x sqrt(n)
    tis = np.minimum(raw, cap[:, None])

    lw, khat = tis.copy(), np.zeros(n_obs)
    n_tail = int(np.ceil(min(0.2 * n, 3.0 * np.sqrt(n / r_eff))))
    usable = np.ptp(np.sort(raw, axis=-1)[:, -n_tail:], axis=-1) > 1e-4      # tail spread PSIS can fit
    if usable.any():
        try:
            with np.errstate(all="ignore"):
                # psislw takes the LOG-LIKELIHOOD and negates it itself to form the LOO log ratios
                lw_u, k_u = array_stats.psislw(log_lik_draws[:, usable], r_eff=r_eff, axis=0)
            lw_u, k_u = np.array(lw_u, dtype=float), np.array(k_u, dtype=float)
            bad = ~(np.all(np.isfinite(lw_u), axis=-1) & np.isfinite(k_u))
            lw_u[bad], k_u[bad] = tis[usable][bad], np.inf                    # PSIS failed here: k-hat inf
        except ValueError:                                                     # PSIS failed for all of them
            lw_u, k_u = tis[usable], np.full(usable.sum(), np.inf)
        lw[usable], khat[usable] = lw_u, k_u
    return lw - logsumexp(lw, axis=-1)[:, None], khat


def loo(log_lik, r_eff=1.0):
    """
    Leave-one-out cross-validation (PSIS-LOO) of a NUTS fit: how well the fit, made without each
    observation (one channel at one retained Fourier bin), predicts it, log p(y_i | y_-i), estimated
    from the posterior draws by Pareto-smoothed importance sampling. The sum over observations, the
    ELPD, compares fits of the same data (e.g. different numbers of pulses): the higher ELPD
    predicts better, and a difference is meaningful only if large compared with its standard error,
    sqrt(n) times the standard deviation over observations of the difference of their pointwise
    values. Observations whose largest log-likelihoods are equal in every draw, where ArviZ's
    az.loo would refuse the whole fit, get truncated importance sampling instead (_psis_log_weights).
    Returns a dict: 'elpd', 'se' (its standard error), 'pointwise' (log p(y_i | y_-i) per
    observation), 'khat' (Pareto k-hat per observation) and 'frac_khat_bad' (the fraction above
    0.7, where the estimate for that observation is unreliable).

    log_lik: pointwise_log_lik's output, shape (chains, draws, observations), or (draws, observations)
    r_eff:   relative efficiency of the draws (ESS / draws); 1 is fine for thinned draws
    """
    log_lik = np.asarray(log_lik, dtype=float)
    log_lik = log_lik.reshape(-1, log_lik.shape[-1])                          # (draws, observations)
    lw, khat = _psis_log_weights(log_lik, r_eff)
    pointwise = logsumexp(lw + log_lik.T, axis=-1)
    return dict(elpd=float(pointwise.sum()), se=float(np.sqrt(pointwise.size) * np.std(pointwise, ddof=1)),
                pointwise=pointwise, khat=khat, frac_khat_bad=float(np.mean(khat > 0.7)))


# -------------------------------------------------------------------
# 3. Grouping chains into modes
# -------------------------------------------------------------------

def _split_rank_rhat(x):
    """
    Rank-normalized split R-hat (Vehtari, Gelman, Simpson, Carpenter & Buerkner 2021) of x, shape
    (chains, draws): each chain is split in half, the draws are replaced by the normal scores of their
    ranks, and the classic R-hat is taken of those (bulk) and of the scores of |x - median| (folded,
    which catches chains that differ in spread); the larger is returned. A split chain that never
    moves gives inf; fewer than 8 draws per chain gives NaN.

    x: draws of one quantity, shape (chains, draws)
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
        """Classic R-hat of u, shape (chains, draws)."""
        k = u.shape[1]
        W = u.var(axis=1, ddof=1).mean()
        if W <= 0.0:
            return np.inf
        B = k * u.mean(axis=1).var(ddof=1)
        return float(np.sqrt(((k - 1) / k * W + B / k) / W))

    def scores(u):
        """Normal scores of the ranks of u."""
        r = rankdata(u, method="average").reshape(u.shape)
        return ndtri((r - 0.375) / (u.size + 0.25))

    return max(rhat(scores(y)), rhat(scores(np.abs(y - np.median(y)))))


def group_chains(samples, log_lik=None, rhat_max=1.1, sites=GROUPING_SITES, verbose=True):
    """
    Groups chains that sampled the same mode. The pulses are first labelled consistently
    (order_pulses, by arrival time); then for every pair of chains the rank-normalized split R-hat
    (_split_rank_rhat) is computed for every per-pulse parameter (once for a parameter shared by all
    pulses) and for the total log-likelihood, and the pair is linked if the largest is below
    rhat_max. The modes are the connected groups of linked chains. Between two chains, R-hat is about
    sqrt(1 + d^2 / 4) for posteriors offset by d standard deviations, so 1.1 links chains within
    about 0.9 standard deviations; it is looser than the 1.01 used to declare convergence because the
    question is coarser. Returns (modes, rhat): lists of chain indices, best mean log-likelihood first
    when log_lik is given, and the largest R-hat of each pair, shape (chains, chains).

    samples:  posterior samples grouped by chain, mcmc.get_samples(group_by_chain=True)
    log_lik:  pointwise_log_lik for the same chains, whose total per draw is compared too; None leaves it out
    rhat_max: two chains are the same mode if every compared quantity has R-hat below this
    sites:    per-pulse sites compared (those present)
    verbose:  print the modes found
    """
    s = order_pulses({k: np.asarray(v) for k, v in samples.items()}, verbose=False)
    if s["mean_arrival_time"].ndim != 3:
        raise ValueError("samples must be grouped by chain: mcmc.get_samples(group_by_chain=True).")
    n_chains, _, n_pulses = s["mean_arrival_time"].shape

    # (name, values of shape (chains, draws)) for every quantity compared
    quantities = []
    for k in sites:
        if k not in s or s[k].ndim != 3 or s[k].shape[2] != n_pulses:
            continue
        v = s[k]
        if np.allclose(v, v[..., :1]):
            quantities.append((k, v[..., 0]))                          # shared by every pulse
        else:
            quantities += [(k, v[..., p]) for p in range(n_pulses)]
    if log_lik is not None:
        L = np.asarray(log_lik, dtype=float)
        if L.shape[0] != n_chains:
            raise ValueError(f"log_lik has {L.shape[0]} chains, samples {n_chains}.")
        quantities.append(("log_lik", L.sum(axis=2)))

    rhat = np.zeros((n_chains, n_chains))
    for i in range(n_chains):
        for j in range(i + 1, n_chains):
            worst = 0.0
            for _, v in quantities:
                r = _split_rank_rhat(v[[i, j]])
                worst = max(worst, np.inf if np.isnan(r) else r)
            rhat[i, j] = rhat[j, i] = worst

    # Connected groups of linked chains (union-find)
    parent = list(range(n_chains))

    def root(a):
        """The representative chain of a's group."""
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


def combine_chains(samples, log_lik, potential_energy, rhat_max=1.1, near_tie=10.0, verbose=True):
    """
    One posterior from chains that may not have mixed: chains are grouped into modes
    (group_chains), the modes are ranked by their highest log posterior over their draws, and every
    draw of the best mode's chains is returned. With more than one mode a warning says that the
    choice ignores each mode's width, i.e. its posterior mass, and a second warning is printed
    when the runner-up's peak is within near_tie nats of the best. Returns (combined, info):
    a flat dict of arrays shaped (draws, ...), pulses labelled by order_pulses, and a dict with
    'modes' (chain indices, best first), 'peak_log_post' and 'mean_log_lik' (per mode, relative to
    the best), 'gap' (nats from the best peak to the runner-up's, inf with one mode), 'near_tie',
    'rhat' (from group_chains), and 'chain' and 'draw' (where each returned draw came from, as
    indices into samples).

    samples:          posterior samples grouped by chain, mcmc.get_samples(group_by_chain=True)
    log_lik:          pointwise_log_lik for the same chains, shape (chains, kept draws, observations)
    potential_energy: minus the log posterior of every draw, shape (chains, draws), from mcmc.get_extra_fields(group_by_chain=True)["potential_energy"]
    rhat_max:         passed to group_chains
    near_tie:         warn when the runner-up's peak log posterior is within this many nats of the best
    verbose:          print each mode's chains, peak and mean log-likelihood, and the draws returned
    """
    s = order_pulses({k: np.asarray(v) for k, v in samples.items()}, verbose=False)
    L = np.asarray(log_lik, dtype=float)
    U = np.asarray(potential_energy, dtype=float)
    n_chains, n_per = s["mean_arrival_time"].shape[:2]
    if U.shape[:2] != (n_chains, n_per):
        raise ValueError(f"potential_energy has shape {U.shape}; expected ({n_chains}, {n_per}).")
    modes, rhat = group_chains(s, L, rhat_max=rhat_max, verbose=False)

    # Rank modes by their highest log posterior, best first
    peak = np.array([-U[m].min() for m in modes])
    order = np.argsort(-peak)
    modes, peak = [modes[i] for i in order], peak[order] - peak.max()
    mean_ll = np.array([L[m].sum(axis=2).mean() for m in modes])
    mean_ll -= mean_ll[0]
    gap = float(-peak[1]) if len(modes) > 1 else np.inf
    near = gap < near_tie

    best = modes[0]
    chain = np.repeat(np.asarray(best), n_per)
    draw = np.tile(np.arange(n_per), len(best))
    combined = {k: v[chain, draw] for k, v in s.items() if v.ndim >= 2 and v.shape[:2] == (n_chains, n_per)}

    if verbose:
        print(f"combine_chains: {n_chains} chains -> {len(modes)} mode(s) (pairs linked where every "
              f"R-hat < {rhat_max:g})")
        for m, chains in enumerate(modes):
            within = max((rhat[a, b] for a in chains for b in chains if a < b), default=0.0)
            print(f"  mode {m}: chains {chains}" + (f" (largest R-hat within {within:.3f})" if len(chains) > 1 else "")
                  + f", peak log post {peak[m]:+.1f}, mean log-lik {mean_ll[m]:+.1f}"
                  + (f" -> kept, {len(chain)} draws" if m == 0 else ""))
    if len(modes) > 1:
        print("warning: mode 0 was chosen by its peak log posterior alone, ignoring each mode's width and so "
              "its posterior mass. TO-DO: compare the modes' posterior masses, probably by bridge sampling.")
    if near:
        print(f"warning: the runner-up mode's peak is only {gap:.1f} nats below the best (< {near_tie:g}), so "
              f"the modes' widths could reverse their ranking; inspect both.")
    info = dict(modes=modes, peak_log_post=peak, mean_log_lik=mean_ll, gap=gap, near_tie=bool(near),
                rhat=rhat, chain=chain, draw=draw)
    return combined, info


# -------------------------------------------------------------------
# 4. Excess-power search of the residual (Anderson, Brady, Creighton & Flanagan 2001)
# -------------------------------------------------------------------

def _format_time(box):
    """
    A box's time extent in ms.

    box: a box dict from excess_power_search or boxes_explained
    """
    return f"{box['t_range'][0] * 1e3:.2f}-{box['t_range'][1] * 1e3:.2f} ms"


def _dyadic_ranges(n, min_width=1):
    """
    Contiguous index ranges [start, stop) of widths min_width, 2 min_width, 4 min_width, ... (and n),
    each width at starts spaced by half the width, so every interval at least min_width long is
    covered by a range at most about twice as long. Returns (starts, stops).

    n:         length of the axis
    min_width: narrowest range
    """
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
    """
    Complex cells, shape (channels, bands, times): the retained bins of each channel cut into bands
    of n_b consecutive bins, each band inverse-DFT'd (unitary) into n_b time cells; a final partial
    band is left out. Bin q of a band is first multiplied by exp(i pi q / n_b), a half-cell time
    shift, so cell j is the band's signal at (j + 1/2) W / n_b and covers [j W / n_b, (j + 1) W / n_b]
    of the segment of length W, with no cell straddling its ends. The transform is unitary, so sums
    of products over cells equal the same sums over the Fourier bins they come from.

    X:   whitened Fourier-domain array, shape (channels, retained bins)
    n_b: bins per band
    """
    n_chan, n_ret = X.shape
    n_bands = n_ret // n_b
    bands = X[:, : n_bands * n_b].reshape(n_chan, n_bands, n_b)
    half_cell = np.exp(1j * np.pi * np.arange(n_b) / n_b)
    return np.fft.ifft(bands * half_cell, axis=-1) * np.sqrt(n_b)


def _cell_powers(X, n_b):
    """
    Cell powers |z|^2 of _cells.

    X:   whitened Fourier-domain array, shape (channels, retained bins)
    n_b: bins per band
    """
    return np.abs(_cells(X, n_b)) ** 2


def min_box_channels(burst, priors):
    """
    The fewest adjacent channels a residual box may span: the narrowest band in which the spectral
    prior lets one pulse put its power. The running power law ln F = gamma x + beta x^2,
    x = ln(nu / nu_ref), is a Gaussian in x for beta < 0, and F^2 then has full width at half maximum
    FWHM_x = sqrt(2 ln 2) / sqrt|beta_min| for the most negative beta allowed. Returns the fewest
    channel centres inside a window [nu, nu exp(FWHM_x)] over every start channel whose window fits in
    the band (at least 1; 1 if beta is not bounded below zero). Spectra so steep that a pulse's power
    sits in a sliver at the band edge are not covered, but such a sliver still falls inside boxes of
    this width at that edge.

    burst:  the ConditionedData of the burst, for the channel frequencies
    priors: the priors of the fit (run.set_priors), for the lower bound of 'beta'
    """
    if "beta" not in priors:
        return 1
    beta_min = float(priors["beta"].low)
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
    explains? An excess-power search (Anderson, Brady, Creighton & Flanagan 2001) over boxes in
    time, Fourier frequency and channel. The residual and the model are whitened by sigma_noise_fd;
    for each resolution n_b the retained bins are cut into bands of n_b bins, each inverse-DFT'd into
    n_b time cells, so a residual cell's power is chi^2 with 2 degrees of freedom under noise and a
    model cell's power is its SNR^2. A box of V cells with residual power E is tested against noise
    plus the largest tolerated misfit, L = epsilon^2 x (the model's power in the box): its p-value
    p_single = P(E' >= E) for E' non-central chi^2 with 2V degrees of freedom and non-centrality L
    (scipy.stats.ncx2, exact), and a box is significant if p_single times the number
    of boxes (Bonferroni, conservative since boxes overlap) is below p_false_alarm and its unexplained
    SNR, sqrt(max(E - 2V - L, 0)), reaches min_snr.

    Significant boxes are then reduced to separate features: taken fewest cells first (ties to the
    smaller p-value, then to the larger unexplained SNR, since p-values below ~1e-308 underflow to
    0), a box is a new feature only if no kept feature's centre lies in its time span and its centre
    c and width w are clear of every kept feature's, |c - c_k| >= x_min (w + w_k), the repulsion
    wall's condition (channels play no part, since every pulse spans the band); c and w are the mean
    and sqrt(2) x the standard deviation over time of the box's unexplained power at the data's
    sample spacing. Each feature is represented by the most significant box holding its centre and
    no other feature's. A box spanning the whole segment says nothing about where its excess is, so
    it neither starts nor represents a feature unless no other box is significant.

    burst:         the ConditionedData the fit used
    M_FD:          the fitted model on the burst's retained Fourier bins (plotting.model_from_draw)
    epsilon:       tolerated model error as a fraction of the model's amplitude (leftover power up to epsilon^2 times the model's is excused)
    min_snr:       unexplained SNR a box must also reach to be significant; 0 decides on the p-value alone
    resolutions:   bins per band n_b, each a tiling with cells of duration W / n_b; None uses 1, 2, 4, ... up to all retained bins
    p_false_alarm: false-alarm probability of the whole search, after the trials correction
    x_min:         separation of two features in units of their summed widths; use the likelihood's repulsion x_min
    min_channels:  fewest adjacent channels a box may span (min_box_channels)
    top:           when nothing is significant, how many of the strongest boxes to return for display
    verbose:       print the outcome

    Returns a dict: 'add_pulse' (p < p_false_alarm and some box significant), 'p' (corrected),
    'n_tiles', 'n_significant', 'tiles' (the separate features, most significant first, or if there
    are none the `top` strongest boxes, apart in time; each with t_range (s), channel_range (MHz),
    channel_index, band_index, time_index, fourier_band (Hz), significant, n_cells, snr_unexplained,
    snr_residual, snr_model, log10_p_single, resolution, and for features centre and width (s)),
    'epsilon' and 'p_false_alarm'. plotting.plot_residual_tiles draws the boxes.
    """
    from scipy.stats import ncx2

    sigma = np.broadcast_to(np.asarray(burst.sigma_noise_fd, dtype=float), np.shape(burst.data_fd))
    Z = (np.asarray(burst.data_fd) - np.asarray(M_FD)) / sigma          # whitened residual
    Zm = np.asarray(M_FD) / sigma                                       # whitened model
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

    def table(P):
        """Summed-volume table of cell powers, so any box's sum takes eight lookups."""
        S = np.zeros((P.shape[0] + 1, P.shape[1] + 1, P.shape[2] + 1))
        S[1:, 1:, 1:] = P.cumsum(0).cumsum(1).cumsum(2)
        return S

    ranges = {n_b: (_dyadic_ranges(n_chan, min_channels), _dyadic_ranges(n_ret // n_b), _dyadic_ranges(n_b))
              for n_b in resolutions}
    n_tiles = sum(len(c[0]) * len(b[0]) * len(t[0]) for c, b, t in ranges.values())
    log_limit = np.log(p_false_alarm) - np.log(n_tiles)               # significant on its own

    def make_tile(n_b, c0, c1, b0, b1, t0, t1, e, v, lam, pm, logp):
        """The dict describing one box."""
        cell = W / n_b                                                 # cell m covers [m, m + 1) cells
        k0, k1 = b0 * n_b, b1 * n_b
        return dict(
            t_range=(t0 * cell, t1 * cell),
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
        S, Sm = table(_cell_powers(Z, n_b)), table(_cell_powers(Zm, n_b))
        c0, c1 = cs[:, None, None], ce[:, None, None]
        b0, b1 = bs[None, :, None], be[None, :, None]
        t0, t1 = ts[None, None, :], te[None, None, :]

        def box(S):
            """Sum over every box of this resolution, from the summed-volume table."""
            return (S[c1, b1, t1] - S[c0, b1, t1] - S[c1, b0, t1] - S[c1, b1, t0]
                    + S[c0, b0, t1] + S[c0, b1, t0] + S[c1, b0, t0] - S[c0, b0, t0])

        E, Pm = box(S), np.maximum(box(Sm), 0.0)
        V = np.broadcast_to((c1 - c0) * (b1 - b0) * (t1 - t0), E.shape)
        L = eps2 * Pm                                                  # tolerated misfit power
        k = 2.0 * V
        logp = ncx2.logsf(E, k, L)                                     # E ~ non-central chi^2_k(L)

        def tile_at(i):
            """The box at flat index i of this resolution's arrays."""
            ci, bi, ti = np.unravel_index(i, logp.shape)
            return make_tile(n_b, cs[ci], ce[ci], bs[bi], be[bi], ts[ti], te[ti],
                             E[ci, bi, ti], V[ci, bi, ti], L[ci, bi, ti], Pm[ci, bi, ti], logp[ci, bi, ti])

        n_keep = min(20 * top, logp.size)
        strongest += [tile_at(i) for i in np.argpartition(logp, n_keep - 1, axis=None)[:n_keep]]
        sig = np.flatnonzero((logp < log_limit) & (E - k - L >= min_snr ** 2))
        significant += [tile_at(i) for i in sig]

    def rank(tile):
        """Most significant first: smaller p-value, then (where p-values underflow to 0) larger unexplained SNR."""
        return tile["log10_p_single"], -tile["snr_unexplained"]

    strongest.sort(key=rank)
    best = strongest[0]
    p = float(min(1.0, 10.0 ** (best["log10_p_single"] + np.log10(n_tiles))))

    def in_time(t, tile):
        """Whether time t lies inside the box's time span."""
        a, b = tile["t_range"]
        return a <= t <= b

    k_full = np.rint(f * n_time * dt).astype(int)
    t_axis = np.arange(n_time) * dt

    def centre_width(tile):
        """Mean and sqrt(2) x spread over the box's times of its unexplained power at the data's sample spacing."""
        (c0, c1), n_b = tile["channel_index"], tile["resolution"]
        j0, j1 = tile["band_index"][0] * n_b, tile["band_index"][1] * n_b

        def power(X):
            full = np.zeros((c1 - c0, n_time), dtype=complex)
            full[:, k_full[j0:j1]] = X[c0:c1, j0:j1]
            return np.sum(np.abs(np.fft.ifft(full, axis=-1) * np.sqrt(n_time)) ** 2, axis=0)

        # Residual power less its noise expectation and the tolerated misfit, clipped at 0
        u = np.maximum(power(Z) - 2.0 * (c1 - c0) * (j1 - j0) / n_time - eps2 * power(Zm), 0.0)
        a, b = tile["t_range"]
        u = u * ((t_axis >= a) & (t_axis <= b))
        if u.sum() <= 0:
            return 0.5 * (a + b), (b - a) / np.sqrt(12.0)
        mean = np.sum(u * t_axis) / u.sum()
        return float(mean), float(np.sqrt(2.0 * np.sum(u * (t_axis - mean) ** 2) / u.sum()))

    # Boxes that may start or represent a feature: not those spanning the whole segment, whose centre
    # says nothing about where their excess is, unless no other box is significant
    located = [tile for tile in significant if tile["time_index"] != (0, tile["resolution"])] or significant

    # Which features are separate: significant boxes with the fewest cells first (ties by rank), each
    # kept only if clear of those kept
    found = []
    for tile in sorted(located, key=lambda tile: (tile["n_cells"],) + rank(tile)):
        if any(in_time(kept["centre"], tile) for kept in found):
            continue
        centre, width = centre_width(tile)
        if all(abs(centre - kept["centre"]) >= x_min * (width + kept["width"]) for kept in found):
            found.append(dict(tile, centre=centre, width=width))

    # Which box represents each feature: the most significant box holding its centre and no other
    # feature's (the feature's own box if none does)
    features = []
    for k, feature in enumerate(found):
        others = [f["centre"] for j, f in enumerate(found) if j != k]
        own = [tile for tile in located
               if in_time(feature["centre"], tile) and not any(in_time(c, tile) for c in others)]
        best_box = min(own, key=rank) if own else feature
        features.append(dict(best_box, centre=feature["centre"], width=feature["width"]))
    features.sort(key=rank)

    if features:
        tiles = features
    else:
        tiles = []                         # nothing significant: the strongest boxes, apart in time
        for tile in strongest:
            (x0, x1) = tile["t_range"]
            if not any(x0 < kept["t_range"][1] and kept["t_range"][0] < x1 for kept in tiles):
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
            snr = ", ".join(f"{t['snr_unexplained']:.1f}" for t in tiles)
            print(f"  {len(features)} feature(s) from {len(significant)} significant boxes; unexplained SNR [{snr}]")
        elif tiles:
            t = tiles[0]
            print(f"  nothing significant; strongest box {_format_time(t)}, "
                  f"{t['channel_range'][0]:.0f}-{t['channel_range'][1]:.0f} MHz: unexplained SNR "
                  f"{t['snr_unexplained']:.1f}")
    return out


def boxes_explained(burst, M_FD_old, M_FD_new, result, min_gain, verbose=True):
    """
    Did adding a pulse fit the leftover signal that called for it? For each significant box of the
    old fit's excess_power_search, with whitened old residual r and model change D = M_new - M_old
    in the box's cells, the drop in leftover power is drop = sum |r|^2 - sum |r - D|^2
    = 2 Re sum conj(r) D - sum |D|^2, the box's share of the chi^2 gain (twice the log-likelihood
    gain), measured with the search's own power statistic. A box is explained if its share of the
    log-likelihood gain, drop / 2, is at least min_gain, the same threshold the iterative search
    applies to the whole fit, so the significant box alone must account for the gain a new pulse has to
    earn; for a change unrelated to the leftover the drop averages -sum |D|^2 with spread
    2 sqrt(sum |D|^2), so noise rarely passes. The iterative search accepts a new pulse only if it
    explains at least one box. Returns a dict: 'any_removed', and 'boxes', each with
    t_range, channel_range, channel_index, snr_before (unexplained SNR in the search), gain
    (drop / 2), change_snr (sqrt(sum |D|^2)) and removed (explained).

    burst:        the ConditionedData both fits used
    M_FD_old:     the old fit's model, the one `result` was computed for
    M_FD_new:     the new fit's model, normally with one more pulse
    result:       the dict excess_power_search returned for the old fit
    min_gain:     log-likelihood gain a box needs to count as explained; pass the iterative search's threshold
    verbose:      print the gain in each box and how many are explained
    """
    log10_limit = np.log10(result["p_false_alarm"]) - np.log10(result["n_tiles"])
    sigma = np.broadcast_to(np.asarray(burst.sigma_noise_fd, dtype=float), np.shape(burst.data_fd))
    data = np.asarray(burst.data_fd)
    M_old, M_new = np.asarray(M_FD_old), np.asarray(M_FD_new)
    R_old = (data - M_old) / sigma                                      # whitened old residual
    D = (M_new - M_old) / sigma                                         # whitened model change

    boxes = []
    for t in result["tiles"]:
        if not t.get("significant", t["log10_p_single"] < log10_limit):
            continue
        n_b = t["resolution"]
        (c0, c1), (b0, b1), (t0, t1) = t["channel_index"], t["band_index"], t["time_index"]
        r = _cells(R_old, n_b)[c0:c1, b0:b1, t0:t1]
        d = _cells(D, n_b)[c0:c1, b0:b1, t0:t1]
        change = float(np.sum(np.abs(d) ** 2))
        drop = float(np.sum(np.abs(r) ** 2) - np.sum(np.abs(r - d) ** 2))
        boxes.append(dict(t_range=t["t_range"], channel_range=t["channel_range"],
                          channel_index=t["channel_index"], snr_before=t["snr_unexplained"], gain=drop / 2.0,
                          change_snr=float(np.sqrt(change)), removed=bool(drop / 2.0 >= min_gain)))

    out = dict(any_removed=any(b["removed"] for b in boxes), boxes=boxes)
    if verbose:
        if not boxes:
            print("boxes_explained: the old result has no significant box")
        else:
            gains = ", ".join(f"{b['gain']:.1f}" for b in boxes)
            print(f"  ln L gain per box [{gains}] (need {min_gain:g}): "
                  f"{sum(b['removed'] for b in boxes)} of {len(boxes)} explained")
    return out
