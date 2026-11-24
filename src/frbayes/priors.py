"""
priors.py - Prior bounds for the Fourier-domain FRB model, derived from the data window.

These functions compute numbers rather than register sampling sites; run.set_priors turns them into
distributions. Keeping them separate makes them testable without a sampler, which matters because a
mistake here is a silently wrong prior rather than a crash. The bounds are deliberately loose: each
lets a single parameter, at its bound, just fit inside the window, and nothing constrains the
combination. With the padded model (model.setup_model(..., model_duration_pad=...)) they express the
assumption that the window contains the pulse, and model_duration_pad_for_priors gives the padding
that holds all the emission they allow outside it.
"""

import warnings

import numpy as np


def _bound(distribution, name, side):
    """
    A finite bound of a prior, e.g. of a Uniform; raises if it has none.

    distribution: the prior
    name:         the parameter's name, for the error message
    side:         "low" or "high"
    """
    value = getattr(distribution, side, None)
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"the '{name}' prior has no finite {'lower' if side == 'low' else 'upper'} bound.")
    return float(value)


def width_prior_bounds(burst, single_pulse_fn, toa, delta_prior=None, n_efold_tau=1.0, n_sigma_stds=3,
                       nu_bot=None, verbose=True):
    """
    Bounds for the intrinsic width sigma and the scattering time tau_ref, assuming the pulse arrives
    at `toa`. sigma's floor is the narrowest width the retained Fourier bins resolve, 1 / (2 pi f_max),
    below which the template has not rolled off and the data do not constrain it; its ceiling lets
    n_sigma_stds widths fit between toa and the nearer edge of the window. tau_ref's ceiling lets
    n_efold_tau e-folds of the tail at the bottom of the band, tau_bot = tau_ref (nu_bot / nu_ref)^delta,
    fit between toa and the end of the window. When delta is sampled, the steepest (most negative)
    delta of its prior is used, so the tail fits for every allowed delta, with a warning that this
    also excludes some tau_ref values that would fit for shallower delta. Returns a dict with
    sigma_min, sigma_max, tau_ref_min (0) and tau_ref_max (s).

    burst:           the ConditionedData, for the window length, f_max, the band and nu_ref
    single_pulse_fn: the model closure from model.setup_model, whose constants give the fixed scattering index
    toa:             arrival time the bounds are computed for (s, relative to the segment start)
    delta_prior:     the prior of 'delta' when it is sampled (bounded below); None uses the model's fixed index
    n_efold_tau:     e-folds of the bottom-of-band scattering tail that must fit
    n_sigma_stds:    Gaussian widths that must fit
    nu_bot:          bottom of the band (MHz); None takes the burst's lowest channel
    verbose:         print the bounds
    """
    W = burst.n_time * burst.dt
    f_max = float(burst.fourier_freqs[-1])
    nu_bot = float(burst.freq_channels[0][0]) if nu_bot is None else nu_bot

    sigma_min = 1.0 / (2.0 * np.pi * f_max)
    sigma_max = min(toa, W - toa) / n_sigma_stds

    if delta_prior is None:
        delta = float(single_pulse_fn.constants.delta)
    else:
        delta = _bound(delta_prior, "delta", "low")          # steepest scattering law the prior allows
        warnings.warn(
            f"delta is sampled, so the tau_ref prior is capped using its steepest value, delta = {delta:g}. "
            f"This keeps the scattering tail inside the window for every allowed delta, but also rules out "
            f"some larger tau_ref values that would still fit for shallower delta. TO-DO: a tau_ref prior "
            f"conditional on delta would remove this limitation."
        )

    tau_ref_min = 0.0
    tau_bot_max = (W - toa) / n_efold_tau
    tau_ref_max = tau_bot_max * (nu_bot / burst.freq_channels_ref) ** (-delta)

    if not sigma_min < sigma_max:
        raise ValueError(
            f"no width is both resolvable ({sigma_min:.2e} s) and containable ({sigma_max:.2e} s); "
            f"the segment is too short or the Fourier truncation too aggressive."
        )
    if verbose:
        print("prior being set with bounds:\nsigma_min =", sigma_min, "\nsigma_max =",
              sigma_max, "\ntau_ref_min =", tau_ref_min, "\ntau_ref_max =", tau_ref_max)

    return {
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "tau_ref_min": tau_ref_min,
        "tau_ref_max": tau_ref_max,
    }


def dispersion_prior_bounds(burst, single_pulse_fn):
    """
    Bounds for the residual DM, symmetric about zero: the largest |DM| for which a burst arriving in
    the middle of the window at the top of the band sweeps no further than an edge of the window by
    the bottom of the band, D_max = W / [2 k_DM (nu_bot^epsilon - nu_top^epsilon)]. The data are
    already dedispersed, so this is a residual of either sign. Returns {"DM_min": -D_max, "DM_max": D_max}.

    burst:           the ConditionedData, for the window length and the band edges
    single_pulse_fn: the model closure, whose constants give k_DM and epsilon, so the prior and the model agree
    """
    constants = single_pulse_fn.constants

    W_seg = burst.n_time * burst.dt
    nu_bot = float(burst.freq_channels.min())
    nu_top = float(burst.freq_channels.max())

    # Delay difference across the band per unit DM; positive since epsilon is negative
    sweep_per_DM = constants.k_DM * (nu_bot ** constants.epsilon - nu_top ** constants.epsilon)
    D_max = W_seg / (2.0 * sweep_per_DM)

    return {"DM_min": -D_max,
            "DM_max": D_max}


def spectral_index_bound(band, beta_min):
    """
    Bound of the symmetric spectral index prior, gamma ~ Uniform(-gamma_max, gamma_max). The running
    power law ln F = gamma x + beta x^2, x = ln(nu / nu_ref), is a Gaussian in x for beta < 0, peaking
    at x_p = gamma / (2 |beta|). The narrowest allowed spectrum (beta = beta_min) needs
    |gamma| = 2 |beta_min| |x_p| to peak at a band edge, and |x_p| is largest, ln(nu_top / nu_bot),
    when nu_ref sits at the other edge; so gamma_max = 2 |beta_min| ln(nu_top / nu_bot) lets every
    allowed width peak anywhere in the band wherever nu_ref is, and is the same for every burst
    observed with that band.

    band:     (nu_bot, nu_top), the bottom and top of the telescope's band (MHz)
    beta_min: lower bound of the beta prior, which must be negative
    """
    if not beta_min < 0:
        raise ValueError(f"the gamma bound from the band needs beta's lower bound to be negative; got {beta_min}. "
                         f"Pass gamma_range instead.")
    nu_bot, nu_top = float(band[0]), float(band[1])
    if not 0 < nu_bot < nu_top:
        raise ValueError(f"band must be (bottom, top) in MHz with 0 < bottom < top; got {band}.")
    return float(2.0 * abs(beta_min) * np.log(nu_top / nu_bot))


def model_duration_pad_for_priors(single_pulse_fn, priors, n_tau=3.0, n_sigma=3.0, verbose=True):
    """
    Padding (s) for model.setup_model(..., model_duration_pad=...) that holds all the emission the
    priors allow outside the window [0, W), whatever the spectrum, so none of it wraps back in.
    With arrival times inside the window, emission reaches at most n_sigma sigma_max plus the
    largest dispersion advance (top of the band) before 0, and n_sigma sigma_max plus n_tau tail
    lengths at the bottom of the band plus the largest dispersion delay (bottom of the band) after W;
    the padding holds both, since emission before 0 lands at the end of the longer window.
    extend_model_duration uses it as the upper limit of its search and usually needs much less.

    single_pulse_fn: the model closure; its .grid gives the band edges and its .constants give nu_ref, k_DM, epsilon and delta
    priors:          the priors dict from run.set_priors; 'sigma', 'tau_ref' and 'DM' need finite bounds, and a sampled 'delta' uses its steepest (lowest) value
    n_tau:           e-folds of the scattering tail the padding must hold (e^-3 = 5% of the tail still wraps for 3)
    n_sigma:         Gaussian widths the padding must hold
    verbose:         print the padding and what sets it, as multiples of the window
    """
    constants = single_pulse_fn.constants
    nu = np.asarray(single_pulse_fn.grid.freqs_channels, dtype=float).reshape(-1)
    nu_bot, nu_top, nu_ref = nu.min(), nu.max(), float(constants.freq_channels_ref)
    # The steepest scattering law the prior allows gives the longest tail at the bottom of the band
    delta = _bound(priors["delta"], "delta", "low") if "delta" in priors else float(constants.delta)

    sigma_max = _bound(priors["sigma"], "sigma", "high")
    tau_bot_max = _bound(priors["tau_ref"], "tau_ref", "high") * (nu_bot / nu_ref) ** delta

    DM_max = 0.0
    if "DM" in priors:
        DM_max = max(abs(_bound(priors["DM"], "DM", "low")), abs(_bound(priors["DM"], "DM", "high")))
    eps = constants.epsilon
    delay_bot = constants.k_DM * DM_max * abs(nu_bot ** eps - nu_ref ** eps)
    advance_top = constants.k_DM * DM_max * abs(nu_top ** eps - nu_ref ** eps)

    pad = 2.0 * n_sigma * sigma_max + n_tau * tau_bot_max + delay_bot + advance_top
    if verbose:
        W = single_pulse_fn.segment_duration
        print(f"padding = {pad / W:.2f} x the segment duration: Gaussian widths "
              f"{2 * n_sigma * sigma_max / W:.2f}, scattering tail at the bottom of the band "
              f"{n_tau * tau_bot_max / W:.2f}, dispersion {(delay_bot + advance_top) / W:.2f}")
    return float(pad)
