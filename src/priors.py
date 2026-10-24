"""
priors.py - Prior bounds for the Fourier-domain FRB model.

These compute numbers rather than register sampling sites: each returns bounds derived from the
conditioned data and the model's own constants, to be turned into distributions in the runner.
That keeps them testable without a sampler, which matters because an error here is a silently
wrong prior rather than a crash. The constructions that do register sites -- the ordered arrival
times, the containment penalty -- live in likelihood.py instead.

The bounds are deliberately loose. Their job is to stop the sampler wandering somewhere the
periodic Fourier-domain model cannot represent, not to express knowledge about bursts. Each
parameter alone, at its bound, just fits inside the segment; nothing here constrains the
combination, which is left to the containment penalty.

With the padded model (setup_model(..., model_duration_pad=...)) the model is no longer periodic
on the segment, so these bounds no longer guard against wraparound. They then express the
assumption that the segment contains the pulse, and they set how much padding is needed:
`model_duration_pad_for_priors` computes the padding that holds all emission they allow outside
the segment.
"""

import numpy as np


def width_prior_bounds(burst, toa, n_efold_tau=1.0, n_sigma_stds=3, nu_bot=None, scattering_index=-4.0):
    """
    Loose box bounds for the width and scattering timescale, as entries for the priors dict.

    These are deliberately weak: each parameter alone, at its bound, just fits inside the segment,
    and nothing stops a draw where both are large at once. The joint condition is left to
    `sample_contained_lowestFDbin`, which acts on the evaluated model and so sees the combination.
    The box only prevents the sampler wandering somewhere absurd.

    sigma is bounded so that n_sigma_stds widths fit between the arrival time and whichever
    segment edge is closer, since a Gaussian is symmetric. Its floor is the narrowest resolvable
    width, 1 / (2 pi f_max) from the Fourier truncation: below that the template has not rolled
    off within the retained bins, so the data do not constrain it. (The sampling interval is not
    the relevant floor -- without truncation the resolvable width would be dt / pi, below dt
    itself.)

    tau is bounded at the BOTTOM of the band, where the tail is longest, by requiring
    n_efold_tau e-folds between the arrival time and the end of the segment, then converted to
    tau_ref.

    toa:          arrival time (s, relative to the segment start)
    n_efold_tau:  how many e-folds of the scattering timescale tau must fit
    n_sigma_stds: how many standard deviations of the pulse width sigma must fit
    nu_bot:       bottom of the band (MHz); taken from the burst's lowest channel if None
    """
    W = burst.n_time * burst.dt
    f_max = float(burst.fourier_freqs[-1])
    nu_bot = float(burst.freq_channels[0][0]) if nu_bot is None else nu_bot

    sigma_min = 1.0 / (2.0 * np.pi * f_max)
    sigma_max = min(toa, W - toa) / n_sigma_stds

    tau_ref_min = 0.0
    tau_bot_max = (W - toa) / n_efold_tau
    tau_ref_max = tau_bot_max * (nu_bot / burst.freq_channels_ref) ** (-scattering_index)

    if not sigma_min < sigma_max:
        raise ValueError(
            f"no width is both resolvable ({sigma_min:.2e} s) and containable ({sigma_max:.2e} s); "
            f"the segment is too short or the Fourier truncation too aggressive."
        )
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
    Bounds for the residual dispersion measure, as an entry for the priors dict.

    Symmetric about zero, with the magnitude set so that a burst arriving in the middle of the
    segment at the top of the band would sweep to a segment edge by the bottom of the band:

        D_max = W_seg / [ 2 k_D ( nu_bot^epsilon - nu_top^epsilon ) ]

    Beyond that the sweep runs off the end of the segment, which the periodic Fourier-domain
    model would wrap around. The bound is loose in the sense that it ignores the pulse's own
    width and scattering, which also consume the segment; those are bounded separately by
    `width_prior_bounds`, and the joint case is left to the containment penalty.

    The data are assumed already dedispersed, so this is a residual about zero rather than an
    absolute DM; the symmetric range reflects that the residual may be of either sign.

    burst:           the ConditionedData, for the segment length and the band edges
    single_pulse_fn: the model closure, whose `.constants` supply k_DM and epsilon, so the prior
                     and the model cannot disagree about the dispersion law

    Returns {"DM": (lo, hi)}.
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


def spectral_coeff_bounds(spectral_coeff_map, gamma_range, beta_range):
    """
    Bounds on the sampled c1, c2 of the SED corresponding to a box in the conventional gamma, beta.

    The map is linear, so a box in (gamma, beta) becomes a parallelogram in (c1, c2); this
    returns its bounding box, which is the tightest region expressible as independent uniform
    priors. That admits a little more than intended -- about 8% of the box falls outside the
    gamma-beta limits for a typical basis -- which is harmless when the limits are loose
    catalogue-wide bounds rather than informative constraints.

    Because the map is linear with constant Jacobian, a flat prior on (c1, c2) is also flat on
    (gamma, beta); no correcting factor is needed.
    """
    a1 = spectral_coeff_map["gamma_c1"]
    b2 = spectral_coeff_map["gamma_c2"]
    a2 = spectral_coeff_map["beta_c2"]

    corners = [((g - b2 * (b / a2)) / a1, b / a2) for g in gamma_range for b in beta_range]
    c1s, c2s = zip(*corners)

    print("prior being set with bounds:\nc1_min =", min(c1s), "\nc1_max =",
          max(c1s), "\nc2_min =", min(c2s), "\nc2_max =", max(c2s))

    return {
        "c1_min": min(c1s),
        "c1_max": max(c1s),
        "c2_min": min(c2s),
        "c2_max": max(c2s),
    }

def _upper_bound(distribution, name):
    """Upper bound of a bounded prior, e.g. a Uniform; raises for one without a finite bound."""
    high = getattr(distribution, "high", None)
    if high is None or not np.isfinite(float(high)):
        raise ValueError(f"the '{name}' prior has no finite upper bound, so the padding it needs "
                         f"cannot be computed from it.")
    return float(high)


def model_duration_pad_for_priors(single_pulse_fn, priors, containment=None,
                                  n_tau=3.0, n_sigma=3.0, verbose=True):
    """
    Padding (s) for setup_model(..., model_duration_pad=...) that holds all the emission the
    priors allow outside the segment [0, W), so that none of it wraps back in, whatever the
    pulse's spectrum. single_pulse_fn.extend_model_duration(burst, priors) uses it as the upper
    limit of its search, and usually pads much less: it measures the wrapped emission directly,
    weighting channels by the burst's own SNR.

    With arrival times in [0, W], a pulse's emission reaches at most
        before 0:  n_sigma sigma_max + the largest dispersion ADVANCE (top of the band)
        after W:   n_sigma sigma_max + n_tau tau_bot_max + the largest dispersion DELAY
                   (bottom of the band)
    and the padding has to hold both, since emission before 0 lands at the end of the longer
    window. Scattering is longest at the bottom of the band, tau_bot = tau_ref (nu_bot/nu_ref)^delta,
    which is why it sets the padding rather than tau_ref itself.

    single_pulse_fn: the model closure (unpadded is fine). Its `.grid` gives the band edges and its
                     `.constants` give nu_ref, k_DM, epsilon and delta, so the padding and the
                     model cannot disagree about them.
    priors:          the priors dictionary, as from run.set_priors; 'sigma', 'tau_ref' and 'DM'
                     need finite bounds (Uniform). If 'delta' is sampled, its lower bound (the
                     steepest scattering law) is used.
    containment:     the `containment` settings, when sigma and tau are drawn jointly by
                     sample_contained_widths instead of being in priors: their bounds are then
                     extent_max / n_sigma and extent_max / n_tau (at the bottom of the band),
                     with that method's own n_sigma and n_tau (defaults 3 and 5)
    n_tau, n_sigma:  e-folds of the tail and widths of the Gaussian the padding must hold. The
                     area of an exponential beyond n e-folds is e^-n (5% for 3), and that is
                     what would still wrap, in the bottom channel.
    verbose:         print the padding and what sets it, as multiples of the segment duration

    Returns the padding in seconds.
    """
    constants = single_pulse_fn.constants
    nu = np.asarray(single_pulse_fn.grid.freqs_channels, dtype=float).reshape(-1)
    nu_bot, nu_top, nu_ref = nu.min(), nu.max(), float(constants.freq_channels_ref)
    # The steepest scattering law the prior allows gives the longest tail at the bottom of the band
    delta = float(priors["delta"].low) if "delta" in priors else float(constants.delta)

    if "sigma" in priors and "tau_ref" in priors:
        sigma_max = _upper_bound(priors["sigma"], "sigma")
        tau_bot_max = _upper_bound(priors["tau_ref"], "tau_ref") * (nu_bot / nu_ref) ** delta
    elif containment is not None:
        extent_max = float(containment["extent_max"])
        sigma_max = extent_max / float(containment.get("n_sigma", 3.0))
        tau_bot_max = extent_max / float(containment.get("n_tau", 5.0))
    else:
        raise ValueError("'sigma' and 'tau_ref' are not both in priors; pass the containment "
                         "settings that draw them instead.")

    DM_max = 0.0
    if "DM" in priors:
        DM_max = max(abs(float(priors["DM"].low)), abs(_upper_bound(priors["DM"], "DM")))
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
