"""
NOTE ON PRECISION & GPU COMPATIBILITY:
This module uses float32 for speed and broad GPU compatibility (where FP64 is often
slow or unsupported). Because float32 only retains ~7 decimal digits of precision,
inputs must stay small to prevent rounding away tiny time delays:

1. TIMING: Pass relative timestamps (times[0] ≈ 0). Large absolute times (MJD/Unix)
   round off fraction-of-a-second delays and distort phase.
2. DM: Fit a residual DM around a dedispersed baseline. Large absolute DMs cause
   multi-second delays that hit the same float32 precision limits.
3. SED: The SED functions return ln F, not F. F = exp(ln F) overflows float32 once ln F > 88,
   and the SNR, which squares F, once ln F > 44, both of which wide spectral priors reach.
   single_pulse_model therefore only ever exponentiates normalized or combined logs.

NOTE ON WRAPAROUND (setup_model's model_duration_pad):
The data are a segment cut from a longer recording, so emission past either end of the window is
simply missing (truncated). The analytic template sampled at the data's bin frequencies k / W is
instead the Fourier transform of the pulse repeated with period W, so emission past the end
reappears at the start (wrapped). With model_duration_pad > 0 the template is evaluated on a
longer window W' = W + pad, transformed to time, cut to [0, W) exactly as the data were, and
transformed back onto the data's retained bins: the model is then truncated like the data, and
does not wrap, as long as the padding holds all emission outside [0, W). The usual way to get
it is single_pulse_fn.extend_model_duration(priors), which returns the same model padded for
those priors.
"""

import math
from dataclasses import dataclass
from typing import Callable
import numpy as np
import jax.numpy as jnp


# -------------------------------------------------------------------
# 1. Containers for Constants & Fourier Grid (Standard Dataclasses)
# -------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConstants:
    freq_channels_ref: float  # fixed reference / pivot EM frequency nu_r, set approximately equal to frequency of peak emission for a pulse with Gaussian-like SED in the dynamic spectrum element M_kn (MHz)
    det_response: float = 1.0  # TO-DO: WILL BE FUNCTION LATER !!!! Maybe also convolution is involved if frequencies are not independent between channels
    delta: int | float = -4  # Exponent of frequency dependence for scatter-broadening (tau \propto nu^-4). Kept as a Python int so nu^delta lowers to integer_pow rather than a general pow; it becomes a float32 tracer if delta is sampled instead.
    k_DM: float = (2.41 * 1e-4)**(-1)  # dispersion constant, pc^-1 * cm^3 * MHz^2 * s
    epsilon: int | float = -2  # Exponent of frequency dependence for pulse dispersion - for now it seems the model only actually allows this to be a constant -2, due to units. Kept as an int for the same reason as delta.
    DM_incoherent: float = 0.0  # DM removed only by incoherent dedispersion (pc cm^-3); contributes to intra-channel smearing. Set to 0 for coherently dedispersed data. Enters only as a sinc width (not a phase), so a large value does not violate the float32 note above.
    channel_width: float | jnp.ndarray = 0.0  # frequency channel width Delta nu (MHz), scalar or per-channel (nfreq, 1) after block averaging; channels labeled at their centers; if 0, intra-channel smearing is omitted


@dataclass(frozen=True)
class FDGrid:
    freqs_channels: jnp.ndarray  # frequency array nu_a, shape of (nfreq,1), units of MHz, these are the rows of the model matrix
    normalized_freqs_channels: jnp.ndarray  # nu_a / nu_r
    log_normalized_freqs_channels: jnp.ndarray  # log(nu_a / nu_r)
    fourier_freqs: jnp.ndarray  # (1, nfourier)
    t_start: jnp.ndarray  # time array reference offset, needed given FFT conventions
    dt_bin: jnp.ndarray  # time bin width (s); time bins are labeled at their start

    # The quantities below depend only on the grid and the model constants, never on sampled
    # parameters, so they are built once here rather than on every model evaluation. The
    # scatter-broadening factor (nu_a / nu_r)^delta is NOT among them, since delta may be
    # sampled; it is evaluated in the pulse model instead.
    fourier_freqs_sq: jnp.ndarray = None  # f_c^2, for the Gaussian envelope
    freqs_channels_disp: jnp.ndarray = None  # nu_a^epsilon, for the dispersion delay
    freq_ref_disp: jnp.ndarray = None  # nu_r^epsilon, the reference frequency's dispersion delay
    bin_transfer: jnp.ndarray = None  # sinc(f_c dt) exp(i pi f_c dt), the time-binning transfer function


def build_fd_grid(freqs_channels: jnp.ndarray, times: jnp.ndarray, constants: ModelConstants, fourier_freqs=None) -> FDGrid:
    """
    Constructs the Fourier domain evaluation grid.

    Note: `times` MUST start near zero (|t_start| <= 1.0 s) so float32
    does not round away small timing details.
    fourier_freqs: the Fourier bins the data occupies, from data.py. None rebuilds the full
                   rfft grid. The model evaluates templates at exactly these frequencies, so
                   it needs to know nothing about how the data was conditioned.
    """
    # channel frequency array nu_a, shape of (nfreq,1), units of MHz, these are the rows of the model matrix
    freqs_channels = jnp.asarray(freqs_channels, dtype=jnp.float32)
    if freqs_channels.ndim == 1:
        freqs_channels = freqs_channels[:, None]

    # time array t_b, shape of (1,ntime), units of seconds, these are the columns of the model matrix
    times = jnp.asarray(times, dtype=jnp.float32)
    times_1d = times[0] if times.ndim > 1 else times

    # Guard check to ensure times start near 0
    t_start_val = abs(float(times_1d[0]))
    if t_start_val > 1.0:
        raise ValueError(
            f"times[0] = {times_1d[0]:.3f} s. `times` must be relative to segment start "
            f"(abs(t_start) <= 1.0 s) to prevent float32 rounding errors. "
            f"See module note on precision."
        )

    freq_channels_ref = jnp.float32(constants.freq_channels_ref)

    normalized_freqs = freqs_channels / freq_channels_ref  # nu_a / nu_r
    log_normalized_freqs = jnp.log(normalized_freqs)

    ## Defining things related to Fourier domain
    Nseg = times_1d.shape[0]
    dt = times_1d[1] - times_1d[0]
    t_start = times_1d[0]

    #fourier_freqs is columns of FD evaluation grid
    if fourier_freqs is None:
        fourier_freqs = jnp.fft.rfftfreq(Nseg, d=dt, dtype=jnp.float32)[None, :]
    else:
        fourier_freqs = jnp.asarray(fourier_freqs, dtype=jnp.float32).reshape(1, -1)

    ## Parameter-independent factors of the pulse model, built once
    # epsilon is left as a Python int so these lower to integer_pow
    epsilon = constants.epsilon
    fourier_freqs_sq = fourier_freqs ** 2
    freqs_channels_disp = freqs_channels ** epsilon
    freq_ref_disp = freq_channels_ref ** epsilon

    # Time bins are labeled at their start, so each sample averages over [t_b, t_b + dt_bin]:
    # the boxcar kernel spans [-dt_bin, 0], giving sinc(f dt_bin) times a half-bin phase shift
    bin_transfer = jnp.sinc(fourier_freqs * dt) * jnp.exp(1j * jnp.pi * fourier_freqs * dt)

    return FDGrid(
        freqs_channels=freqs_channels,
        normalized_freqs_channels=normalized_freqs,
        log_normalized_freqs_channels=log_normalized_freqs,
        fourier_freqs=fourier_freqs,
        t_start=t_start,
        dt_bin=dt,
        fourier_freqs_sq=fourier_freqs_sq,
        freqs_channels_disp=freqs_channels_disp,
        freq_ref_disp=freq_ref_disp,
        bin_transfer=bin_transfer,
    )

# -------------------------------------------------------------------
# 2. Spectral Energy Distribution (SED) Functions (ln F)
#
# Every SED function returns ln F, shape (nfreq, 1), NOT F. See note 3 in the module docstring:
# single_pulse_model exponentiates it only after normalizing (or after adding ln A), so F itself
# is never formed and cannot overflow.
# -------------------------------------------------------------------

def running_power_law_log_sed(params_dict: dict, grid: FDGrid, constants: ModelConstants) -> jnp.ndarray:
    """
    Log of the running power-law SED, F_a = (nu_a / nu_r)^kappa_a * det_response:

        ln F_a = kappa_a x_a + ln(det_response),    kappa_a = gamma + beta x_a,    x_a = ln(nu_a / nu_r)
    """
    gamma = jnp.float32(params_dict["gamma"])  # Spectral index for burst component
    beta = jnp.float32(params_dict["beta"])  # Spectral running for burst component
    det_response = jnp.float32(constants.det_response)

    kappa = gamma + (beta * grid.log_normalized_freqs_channels)  # Running power law of SED
    return kappa * grid.log_normalized_freqs_channels + jnp.log(det_response)

def make_orthogonal_poly_log_sed(P1: jnp.ndarray, P2: jnp.ndarray) -> Callable:
    """
    Builds an SED function in an orthogonal polynomial basis:

        ln F_a = c1 P1(x_a) + c2 P2(x_a),    x_a = ln(nu_a / nu_r)

    rather than ln F_a = gamma x_a + beta x_a^2. P1 and P2 are precomputed by
    `orthogonal_spectral_basis`, orthogonal to each other and to the constant in the metric set
    by the per-channel weights. Over a factor-of-two band x and x^2 are nearly parallel, which
    is what makes gamma and beta strongly degenerate; c1 and c2 are constructed not to be, and
    orthogonality to the constant also decouples them from the fluence.

    The basis is fixed before sampling, so this is only a change of coordinates: recover gamma
    and beta afterwards with the coefficients returned alongside P1 and P2. Note that a fixed
    prior on (c1, c2) implies a different prior on (gamma, beta) than a fixed prior on those
    would, and that the coefficients are O(1) by construction, so their priors should be too.

    The returned function gives ln F (plus ln det_response), not F.
    """
    P1 = jnp.asarray(P1, dtype=jnp.float32).reshape(-1, 1)
    P2 = jnp.asarray(P2, dtype=jnp.float32).reshape(-1, 1)

    def orthogonal_poly_log_sed(params_dict: dict, grid: FDGrid, constants: ModelConstants) -> jnp.ndarray:
        c1 = jnp.float32(params_dict["c1"])  # coefficient of the linear basis function
        c2 = jnp.float32(params_dict["c2"])  # coefficient of the quadratic basis function
        det_response = jnp.float32(constants.det_response)

        return c1 * P1 + c2 * P2 + jnp.log(det_response)

    return orthogonal_poly_log_sed

# -------------------------------------------------------------------
# 3. Fourier Domain Pulse Functions (T)
# -------------------------------------------------------------------

def scattered_gaussian_fd_pulse(params_dict: dict, grid: FDGrid, constants: ModelConstants) -> jnp.ndarray:
    """
    Evaluates a Gaussian pulse with thin-screen scattering in the FD.

    Note: `DM` should be a small residual DM (relative to a dedispersed baseline)
    to avoid float32 rounding errors on large dispersion delays.
    """
    if constants.epsilon != -2:
        raise ValueError("Pulse model valid only for epsilon = -2.")

    # delta is taken from params_dict when a prior has been given for it, and is then a float32
    # tracer; otherwise it comes from constants as a Python int, so the power below lowers to
    # integer_pow. This is why (nu_a / nu_r)^delta cannot be precomputed in the grid.
    delta = params_dict.get("delta", constants.delta)

    mean_arrival_time = jnp.float32(params_dict["mean_arrival_time"])  # defined so that the pulse peaks at freq_ref at this time (before scatter or dispersion shifts at other frequencies)
    sigma = jnp.float32(params_dict["sigma"])  # width of gaussian in pulse, without scatter-broadening
    DM = jnp.float32(params_dict["DM"])  # dispersion measure, pc/cm^3 (residual DM)
    tau_ref = jnp.float32(params_dict["tau_ref"])  # scatter broadening timescale at freq_ref, if 0 then no scatter is put in the model, s

    k_DM = jnp.float32(constants.k_DM)  # dispersion constant, (2.41*1e-4)**(-1) #pc^-1 * cm^3 * MHz^2 * s

    # mean time in the gaussian model with no scatter broadening
    mean_time_gaussian = mean_arrival_time + (
        k_DM * DM * (grid.freqs_channels_disp - grid.freq_ref_disp)
    )

    mu_a = mean_time_gaussian - grid.t_start

    tau_a = tau_ref * (grid.normalized_freqs_channels ** delta)  # Timescale of scattering-broadening at nu_a

    T = jnp.exp(
        -2 * ((jnp.pi ** 2) * (sigma ** 2) * grid.fourier_freqs_sq)
        - 2j * (jnp.pi * grid.fourier_freqs * mu_a)
    ) / (1 + 2j * jnp.pi * grid.fourier_freqs * tau_a)

    # --- Instrumental transfer function: intra-channel dispersion smearing and time binning ---
    # jnp.sinc(x) = sin(pi x) / (pi x), the FT of a unit-area boxcar of unit width
    DM_inc = jnp.float32(constants.DM_incoherent)
    channel_width = jnp.float32(constants.channel_width)

    # Intra-channel dispersion smearing: delay difference across the channel (assumes epsilon = -2).
    # Depends on the sampled DM, so unlike the time-binning term it cannot be precomputed.
    # TO-DO: Check this convention -> channels are labeled at their centers, so the smearing boxcar is centered on the channel-center delay (no phase).
    dt_smear = k_DM * (DM_inc + DM) * (
        (2.0 * grid.freqs_channels * channel_width) / ((grid.freqs_channels**2 - (channel_width**2 / 4.0))**2)
    )
    instrmnt_smear_transfer = jnp.sinc(grid.fourier_freqs * dt_smear)

    # Time binning, precomputed in build_fd_grid since it depends only on dt_bin.
    # To-DO: Check this convention -> bins are labeled at their start, so each sample averages over [t_b, t_b + dt_bin].
    instrmnt_bin_transfer = grid.bin_transfer

    T = T * instrmnt_smear_transfer * instrmnt_bin_transfer

    return T

# -------------------------------------------------------------------
# 4. Padding against wraparound: FFT length
# -------------------------------------------------------------------

def _fft_friendly(n):
    """Smallest n' >= n whose only prime factors are 2, 3 and 5, so the FFTs stay fast."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


# -------------------------------------------------------------------
# 5. Main Model Orchestrator
# -------------------------------------------------------------------

def setup_model(
    constants_dict: dict,
    freqs_channels: jnp.ndarray,
    times: jnp.ndarray,
    sed_fn: Callable = running_power_law_log_sed,
    pulse_fn: Callable = scattered_gaussian_fd_pulse,
    fourier_freqs=None,
    model_duration_pad: float = 0.0,
    _wraparoundFDdiagnostic: bool = True,
):
    """
    Setting up the necessary constants for the model itself. The model will be written as a closure within the setup method.

    constants_dict: dict of each constant in the model
    freqs_channels: array of frequency channels in the data
    times: array of time bins in the data
    sed_fn: returns ln F (see Section 2), not F
    fourier_freqs: the Fourier bins the data occupies, from data.py; None uses the full rfft grid
    model_duration_pad: padding (s) against wraparound; 0 (default) keeps the original, wrapped
        model. With model_duration_pad > 0 the template is evaluated on every rfft bin of a window
        longer by at least model_duration_pad (rounded up to an FFT-friendly length), transformed
        to time, cut to the data window, and transformed back onto `fourier_freqs` (see the module
        note on wraparound). It is then truncated exactly like the data, as long as the padding
        holds all emission the priors allow outside the window. Rather than setting it here,
        build the model unpadded and call `.extend_model_duration(priors)` on it once the
        priors exist (they need a model first, for the dispersion bounds), which returns the
        same model padded for them.
        The returned templates have the same shape either way, so nothing downstream changes;
        the cost is evaluating the model on more bins plus one inverse and one forward FFT per
        pulse and channel.
    _wraparoundFDdiagnostic: if True, also builds the same model on the two frequencies
        [f1/2, f1], f1 = 1 / W_seg, and attaches it as `.lowfreq_diagnostic`. The likelihoods use
        it to record the wraparound ratio. The inner call sets this False, so it does not recurse.
        With model_duration_pad > 0 the model no longer wraps, so the ratio is then only a measure
        of the emission's extent relative to the window, and the containment_FD penalty is
        unnecessary.
    """
    ### We are setting up constant values here
    constants = ModelConstants(**constants_dict)
    # channel_width may be a scalar or a per-channel array (see ModelConstants), so reduce
    # the comparison before combining it
    if constants.DM_incoherent > 0 and bool(jnp.all(jnp.asarray(constants.channel_width) == 0)):
        raise ValueError("DM_incoherent > 0 but channel_width = 0: intra-channel smearing would be silently omitted.")
    grid = build_fd_grid(freqs_channels, times, constants, fourier_freqs)

    # ---- Padding against wraparound: the longer grid the model is evaluated on ----
    if model_duration_pad < 0:
        raise ValueError(f"model_duration_pad must be >= 0; got {model_duration_pad}.")
    eval_grid, n_long, truncate = grid, None, None
    if model_duration_pad > 0:
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        n_seg, dt = t.size, float(t[1] - t[0])
        n_long = _fft_friendly(n_seg + int(math.ceil(model_duration_pad / dt)))
        # Same start and spacing as the data, just longer; build_fd_grid then uses every rfft
        # bin of it, DC included, since cutting in time mixes all frequencies
        times_long = (t[0] + dt * np.arange(n_long)).astype(np.float32)
        eval_grid = build_fd_grid(freqs_channels, times_long, constants, fourier_freqs=None)

        # Where each of the data's retained bins sits on the segment's own rfft grid
        f_data = np.asarray(grid.fourier_freqs, dtype=np.float64).reshape(-1)
        bin_index = np.rint(f_data * n_seg * dt).astype(int)
        if not np.allclose(bin_index, f_data * n_seg * dt, atol=1e-3):
            raise ValueError("fourier_freqs are not bins of the segment's rfft grid (k / (N dt)).")
        bin_index = jnp.asarray(bin_index)

        def truncate(M_long):
            # To time on the long window, where it is periodic only on W' and nothing wraps into
            # [0, W); cut to the data window, as the data were cut; back onto the retained bins.
            # The dt factors of the transform convention (X = dt rfft(x), x = irfft(X) / dt)
            # cancel between the inverse and the forward transform, so they are left out.
            x = jnp.fft.irfft(M_long, n=n_long, axis=-1)
            return jnp.fft.rfft(x[..., :n_seg], axis=-1)[..., bin_index]

    def single_pulse_model(params_dict):
        """
        Model of a single pulse, which will be summed in the forward model for multi-pulse signals.

        params_dict: dict of each parameter of the pulse

        With 'A' in params_dict (fluence priors): returns A F T, the pulse at that fluence.

        Without 'A' (A_snr and marginalized likelihoods, which set the amplitude themselves):
        returns the template with its SED scaled to peak at exactly 1 across the channels,

            exp(ln F - max_a ln F_a) T  =  F T / max_a F_a,

        which cannot overflow for any spectral parameters. The factor removed, max_a F_a, is
        one positive number per pulse; those likelihoods only ever use the template times a
        free amplitude, so it cancels from the fit, and they convert the amplitude back to
        fluence with `log_sed_peak` below.
        """
        lnF = sed_fn(params_dict, eval_grid, constants)  # (nfreq, 1), the log of the SED
        T = pulse_fn(params_dict, eval_grid, constants)

        if "A" in params_dict:
            # A F = sign(A) exp(ln|A| + ln F): combined in log space, so a huge F times a small A
            # never overflows on the way to a moderate product. The sign is carried separately so
            # a negative A (e.g. a rejected full-normal fluence draw being plotted) still evaluates.
            A = jnp.float32(params_dict["A"])
            M = jnp.sign(A) * jnp.exp(jnp.log(jnp.abs(A)) + lnF) * T
        else:
            M = jnp.exp(lnF - jnp.max(lnF)) * T

        # With padding: truncated to the data window like the data, on the data's retained bins
        return M if truncate is None else truncate(M)

    def log_sed_peak(params_dict):
        """
        ln max_a F_a: the log of the factor single_pulse_model divides out when 'A' is absent.
        An amplitude B on that normalized template is the fluence A = B exp(-log_sed_peak).
        """
        return jnp.max(sed_fn(params_dict, grid, constants))

    # Recorded so a saved fit can say which SED and pulse forms produced it; the returned
    # closure is always named single_pulse_model, which says nothing about its contents
    # constants and grid exposed for convenience
    single_pulse_model.constants = constants
    single_pulse_model.grid = grid              # the data's grid, on which templates are returned
    single_pulse_model.model_duration_pad = float(model_duration_pad)
    t_seg = jnp.asarray(times, dtype=jnp.float32).reshape(-1)
    single_pulse_model.segment_duration = t_seg.shape[0] * float(t_seg[1] - t_seg[0])   # W (s)
    single_pulse_model.n_long = n_long          # samples in the padded window; None without padding
    single_pulse_model.sed_fn = sed_fn.__name__
    single_pulse_model.pulse_fn = pulse_fn.__name__
    single_pulse_model.log_sed_peak = log_sed_peak

    def extend_model_duration(burst, priors, wrapped_snr=1.0, containment=None, n_tau=3.0,
                              n_sigma=3.0, time_param="mean_arrival_time"):
        """
        Returns this same model, rebuilt with the smallest padding against wraparound (see the
        module note on wraparound) for which no pulse the priors allow wraps detectably back
        into the data window. Optional: leave the call out to fit the original, unpadded model.

            single_pulse_fn = setup_model(...)
            priors = set_priors(burst, single_pulse_fn, n_pulses, ...)
            single_pulse_fn = single_pulse_fn.extend_model_duration(burst, priors)

        HOW THE PADDING IS CHOSEN
        Worst-case pulses are built from the prior bounds: arrival time at either end of its
        prior, the widest Gaussian and longest scattering tail allowed, and the residual DM at
        either bound (or zero), so that emission runs past whichever edge it can. Each is
        compared, for a candidate padding, with the same pulse on a window long enough that
        nothing wraps. The difference is the wrapped emission, measured as a fraction of the
        pulse's noise-weighted power, with channels weighted by this burst's own SNR^2 per
        channel (as data.estimate_channel_weights measures it), so long tails in channels with
        little signal count for little. The padding chosen is the smallest for which
            (wrapped fraction) x SNR^2  <=  wrapped_snr^2
        for every worst-case pulse, SNR being this burst's total optimal SNR estimated from the
        data: wrapped emission below that is buried in the noise. The search runs up to the
        padding priors.model_duration_pad_for_priors gives with n_tau and n_sigma, which is
        used, with a warning, if even that is not enough.

        burst:       the ConditionedData of the fit, for the channel weights and the burst's SNR
        priors:      the priors dictionary of the fit, as returned by run.set_priors
        wrapped_snr: largest SNR the wrapped emission may have; 1 (default) keeps it at the
                     noise level
        containment: the `containment` settings, if sigma and tau are drawn jointly instead of
                     being in priors; the worst cases are then the widest Gaussian with no tail
                     and the longest tail with the narrowest Gaussian
        n_tau, n_sigma: e-folds of the tail and widths of the Gaussian in the upper limit of the
                     search; see priors.model_duration_pad_for_priors
        time_param:  name of the arrival-time prior
        """
        # Imported here rather than at the top: priors.py does not import this module, so there
        # is no cycle, and a model without padding never needs it
        import warnings
        from priors import model_duration_pad_for_priors

        pad_max = model_duration_pad_for_priors(single_pulse_model, priors, containment=containment,
                                                n_tau=n_tau, n_sigma=n_sigma, verbose=False)
        n_seg = int(np.asarray(times).reshape(-1).shape[0])
        W = single_pulse_model.segment_duration

        # ---- SNR^2 per channel from the data: excess power over the noise expectation ----
        d = np.asarray(burst.data_fd)
        s = np.broadcast_to(np.asarray(burst.sigma_noise_fd), d.shape)
        snr2_chan = np.maximum(np.sum(np.abs(d) ** 2 / s ** 2, axis=-1) - 2.0 * d.shape[-1], 0.0)
        snr2 = max(float(snr2_chan.sum()), 1.0)
        weights = snr2_chan / snr2 if snr2_chan.sum() > 0 else np.full(snr2_chan.shape, 1.0 / snr2_chan.size)
        tolerance = wrapped_snr ** 2 / snr2           # largest wrapped fraction of the power

        # ---- worst-case pulses from the prior bounds ----
        def bound(name, side):
            return float(getattr(priors[name], side))
        DM_max = max(abs(bound("DM", "low")), abs(bound("DM", "high"))) if "DM" in priors else 0.0
        if "sigma" in priors and "tau_ref" in priors:
            widths = [(bound("sigma", "high"), bound("tau_ref", "high"))]
        else:
            nu = np.asarray(freqs_channels, dtype=float).reshape(-1)
            bot_over_ref = nu.min() / float(constants.freq_channels_ref)
            extent = float(containment["extent_max"])
            n_s, n_t = float(containment.get("n_sigma", 3.0)), float(containment.get("n_tau", 5.0))
            widths = [(extent / n_s, 0.0),
                      (float(containment["sigma_min"]), extent / n_t * bot_over_ref ** (-constants.delta))]
        flat_sed = dict(gamma=0.0, beta=0.0, c1=0.0, c2=0.0)        # ln F = 0: the shape alone
        if "delta" in priors:
            flat_sed["delta"] = bound("delta", "low")                 # steepest scattering law
        pulses = [dict(flat_sed, mean_arrival_time=t, sigma=sig, tau_ref=tau, DM=dm)
                  for t in (bound(time_param, "low"), bound(time_param, "high"))
                  for sig, tau in widths for dm in (-DM_max, 0.0, DM_max)]

        def build(pad):
            return setup_model(constants_dict, freqs_channels, times, sed_fn, pulse_fn,
                               fourier_freqs=fourier_freqs, model_duration_pad=pad,
                               _wraparoundFDdiagnostic=False)

        # Reference: a window long enough that nothing the priors allow wraps
        reference = build(4.0 * pad_max + W)
        refs = [np.asarray(reference(p)) for p in pulses]

        def wrapped_fraction(pad):
            fn = build(pad)
            worst = 0.0
            for p, ref in zip(pulses, refs):
                per_channel = (np.sum(np.abs(np.asarray(fn(p)) - ref) ** 2, axis=-1)
                               / np.maximum(np.sum(np.abs(ref) ** 2, axis=-1), np.finfo(float).tiny))
                worst = max(worst, float(np.sum(weights * per_channel)))
            return worst

        # ---- smallest adequate padding: bisection on the FFT-friendly window lengths ----
        dt = W / n_seg
        lengths = sorted({_fft_friendly(n_seg + k) for k in range(1, int(math.ceil(pad_max / dt)) + 1)})
        if wrapped_fraction(pad_max) > tolerance:
            warnings.warn("even the padding from the prior bounds leaves wrapped emission above "
                          f"wrapped_snr = {wrapped_snr}; using it anyway. Increase n_tau, or check "
                          "the priors.")
            pad = pad_max
        else:
            lo, hi = 0, len(lengths) - 1                   # lengths[hi] is known to be adequate
            while lo < hi:
                mid = (lo + hi) // 2
                if wrapped_fraction((lengths[mid] - n_seg) * dt) <= tolerance:
                    hi = mid
                else:
                    lo = mid + 1
            pad = (lengths[hi] - n_seg) * dt

        extended = setup_model(constants_dict, freqs_channels, times, sed_fn, pulse_fn,
                               fourier_freqs=fourier_freqs, model_duration_pad=pad,
                               _wraparoundFDdiagnostic=_wraparoundFDdiagnostic)
        print(f"extend_model_duration: extending the model duration by a factor of "
              f"{extended.n_long / n_seg:.2f} given the priors")
        return extended

    single_pulse_model.extend_model_duration = extend_model_duration

    if _wraparoundFDdiagnostic:
        t = jnp.asarray(times, dtype=jnp.float32).reshape(-1)
        W_seg = t.shape[0] * float(t[1] - t[0])
        single_pulse_model.lowfreq_diagnostic = setup_model(
            constants_dict, freqs_channels, times, sed_fn, pulse_fn,
            fourier_freqs=jnp.array([0.5 / W_seg, 1.0 / W_seg], dtype=jnp.float32),
            _wraparoundFDdiagnostic=False,
        )

    return single_pulse_model
