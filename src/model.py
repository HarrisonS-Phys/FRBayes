"""
model.py - The pulse model in the Fourier domain. setup_model returns single_pulse_fn, which maps one
pulse's parameters to its template on every channel and retained Fourier bin of the data.

A pulse is a Gaussian in time, dispersed, scattered by a one-sided exponential tail, smeared within
each channel and averaged over each time bin, with a running power-law spectrum; its template is the
analytic Fourier transform of that, evaluated at the data's bins. Everything is float32 (fast on
GPUs, ~7 significant digits), so times must be relative to the segment start, DM must be a small
residual about the DM the data were dedispersed with, and SEDs are returned as ln F and exponentiated
only after normalizing, since F itself overflows float32 once ln F > 88.

The analytic template at the bins k / W is the transform of the pulse repeated with period W, so
emission past one end of the window wraps around to the other, whereas the data were simply cut
from a longer recording. The padded model (model_duration_pad > 0, usually set with
single_pulse_fn.extend_model_duration) evaluates the template on a longer window, transforms it to
time, cuts it to the data window as the data were cut, and transforms it back, so nothing wraps.
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import jax.numpy as jnp


# -------------------------------------------------------------------
# 1. Constants and the evaluation grid
# -------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConstants:
    """Fixed constants of the model, given to setup_model as a dict of these names."""

    freq_channels_ref: float  # reference frequency nu_r (MHz), at which arrival times, tau_ref and the SED's x = ln(nu / nu_r) are defined
    det_response: float = 1.0  # detector response multiplying the SED; a constant for now (TO-DO: frequency-dependent beam response)
    delta: int | float = -4  # scattering index, tau proportional to nu^delta; a Python int so nu^delta lowers to an integer power (a sampled 'delta' overrides it)
    k_DM: float = (2.41 * 1e-4) ** (-1)  # dispersion constant (pc^-1 cm^3 MHz^2 s)
    epsilon: int | float = -2  # dispersion index, delay proportional to nu^epsilon; the pulse model requires -2
    DM_incoherent: float = 0.0  # DM removed only by incoherent dedispersion (pc cm^-3), which smears each channel; 0 for coherent dedispersion
    channel_width: float | jnp.ndarray = 0.0  # channel width (MHz), a scalar or (n_chan, 1) per block; 0 leaves out intra-channel smearing


@dataclass(frozen=True)
class FDGrid:
    """The channels and Fourier bins a template is evaluated on, and the factors that depend only on them, built once by build_fd_grid."""

    freqs_channels: jnp.ndarray  # channel centres nu_a (MHz), shape (n_chan, 1)
    normalized_freqs_channels: jnp.ndarray  # nu_a / nu_r, shape (n_chan, 1)
    log_normalized_freqs_channels: jnp.ndarray  # x_a = ln(nu_a / nu_r), the SED's variable, shape (n_chan, 1)
    fourier_freqs: jnp.ndarray  # Fourier bins f_c (Hz), shape (1, n_fourier)
    t_start: jnp.ndarray  # time of the first sample (s), the origin of the template's phase
    # The scattering factor (nu_a / nu_r)^delta is not precomputed, since delta may be sampled
    fourier_freqs_sq: jnp.ndarray = None  # f_c^2, for the Gaussian envelope
    freqs_channels_disp: jnp.ndarray = None  # nu_a^epsilon, for the dispersion delay
    freq_ref_disp: jnp.ndarray = None  # nu_r^epsilon, the reference frequency's dispersion delay
    bin_transfer: jnp.ndarray = None  # sinc(f_c dt) exp(i pi f_c dt), the effect of averaging over a time bin


def build_fd_grid(freqs_channels: jnp.ndarray, times: jnp.ndarray, constants: ModelConstants, fourier_freqs=None) -> FDGrid:
    """
    Builds the FDGrid for the given channels and time samples. Raises if times[0] is more than 1 s
    from zero, since float32 would then round away small delays (see the module docstring).

    freqs_channels: channel centres (MHz), shape (n_chan,) or (n_chan, 1)
    times:          sample times relative to the segment start (s); the first, the spacing and the count are used
    constants:      the ModelConstants, for nu_r and epsilon
    fourier_freqs:  the Fourier bins to evaluate on (Hz), from data.prepare_burst; None uses every rfft bin of `times`
    """
    freqs_channels = jnp.asarray(freqs_channels, dtype=jnp.float32)
    if freqs_channels.ndim == 1:
        freqs_channels = freqs_channels[:, None]

    times = jnp.asarray(times, dtype=jnp.float32)
    times_1d = times[0] if times.ndim > 1 else times

    if abs(float(times_1d[0])) > 1.0:
        raise ValueError(
            f"times[0] = {times_1d[0]:.3f} s. `times` must be relative to segment start "
            f"(abs(t_start) <= 1.0 s) to prevent float32 rounding errors. "
            f"See module note on precision."
        )

    freq_channels_ref = jnp.float32(constants.freq_channels_ref)
    normalized_freqs = freqs_channels / freq_channels_ref                  # nu_a / nu_r

    Nseg = times_1d.shape[0]
    dt = times_1d[1] - times_1d[0]
    if fourier_freqs is None:
        fourier_freqs = jnp.fft.rfftfreq(Nseg, d=dt, dtype=jnp.float32)[None, :]
    else:
        fourier_freqs = jnp.asarray(fourier_freqs, dtype=jnp.float32).reshape(1, -1)

    # epsilon stays a Python int, so these lower to integer powers
    epsilon = constants.epsilon

    return FDGrid(
        freqs_channels=freqs_channels,
        normalized_freqs_channels=normalized_freqs,
        log_normalized_freqs_channels=jnp.log(normalized_freqs),
        fourier_freqs=fourier_freqs,
        t_start=times_1d[0],
        fourier_freqs_sq=fourier_freqs ** 2,
        freqs_channels_disp=freqs_channels ** epsilon,
        freq_ref_disp=freq_channels_ref ** epsilon,
        # Samples are labelled at the start of their bin, so each averages over [t_b, t_b + dt]: a
        # boxcar whose transform is sinc(f dt) with a half-bin phase shift
        bin_transfer=jnp.sinc(fourier_freqs * dt) * jnp.exp(1j * jnp.pi * fourier_freqs * dt),
    )


# -------------------------------------------------------------------
# 2. Spectral energy distribution (returned as ln F)
# -------------------------------------------------------------------

def running_power_law_log_sed(params_dict: dict, grid: FDGrid, constants: ModelConstants) -> jnp.ndarray:
    """
    ln F_a of the running power law F_a = (nu_a / nu_r)^(gamma + beta x_a) * det_response, with
    x_a = ln(nu_a / nu_r), i.e. ln F_a = (gamma + beta x_a) x_a + ln(det_response); shape (n_chan, 1).

    params_dict: one pulse's parameters; reads 'gamma' (spectral index) and 'beta' (spectral running)
    grid:        the FDGrid, for x_a
    constants:   the ModelConstants, for det_response
    """
    gamma = jnp.float32(params_dict["gamma"])
    beta = jnp.float32(params_dict["beta"])
    det_response = jnp.float32(constants.det_response)

    kappa = gamma + beta * grid.log_normalized_freqs_channels              # running spectral index
    return kappa * grid.log_normalized_freqs_channels + jnp.log(det_response)


# -------------------------------------------------------------------
# 3. Pulse shape in the Fourier domain
# -------------------------------------------------------------------

def scattered_gaussian_fd_pulse(params_dict: dict, grid: FDGrid, constants: ModelConstants) -> jnp.ndarray:
    """
    The Fourier transform of one pulse's shape in every channel, before its SED, shape
    (n_chan, n_fourier): a Gaussian of width sigma centred at the dispersed arrival time mu_a,
    convolved with a one-sided exponential of time constant tau_a = tau_ref (nu_a / nu_r)^delta, then
    with the boxcars of intra-channel dispersion smearing and of the time bin,

        T_ac = exp(-2 pi^2 sigma^2 f_c^2 - 2 pi i f_c mu_a) / (1 + 2 pi i f_c tau_a)
               x sinc(f_c dt_smear,a) x sinc(f_c dt) exp(i pi f_c dt),

    with sinc(x) = sin(pi x) / (pi x) and mu_a = t + k_DM DM (nu_a^-2 - nu_r^-2) - t_start.

    params_dict: one pulse's parameters; reads 'mean_arrival_time' (s, at nu_r), 'sigma' (s), 'DM' (residual, pc cm^-3), 'tau_ref' (s), and 'delta' if sampled
    grid:        the FDGrid the template is evaluated on
    constants:   the ModelConstants, for k_DM, epsilon, delta, DM_incoherent and channel_width
    """
    if constants.epsilon != -2:
        raise ValueError("Pulse model valid only for epsilon = -2.")

    # A sampled delta is a float32 tracer; the fixed one is a Python int, so the power below is an integer power
    delta = params_dict.get("delta", constants.delta)

    mean_arrival_time = jnp.float32(params_dict["mean_arrival_time"])   # centre of the Gaussian at nu_r
    sigma = jnp.float32(params_dict["sigma"])                           # intrinsic width
    DM = jnp.float32(params_dict["DM"])                                 # residual dispersion measure
    tau_ref = jnp.float32(params_dict["tau_ref"])                       # scattering time at nu_r; 0 for no scattering
    k_DM = jnp.float32(constants.k_DM)

    # Centre of the Gaussian in each channel, relative to the first sample
    mu_a = mean_arrival_time + k_DM * DM * (grid.freqs_channels_disp - grid.freq_ref_disp) - grid.t_start
    tau_a = tau_ref * (grid.normalized_freqs_channels ** delta)          # scattering time in each channel

    T = jnp.exp(
        -2 * ((jnp.pi ** 2) * (sigma ** 2) * grid.fourier_freqs_sq)
        - 2j * (jnp.pi * grid.fourier_freqs * mu_a)
    ) / (1 + 2j * jnp.pi * grid.fourier_freqs * tau_a)

    # Intra-channel dispersion smearing: the delay difference across each channel (epsilon = -2), a
    # boxcar centred on the channel's own delay. It depends on the sampled DM, so it is computed here.
    # TO-DO: check this convention -> channels are labelled at their centres, so the smearing boxcar is centred on the channel-centre delay (no phase).
    DM_inc = jnp.float32(constants.DM_incoherent)
    channel_width = jnp.float32(constants.channel_width)
    dt_smear = k_DM * (DM_inc + DM) * (
        (2.0 * grid.freqs_channels * channel_width) / ((grid.freqs_channels ** 2 - (channel_width ** 2 / 4.0)) ** 2)
    )
    instrmnt_smear_transfer = jnp.sinc(grid.fourier_freqs * dt_smear)

    # Time binning, precomputed in build_fd_grid.
    # TO-DO: check this convention -> bins are labelled at their start, so each sample averages over [t_b, t_b + dt_bin].
    return T * instrmnt_smear_transfer * grid.bin_transfer


# -------------------------------------------------------------------
# 4. Padding against wraparound
# -------------------------------------------------------------------

def _fft_friendly(n):
    """
    Smallest n' >= n whose only prime factors are 2, 3 and 5, so the FFTs of the padded window stay fast.

    n: the shortest acceptable length, in samples
    """
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


# -------------------------------------------------------------------
# 5. The model closure
# -------------------------------------------------------------------

def setup_model(
    constants_dict: dict,
    freqs_channels: jnp.ndarray,
    times: jnp.ndarray,
    sed_fn: Callable = running_power_law_log_sed,
    pulse_fn: Callable = scattered_gaussian_fd_pulse,
    fourier_freqs=None,
    model_duration_pad: float = 0.0,
):
    """
    Builds single_pulse_fn(params_dict), which returns one pulse's template on every channel and
    retained Fourier bin, shape (n_chan, n_fourier). With 'A' in params_dict the template is A F T,
    the pulse at that fluence; without it (A_snr fits) the SED is scaled to peak at 1 across the
    channels, exp(ln F - max ln F) T, which cannot overflow, and likelihood.FD_gaussianlikelihood
    sets the amplitude. The closure also carries .constants, .grid (the data's FDGrid),
    .log_sed_peak, .sed_fn and .pulse_fn (the forms' names, recorded by results.save_run),
    .segment_duration, .model_duration_pad, .n_long and .extend_model_duration.

    constants_dict:     keyword arguments of ModelConstants
    freqs_channels:     channel centres (MHz), shape (n_chan,) or (n_chan, 1)
    times:              sample times relative to the segment start (s)
    sed_fn:             function (params_dict, grid, constants) -> ln F, shape (n_chan, 1)
    pulse_fn:           function (params_dict, grid, constants) -> T, shape (n_chan, n_fourier)
    fourier_freqs:      the data's Fourier bins (Hz), from data.prepare_burst; None uses every rfft bin
    model_duration_pad: padding against wraparound (s), rounded up to an FFT-friendly window; 0 (default) for none; usually set with .extend_model_duration
    """
    constants = ModelConstants(**constants_dict)
    # channel_width may be a scalar or a per-channel array, so reduce the comparison before testing it
    if constants.DM_incoherent > 0 and bool(jnp.all(jnp.asarray(constants.channel_width) == 0)):
        raise ValueError("DM_incoherent > 0 but channel_width = 0: intra-channel smearing would be silently omitted.")
    grid = build_fd_grid(freqs_channels, times, constants, fourier_freqs)

    # ---- Padding: the longer window the template is evaluated on, and the cut back to the data ----
    if model_duration_pad < 0:
        raise ValueError(f"model_duration_pad must be >= 0; got {model_duration_pad}.")
    eval_grid, n_long, truncate = grid, None, None
    if model_duration_pad > 0:
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        n_seg, dt = t.size, float(t[1] - t[0])
        n_long = _fft_friendly(n_seg + int(math.ceil(model_duration_pad / dt)))
        # Same start and spacing as the data, only longer; every rfft bin of it, DC included, since
        # cutting in time mixes all frequencies
        times_long = (t[0] + dt * np.arange(n_long)).astype(np.float32)
        eval_grid = build_fd_grid(freqs_channels, times_long, constants, fourier_freqs=None)

        # Where each of the data's retained bins sits on the segment's own rfft grid
        f_data = np.asarray(grid.fourier_freqs, dtype=np.float64).reshape(-1)
        bin_index = np.rint(f_data * n_seg * dt).astype(int)
        if not np.allclose(bin_index, f_data * n_seg * dt, atol=1e-3):
            raise ValueError("fourier_freqs are not bins of the segment's rfft grid (k / (N dt)).")
        bin_index = jnp.asarray(bin_index)

        def truncate(M_long):
            # To time on the long window, cut to [0, W) as the data were cut, and back onto the data's
            # retained bins; the dt factors of X = dt rfft(x) cancel between the two transforms
            x = jnp.fft.irfft(M_long, n=n_long, axis=-1)
            return jnp.fft.rfft(x[..., :n_seg], axis=-1)[..., bin_index]

    def single_pulse_model(params_dict):
        """
        One pulse's template on the data's channels and retained Fourier bins, shape
        (n_chan, n_fourier): A F T with 'A' in params_dict, otherwise the template with its SED peak
        scaled to 1 (see setup_model).

        params_dict: one pulse's parameters as scalars; the SED and pulse functions read the ones they need
        """
        lnF = sed_fn(params_dict, eval_grid, constants)                    # (n_chan, 1)
        T = pulse_fn(params_dict, eval_grid, constants)

        if "A" in params_dict:
            # A F = sign(A) exp(ln|A| + ln F), combined in log space so a large F times a small A
            # cannot overflow; the sign is kept apart because the log needs |A|
            A = jnp.float32(params_dict["A"])
            M = jnp.sign(A) * jnp.exp(jnp.log(jnp.abs(A)) + lnF) * T
        else:
            M = jnp.exp(lnF - jnp.max(lnF)) * T                            # SED peak scaled to 1

        # With padding: cut to the data window like the data, on the data's retained bins
        return M if truncate is None else truncate(M)

    def log_sed_peak(params_dict):
        """
        ln max_a F_a, the factor single_pulse_model divides out when 'A' is absent: an amplitude B on
        that template is the fluence B exp(-log_sed_peak).

        params_dict: one pulse's parameters as scalars
        """
        return jnp.max(sed_fn(params_dict, grid, constants))

    t_seg = jnp.asarray(times, dtype=jnp.float32).reshape(-1)
    single_pulse_model.constants = constants
    single_pulse_model.grid = grid                                         # the data's grid, on which templates are returned
    single_pulse_model.model_duration_pad = float(model_duration_pad)      # padding (s) requested; 0 without
    single_pulse_model.segment_duration = t_seg.shape[0] * float(t_seg[1] - t_seg[0])   # W (s)
    single_pulse_model.n_long = n_long                                     # samples in the padded window; None without padding
    single_pulse_model.sed_fn = sed_fn.__name__                            # names of the forms, recorded by results.save_run
    single_pulse_model.pulse_fn = pulse_fn.__name__
    single_pulse_model.log_sed_peak = log_sed_peak

    def extend_model_duration(burst=None, priors=None, wrapped_snr=1.0, n_tau=3.0, n_sigma=3.0,
                              time_param="mean_arrival_time", factor=None):
        """
        Returns this model rebuilt with padding against wraparound: the smallest padding for which no
        pulse the priors allow wraps detectably back into the window, or, with `factor`, a fixed
        padding. The search builds worst-case pulses from the prior bounds (arrival time at either
        end, the widest sigma, the longest tau_ref, residual DM at either bound or zero, flat
        spectrum) and compares each with the same pulse on a window long enough that nothing wraps;
        weighting each channel by this burst's share of SNR^2 there, it bisects over FFT-friendly
        lengths for the smallest padding with (wrapped fraction of the power) x SNR^2 <= wrapped_snr^2
        for every worst case, up to the padding priors.model_duration_pad_for_priors gives (used, with
        a warning, if even that is not enough). Usage:
            single_pulse_fn = single_pulse_fn.extend_model_duration(burst, priors)   # or (factor=2)

        burst:       the ConditionedData of the fit, for the SNR^2 per channel (not needed with factor)
        priors:      the priors of the fit, from run.set_priors, whose bounds give the worst cases (not needed with factor)
        wrapped_snr: largest SNR the wrapped emission may have; 1 (default) keeps it at the noise level
        n_tau:       e-folds of the scattering tail the largest padding searched holds
        n_sigma:     Gaussian widths the largest padding searched holds
        time_param:  name of the arrival-time prior
        factor:      if given (> 1), skips the search and pads the window to factor x the segment, rounded up to an FFT-friendly length, without checking it is enough
        """
        if factor is not None:
            factor = float(factor)
            if factor <= 1.0:
                raise ValueError(f"factor is the extended duration over the segment's and must be > 1; "
                                 f"got {factor}.")
            extended = setup_model(constants_dict, freqs_channels, times, sed_fn, pulse_fn,
                                   fourier_freqs=fourier_freqs,
                                   model_duration_pad=(factor - 1.0) * single_pulse_model.segment_duration)
            n_seg = int(np.asarray(times).reshape(-1).shape[0])
            print(f"extend_model_duration: extending the model duration by a factor of "
                  f"{extended.n_long / n_seg:.2f} (requested {factor:g})")
            return extended
        if burst is None or priors is None:
            raise ValueError("extend_model_duration needs burst and priors, or a fixed factor.")

        # Imported here: priors.py does not import this module, so there is no cycle
        import warnings
        from priors import model_duration_pad_for_priors

        pad_max = model_duration_pad_for_priors(single_pulse_model, priors, n_tau=n_tau, n_sigma=n_sigma,
                                                verbose=False)
        n_seg = int(np.asarray(times).reshape(-1).shape[0])
        W = single_pulse_model.segment_duration

        # ---- this burst's SNR^2 per channel: excess power over the noise expectation of 2 per bin ----
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
        flat_sed = dict(gamma=0.0, beta=0.0)                       # ln F = 0: the shape alone
        if "delta" in priors:
            flat_sed["delta"] = bound("delta", "low")              # steepest scattering law
        pulses = [dict(flat_sed, mean_arrival_time=t, sigma=bound("sigma", "high"),
                       tau_ref=bound("tau_ref", "high"), DM=dm)
                  for t in (bound(time_param, "low"), bound(time_param, "high"))
                  for dm in (-DM_max, 0.0, DM_max)]

        def build(pad):
            return setup_model(constants_dict, freqs_channels, times, sed_fn, pulse_fn,
                               fourier_freqs=fourier_freqs, model_duration_pad=pad)

        # Reference: a window long enough that nothing the priors allow wraps
        reference = build(4.0 * pad_max + W)
        refs = [np.asarray(reference(p)) for p in pulses]

        def wrapped_fraction(pad):
            """Largest SNR^2-weighted fraction of a worst-case pulse's power that wraps with this padding (s)."""
            fn = build(pad)
            worst = 0.0
            for p, ref in zip(pulses, refs):
                per_channel = (np.sum(np.abs(np.asarray(fn(p)) - ref) ** 2, axis=-1)
                               / np.maximum(np.sum(np.abs(ref) ** 2, axis=-1), np.finfo(float).tiny))
                worst = max(worst, float(np.sum(weights * per_channel)))
            return worst

        # ---- smallest adequate padding: bisection over the FFT-friendly window lengths ----
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

        extended = build(pad)
        print(f"extend_model_duration: extending the model duration by a factor of "
              f"{extended.n_long / n_seg:.2f} given the priors")
        return extended

    single_pulse_model.extend_model_duration = extend_model_duration

    return single_pulse_model
