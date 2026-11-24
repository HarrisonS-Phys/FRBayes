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
    DM_incoherent: float = 0.0  # DM removed only by incoherent dedispersion (pc cm^-3), which smears each raw channel: the total dedispersion DM less any DM removed coherently, negative if the incoherent step lowered the DM; 0 for fully coherent dedispersion
    freq_members: jnp.ndarray = None  # (n_chan, n_mem) centres (MHz) of the raw channels averaged into each channel, from data.prepare_burst; None leaves out intra-channel smearing
    raw_channel_width: float = 0.0  # width of one raw channel (MHz), burst.raw_dnu


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
    smear_runs: dict = None  # constants of intra-channel smearing, from _smearing_runs; None without it


def _dirichlet(x, n):
    """
    D_n(x) = sin(pi n x) / (n sin(pi x)), the Fourier transform of n unit impulses spaced by 1,
    averaged: the comb of n raw channels' delays. Near integer x, where sin(pi x) vanishes, it
    takes the limit cos(pi n x) / cos(pi x).

    x: the spacing of the impulses times the Fourier frequency
    n: the number of impulses (a positive integer, as a float array broadcasting with x)
    """
    s = jnp.sin(jnp.pi * x)
    small = jnp.abs(s) < 1e-4
    ratio = jnp.sin(jnp.pi * n * x) / (n * jnp.where(small, 1.0, s))
    return jnp.where(small, jnp.cos(jnp.pi * n * x) / jnp.cos(jnp.pi * x), ratio)


def _smearing_runs(freqs_channels, constants):
    """
    The constants of intra-channel dispersion smearing. Each channel's members (the raw channels
    averaged into it, constants.freq_members) are split into runs of adjacent raw channels, a new
    run starting wherever bad channels leave a gap. For a run with outer edges nu_lo and nu_hi
    (members' extremes +- half a raw channel width) and channel centre nu_a, all in MHz:
    'weight', its share of the channel's members; 'width', nu_lo^-2 - nu_hi^-2, the spread of its
    delays per unit k_DM DM; 'offset', 1 / (nu_lo nu_hi) - nu_a^-2, its mean delay (the average of
    nu^-2 over the run) less that at the channel centre; and, for incoherent dedispersion, which
    aligns raw channels at their own centres nu_m, the same two quantities for one raw channel
    at the run's centre, 'raw_width' and 'raw_offset'; and 'n_raw', its number of raw channels. Runs
    are padded with zero weight (and n_raw = 1) to the most any channel has. Computed in float64
    and returned as (n_chan, n_runs, 1) float32 arrays,
    with 'multi', the indices of the channels with more than one run (those straddling bad
    channels); None without freq_members.

    freqs_channels: channel centres nu_a (MHz), shape (n_chan,) or (n_chan, 1)
    constants:      the ModelConstants, for freq_members and raw_channel_width
    """
    if constants.freq_members is None:
        return None
    members = np.sort(np.asarray(constants.freq_members, dtype=np.float64), axis=1)
    nu_a = np.asarray(freqs_channels, dtype=np.float64).reshape(-1)
    dnu = float(constants.raw_channel_width)
    if members.shape[0] != nu_a.size:
        raise ValueError(f"freq_members has {members.shape[0]} rows but there are {nu_a.size} channels.")
    runs = [np.split(m, np.where(np.diff(m) > 1.5 * dnu)[0] + 1) for m in members]   # split at gaps
    n_runs = max(len(r) for r in runs)
    names = ("weight", "width", "offset", "raw_width", "raw_offset", "n_raw")
    out = {name: np.zeros((nu_a.size, n_runs)) for name in names}
    out["n_raw"][:] = 1.0                                   # padding runs: zero weight, and no 0 / 0 below
    for c, channel_runs in enumerate(runs):
        for r, run in enumerate(channel_runs):
            lo, hi = run.min() - 0.5 * dnu, run.max() + 0.5 * dnu
            mid = 0.5 * (run.min() + run.max())
            out["weight"][c, r] = run.size / members.shape[1]
            out["width"][c, r] = lo ** -2 - hi ** -2
            out["offset"][c, r] = 1.0 / (lo * hi) - nu_a[c] ** -2
            out["raw_width"][c, r] = (mid - 0.5 * dnu) ** -2 - (mid + 0.5 * dnu) ** -2
            out["raw_offset"][c, r] = 1.0 / ((mid - 0.5 * dnu) * (mid + 0.5 * dnu)) - mid ** -2
            out["n_raw"][c, r] = run.size
    result = {name: jnp.asarray(v[..., None], dtype=jnp.float32) for name, v in out.items()}
    result["multi"] = np.array([c for c, channel_runs in enumerate(runs) if len(channel_runs) > 1], dtype=int)
    return result


def build_fd_grid(freqs_channels: jnp.ndarray, times: jnp.ndarray, constants: ModelConstants, fourier_freqs=None) -> FDGrid:
    """
    Builds the FDGrid for the given channels and time samples. Raises if times[0] is more than 1 s
    from zero, since float32 would then round away small delays (see the module docstring).

    freqs_channels: channel centres (MHz), shape (n_chan,) or (n_chan, 1)
    times:          sample times relative to the segment start (s); the first, the spacing and the count are used
    constants:      the ModelConstants, for nu_r, epsilon and the intra-channel smearing (_smearing_runs)
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
        smear_runs=_smearing_runs(freqs_channels, constants),
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

def gaussian_arrival_fd(params_dict: dict, grid: FDGrid) -> jnp.ndarray:
    """
    The Gaussian envelope and the arrival time at nu_r of one pulse in the Fourier domain,
    exp(-2 pi^2 sigma^2 f_c^2 - 2 pi i f_c t), shape (1, n_fourier): the part of
    scattered_gaussian_fd_pulse that differs between pulses whose DM and scattering are shared. It
    depends on the Fourier frequency alone, not on the channel.

    params_dict: one pulse's parameters; reads 'mean_arrival_time' (s, at nu_r) and 'sigma' (s)
    grid:        the FDGrid the template is evaluated on
    """
    t = jnp.float32(params_dict["mean_arrival_time"])                   # centre of the Gaussian at nu_r
    sigma = jnp.float32(params_dict["sigma"])                           # intrinsic width
    return jnp.exp(-2 * (jnp.pi ** 2) * (sigma ** 2) * grid.fourier_freqs_sq - 2j * jnp.pi * grid.fourier_freqs * t)


def shared_transfer_fd(params_dict: dict, grid: FDGrid, constants: ModelConstants) -> jnp.ndarray:
    """
    The rest of scattered_gaussian_fd_pulse, shape (n_chan, n_fourier): the dispersion delay of each
    channel relative to nu_r (less the first sample's time), the scattering tail, the intra-channel
    smearing and the time bin. It depends only on DM, tau_ref and delta, so pulses that share them
    share it.

    params_dict: one pulse's parameters; reads 'DM' (residual, pc cm^-3), 'tau_ref' (s), and 'delta' if sampled
    grid:        the FDGrid the template is evaluated on
    constants:   the ModelConstants, for k_DM, epsilon, delta and DM_incoherent
    """
    if constants.epsilon != -2:
        raise ValueError("Pulse model valid only for epsilon = -2.")

    # A sampled delta is a float32 tracer; the fixed one is a Python int, so the power below is an integer power
    delta = params_dict.get("delta", constants.delta)

    DM = jnp.float32(params_dict["DM"])                                 # residual dispersion measure
    tau_ref = jnp.float32(params_dict["tau_ref"])                       # scattering time at nu_r; 0 for no scattering
    k_DM = jnp.float32(constants.k_DM)

    # Each channel's delay relative to nu_r, measured from the first sample
    delay_a = k_DM * DM * (grid.freqs_channels_disp - grid.freq_ref_disp) - grid.t_start
    tau_a = tau_ref * (grid.normalized_freqs_channels ** delta)          # scattering time in each channel

    # Intra-channel dispersion smearing S_ac (see scattered_gaussian_fd_pulse): it depends on the
    # sampled DM, so it is computed here, from the run constants _smearing_runs precomputed in build_fd_grid
    runs, smear = grid.smear_runs, 1.0
    if runs is not None:
        DM_inc = jnp.float32(constants.DM_incoherent)
        shift = k_DM * (DM * runs["offset"] + DM_inc * runs["raw_offset"])   # each run's mean delay, (n_chan, n_runs, 1)

        def spread(width, raw_width, n_raw, f):
            """
            The run's smearing in the Fourier domain. Its n raw channels are n copies of one raw
            channel's sweep, of width DM (width / n) + DM_inc raw_width, with the residual DM
            spacing them by DM (width / n): D_n(f k DM width / n) sinc(f k (DM width / n + DM_inc
            raw_width)). With DM_inc = 0 that is one boxcar over the whole run, sinc(f k DM width).
            """
            if constants.DM_incoherent == 0:
                return jnp.sinc(f * k_DM * DM * width)
            step = width / n_raw
            return (_dirichlet(f * k_DM * DM * step, n_raw)
                    * jnp.sinc(f * k_DM * (DM * step + DM_inc * raw_width)))

        # Every channel's first run: its mean-delay shift moves the whole template, so it joins the
        # channel's delay in the phase computed below instead of needing a phase of its own
        delay_a = delay_a + shift[:, 0, :]
        smear = runs["weight"][:, 0, :] * spread(runs["width"][:, 0, :], runs["raw_width"][:, 0, :],
                                                 runs["n_raw"][:, 0, :], grid.fourier_freqs)
        # The other runs exist only in the few channels straddling bad channels, so they are
        # computed for those alone, each with its delay relative to the first run
        multi = runs["multi"]
        if multi.size:
            f = grid.fourier_freqs[None]                                 # (1, 1, n_fourier)
            extra = jnp.sum(runs["weight"][multi, 1:]
                            * spread(runs["width"][multi, 1:], runs["raw_width"][multi, 1:],
                                     runs["n_raw"][multi, 1:], f)
                            * jnp.exp(-2j * jnp.pi * f * (shift[multi, 1:] - shift[multi, :1])), axis=1)
            smear = smear.astype(jnp.complex64).at[multi].add(extra)

    # Time binning, precomputed in build_fd_grid: samples are labelled at the start of their bin, so
    # each averages over [t_b, t_b + dt]
    return (jnp.exp(-2j * jnp.pi * grid.fourier_freqs * delay_a) / (1 + 2j * jnp.pi * grid.fourier_freqs * tau_a)
            * smear * grid.bin_transfer)


def scattered_gaussian_fd_pulse(params_dict: dict, grid: FDGrid, constants: ModelConstants) -> jnp.ndarray:
    """
    The Fourier transform of one pulse's shape in every channel, before its SED, shape
    (n_chan, n_fourier): a Gaussian of width sigma centred at the dispersed arrival time mu_a,
    convolved with a one-sided exponential of time constant tau_a = tau_ref (nu_a / nu_r)^delta, then
    with the intra-channel dispersion smearing S_ac and the boxcar of the time bin,

        T_ac = exp(-2 pi^2 sigma^2 f_c^2 - 2 pi i f_c mu_a) / (1 + 2 pi i f_c tau_a)
               x S_ac x sinc(f_c dt) exp(i pi f_c dt),

    with sinc(x) = sin(pi x) / (pi x) and mu_a = t + k_DM DM (nu_a^-2 - nu_r^-2) - t_start. Each run r
    of n adjacent raw channels in channel a (_smearing_runs) is n copies of one raw channel's sweep,
    of width DM W_ar / n + DM_inc V_ar (the residual DM, and with incoherent dedispersion the DM
    removed only incoherently), spaced by the residual DM's delay between neighbours, DM W_ar / n:

        S_ac = sum_r w_ar D_n(f_c k_DM DM W_ar / n) sinc(f_c k_DM (DM W_ar / n + DM_inc V_ar))
               x exp(-2 pi i f_c k_DM (DM O_ar + DM_inc P_ar)),

    with D_n(x) = sin(pi n x) / (n sin(pi x)) and w, W, O, V, P and n the run's 'weight', 'width',
    'offset', 'raw_width', 'raw_offset' and 'n_raw'. With DM_inc = 0 a run is one boxcar,
    D_n(x) sinc(x) = sinc(n x), so S_ac = sum_r w_ar sinc(f_c k_DM DM W_ar) exp(-2 pi i f_c k_DM DM O_ar). It is
    computed as gaussian_arrival_fd x shared_transfer_fd: the first holds everything that differs
    between pulses whose DM and scattering are shared, the second everything that does not, so a
    burst of such pulses can build the second once (likelihood.FD_gaussianlikelihood).

    params_dict: one pulse's parameters; reads 'mean_arrival_time' (s, at nu_r), 'sigma' (s), 'DM' (residual, pc cm^-3), 'tau_ref' (s), and 'delta' if sampled
    grid:        the FDGrid the template is evaluated on
    constants:   the ModelConstants, for k_DM, epsilon, delta and DM_incoherent
    """
    return gaussian_arrival_fd(params_dict, grid) * shared_transfer_fd(params_dict, grid, constants)


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
    .segment_duration, .model_duration_pad, .n_long, .extend_model_duration, .pulse_factors and
    .shared_factor (the factored form of the default pulse, else None) and .cut (the cut of a padded
    template to the data window, None without padding).

    constants_dict:     keyword arguments of ModelConstants
    freqs_channels:     channel centres (MHz), shape (n_chan,) or (n_chan, 1)
    times:              sample times relative to the segment start (s)
    sed_fn:             function (params_dict, grid, constants) -> ln F, shape (n_chan, 1)
    pulse_fn:           function (params_dict, grid, constants) -> T, shape (n_chan, n_fourier)
    fourier_freqs:      the data's Fourier bins (Hz), from data.prepare_burst; None uses every rfft bin
    model_duration_pad: padding against wraparound (s), rounded up to an FFT-friendly window; 0 (default) for none; usually set with .extend_model_duration
    """
    constants = ModelConstants(**constants_dict)
    if constants.DM_incoherent != 0 and (constants.freq_members is None or constants.raw_channel_width <= 0):
        raise ValueError("a non-zero DM_incoherent needs freq_members and raw_channel_width: intra-channel "
                         "smearing would be silently omitted.")
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

    # The factored form of the default pulse, for bursts whose pulses share DM and scattering: each
    # pulse's ln F (n_chan, 1) and Gaussian-and-arrival factor (1, n_eval), and the factor they all
    # share (n_chan, n_eval), on the grid templates are evaluated on (the padded window's, with
    # padding); .cut takes a padded template to the data's window and retained bins (None without
    # padding)
    factored = pulse_fn is scattered_gaussian_fd_pulse
    single_pulse_model.pulse_factors = (
        (lambda params_dict: (sed_fn(params_dict, eval_grid, constants), gaussian_arrival_fd(params_dict, eval_grid)))
        if factored else None)
    single_pulse_model.shared_factor = (
        (lambda params_dict: shared_transfer_fd(params_dict, eval_grid, constants)) if factored else None)
    single_pulse_model.cut = truncate

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
        from .priors import model_duration_pad_for_priors

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
