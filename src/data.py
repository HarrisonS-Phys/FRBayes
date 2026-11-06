"""
data.py - Conditioning of a dedispersed dynamic spectrum for the Fourier-domain model.

prepare_burst loads a dedispersed burst, averages its channels into blocks, estimates the noise of
each block from the data before the burst, transforms each block to the Fourier domain as
X = dt * rfft(x), keeps only the low Fourier bins (dropping DC and the upper band, Nyquist
included), and picks the reference frequency. It returns a ConditionedData holding everything the
model and the likelihood need, so neither has to know how the conditioning was done. sigma_noise_fd
is the standard deviation of the real (equally, the imaginary) part of each retained bin, dt * sigma_td
* sqrt(N / 2) for white noise of standard deviation sigma_td in N samples, and times are relative to
the segment start, as model.py requires for float32.
"""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp


@dataclass(frozen=True)
class ConditionedData:
    """Everything the model, the likelihood and the plots need from a conditioned burst, as returned by prepare_burst."""

    # --- used by the likelihood ---
    data_fd: jnp.ndarray  # complex (n_chan, n_fourier): dt * rfft of the baseline-subtracted data, retained bins only
    sigma_noise_fd: jnp.ndarray  # (n_chan, 1) std of the real/imaginary part of each retained bin

    # --- used by model.setup_model ---
    freq_channels: jnp.ndarray  # (n_chan, 1) block centre frequencies (MHz)
    times: jnp.ndarray  # (n_time,) sample times relative to the segment start (s)
    fourier_freqs: jnp.ndarray  # (n_fourier,) the retained Fourier bins (Hz)
    freq_channels_ref: float  # reference frequency nu_r (MHz), from dispersion_orthogonal_reference
    channel_width: jnp.ndarray  # (n_chan, 1) width of each block (MHz), for intra-channel smearing

    # --- per-channel weights ---
    channel_weights: jnp.ndarray  # (n_chan,) each channel's share of the burst's SNR^2, from estimate_channel_weights

    # --- kept for plotting and diagnostics ---
    data_td: jnp.ndarray  # (n_chan, n_time) time-domain data, each channel's off-pulse median subtracted
    sigma_noise_td: jnp.ndarray  # (n_chan, 1) per-channel time-domain noise std
    dt: float  # time bin width (s)
    n_time: int  # number of time samples in the segment
    freq_span: jnp.ndarray  # (n_chan,) spread of the native channel centres within each block (MHz)
    native_dnu: float  # native channel width before averaging (MHz)
    n_noise_bins: int  # time bins the off-pulse noise was estimated from

    # --- where the data came from ---
    event_id: str  # identifier of the burst, carried to the saved fit
    data_path: str  # absolute path of the file loaded

    # --- for plotting: the native channels behind each block ---
    freq_members: jnp.ndarray = None  # (n_chan, n_avg) native centres (MHz) of the good channels averaged into each block; None for bursts prepared before this field existed


def block_average_channels(power, freq, n_avg):
    """
    Averages consecutive blocks of n_avg channels. Bad channels must already have been DISCARDED,
    so every block averages exactly n_avg good channels (keeping the noise per block uniform); a
    block spanning a discarded region then covers a wider range of frequency, which freq_span
    reports and which widens its intra-channel smearing. Channels at the top of the band that do not
    fill a block are dropped. Returns (power_blocks, freq_blocks, freq_span, n_dropped).

    power: (n_chan, n_time) good channels only, ordered from the bottom of the band up
    freq:  (n_chan,) their centre frequencies (MHz)
    n_avg: channels averaged per block
    """
    n_chan = freq.shape[0]
    n_blocks = n_chan // n_avg
    n_dropped = n_chan - n_blocks * n_avg

    freq_grouped = freq[: n_blocks * n_avg].reshape(n_blocks, n_avg)
    power_grouped = power[: n_blocks * n_avg].reshape(n_blocks, n_avg, -1)

    return (
        power_grouped.mean(axis=1),
        freq_grouped.mean(axis=1),  # block centre = mean of its members' frequencies
        freq_grouped.max(axis=1) - freq_grouped.min(axis=1),
        n_dropped,
    )


def estimate_noise(data_td, dt, toa=None, exclude_fraction=0.2, noise_fraction=0.2, clip=7.0,
                   verbose=True):
    """
    Per-channel median and standard deviation of the noise, in the time domain, from bins BEFORE the
    burst only (the scattering tail can reach the end of the segment). The window is the lead-in
    [0, toa] less its last exclude_fraction, or, without a toa, the first noise_fraction of the
    segment. The standard deviation, not 1.4826 x MAD, is used because each Fourier bin sums all N
    samples and so has the true variance of the samples whatever their distribution, and intensity
    noise is skewed enough that the MAD falls short of it (by 8% for 8 degrees of freedom); samples
    beyond `clip` standard deviations are left out iteratively so an RFI spike cannot inflate it.
    Raises if any channel's estimate is non-finite or non-positive. Returns (median_noise_td,
    sigma_noise_td, n_noise_bins), the first two of shape (n_chan, 1).

    data_td:          (n_chan, n_time) time-domain data
    dt:               time bin width (s)
    toa:              burst arrival time (s, relative to the segment start) from an upstream pipeline; None uses noise_fraction
    exclude_fraction: fraction of the lead-in, nearest the toa, left out because the pulse's leading edge may reach into it
    noise_fraction:   fraction of the segment used when toa is None; only safe if the burst arrives later than that
    clip:             samples further than this many standard deviations from the median are left out; None keeps every sample
    verbose:          print the window used and how many samples were clipped
    """
    n_bins = data_td.shape[-1]

    if toa is None:
        n_noise_bins = int(n_bins * noise_fraction)
        method = f"first {noise_fraction:.0%} of the segment"
    else:
        n_noise_bins = int(toa / dt * (1.0 - exclude_fraction))
        method = f"lead-in before the TOA, less its last {exclude_fraction:.0%}"

    n_noise_bins = max(min(n_noise_bins, n_bins), 1)
    if verbose:
        print(f"noise estimated from {n_noise_bins} of {n_bins} time bins ({method})")

    off_pulse = data_td[:, :n_noise_bins]

    median_noise_td = jnp.median(off_pulse, axis=1, keepdims=True)
    deviation = off_pulse - median_noise_td

    # Start from the MAD-based scale, then iterate the clipped standard deviation, so for clean noise
    # every sample ends up included
    sigma_noise_td = 1.4826 * jnp.median(jnp.abs(deviation), axis=1, keepdims=True)
    keep = jnp.ones_like(deviation, dtype=bool)
    if clip is not None:
        for _ in range(5):
            keep = jnp.abs(deviation) <= clip * sigma_noise_td
            n_keep = jnp.maximum(jnp.sum(keep, axis=1, keepdims=True), 2)
            mean_keep = jnp.sum(jnp.where(keep, off_pulse, 0.0), axis=1, keepdims=True) / n_keep
            sigma_noise_td = jnp.sqrt(jnp.sum(jnp.where(keep, (off_pulse - mean_keep) ** 2, 0.0),
                                              axis=1, keepdims=True) / (n_keep - 1))
    else:
        sigma_noise_td = jnp.std(off_pulse, axis=1, keepdims=True, ddof=1)
    if verbose and clip is not None:
        n_clipped = int(jnp.sum(~keep))
        if n_clipped:
            print(f"  {n_clipped} of {keep.size} off-pulse samples beyond {clip:g} sigma left out of "
                  f"the noise estimate")

    # A dead channel left in the file, or a window so short the spread is zero, would otherwise give
    # an infinite weight downstream
    bad = ~(jnp.isfinite(sigma_noise_td) & (sigma_noise_td > 0))
    if bool(jnp.any(bad)):
        raise ValueError(
            f"{int(jnp.sum(bad))} of {sigma_noise_td.size} channels have a non-finite or "
            f"non-positive noise estimate, from {n_noise_bins} off-pulse bins. Check that bad "
            f"channels were discarded and that the off-pulse window is long enough."
        )

    return median_noise_td, sigma_noise_td, n_noise_bins


def truncate_fourier(data_fd, fourier_freqs_full, fd_downsample):
    """
    Keeps the lowest 1 / fd_downsample of the Fourier bins, dropping DC. DC is dropped because it
    also carries the error of the subtracted baseline, which is the same in every sample and so
    lands entirely in DC (inflating its variance ~9x for a noise window of a fifth of the segment),
    while holding only ~1-2% of the SNR^2; the upper bins are dropped because the pulse's Gaussian
    envelope has rolled off there, leaving only noise. Unlike downsampling in time this is an exact
    low-pass with no aliasing. Returns (data_fd, fourier_freqs) on the retained bins.

    data_fd:            (n_chan, n_rfft) dt * rfft of the data, every bin
    fourier_freqs_full: (n_rfft,) the frequencies of those bins (Hz)
    fd_downsample:      keep the lowest 1 / fd_downsample of the bins
    """
    n_keep = fourier_freqs_full.size // fd_downsample
    keep = slice(1, n_keep)
    return data_fd[:, keep], fourier_freqs_full[keep]


def estimate_channel_weights(data_fd, sigma_noise_fd):
    """
    Each channel's share of the burst's SNR^2, w_a, normalized to sum to 1: its excess Fourier power
    over the noise expectation of 2 sigma^2 per bin, in units of sigma^2, floored at 0. Summing over
    every retained bin needs no burst window and leaves the estimate precise enough without
    smoothing across channels. Used to choose the reference frequency
    (dispersion_orthogonal_reference) and by plotting.plot_noise_window.

    data_fd:        (n_chan, n_fourier) data on the retained Fourier bins
    sigma_noise_fd: (n_chan, 1) noise std of the real/imaginary part of each bin
    """
    d = jnp.asarray(data_fd)
    s = jnp.asarray(sigma_noise_fd).reshape(-1)
    n_bins = d.shape[-1]

    # Noise alone gives E|d|^2 = 2 sigma^2 per bin (real and imaginary parts), so subtract it
    excess_power = jnp.sum(jnp.abs(d) ** 2, axis=-1) - 2.0 * n_bins * s ** 2
    weights = jnp.maximum(excess_power, 0.0) / s ** 2       # negative: consistent with pure noise

    return weights / weights.sum()


def dispersion_orthogonal_reference(freq, weights):
    """
    The reference frequency nu_r with nu_r^-2 equal to the weighted mean of nu^-2 over the channels,
    so the dispersion delay nu^-2 - nu_r^-2 averages to zero under the channel weights. This
    decouples DM from the arrival time, which otherwise trade off strongly since both shift the
    pulse in time; the arrival time is then that at the frequency of the mean dispersion delay.

    freq:    (n_chan,) channel centre frequencies (MHz)
    weights: (n_chan,) channel weights, from estimate_channel_weights
    """
    nu_inv2_mean = jnp.sum(weights * jnp.asarray(freq) ** -2) / jnp.sum(weights)
    return float(nu_inv2_mean ** -0.5)


def prepare_burst(path, event_id=None, n_avg=8, fd_downsample=4, toa=None, exclude_fraction=0.2,
                  noise_fraction=0.2, noise_clip=7.0, verbose=True):
    """
    Loads a dedispersed burst and conditions it for the Fourier-domain model (see the module
    docstring), returning a ConditionedData. The file must hold only good channels: bad channels
    DISCARDED rather than flagged, since which channels survive decides where the block boundaries,
    and so each block's span and smearing, fall.

    path:             .npz holding 'power' (n_chan, n_time), 'freq' (n_chan,) and 'time' (n_time,), already dedispersed
    event_id:         identifier of the burst, carried to the saved fit; not used in the conditioning
    n_avg:            native channels averaged per block
    fd_downsample:    keep the lowest 1 / fd_downsample of the Fourier bins
    toa:              burst arrival time (s, relative to the segment start) for the noise window; see estimate_noise
    exclude_fraction: fraction of the pre-toa lead-in, nearest the toa, left out of the noise window
    noise_fraction:   fraction of the segment used for the noise when toa is None
    noise_clip:       off-pulse samples beyond this many standard deviations are left out of the noise estimate
    verbose:          print the blocks, noise window, Fourier truncation and reference frequency
    """
    raw = jnp.load(path)
    power, freq, time = raw["power"], raw["freq"], raw["time"]

    # Order channels from the bottom of the band upward
    freq = freq[::-1]
    power = power[::-1, :]
    native_dnu = float(jnp.median(jnp.abs(jnp.diff(freq))))

    freq_native = freq
    power, freq, freq_span, n_dropped = block_average_channels(power, freq, n_avg)
    # The native centres averaged into each block, in block order, so plots can draw each block over
    # its own members with any discarded gap left visible
    freq_members = jnp.asarray(freq_native[: power.shape[0] * n_avg], dtype=jnp.float32).reshape(power.shape[0], n_avg)
    if verbose:
        print(f"{power.shape[0] * n_avg + n_dropped} good channels -> {power.shape[0]} blocks of "
              f"{n_avg} ({n_dropped} dropped); block span median "
              f"{float(jnp.median(freq_span)):.3f} MHz, max {float(freq_span.max()):.3f} MHz")

    # Times relative to the segment start, so float32 does not round away small delays
    times = jnp.asarray(time - time[0], dtype=jnp.float32)
    freq_channels = jnp.asarray(freq, dtype=jnp.float32)[:, None]
    data_td = jnp.asarray(power, dtype=jnp.float32)

    n_time = times.shape[0]
    dt = float(times[1] - times[0])

    median_noise_td, sigma_noise_td, n_noise_bins = estimate_noise(
        data_td, dt, toa=toa, exclude_fraction=exclude_fraction, noise_fraction=noise_fraction,
        clip=noise_clip, verbose=verbose,
    )
    # For display only: subtracting a constant changes only a channel's DC bin, which
    # truncate_fourier drops, so the fit is identical with or without it
    data_td = data_td - median_noise_td

    data_fd = jnp.float32(dt) * jnp.fft.rfft(data_td, axis=-1)

    # Time-domain noise std to the std of the real/imaginary part of each Fourier bin
    sigma_noise_fd = jnp.float32(dt) * sigma_noise_td * jnp.sqrt(n_time / 2.0)

    fourier_freqs_full = jnp.fft.rfftfreq(n_time, d=dt)
    data_fd, fourier_freqs = truncate_fourier(data_fd, fourier_freqs_full, fd_downsample)
    if verbose:
        print(f"Fourier truncation: keeping {fourier_freqs.size} of {fourier_freqs_full.size} bins "
              f"(up to {float(fourier_freqs[-1]) / 1e3:.1f} kHz of "
              f"{float(fourier_freqs_full[-1]) / 1e3:.1f} kHz)")

    channel_weights = estimate_channel_weights(data_fd, sigma_noise_fd)
    freq_channels_ref = dispersion_orthogonal_reference(freq, channel_weights)
    if verbose:
        print(f"weighted mean frequency {float(jnp.sum(channel_weights * jnp.asarray(freq))):.1f} MHz, "
              f"reference frequency {freq_channels_ref:.1f} MHz")

    # freq_span runs centre to centre, one native channel short of the block's full width
    channel_width = jnp.asarray(freq_span + native_dnu, dtype=jnp.float32)[:, None]

    return ConditionedData(
        event_id=str(event_id) if event_id else "",
        data_path=str(Path(path).resolve()),
        data_fd=data_fd,
        sigma_noise_fd=sigma_noise_fd,
        freq_channels=freq_channels,
        times=times,
        fourier_freqs=fourier_freqs,
        freq_channels_ref=freq_channels_ref,
        channel_width=channel_width,
        channel_weights=channel_weights,
        data_td=data_td,
        sigma_noise_td=sigma_noise_td,
        dt=dt,
        n_time=n_time,
        n_noise_bins=n_noise_bins,
        freq_span=freq_span,
        native_dnu=native_dnu,
        freq_members=freq_members,
    )
