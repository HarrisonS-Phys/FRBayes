"""
data.py - Conditioning of a dedispersed dynamic spectrum for the Fourier-domain model.

prepare_burst loads a dedispersed burst, averages its channels into blocks, estimates the noise of
each block from the data before the burst, transforms each block to the Fourier domain as
X = dt * rfft(x), keeps only the low Fourier bins (dropping DC and the upper band, Nyquist
included), and picks the reference frequency from the channels' signal power inside the boxes an
excess-power search of the data finds significant. It returns a ConditionedData holding everything the
model and the likelihood need, so neither has to know how the conditioning was done. sigma_noise_fd
is the standard deviation of the real (equally, the imaginary) part of each retained bin, dt * sigma_td
* sqrt(N / 2) for white noise of standard deviation sigma_td in N samples, and times are relative to
the segment start, as model.py requires for float32.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np


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
    freq_channels_ref: float  # reference frequency nu_r (MHz), from dispersion_orthogonal_reference with channel_weights
    channel_width: jnp.ndarray  # (n_chan, 1) full extent of each block (MHz), member edge to member edge, gaps included; recorded with the results (the model's smearing uses freq_members)

    # --- per-channel weights ---
    channel_weights: jnp.ndarray  # (n_chan,) each channel's share of the burst's SNR^2, from box_channel_weights (estimate_channel_weights if no box is significant)

    # --- kept for plotting and diagnostics ---
    data_td: jnp.ndarray  # (n_chan, n_time) time-domain data, each channel's noise-window mean subtracted
    sigma_noise_td: jnp.ndarray  # (n_chan, 1) per-channel time-domain noise std
    dt: float  # time bin width (s)
    n_time: int  # number of time samples in the segment
    freq_span: jnp.ndarray  # (n_chan,) spread of the raw channel centres within each block (MHz)
    raw_dnu: float  # raw channel width before averaging (MHz)
    n_noise_bins: int  # time bins the off-pulse noise was estimated from

    # --- where the data came from ---
    event_id: str  # identifier of the burst, carried to the saved fit
    data_path: str  # absolute path of the file loaded

    # --- for plotting: the raw channels behind each block ---
    freq_members: jnp.ndarray = None  # (n_chan, n_avg) raw centres (MHz) of the good channels averaged into each block


def block_average_channels(power, freq, n_avg):
    """
    Averages consecutive blocks of n_avg channels. Bad channels must already have been REMOVED,
    so every block averages exactly n_avg good channels (keeping the noise per block uniform); a
    block spanning a region of bad channels then covers a wider range of frequency, which freq_span
    reports; the model smears each unbroken group of its raw channels separately. Channels at the top of the band that do not
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


def estimate_noise(data_td, dt, toa=None, exclude_fraction=0.2, noise_fraction=0.2, verbose=True):
    """
    Per-channel mean and standard deviation of the noise, in the time domain, from bins BEFORE the
    burst only (the scattering tail can reach the end of the segment). The window is the lead-in
    [0, toa] less its last exclude_fraction, or, without a toa, the first noise_fraction of the
    segment. The standard deviation is used because each Fourier bin sums all N samples, so its
    variance is set by the variance of the samples whatever their distribution; every sample in the
    window counts, so the window must be free of RFI. The mean is the baseline subtracted for
    display. Raises if any channel's estimate is non-finite or non-positive. Returns
    (mean_noise_td, sigma_noise_td, n_noise_bins), the first two of shape (n_chan, 1).

    data_td:          (n_chan, n_time) time-domain data
    dt:               time bin width (s)
    toa:              burst arrival time (s, relative to the segment start) from an upstream pipeline; None uses noise_fraction
    exclude_fraction: fraction of the lead-in, nearest the toa, left out because the pulse's leading edge may reach into it
    noise_fraction:   fraction of the segment used when toa is None; only safe if the burst arrives later than that
    verbose:          print the window used
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

    mean_noise_td = jnp.mean(off_pulse, axis=1, keepdims=True)
    sigma_noise_td = jnp.std(off_pulse, axis=1, keepdims=True, ddof=1)

    # A dead channel left in the file, or a window so short the spread is zero, would otherwise give
    # an infinite weight downstream
    bad = ~(jnp.isfinite(sigma_noise_td) & (sigma_noise_td > 0))
    if bool(jnp.any(bad)):
        raise ValueError(
            f"{int(jnp.sum(bad))} of {sigma_noise_td.size} channels have a non-finite or "
            f"non-positive noise estimate, from {n_noise_bins} off-pulse bins. Check that bad "
            f"channels were removed and that the off-pulse window is long enough."
        )

    return mean_noise_td, sigma_noise_td, n_noise_bins


def truncate_fourier(data_fd, fourier_freqs_full, fd_downsample, n_time):
    """
    Keeps the lowest 1 / fd_downsample of the Fourier bins, dropping DC and, when n_time is even,
    the Nyquist bin. DC is dropped because it also carries the error of the subtracted baseline,
    which is the same in every sample and so lands entirely in DC (4x its variance without the
    subtraction, for a noise window of a fifth of the segment), while holding only ~1-2% of the
    SNR^2. The Nyquist bin is never kept because, like DC, it is purely real, so its noise is not
    split evenly between a real and an imaginary part as sigma_noise_fd assumes. The upper bins are
    dropped because the pulse's Gaussian envelope has rolled off there, leaving only noise. Unlike
    downsampling in time this is an exact low-pass with no aliasing. Returns (data_fd,
    fourier_freqs) on the retained bins.

    data_fd:            (n_chan, n_rfft) dt * rfft of the data, every bin
    fourier_freqs_full: (n_rfft,) the frequencies of those bins (Hz)
    fd_downsample:      keep the lowest 1 / fd_downsample of the bins
    n_time:             number of time samples the rfft was taken over; when even, the last bin is Nyquist
    """
    n_rfft = fourier_freqs_full.size
    n_complex = n_rfft - 1 if n_time % 2 == 0 else n_rfft   # bins before Nyquist, which has none when n_time is odd
    keep = slice(1, min(n_rfft // fd_downsample, n_complex))
    return data_fd[:, keep], fourier_freqs_full[keep]


def estimate_channel_weights(data_fd, sigma_noise_fd):
    """
    Each channel's share of the burst's SNR^2, w_a, normalized to sum to 1: its excess Fourier power
    over the noise expectation of 2 sigma^2 per bin, in units of sigma^2, floored at 0. Summing over
    every retained bin needs no burst window, but every bin's noise enters, so at low SNR the floor
    gives channels without signal a positive weight. prepare_burst uses it only when
    box_channel_weights finds no significant box.

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


def significant_boxes(burst, min_snr=8.0, p_false_alarm=0.01):
    """
    The boxes an excess-power search of the data finds significant (postprocess.excess_power_search
    with no model: one box per separate feature), by the same test the fits use to add a pulse.
    They give the channel weights (box_channel_weights) and the check of the Fourier truncation
    (truncation_check).

    burst:         a ConditionedData; its data, noise and grid are searched
    min_snr:       SNR a box needs, besides its p-value, to be significant, as in the fits' search
    p_false_alarm: false-alarm probability of the search, after its correction for the number of boxes
    """
    from postprocess import excess_power_search

    data_fd = np.asarray(burst.data_fd)
    result = excess_power_search(burst, np.zeros_like(data_fd), min_snr=min_snr,
                                 p_false_alarm=p_false_alarm, verbose=False)
    return [t for t in result["tiles"] if t["significant"]]


def box_channel_weights(burst, boxes):
    """
    Each channel's share of the burst's SNR^2, measured only inside the significant boxes of the
    data: the channel's power in each box's cells less the noise expectation of 2 per cell, summed
    over the boxes, floored at 0 and normalized to sum to 1. Summing over the boxes' cells rather
    than every retained bin leaves far less noise in each channel, so the floor hardly biases the
    weights, and channels outside every box get none. Returns None if there are no boxes.

    burst: a ConditionedData
    boxes: its significant boxes, from significant_boxes
    """
    from postprocess import _cell_powers

    if not boxes:
        return None
    data_fd = np.asarray(burst.data_fd)
    Z = data_fd / np.broadcast_to(np.asarray(burst.sigma_noise_fd), data_fd.shape)   # whitened data
    excess = np.zeros(Z.shape[0])
    for box in boxes:
        (c0, c1), (b0, b1), (t0, t1) = box["channel_index"], box["band_index"], box["time_index"]
        cells = _cell_powers(Z, box["resolution"])[c0:c1, b0:b1, t0:t1]
        excess[c0:c1] += np.sum(cells - 2.0, axis=(1, 2))
    weights = np.maximum(excess, 0.0)
    return jnp.asarray(weights / weights.sum(), dtype=jnp.float32)


def truncation_check(burst, data_fd_all, boxes, n_sigma=3.0, verbose=True):
    """
    Does the Fourier truncation throw away a measurable part of the burst? In the channels and time
    cells of each significant box, at the box's own resolution, it compares the excess power (power
    less the noise expectation of 2 per cell) in every retained band, rho_kept^2, with that in the
    bands above the cut (Nyquist left out). With R their ratio, keeping every bin would raise the
    burst's SNR by rho_kept (sqrt(1 + R) - 1); it warns that a smaller fd_downsample is needed if that
    gain exceeds 1, the uncertainty of a measured SNR, and the excess above the cut exceeds n_sigma
    times its noise standard deviation, 2 sqrt(V) for V cells, so that noise alone rarely sets it
    off. Measured on the data rather than a fitted model, it is not limited by the width prior's
    floor, the narrowest width the retained bins resolve. Returns a dict: 'snr_kept', 'snr_all',
    'gain', 'excess_above', 'noise_above' and 'too_few_bins'; None if there are no boxes.

    burst:       the ConditionedData, on its retained bins
    data_fd_all: (n_chan, n_rfft) dt * rfft of the same data, every bin
    boxes:       its significant boxes, from significant_boxes
    n_sigma:     noise standard deviations the excess above the cut must exceed for a warning
    verbose:     print the result
    """
    import warnings
    from postprocess import _cell_powers

    if not boxes:
        return None
    sigma = np.asarray(burst.sigma_noise_fd, dtype=float)
    n_kept = np.asarray(burst.data_fd).shape[-1]                   # retained bins are k = 1 ... n_kept
    k_top = (int(burst.n_time) - 1) // 2                           # last bin below Nyquist
    Z_kept = np.asarray(burst.data_fd) / sigma
    Z_above = np.asarray(data_fd_all)[:, n_kept + 1:k_top + 1] / sigma

    below = above = n_above = 0.0
    cells_kept, cells_above = {}, {}
    for box in boxes:
        n_b = box["resolution"]
        (c0, c1), (t0, t1) = box["channel_index"], box["time_index"]
        if n_b not in cells_kept:
            cells_kept[n_b] = _cell_powers(Z_kept, n_b)
            cells_above[n_b] = _cell_powers(Z_above, n_b) if Z_above.shape[-1] >= n_b else None
        below += float(np.sum(cells_kept[n_b][c0:c1, :, t0:t1] - 2.0))
        if cells_above[n_b] is not None:
            P = cells_above[n_b][c0:c1, :, t0:t1]
            above += float(np.sum(P - 2.0))
            n_above += P.size

    snr_kept = float(np.sqrt(max(below, 0.0)))
    snr_all = float(np.sqrt(max(below, 0.0) + max(above, 0.0)))
    gain = snr_all - snr_kept
    noise_above = 2.0 * np.sqrt(n_above)
    too_few = bool(gain > 1.0 and above > n_sigma * noise_above)
    if verbose:
        print(f"Fourier truncation check: in the significant boxes, SNR {snr_kept:.1f} in the retained bins, "
              f"{snr_all:.1f} with every bin (gain {gain:.2f}); excess power above the cut {above:.0f}, "
              f"noise sd {noise_above:.0f}")
    if too_few:
        warnings.warn(f"the bins above the Fourier cut hold part of the burst: keeping them would raise its SNR "
                      f"from {snr_kept:.1f} to {snr_all:.1f}. Prepare the burst with a smaller fd_downsample.")
    return dict(snr_kept=snr_kept, snr_all=snr_all, gain=gain, excess_above=above,
                noise_above=float(noise_above), too_few_bins=too_few)


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
                  noise_fraction=0.2, verbose=True):
    """
    Loads a dedispersed burst and conditions it for the Fourier-domain model (see the module
    docstring), returning a ConditionedData. The file must hold only good channels: bad channels
    REMOVED, not zeroed or masked, since which channels remain decides where the block boundaries,
    and so each block's span and smearing, fall.

    path:             .npz holding 'power' (n_chan, n_time), 'freq' (n_chan,) and 'time' (n_time,), already dedispersed; channels in any order
    event_id:         identifier of the burst, carried to the saved fit; not used in the conditioning
    n_avg:            raw channels averaged per block
    fd_downsample:    keep the lowest 1 / fd_downsample of the Fourier bins
    toa:              burst arrival time (s, relative to the segment start) for the noise window; see estimate_noise
    exclude_fraction: fraction of the pre-toa lead-in, nearest the toa, left out of the noise window
    noise_fraction:   fraction of the segment used for the noise when toa is None
    verbose:         print the blocks, noise window, Fourier truncation and reference frequency
    """
    raw = jnp.load(path)
    power, freq, time = raw["power"], raw["freq"], raw["time"]

    # Order channels from the bottom of the band upward, whatever order the file holds them in
    order = np.argsort(np.asarray(freq))
    freq, power = freq[order], power[order, :]
    raw_dnu = float(jnp.median(jnp.abs(jnp.diff(freq))))

    freq_raw = freq
    power, freq, freq_span, n_dropped = block_average_channels(power, freq, n_avg)
    # The raw centres averaged into each block, in block order, so plots can draw each block over
    # its own members with any gap of bad channels left visible
    freq_members = jnp.asarray(freq_raw[: power.shape[0] * n_avg], dtype=jnp.float32).reshape(power.shape[0], n_avg)
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

    mean_noise_td, sigma_noise_td, n_noise_bins = estimate_noise(
        data_td, dt, toa=toa, exclude_fraction=exclude_fraction, noise_fraction=noise_fraction,
        verbose=verbose,
    )
    # For display only: subtracting a constant changes only a channel's DC bin, which
    # truncate_fourier drops, so the fit is identical with or without it
    data_td = data_td - mean_noise_td

    data_fd = jnp.float32(dt) * jnp.fft.rfft(data_td, axis=-1)
    data_fd_all = data_fd                       # every bin, for truncation_check

    # Time-domain noise std to the std of the real/imaginary part of each Fourier bin
    sigma_noise_fd = jnp.float32(dt) * sigma_noise_td * jnp.sqrt(n_time / 2.0)

    fourier_freqs_full = jnp.fft.rfftfreq(n_time, d=dt)
    data_fd, fourier_freqs = truncate_fourier(data_fd, fourier_freqs_full, fd_downsample, n_time)
    if verbose:
        print(f"Fourier truncation: keeping {fourier_freqs.size} of {fourier_freqs_full.size} bins "
              f"(up to {float(fourier_freqs[-1]) / 1e3:.1f} kHz of "
              f"{float(fourier_freqs_full[-1]) / 1e3:.1f} kHz)")

    # Whole-band weights first: the box search needs a complete ConditionedData, and they are the
    # fallback if it finds nothing
    channel_weights = estimate_channel_weights(data_fd, sigma_noise_fd)
    freq_channels_ref = dispersion_orthogonal_reference(freq, channel_weights)

    # freq_span runs centre to centre, one raw channel short of the block's full width
    channel_width = jnp.asarray(freq_span + raw_dnu, dtype=jnp.float32)[:, None]

    burst = ConditionedData(
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
        raw_dnu=raw_dnu,
        freq_members=freq_members,
    )

    boxes = significant_boxes(burst)
    box_weights = box_channel_weights(burst, boxes)
    if box_weights is None:
        if verbose:
            print("no significant box in the data: channel weights from every retained bin instead")
    else:
        channel_weights = box_weights
        freq_channels_ref = dispersion_orthogonal_reference(freq, channel_weights)
        burst = replace(burst, channel_weights=channel_weights, freq_channels_ref=freq_channels_ref)
    if verbose:
        print((f"channel weights from {len(boxes)} significant box(es) of an excess-power search of the data; "
               if boxes else "") + f"weighted mean frequency {float(jnp.sum(channel_weights * jnp.asarray(freq))):.1f} MHz, "
              f"reference frequency {freq_channels_ref:.1f} MHz")
    truncation_check(burst, data_fd_all, boxes, verbose=verbose)
    return burst
