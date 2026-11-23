"""
plotting.py - Plots of prepared bursts, of models evaluated on the same grid, and of fits.

Single plots take the burst for its grid (channels, times, retained Fourier bins, noise) and plot
its data by default; pass X_fd to plot anything else on that grid instead -- a model, a single
pulse, or a residual. Each has the waterfall in the middle, the channel-summed profile (or
spectrum) above it, and a marginal along the frequency channels to its right.

    plot_burst_TD(burst)                                    data, in time
    plot_burst_TD(burst, M_FD, title="Model")               model, in time
    plot_burst_FD(burst, burst.data_fd - M_FD, "Residual")  residual, in Fourier space
    plot_noise_window(burst)                                the window the noise was estimated from
    plot_residual_tiles(burst, M_FD, result)                boxes found by excess_power_search
    plot_boxes_explained(burst, M_old, M_new, check)        whether a new pulse explains them

Fits are shown as data | model | residual side by side:

    M_FD, M_FD_pulses = model_from_draw(single_pulse_fn, samples, draw_idx)
    plot_fit_TD(burst, M_FD, M_FD_pulses, title_model=f"Model (draw {draw_idx})")
    plot_fit_FD(burst, M_FD, title_model=f"Re(Model) (draw {draw_idx})")

Options shared by every plot:
    freq_unit  unit of the frequency-channel axis: "MHz" (default, the CHIME band reads 400-800),
               "GHz", "kHz" or "Hz". Channel frequencies are read in MHz, the model's convention.
    marginal   "hist" (default) draws the raw value of every bin as a step histogram over the
               bin's extent; "line" draws the marginals as curves through each bin.

To compare two single plots by eye, give them the same colour limits with vlim=(low, high).

FOURIER BINS. The data of a prepared burst live in the Fourier domain, on the retained Fourier bins
only. To show them in time, each bin is put back at its own index on the full rfft grid and
inverted; irfft of the truncated array would reinterpret bin k as frequency k / (n_time dt), which
is not where it came from.

BIN EXTENTS. Time samples are labeled at their start, as in the model, so sample b covers
[t_b, t_b + dt]; channels are labeled at their centres. Waterfalls are drawn over those extents,
curves through the bins' centres, and histograms over the bins' edges, so all three line up.

BAD CHANNELS. The fit's channels are blocks of n_avg consecutive GOOD raw channels (see
data.block_average_channels), so a block next to a region of bad channels spans it, and its centre -- the
mean of its members -- can fall inside the gap. When the burst carries the raw centres of each
block's members (burst.freq_members, from data.prepare_burst), the waterfalls and the marginals
along the channels are drawn over the raw channels each block was averaged from: a block that
straddles a region of bad channels is drawn in pieces on either side of it, and the bad raw
channels are left grey. Every value shown is still one of the fit's blocks -- nothing is drawn at
raw resolution except where each block sits -- and every block covers the same number of raw
channels, so the fit looks exactly as it did; only the frequency axis is now exact and the regions
of bad channels are visible. Sums over channels (the profiles above the waterfalls) are taken over the
blocks, as before. Without freq_members, each block is drawn as one band around its centre.

THE MISSING DC BIN. The zero-frequency (DC) bin is not among the retained bins, so every channel's
time series has zero mean. A channel with bright emission therefore sits below zero away from the
burst: bright channels show as dark horizontal bands, and the profile's baseline is negative. Data
and model share this exactly, so the fit is unaffected. The time-domain plots therefore subtract
each channel's mean over the noise window -- its first burst.n_noise_bins samples, the same ones its
noise level was measured from -- by default (rebaseline=True; pass False to see the data exactly as
fitted), which restores the off-pulse level for display as long as the window holds no emission.
The same reason makes a plain sum over time zero in every channel, so the time-domain marginal
along the channels (the fluence spectrum) is always computed with each channel's window mean
removed.
"""

from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# Parameters single_pulse_fn may read; it reads only the ones its SED and pulse forms need. 'A' is
# the fluence, which single_pulse_fn applies itself.
MODEL_PARAMS = ("mean_arrival_time", "sigma", "DM", "tau_ref", "c1", "c2", "gamma", "beta", "delta", "A")

# Factor from MHz, the unit the channel frequencies are read in, to each display unit
FREQ_UNITS = {"Hz": 1e6, "kHz": 1e3, "MHz": 1.0, "GHz": 1e-3}

MARGINAL_STYLES = ("line", "hist")

# Colour of the bad channels in the waterfalls
BAD_COLOUR = "0.8"

# How the fit's channels are laid out along the frequency axis; see _channel_layout
_Channels = namedtuple("_Channels", "block edges centres label gaps")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def to_time_domain(burst, X_fd):
    """
    Inverts Fourier-domain arrays on the burst's retained bins back to the time domain.

    burst: the ConditionedData the arrays belong to; its n_time, dt and fourier_freqs give where
           each retained bin sits on the full rfft grid
    X_fd:  array of shape (..., n_fourier), e.g. burst.data_fd or a model evaluated on that grid

    Returns an array of shape (..., n_time).
    """
    Nseg, dt = burst.n_time, burst.dt
    bin_index = jnp.rint(jnp.asarray(burst.fourier_freqs).reshape(-1) * Nseg * dt).astype(int)
    full = jnp.zeros(X_fd.shape[:-1] + (Nseg // 2 + 1,), X_fd.dtype)
    return (1.0 / dt) * jnp.fft.irfft(full.at[..., bin_index].set(X_fd), n=Nseg, axis=-1)


def model_from_draw(single_pulse_fn, samples, draw_idx):
    """
    The model of one posterior draw on the burst's Fourier grid.

    single_pulse_fn: the model closure the fit used
    samples:         posterior samples with a flat draw axis, each shaped (n_draws, n_pulses), e.g.
                     dict(mcmc.get_samples()). Must contain 'A', the fluence: for a marginalized
                     fit, add the draws from posthoc_samples first, flattened to (n_draws, n_pulses).
    draw_idx:        index of the draw on that axis

    Returns (M_FD, M_FD_pulses): the summed model, shape (n_chan, n_fourier), and each pulse
    separately, shape (n_pulses, n_chan, n_fourier).
    """
    if "A" not in samples:
        raise KeyError("samples has no 'A'. For a marginalized fit, add the fluences drawn by "
                       "posthoc_samples first, flattened to (n_draws, n_pulses).")
    n_pulses = np.shape(samples["mean_arrival_time"])[-1]
    # A parameter stored once per draw rather than per pulse is given to every pulse
    params = {k: jnp.broadcast_to(jnp.asarray(samples[k][draw_idx]).reshape(-1), (n_pulses,))
              for k in MODEL_PARAMS if k in samples}
    M_FD_pulses = jax.vmap(single_pulse_fn)(params)
    return jnp.sum(M_FD_pulses, axis=0), M_FD_pulses


def _announce_rebaseline():
    print("rebaseline: each channel's noise-window mean subtracted, for display only.")


def _subtract_baseline(X_td, n_win):
    """Each channel (last axis: time) minus its mean over the noise window, its first n_win samples."""
    return X_td - np.mean(X_td[..., :n_win], axis=-1, keepdims=True)


def _fluence_spectrum(X_td, dt, n_win):
    """Sum over time of each channel with its noise-window mean (first n_win samples) removed, times
    dt: the fluence per channel. A plain sum would be zero in every channel, since the DC bin is
    missing."""
    return np.nansum(_subtract_baseline(X_td, n_win), axis=-1) * dt


def _edges_from_centres(centres):
    """Bin edges halfway between neighbouring centres, extended by half a spacing at both ends."""
    c = np.asarray(centres, dtype=float).reshape(-1)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])


def _time_axis(burst):
    """(centres, edges) of the time samples, which are labeled at their start."""
    t = np.asarray(burst.times).reshape(-1)
    return t + 0.5 * burst.dt, np.append(t, t[-1] + burst.dt)


def _channel_layout(burst, freq_unit):
    """
    How the fit's channels are drawn along the frequency axis, in `freq_unit`.

    With burst.freq_members -- the raw centres (MHz) of the good channels averaged into each
    block, shape (n_chan, n_avg) -- the raw grid from the lowest member to the highest, with
    spacing burst.raw_dnu, is cut into segments of consecutive raw channels that belong to
    the same block, or to none. A block is then drawn over exactly the raw channels it was
    averaged from, in several pieces if it straddles a region of bad channels, and the bad raw
    channels form segments of their own, drawn grey. Without freq_members, or if the members do
    not sit on an even raw grid, each block is one segment around its centre.

    Returns a _Channels:
        block    (n_seg,) the block drawn on each segment; -1 where no block is (bad channels)
        edges    (n_seg + 1,) segment edges in freq_unit
        centres  (n_seg,) segment centres in freq_unit
        label    the frequency-axis label
        gaps     True if any segment covers bad channels
    """
    if freq_unit not in FREQ_UNITS:
        raise ValueError(f"freq_unit must be one of {list(FREQ_UNITS)}; got {freq_unit!r}.")
    scale, label = FREQ_UNITS[freq_unit], f"Frequency [{freq_unit}]"
    n_chan = np.asarray(burst.freq_channels).size

    members = getattr(burst, "freq_members", None)
    if members is not None:
        members = np.asarray(members, dtype=np.float64)
        dnu = float(getattr(burst, "raw_dnu", 0.0))
        f0 = members.min()
        idx = np.rint((members - f0) / dnu).astype(int) if dnu > 0 else None
        if (idx is not None and members.ndim == 2 and members.shape[0] == n_chan
                and np.allclose(members, f0 + dnu * idx, rtol=0.0, atol=0.25 * dnu)
                and np.unique(idx).size == idx.size):
            # The block each raw channel was averaged into, -1 for bad channels
            block_of_row = np.full(int(idx.max()) + 1, -1)
            block_of_row[idx.ravel()] = np.repeat(np.arange(n_chan), members.shape[1])
            # Merge runs of consecutive raw channels drawn with the same block (or with none)
            starts = np.concatenate([[0], np.flatnonzero(np.diff(block_of_row)) + 1])
            row_edges = f0 + dnu * (np.arange(block_of_row.size + 1) - 0.5)
            edges = row_edges[np.append(starts, block_of_row.size)] * scale
            block = block_of_row[starts]
            return _Channels(block, edges, 0.5 * (edges[1:] + edges[:-1]), label, bool(np.any(block < 0)))
        print("plotting: the raw channels of the blocks are not on an even grid; drawing each "
              "channel block as one band around its centre.")

    centres = np.asarray(burst.freq_channels).reshape(-1) * scale
    return _Channels(np.arange(n_chan), _edges_from_centres(centres), centres, label, False)


def _on_channels(values, ch):
    """Per-block values, shape (n_chan, ...), laid out on the drawn segments `ch`: each segment
    takes the value of its block, and segments of bad channels are NaN."""
    v = np.asarray(values, dtype=float)
    out = v[np.maximum(ch.block, 0)]
    out[ch.block < 0] = np.nan
    return out


def _draw_waterfall(ax, values, x_edges, ch, vmin, vmax):
    """Waterfall of per-block `values`, shape (n_chan, n_x), over the x bins' edges and the
    channel layout `ch`. Bad channels are drawn in BAD_COLOUR."""
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(BAD_COLOUR)
    Z = np.ma.masked_invalid(_on_channels(values, ch))
    return ax.pcolormesh(x_edges, ch.edges, Z, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax,
                         rasterized=True)


def _check_marginal(marginal):
    if marginal not in MARGINAL_STYLES:
        raise ValueError(f"marginal must be one of {MARGINAL_STYLES}; got {marginal!r}.")


def _draw_top(ax, values, centres, edges, marginal, **kw):
    """Marginal above a waterfall: a curve through the bins' centres, or a step histogram of the
    raw bin values over the bins' edges."""
    if marginal == "hist":
        return ax.stairs(values, edges, baseline=None, **kw)
    return ax.plot(centres, values, **kw)[0]


def _draw_side(ax, values, ch, marginal, **kw):
    """Marginal along the channels, to the right of a waterfall: per-block `values` laid out on
    the channel layout `ch`, as a curve through the segments' centres or a step histogram over
    their edges. Both break across bad channels. The histogram is closed down to zero, so a
    block isolated between two bad channels still reads as a bar rather than a lone vertical
    stroke."""
    v = _on_channels(values, ch)
    if marginal == "hist":
        return ax.stairs(v, ch.edges, orientation="horizontal", baseline=0.0, **kw)
    return ax.plot(v, ch.centres, **kw)[0]


def _format_side(ax):
    """Few, small ticks on a narrow marginal panel, in scientific notation when the values are."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    ax.tick_params(axis="x", labelsize=7)


def _noise(burst, shape):
    return np.broadcast_to(np.asarray(burst.sigma_noise_fd), shape)


def _single_layout():
    """Profile on top, waterfall below, marginal along the channels to its right."""
    fig = plt.figure(figsize=(7.5, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[4, 1],
                           hspace=0.05, wspace=0.05, figure=fig)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_wf = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_side = fig.add_subplot(gs[1, 1], sharey=ax_wf)
    ax_top.tick_params(labelbottom=False)
    ax_side.tick_params(labelleft=False)
    _format_side(ax_side)
    return fig, ax_top, ax_wf, ax_side


def _triple_layout():
    """Three panels side by side (data, model, residual), each as in _single_layout. Returns the
    figure, the three (top, waterfall, side) axes tuples, and the empty cell above the model's
    side panel, used for the legend of the pulses."""
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 3], width_ratios=[4, 1, 4, 1, 4, 1],
                           hspace=0.05, wspace=0.08, figure=fig)
    panels = []
    for i in range(3):
        top = fig.add_subplot(gs[0, 2 * i], sharey=panels[0][0] if panels else None)
        wf = fig.add_subplot(gs[1, 2 * i], sharex=top, sharey=panels[0][1] if panels else None)
        side = fig.add_subplot(gs[1, 2 * i + 1], sharey=wf, sharex=panels[0][2] if panels else None)
        top.tick_params(labelbottom=False)
        side.tick_params(labelleft=False)
        _format_side(side)
        if i > 0:
            top.tick_params(labelleft=False)
            wf.tick_params(labelleft=False)
        panels.append((top, wf, side))
    legend_cell = fig.add_subplot(gs[0, 3])
    legend_cell.axis("off")
    for i in (0, 2):                       # the cells above the data and residual side panels
        fig.add_subplot(gs[0, 2 * i + 1]).axis("off")
    return fig, panels, legend_cell


# -------------------------------------------------------------------
# Single plots
# -------------------------------------------------------------------

def plot_burst_TD(burst, X_fd=None, title="Data", vlim=None, pct=(1.0, 99.0), rebaseline=True,
                  freq_unit="MHz", marginal="hist"):
    """
    Waterfall in TIME of Fourier-domain values on the burst's grid, with the frequency-summed
    profile above and the fluence spectrum (sum over time, each channel's noise-window mean removed)
    to the right. Bad channels are grey (see the module note).

    burst:      the ConditionedData whose grid is used, e.g. as returned by data.prepare_burst
    X_fd:       what to plot, shape (n_chan, n_fourier): a model, a residual, ... Default
                burst.data_fd.
    title:      text at the top; the number of channels, samples and the time resolution are
                appended, so bursts prepared with different downsampling can be compared
    vlim:       (low, high) colour limits of the waterfall. Default: the `pct` percentiles of what
                is plotted. Pass the same vlim to two plots to compare them, e.g. data and model.
    pct:        lower and upper percentiles that set the colour limits when vlim is not given
    rebaseline: subtract each channel's mean over the noise window before plotting the waterfall
                and profile (see the module note on the missing DC bin). Default True, for display
                only; False shows the data exactly as fitted, with bright channels as dark bands.
    freq_unit:  unit of the frequency-channel axis, "MHz" (default), "GHz", "kHz" or "Hz"
    marginal:   "hist" (default) for the raw value of every bin as a step histogram, "line" for
                curves

    Returns the figure.
    """
    _check_marginal(marginal)
    X_fd = burst.data_fd if X_fd is None else X_fd
    X_td = np.asarray(to_time_domain(burst, jnp.asarray(X_fd)))
    spectrum = _fluence_spectrum(X_td, burst.dt, burst.n_noise_bins)
    if rebaseline:
        _announce_rebaseline()
        X_td = _subtract_baseline(X_td, burst.n_noise_bins)

    t_c, t_e = _time_axis(burst)
    ch = _channel_layout(burst, freq_unit)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(X_td, pct)

    fig, ax_ts, ax_wf, ax_side = _single_layout()
    _draw_top(ax_ts, np.nansum(X_td, axis=0), t_c, t_e, marginal, color="k", lw=0.8)
    ax_ts.set_title(title)

    _draw_waterfall(ax_wf, X_td, t_e, ch, vmin, vmax)
    ax_wf.set_xlabel("Time [s]")
    ax_wf.set_ylabel(ch.label)

    _draw_side(ax_side, spectrum, ch, marginal, color="k", lw=0.8)
    ax_side.axvline(0, color="gray", lw=0.6, ls="--")
    ax_side.set_xlabel("Fluence\n(window mean removed)")
    plt.show()
    return fig


def plot_burst_FD(burst, X_fd=None, title="Data", vlim=None, pct=(1.0, 99.0), freq_unit="MHz",
                  marginal="hist"):
    """
    The burst in FOURIER space, in units of the noise: the amplitude |X| / sigma_noise_fd of every
    channel and retained Fourier bin, which is what the likelihood weighs. Only the amplitude is
    shown; the phase carries the arrival times. Bad channels are grey (see the module note).

    Above: the same for the frequency-summed series, |sum_c X_c| / sqrt(sum_c sigma_c^2), i.e. the
    Fourier transform of the profile shown by plot_burst_TD. Right: the RMS over Fourier bins of
    each channel, which shows where in the band the power is.

    burst:     the ConditionedData whose grid and noise are used, e.g. as returned by
               data.prepare_burst
    X_fd:      what to plot, shape (n_chan, n_fourier): a model, a residual, ... Default
               burst.data_fd.
    title:     text at the top; the number of channels, samples and the time resolution are
               appended
    vlim:      (low, high) colour limits of the waterfall, in noise units. Default: the `pct`
               percentiles of what is plotted. Pass the same vlim to two plots to compare them.
    pct:       lower and upper percentiles that set the colour limits when vlim is not given
    freq_unit: unit of the frequency-channel axis, "MHz" (default), "GHz", "kHz" or "Hz"
    marginal:  "hist" (default) for the raw value of every bin as a step histogram, "line" for
               curves

    Returns the figure.
    """
    _check_marginal(marginal)
    X_fd = np.asarray(burst.data_fd if X_fd is None else X_fd)
    sigma = _noise(burst, X_fd.shape)
    amp = np.abs(X_fd) / sigma
    summed = np.abs(np.nansum(X_fd, axis=0)) / np.sqrt(np.nansum(sigma ** 2, axis=0))

    k_c = np.asarray(burst.fourier_freqs).reshape(-1) / 1e3
    k_e = _edges_from_centres(k_c)
    ch = _channel_layout(burst, freq_unit)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(amp, pct)

    fig, ax_ts, ax_wf, ax_side = _single_layout()
    _draw_top(ax_ts, summed, k_c, k_e, marginal, color="k", lw=0.8)
    ax_ts.set_ylabel("|X| / noise")
    ax_ts.set_title(title)

    # Each Fourier bin is drawn over its own edges, even if the retained bins are uneven
    _draw_waterfall(ax_wf, amp, k_e, ch, vmin, vmax)
    ax_wf.set_xlabel("Fourier frequency [kHz]")
    ax_wf.set_ylabel(ch.label)

    _draw_side(ax_side, np.sqrt(np.nanmean(amp ** 2, axis=1)), ch, marginal, color="k", lw=0.8)
    ax_side.set_xlabel("RMS |X| / noise")
    plt.show()
    return fig


def plot_noise_window(burst, vlim=None, pct=(1.0, 99.0), freq_unit="MHz", marginal="hist",
                      tail_fraction=0.2):
    """
    The data in TIME with the off-pulse window the noise was estimated from shaded, to check that
    the window holds no burst emission and that the noise level holds across the segment. Drawn
    exactly like plot_burst_TD: the waterfall in data units, the sum over channels above and the
    fluence spectrum (sum over time, each channel's noise-window mean removed) to the right.
    Bad channels are grey (see the module note).

    Uses burst.data_td, the time-domain data at full time resolution that data.estimate_noise
    worked on (before the Fourier truncation), with each channel's mean over the window removed,
    as in the other time-domain plots. The window is its first burst.n_noise_bins samples.

    burst:         the ConditionedData, e.g. as returned by data.prepare_burst
    vlim:          (low, high) colour limits of the waterfall. Default: the `pct` percentiles of the
                   data.
    pct:           lower and upper percentiles that set the colour limits when vlim is not given
    freq_unit:     unit of the frequency-channel axis, "MHz" (default), "GHz", "kHz" or "Hz"
    marginal:      "hist" (default) for the raw value of every bin as a step histogram, "line" for
                   curves
    tail_fraction: the last fraction of the segment, used as a second off-pulse window for the
                   printed comparison. It may still hold the scattering tail; check the plot.

    Prints two checks on the SNR-weighted profile sum_c sqrt(w_c) x_c(t) / sigma_c, with w the
    channel weights, which has unit variance in every sample for white noise of the estimated
    level: its mean over the last third of the window less its mean over the whole window, in units
    of the standard error of that difference (clearly positive means emission reaches into the
    window), and its standard deviation in the window and at the end of the segment (1 if the
    noise estimate holds; larger at the end means the noise is underestimated there, or the tail
    reaches it).
    Returns the figure.
    """
    _check_marginal(marginal)
    n_win = int(burst.n_noise_bins)
    X_td = _subtract_baseline(np.asarray(burst.data_td), n_win)
    spectrum = _fluence_spectrum(X_td, burst.dt, n_win)

    z = X_td / np.asarray(burst.sigma_noise_td).reshape(-1, 1)
    w = np.asarray(burst.channel_weights, dtype=float).reshape(-1)
    profile = np.tensordot(np.sqrt(w / w.sum()), z, axes=(0, 0))          # unit variance for white noise

    t_c, t_e = _time_axis(burst)
    ch = _channel_layout(burst, freq_unit)
    win_end = t_e[n_win]
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(X_td, pct)

    # The profile's window mean is zero, so the last third's mean is its difference from the window
    # mean, whose variance is 1/n_last - 1/n_win for unit-variance samples
    n_last = max(n_win // 3, 1)
    last = profile[n_win - n_last:n_win]
    n_tail = max(int(tail_fraction * profile.size), 2)
    print(f"noise window: first {n_win} of {profile.size} samples (to {win_end * 1e3:.2f} ms)")
    print(f"  profile mean over the last third of the window, less the window mean: "
          f"{last.mean() / np.sqrt(1.0 / n_last - 1.0 / n_win):+.2f} standard errors "
          f"(emission leaking in shows as clearly positive)")
    print(f"  profile sd in the window: {profile[:n_win].std():.2f}   "
          f"in the last {tail_fraction:.0%} of the segment: {profile[-n_tail:].std():.2f}   (expect 1)")

    fig, ax_ts, ax_wf, ax_side = _single_layout()
    _draw_top(ax_ts, np.nansum(X_td, axis=0), t_c, t_e, marginal, color="k", lw=0.8)
    ax_ts.axvspan(t_e[0], win_end, color="tab:orange", alpha=0.2, lw=0, label="noise window")
    ax_ts.legend(loc="upper right", fontsize=7, frameon=False)
    ax_ts.set_title("Noise window")

    _draw_waterfall(ax_wf, X_td, t_e, ch, vmin, vmax)
    ax_wf.axvline(win_end, color="tab:orange", lw=1.2)
    ax_wf.set_xlabel("Time [s]")
    ax_wf.set_ylabel(ch.label)

    _draw_side(ax_side, spectrum, ch, marginal, color="k", lw=0.8)
    ax_side.axvline(0, color="gray", lw=0.6, ls="--")
    ax_side.set_xlabel("Fluence\n(window mean removed)")
    plt.show()
    return fig


# -------------------------------------------------------------------
# Fits: data | model | residual
# -------------------------------------------------------------------

def plot_fit_TD(burst, M_FD, M_FD_pulses=None, title_model="Model", vlim=None, pct=(1.0, 99.0),
                rebaseline=True, freq_unit="MHz", marginal="hist"):
    """
    Data, model and residual side by side in TIME: each a waterfall with the frequency-summed
    profile above and the fluence spectrum (sum over time, each channel's noise-window mean removed)
    to the right. Data, model and residual share the colour limits and the axes, so they can be compared
    directly. Bad channels are grey (see the module note).

    burst:       the ConditionedData the fit used
    M_FD:        the model on the burst's Fourier grid, shape (n_chan, n_fourier), e.g. from
                 model_from_draw
    M_FD_pulses: optional, each pulse separately, shape (n_pulses, n_chan, n_fourier); with more
                 than one pulse their profiles are drawn on the model's profile and listed in a
                 legend beside it
    title_model: title of the model panel, e.g. f"Model (draw {draw_idx})"
    vlim:        (low, high) colour limits of all three waterfalls. Default: the `pct`
                 percentiles of the data.
    pct:         lower and upper percentiles that set the colour limits when vlim is not given
    rebaseline:  subtract each channel's mean over the noise window from data and model before plotting
                 (see the module note on the missing DC bin); the residual is their difference.
                 Default True, for display only; False shows them exactly as fitted. With it, the pulses' profiles are rebaselined each on its own,
                 so their sum matches the model's profile only approximately.
    freq_unit:   unit of the frequency-channel axis, "MHz" (default), "GHz", "kHz" or "Hz"
    marginal:    "hist" (default) for the raw value of every bin as a step histogram, "line" for
                 curves

    Returns the figure.
    """
    _check_marginal(marginal)
    data_td = np.asarray(to_time_domain(burst, jnp.asarray(burst.data_fd)))
    model_td = np.asarray(to_time_domain(burst, jnp.asarray(M_FD)))
    pulses_td = None if M_FD_pulses is None else np.asarray(to_time_domain(burst, jnp.asarray(M_FD_pulses)))

    spectra = [_fluence_spectrum(data_td, burst.dt, burst.n_noise_bins),
               _fluence_spectrum(model_td, burst.dt, burst.n_noise_bins)]
    spectra.append(spectra[0] - spectra[1])

    if rebaseline:
        _announce_rebaseline()
        data_td = _subtract_baseline(data_td, burst.n_noise_bins)
        model_td = _subtract_baseline(model_td, burst.n_noise_bins)
        if pulses_td is not None:
            pulses_td = _subtract_baseline(pulses_td, burst.n_noise_bins)
    resid_td = data_td - model_td

    t_c, t_e = _time_axis(burst)
    ch = _channel_layout(burst, freq_unit)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(data_td, pct)

    fig, panels, legend_cell = _triple_layout()
    titles = ("Data", title_model, "Residual")
    for (ax_ts, ax_wf, ax_side), X_td, spectrum, name in zip(panels, (data_td, model_td, resid_td), spectra, titles):
        _draw_top(ax_ts, np.nansum(X_td, axis=0), t_c, t_e, marginal, color="k", lw=0.8, label="sum")
        ax_ts.set_title(name)
        _draw_waterfall(ax_wf, X_td, t_e, ch, vmin, vmax)
        ax_wf.set_xlabel("Time [s]")
        _draw_side(ax_side, spectrum, ch, marginal, color="k", lw=0.8)
        ax_side.axvline(0, color="gray", lw=0.6, ls="--")
        ax_side.set_xlabel("Fluence\n(window mean removed)")
    panels[0][1].set_ylabel(ch.label)
    panels[2][0].axhline(0, color="gray", lw=0.6, ls="--")

    if pulses_td is not None and pulses_td.shape[0] > 1:
        colors = plt.cm.tab10(np.arange(pulses_td.shape[0]) % 10)
        for j, pulse in enumerate(pulses_td):
            _draw_top(panels[1][0], np.nansum(pulse, axis=0), t_c, t_e, marginal,
                      color=colors[j], lw=0.8, ls="--", label=f"pulse {j}")
        legend_cell.legend(*panels[1][0].get_legend_handles_labels(), loc="center", fontsize=7,
                           frameon=False)

    fig.suptitle("Fit", fontsize=10)
    plt.show()
    return fig


def plot_fit_FD(burst, M_FD, title_model="Re(Model)", vlim=None, pct=(1.0, 99.0), freq_unit="MHz",
                marginal="hist"):
    """
    Data, model and residual side by side in FOURIER space, whitened: the real part of each
    divided by the noise, so all three are in units of sigma and the residual should look like
    unit-variance noise. Above each: the sum over channels. Right of each: the RMS over Fourier
    bins of each channel; for the residual it should sit at the noise level, 1 (dashed), in every
    channel, and channels above it are where the model misses. Bad channels are grey (see the
    module note).

    burst:       the ConditionedData the fit used; its noise whitens all three
    M_FD:        the model on the burst's Fourier grid, shape (n_chan, n_fourier), e.g. from
                 model_from_draw
    title_model: title of the model panel, e.g. f"Re(Model) (draw {draw_idx})"
    vlim:        (low, high) colour limits of the data and model waterfalls, in noise units.
                 Default: the `pct` percentiles of the data. The residual gets its own limits,
                 symmetric about zero, from its 99th percentile of |residual|.
    pct:         lower and upper percentiles that set the data and model colour limits when vlim
                 is not given
    freq_unit:   unit of the frequency-channel axis, "MHz" (default), "GHz", "kHz" or "Hz"
    marginal:    "hist" (default) for the raw value of every bin as a step histogram, "line" for
                 curves

    Prints the standard deviation of the whitened residual, which should be close to 1.
    Returns the figure.
    """
    _check_marginal(marginal)
    data_fd = np.asarray(burst.data_fd)
    sigma = _noise(burst, data_fd.shape)
    data_w = (data_fd / sigma).real
    model_w = (np.asarray(M_FD) / sigma).real
    resid_w = data_w - model_w
    print(f"whitened residual sd: {float(np.nanstd(resid_w)):.3f}  (expect 1)")

    k_c = np.asarray(burst.fourier_freqs).reshape(-1) / 1e3
    k_e = _edges_from_centres(k_c)
    ch = _channel_layout(burst, freq_unit)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(data_w, pct)
    vres = np.nanpercentile(np.abs(resid_w), 99.0)
    limits = ((vmin, vmax), (vmin, vmax), (-vres, vres))

    fig, panels, _ = _triple_layout()
    titles = ("Re(Data)", title_model, "Re(Residual)")
    for (ax_ts, ax_wf, ax_side), X, (lo, hi), name in zip(panels, (data_w, model_w, resid_w), limits, titles):
        _draw_top(ax_ts, np.nansum(X, axis=0), k_c, k_e, marginal, color="k", lw=0.8)
        ax_ts.set_title(name)
        # Each Fourier bin is drawn over its own edges, even if the retained bins are uneven
        _draw_waterfall(ax_wf, X, k_e, ch, lo, hi)
        ax_wf.set_xlabel("Fourier frequency [kHz]")
        _draw_side(ax_side, np.sqrt(np.nanmean(X ** 2, axis=1)), ch, marginal, color="k", lw=0.8)
        ax_side.set_xlabel("RMS / noise")
    panels[0][1].set_ylabel(ch.label)
    panels[2][0].axhline(0, color="gray", lw=0.6, ls="--")
    panels[2][2].axvline(1.0, color="gray", lw=0.6, ls="--")

    fig.suptitle("Fit", fontsize=10)
    plt.show()
    return fig


def _draw_box(ax, ch, box, label, colour, linestyle="-", linewidth=1.5):
    """Outlines a box of postprocess.excess_power_search on a time waterfall: whole channels, from
    the lower edge of its first to the upper edge of its last, over its time extent."""
    from matplotlib.patches import Rectangle

    c0, c1 = box["channel_index"]
    seg = np.flatnonzero((ch.block >= c0) & (ch.block < c1))
    lo, hi = ch.edges[seg.min()], ch.edges[seg.max() + 1]
    t0, t1 = box["t_range"]
    ax.add_patch(Rectangle((t0, lo), t1 - t0, hi - lo, fill=False, edgecolor=colour,
                           linewidth=linewidth, linestyle=linestyle))
    ax.text(t0, hi, label, color=colour, fontsize=9, fontweight="bold", va="bottom", ha="left",
            clip_on=True)


def plot_residual_tiles(burst, M_FD, result, n_show=None, vlim=None, pct=(1.0, 99.0), rebaseline=True,
                        freq_unit="MHz", colour="tab:red"):
    """
    The data and the residual in TIME, with the boxes postprocess.excess_power_search found
    outlined: where the leftover signal that neither noise nor the tolerated model error explains
    sits. Boxes are numbered by significance, as in the list excess_power_search prints. Solid
    outlines are boxes significant on their own after the trials correction; dashed ones are not.

    burst:     the ConditionedData the fit used
    M_FD:      the model the search was run on
    result:    the dict excess_power_search returned
    n_show:    how many of the boxes in result['tiles'] to draw (default: all of them)
    vlim, pct: colour limits of the waterfalls (shared), as in plot_burst_TD
    rebaseline: subtract each channel's mean over the noise window (display only; see the module note)
    freq_unit: unit of the frequency-channel axis
    colour:    of the outlines

    A box covers whole radio channels, so its frequency extent is drawn from the lower edge of its
    first channel to the upper edge of its last. Its time extent is that of its time cells, whose
    edges are blurred by about one cell (see excess_power_search).

    Returns the figure.
    """
    tiles = result["tiles"][:n_show]
    log10_limit = np.log10(result["p_false_alarm"]) - np.log10(result["n_tiles"])

    data_td = np.asarray(to_time_domain(burst, jnp.asarray(burst.data_fd)))
    panels = [("Data", data_td),
              ("Residual", data_td - np.asarray(to_time_domain(burst, jnp.asarray(M_FD))))]
    if rebaseline:
        _announce_rebaseline()
        panels = [(name, _subtract_baseline(X, burst.n_noise_bins)) for name, X in panels]

    t_c, t_e = _time_axis(burst)
    ch = _channel_layout(burst, freq_unit)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(panels[0][1], pct)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, squeeze=False)
    for ax, (name, X) in zip(axes[0], panels):
        _draw_waterfall(ax, X, t_e, ch, vmin, vmax)
        for rank, t in reversed(list(enumerate(tiles, start=1))):
            significant = t["log10_p_single"] < log10_limit
            _draw_box(ax, ch, t, f"{rank}", colour, linestyle="-" if significant else "--",
                      linewidth=2.0 if rank == 1 else 1.2)
        ax.set_title(name)
        ax.set_xlabel("Time [s]")
    axes[0][0].set_ylabel(ch.label)
    fig.suptitle("Residual boxes (dashed: not significant)", fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig


def plot_boxes_explained(burst, M_FD_old, M_FD_new, check, vlim=None, pct=(1.0, 99.0),
                         rebaseline=True, freq_unit="MHz"):
    """
    The check of postprocess.boxes_explained, in TIME: the residual of the old fit, the residual of
    the new fit (normally one more pulse), and the change in the model between them, each with the
    old fit's significant boxes outlined -- white where the change explains leftover signal (its
    log-likelihood gain in the box reaches min_gain), red where it does not, dashed grey where the
    model hardly changed inside the box (model-change SNR below 5). Boxes are numbered as in the
    list boxes_explained prints.

    burst:      the ConditionedData both fits used
    M_FD_old, M_FD_new: the two models, as passed to boxes_explained
    check:      the dict boxes_explained returned
    vlim, pct:  colour limits (shared by all three panels); default the `pct` percentiles of the
                old residual
    rebaseline: subtract each channel's mean over the noise window (display only; see the module note)
    freq_unit:  unit of the frequency-channel axis

    Returns the figure.
    """
    data_td = np.asarray(to_time_domain(burst, jnp.asarray(burst.data_fd)))
    old_td = np.asarray(to_time_domain(burst, jnp.asarray(M_FD_old)))
    new_td = np.asarray(to_time_domain(burst, jnp.asarray(M_FD_new)))
    panels = [("Residual, old fit", data_td - old_td), ("Residual, new fit", data_td - new_td),
              ("Model change (new - old)", new_td - old_td)]
    if rebaseline:
        _announce_rebaseline()
        panels = [(name, _subtract_baseline(X, burst.n_noise_bins)) for name, X in panels]

    t_c, t_e = _time_axis(burst)
    ch = _channel_layout(burst, freq_unit)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(panels[0][1], pct)

    def style(b):
        if b["removed"]:
            return "white", "-"
        if b["change_snr"] >= 5.0:
            return "red", "-"
        return "0.7", "--"

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True, squeeze=False)
    for ax, (name, X) in zip(axes[0], panels):
        _draw_waterfall(ax, X, t_e, ch, vmin, vmax)
        for rank, b in reversed(list(enumerate(check["boxes"], start=1))):
            c, ls = style(b)
            _draw_box(ax, ch, b, f"{rank}", c, linestyle=ls, linewidth=2.5)
        ax.set_title(name)
        ax.set_xlabel("Time [s]")
    axes[0][0].set_ylabel(ch.label)
    fig.suptitle("Boxes explained by the new fit (white: explained, red: not explained, "
                 "dashed grey: model unchanged)", fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig
