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
    plot_hdi_mask(burst, M_FD=M_FD, single_pulse_fn=fn, params=params, hdi_mass=0.95)
                                                            what postprocess.residual_test blocks out
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

FLAGGED CHANNELS. The fit's channels are blocks of n_avg consecutive GOOD native channels (see
data.block_average_channels), so a block next to a flagged region spans it, and its centre -- the
mean of its members -- can fall inside the gap. When the burst carries the native centres of each
block's members (burst.freq_members, from data.prepare_burst), the waterfalls and the marginals
along the channels are drawn over the native channels each block was averaged from: a block that
straddles a flagged region is drawn in pieces on either side of it, and the flagged native
channels are left grey. Every value shown is still one of the fit's blocks -- nothing is drawn at
native resolution except where each block sits -- and every block covers the same number of native
channels, so the fit looks exactly as it did; only the frequency axis is now exact and the flagged
regions are visible. Sums over channels (the profiles above the waterfalls) are taken over the
blocks, as before. Without freq_members, each block is drawn as one band around its centre.

THE MISSING DC BIN. The zero-frequency (DC) bin is not among the retained bins, so every channel's
time series has zero mean. A channel with bright emission therefore sits below zero away from the
burst: bright channels show as dark horizontal bands, and the profile's baseline is negative. Data
and model share this exactly, so the fit is unaffected. The time-domain plots therefore subtract
each channel's median over time by default (rebaseline=True; pass False to see the data exactly as
fitted), which restores the off-pulse level for display as long as the burst covers less than half
the window. The same reason makes a plain sum over time zero in every channel, so the
time-domain marginal along the channels (the fluence spectrum) is always computed with each
channel's median removed.
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

# Colour of the flagged channels in the waterfalls
FLAGGED_COLOUR = "0.8"

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
    print("rebaseline: subtracting each channel's median over time. The DC (zero-frequency) bin is "
          "purposely removed from the data, so every channel's time series has zero mean and bright channels sit "
          "below zero off-pulse. This restores the off-pulse level for display only; the fit is "
          "unaffected.")


def _subtract_median(X_td):
    """Each channel (row) minus its median over time."""
    return X_td - np.median(X_td, axis=-1, keepdims=True)


def _fluence_spectrum(X_td, dt):
    """Sum over time of each channel with its median removed, times dt: the fluence per channel.
    A plain sum would be zero in every channel, since the DC bin is missing."""
    return np.nansum(_subtract_median(X_td), axis=-1) * dt


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

    With burst.freq_members -- the native centres (MHz) of the good channels averaged into each
    block, shape (n_chan, n_avg) -- the native grid from the lowest member to the highest, with
    spacing burst.native_dnu, is cut into segments of consecutive native channels that belong to
    the same block, or to none. A block is then drawn over exactly the native channels it was
    averaged from, in several pieces if it straddles a flagged region, and the flagged native
    channels form segments of their own, drawn grey. Without freq_members, or if the members do
    not sit on an even native grid, each block is one segment around its centre.

    Returns a _Channels:
        block    (n_seg,) the block drawn on each segment; -1 where no block is (flagged)
        edges    (n_seg + 1,) segment edges in freq_unit
        centres  (n_seg,) segment centres in freq_unit
        label    the frequency-axis label
        gaps     True if any segment is flagged
    """
    if freq_unit not in FREQ_UNITS:
        raise ValueError(f"freq_unit must be one of {list(FREQ_UNITS)}; got {freq_unit!r}.")
    scale, label = FREQ_UNITS[freq_unit], f"Frequency [{freq_unit}]"
    n_chan = np.asarray(burst.freq_channels).size

    members = getattr(burst, "freq_members", None)
    if members is not None:
        members = np.asarray(members, dtype=np.float64)
        dnu = float(getattr(burst, "native_dnu", 0.0))
        f0 = members.min()
        idx = np.rint((members - f0) / dnu).astype(int) if dnu > 0 else None
        if (idx is not None and members.ndim == 2 and members.shape[0] == n_chan
                and np.allclose(members, f0 + dnu * idx, rtol=0.0, atol=0.25 * dnu)
                and np.unique(idx).size == idx.size):
            # The block each native channel was averaged into, -1 for flagged channels
            block_of_row = np.full(int(idx.max()) + 1, -1)
            block_of_row[idx.ravel()] = np.repeat(np.arange(n_chan), members.shape[1])
            # Merge runs of consecutive native channels drawn with the same block (or with none)
            starts = np.concatenate([[0], np.flatnonzero(np.diff(block_of_row)) + 1])
            row_edges = f0 + dnu * (np.arange(block_of_row.size + 1) - 0.5)
            edges = row_edges[np.append(starts, block_of_row.size)] * scale
            block = block_of_row[starts]
            return _Channels(block, edges, 0.5 * (edges[1:] + edges[:-1]), label, bool(np.any(block < 0)))
        print("plotting: the native channels of the blocks are not on an even grid; drawing each "
              "channel block as one band around its centre.")

    centres = np.asarray(burst.freq_channels).reshape(-1) * scale
    return _Channels(np.arange(n_chan), _edges_from_centres(centres), centres, label, False)


def _on_channels(values, ch):
    """Per-block values, shape (n_chan, ...), laid out on the drawn segments `ch`: each segment
    takes the value of its block, and flagged segments are NaN."""
    v = np.asarray(values, dtype=float)
    out = v[np.maximum(ch.block, 0)]
    out[ch.block < 0] = np.nan
    return out


def _draw_waterfall(ax, values, x_edges, ch, vmin, vmax):
    """Waterfall of per-block `values`, shape (n_chan, n_x), over the x bins' edges and the
    channel layout `ch`. Flagged channels are drawn in FLAGGED_COLOUR."""
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(FLAGGED_COLOUR)
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
    their edges. Both break across flagged channels. The histogram is closed down to zero, so a
    block isolated between two flagged channels still reads as a bar rather than a lone vertical
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


def _title(burst, title, n_chan, gaps=False):
    return (f"{title}  ({n_chan} channels x {burst.n_time} samples, "
            f"dt = {burst.dt * 1e6:.1f} us{'; grey: flagged' if gaps else ''})")


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
    profile above and the fluence spectrum (sum over time, each channel's median removed) to the
    right. Flagged channels are grey (see the module note).

    burst:      the ConditionedData whose grid is used, e.g. as returned by data.prepare_burst
    X_fd:       what to plot, shape (n_chan, n_fourier): a model, a residual, ... Default
                burst.data_fd.
    title:      text at the top; the number of channels, samples and the time resolution are
                appended, so bursts prepared with different downsampling can be compared
    vlim:       (low, high) colour limits of the waterfall. Default: the `pct` percentiles of what
                is plotted. Pass the same vlim to two plots to compare them, e.g. data and model.
    pct:        lower and upper percentiles that set the colour limits when vlim is not given
    rebaseline: subtract each channel's median over time before plotting the waterfall and
                profile (see the module note on the missing DC bin). Default True, for display
                only; False shows the data exactly as fitted, with bright channels as dark bands.
    freq_unit:  unit of the frequency-channel axis, "MHz" (default), "GHz", "kHz" or "Hz"
    marginal:   "hist" (default) for the raw value of every bin as a step histogram, "line" for
                curves

    Returns the figure.
    """
    _check_marginal(marginal)
    X_fd = burst.data_fd if X_fd is None else X_fd
    X_td = np.asarray(to_time_domain(burst, jnp.asarray(X_fd)))
    spectrum = _fluence_spectrum(X_td, burst.dt)
    if rebaseline:
        _announce_rebaseline()
        X_td = _subtract_median(X_td)

    t_c, t_e = _time_axis(burst)
    ch = _channel_layout(burst, freq_unit)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(X_td, pct)

    fig, ax_ts, ax_wf, ax_side = _single_layout()
    _draw_top(ax_ts, np.nansum(X_td, axis=0), t_c, t_e, marginal, color="k", lw=0.8)
    ax_ts.set_title(_title(burst, title, X_td.shape[0], ch.gaps))

    _draw_waterfall(ax_wf, X_td, t_e, ch, vmin, vmax)
    ax_wf.set_xlabel("Time [s]")
    ax_wf.set_ylabel(ch.label)

    _draw_side(ax_side, spectrum, ch, marginal, color="k", lw=0.8)
    ax_side.axvline(0, color="gray", lw=0.6, ls="--")
    ax_side.set_xlabel("Fluence\n(median removed)")
    plt.show()
    return fig


def plot_burst_FD(burst, X_fd=None, title="Data", vlim=None, pct=(1.0, 99.0), freq_unit="MHz",
                  marginal="hist"):
    """
    The burst in FOURIER space, in units of the noise: the amplitude |X| / sigma_noise_fd of every
    channel and retained Fourier bin, which is what the likelihood weighs. Only the amplitude is
    shown; the phase carries the arrival times. Flagged channels are grey (see the module note).

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
    ax_ts.set_title(_title(burst, title, X_fd.shape[0], ch.gaps))

    # Each Fourier bin is drawn over its own edges, even if the retained bins are uneven
    _draw_waterfall(ax_wf, amp, k_e, ch, vmin, vmax)
    ax_wf.set_xlabel("Fourier frequency [kHz]")
    ax_wf.set_ylabel(ch.label)

    _draw_side(ax_side, np.sqrt(np.nanmean(amp ** 2, axis=1)), ch, marginal, color="k", lw=0.8)
    ax_side.set_xlabel("RMS |X| / noise")
    plt.show()
    return fig


def plot_noise_window(burst, vlim=(-3.0, 3.0), freq_unit="MHz", marginal="hist", tail_fraction=0.2):
    """
    The data in units of each channel's noise estimate, with the off-pulse window the estimate
    came from marked, to check that the window holds no burst emission and that the noise level
    holds across the segment.

    Uses burst.data_td, the time-domain data at full time resolution that data.estimate_noise
    worked on (before the Fourier truncation), divided by burst.sigma_noise_td per channel. The
    window is its first burst.n_noise_bins samples.

    Above: the SNR-weighted profile, sum_c sqrt(w_c) z_c(t) with w the channel weights. For white
    noise of the estimated level it has unit variance in every sample, so emission shows as a
    run above zero, and a noise level that changes along the segment as a change in its spread.
    The window is shaded, with +-3 dashed. Right: the mean of each channel over the window, in
    units of the standard error of that mean; emission leaking into the window pushes channels
    positive. The median subtracted in data.prepare_burst was taken over the same window, so even
    for pure noise these scatter about zero with a spread of ~0.75 rather than 1.

    burst:         the ConditionedData, e.g. as returned by data.prepare_burst
    vlim:          colour limits of the waterfall, in units of the noise (default +-3)
    freq_unit:     unit of the frequency-channel axis, "MHz" (default), "GHz", "kHz" or "Hz"
    marginal:      "hist" (default) for the raw value of every bin as a step histogram, "line" for
                   curves
    tail_fraction: the last fraction of the segment, used as a second off-pulse window for the
                   printed comparison. It may still hold the scattering tail; check the plot.

    Prints the profile's mean over the last third of the window (in units of its standard error;
    clearly positive means emission reaches into the window) and the profile's standard deviation
    in the window and at the end of the segment (1 if the noise estimate holds; larger at the end
    means the noise is underestimated there, or the tail reaches it).
    Returns the figure.
    """
    _check_marginal(marginal)
    n_win = int(burst.n_noise_bins)
    z = np.asarray(burst.data_td) / np.asarray(burst.sigma_noise_td).reshape(-1, 1)
    w = np.asarray(burst.channel_weights, dtype=float).reshape(-1)
    profile = np.tensordot(np.sqrt(w / w.sum()), z, axes=(0, 0))          # unit variance for white noise

    t_c, t_e = _time_axis(burst)
    ch = _channel_layout(burst, freq_unit)
    win_end = t_e[n_win]
    window_mean = z[:, :n_win].mean(axis=1) * np.sqrt(n_win)              # per channel, standard errors

    last = profile[n_win - max(n_win // 3, 1):n_win]
    n_tail = max(int(tail_fraction * profile.size), 2)
    print(f"noise window: first {n_win} of {profile.size} samples (to {win_end * 1e3:.2f} ms)")
    print(f"  profile mean over the last third of the window: {last.mean() * np.sqrt(last.size):+.2f} "
          f"standard errors (emission leaking in shows as clearly positive)")
    print(f"  profile sd in the window: {profile[:n_win].std():.2f}   "
          f"in the last {tail_fraction:.0%} of the segment: {profile[-n_tail:].std():.2f}   (expect 1)")

    fig, ax_ts, ax_wf, ax_side = _single_layout()
    _draw_top(ax_ts, profile, t_c, t_e, marginal, color="k", lw=0.8)
    ax_ts.axvspan(t_e[0], win_end, color="tab:orange", alpha=0.2, lw=0, label="noise window")
    for level in (-3.0, 3.0):
        ax_ts.axhline(level, color="gray", lw=0.6, ls="--")
    ax_ts.axhline(0.0, color="gray", lw=0.6)
    ax_ts.set_ylabel("weighted\nprofile [noise]")
    ax_ts.legend(loc="upper right", fontsize=7, frameon=False)
    ax_ts.set_title(_title(burst, "Noise window", z.shape[0], ch.gaps))

    _draw_waterfall(ax_wf, z, t_e, ch, vlim[0], vlim[1])
    ax_wf.axvline(win_end, color="tab:orange", lw=1.2)
    ax_wf.set_xlabel("Time [s]")
    ax_wf.set_ylabel(ch.label)

    _draw_side(ax_side, window_mean, ch, marginal, color="k", lw=0.8)
    for level in (-3.0, 3.0):
        ax_side.axvline(level, color="gray", lw=0.6, ls="--")
    ax_side.axvline(0.0, color="gray", lw=0.6)
    ax_side.set_xlabel("window mean\n[std. errors]")
    plt.show()
    return fig


# -------------------------------------------------------------------
# Fits: data | model | residual
# -------------------------------------------------------------------

def plot_fit_TD(burst, M_FD, M_FD_pulses=None, title_model="Model", vlim=None, pct=(1.0, 99.0),
                rebaseline=True, freq_unit="MHz", marginal="hist"):
    """
    Data, model and residual side by side in TIME: each a waterfall with the frequency-summed
    profile above and the fluence spectrum (sum over time, each channel's median removed) to the
    right. Data, model and residual share the colour limits and the axes, so they can be compared
    directly. Flagged channels are grey (see the module note).

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
    rebaseline:  subtract each channel's median over time from data and model before plotting
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

    spectra = [_fluence_spectrum(data_td, burst.dt), _fluence_spectrum(model_td, burst.dt)]
    spectra.append(spectra[0] - spectra[1])

    if rebaseline:
        _announce_rebaseline()
        data_td, model_td = _subtract_median(data_td), _subtract_median(model_td)
        if pulses_td is not None:
            pulses_td = _subtract_median(pulses_td)
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
        ax_side.set_xlabel("Fluence\n(median removed)")
    panels[0][1].set_ylabel(ch.label)
    panels[2][0].axhline(0, color="gray", lw=0.6, ls="--")

    if pulses_td is not None and pulses_td.shape[0] > 1:
        colors = plt.cm.tab10(np.arange(pulses_td.shape[0]) % 10)
        for j, pulse in enumerate(pulses_td):
            _draw_top(panels[1][0], np.nansum(pulse, axis=0), t_c, t_e, marginal,
                      color=colors[j], lw=0.8, ls="--", label=f"pulse {j}")
        legend_cell.legend(*panels[1][0].get_legend_handles_labels(), loc="center", fontsize=7,
                           frameon=False)

    fig.suptitle(_title(burst, "Fit", data_td.shape[0], ch.gaps), fontsize=10)
    plt.show()
    return fig


def plot_fit_FD(burst, M_FD, title_model="Re(Model)", vlim=None, pct=(1.0, 99.0), freq_unit="MHz",
                marginal="hist"):
    """
    Data, model and residual side by side in FOURIER space, whitened: the real part of each
    divided by the noise, so all three are in units of sigma and the residual should look like
    unit-variance noise. Above each: the sum over channels. Right of each: the RMS over Fourier
    bins of each channel; for the residual it should sit at the noise level, 1 (dashed), in every
    channel, and channels above it are where the model misses. Flagged channels are grey (see the
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

    fig.suptitle(_title(burst, "Fit", data_fd.shape[0], ch.gaps), fontsize=10)
    plt.show()
    return fig


def plot_hdi_mask(burst, mask=None, M_FD=None, single_pulse_fn=None, params=None, hdi_mass=None,
                  active_snr=3.0, vlim=None, pct=(1.0, 99.0), rebaseline=True, freq_unit="MHz",
                  colour="tab:red", alpha=0.35):
    """
    The data in TIME, and with M_FD also the residual, with the pixels that
    postprocess.residual_test blocks out shaded: each fitted pulse's highest-density interval,
    channel by channel. Use it to check that the blocked region covers each pulse, including its
    tail at the bottom of the band, and nothing else.

    burst:      the ConditionedData the fit used
    mask:       boolean (n_chan, n_time), e.g. residual_test(...)["mask"]. Leave it out to build
                it here from single_pulse_fn, params and hdi_mass instead.
    M_FD:       optional, the model the mask was built for; adds a residual panel
    single_pulse_fn, params, hdi_mass, active_snr:
                used only when mask is not given: the model closure, one draw's per-pulse
                parameters (postprocess.draw_params(...) or test_params), the probability mass of
                each pulse's per-channel profile to block (default 0.95), and the A_snr below
                which a pulse is not blocked; see postprocess.pulse_footprints. A mask passed in
                was built with its own hdi_mass, so giving both raises an error.
    vlim, pct:  colour limits of the waterfalls (shared), as in plot_burst_TD
    rebaseline: subtract each channel's median over time (display only; see the module note)
    freq_unit:  unit of the frequency-channel axis
    colour, alpha: of the shading

    Returns the figure.
    """
    from matplotlib.colors import ListedColormap

    if mask is not None and hdi_mass is not None:
        raise ValueError("a mask was passed, so hdi_mass would have no effect: the mask was built with "
                         "its own hdi_mass. Leave out the mask (and pass single_pulse_fn and params), or "
                         "rebuild it with residual_test(..., hdi_mass=...).")
    if mask is None:
        if single_pulse_fn is None or params is None:
            raise ValueError("pass a mask, or single_pulse_fn and params to build one with hdi_mass.")
        hdi_mass = 0.95 if hdi_mass is None else hdi_mass
        from postprocess import pulse_footprints
        mask = pulse_footprints(burst, single_pulse_fn, params, hdi_mass=hdi_mass, active_snr=active_snr)
        label = f"{hdi_mass:.0%} HDIs"
    else:
        label = "HDIs"

    data_td = np.asarray(to_time_domain(burst, jnp.asarray(burst.data_fd)))
    panels = [("Data", data_td)]
    if M_FD is not None:
        panels.append(("Residual", data_td - np.asarray(to_time_domain(burst, jnp.asarray(M_FD)))))
    if rebaseline:
        _announce_rebaseline()
        panels = [(name, _subtract_median(X)) for name, X in panels]

    t_c, t_e = _time_axis(burst)
    ch = _channel_layout(burst, freq_unit)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(panels[0][1], pct)
    shade = np.ma.masked_invalid(_on_channels(np.where(np.asarray(mask, bool), 1.0, np.nan), ch))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5), sharey=True, squeeze=False)
    for ax, (name, X) in zip(axes[0], panels):
        _draw_waterfall(ax, X, t_e, ch, vmin, vmax)
        ax.pcolormesh(t_e, ch.edges, shade, shading="flat", cmap=ListedColormap([colour]), alpha=alpha,
                      rasterized=True)
        ax.set_title(name)
        ax.set_xlabel("Time [s]")
    axes[0][0].set_ylabel(ch.label)
    fig.suptitle(_title(burst, f"Blocked by the pulses' {label} ({float(np.mean(mask)):.0%} of pixels, shaded)",
                        np.asarray(mask).shape[0], ch.gaps), fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig


def _draw_box(ax, ch, box, label, colour, linestyle="-", linewidth=1.5):
    """Outlines a box of postprocess.excess_power_search on a time waterfall: whole channels, from
    the lower edge of its first to the upper edge of its last, over its time extent, plus the part
    at the end of the segment for a box whose first time cell wraps there (t_wrap)."""
    from matplotlib.patches import Rectangle

    c0, c1 = box["channel_index"]
    seg = np.flatnonzero((ch.block >= c0) & (ch.block < c1))
    lo, hi = ch.edges[seg.min()], ch.edges[seg.max() + 1]
    for t0, t1 in [box["t_range"]] + ([box["t_wrap"]] if box.get("t_wrap") else []):
        ax.add_patch(Rectangle((t0, lo), t1 - t0, hi - lo, fill=False, edgecolor=colour,
                               linewidth=linewidth, linestyle=linestyle))
        ax.text(t0, hi, label, color=colour, fontsize=9, fontweight="bold", va="bottom", ha="left",
                clip_on=True)


def plot_residual_tiles(burst, M_FD, result, n_show=None, vlim=None, pct=(1.0, 99.0), rebaseline=True,
                        freq_unit="MHz", colour="tab:red"):
    """
    The data and the residual in TIME, with the boxes postprocess.excess_power_search found
    outlined: where the leftover signal that neither noise nor the tolerated model error explains
    sits. Boxes are numbered by significance and listed, with their unexplained SNR, in the
    corner of the residual panel. Solid outlines are boxes significant on their own after the
    trials correction; dashed ones are not.

    burst:     the ConditionedData the fit used
    M_FD:      the model the search was run on
    result:    the dict excess_power_search returned
    n_show:    how many of the boxes in result['tiles'] to draw (default: all of them)
    vlim, pct: colour limits of the waterfalls (shared), as in plot_burst_TD
    rebaseline: subtract each channel's median over time (display only; see the module note)
    freq_unit: unit of the frequency-channel axis
    colour:    of the outlines

    A box covers whole radio channels, so its frequency extent is drawn from the lower edge of its
    first channel to the upper edge of its last. Its time extent is that of its time cells, whose
    edges are blurred by about one cell (see excess_power_search); a box including the first time
    cell, which straddles the start of the segment, is also drawn at the segment's end.

    Returns the figure.
    """
    tiles = result["tiles"][:n_show]
    log10_limit = np.log10(result["p_false_alarm"]) - np.log10(result["n_tiles"])

    data_td = np.asarray(to_time_domain(burst, jnp.asarray(burst.data_fd)))
    panels = [("Data", data_td),
              ("Residual", data_td - np.asarray(to_time_domain(burst, jnp.asarray(M_FD))))]
    if rebaseline:
        _announce_rebaseline()
        panels = [(name, _subtract_median(X)) for name, X in panels]

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
    lines = [f"{rank}: SNR {t['snr_unexplained']:.1f}, {t['t_range'][0] * 1e3:.2f}-{t['t_range'][1] * 1e3:.2f} ms"
             + (f" + {t['t_wrap'][0] * 1e3:.2f}-{t['t_wrap'][1] * 1e3:.2f} ms" if t.get("t_wrap") else "")
             + f", {t['channel_range'][0]:.0f}-{t['channel_range'][1]:.0f} MHz"
             + ("" if t["log10_p_single"] < log10_limit else " (not significant)")
             for rank, t in enumerate(tiles, start=1)]
    axes[0][1].text(0.99, 0.99, "\n".join(lines), transform=axes[0][1].transAxes, ha="right", va="top",
                    fontsize=7, color="k", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    axes[0][0].set_ylabel(ch.label)
    verdict = "add a pulse" if result["add_pulse"] else "stop"
    fig.suptitle(_title(burst, f"Unexplained residual, tolerating {result['epsilon']:.0%} model error "
                               f"(p = {result['p']:.2g}: {verdict}; solid = significant)",
                        np.asarray(burst.freq_channels).size, ch.gaps), fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig


def plot_boxes_explained(burst, M_FD_old, M_FD_new, check, vlim=None, pct=(1.0, 99.0),
                         rebaseline=True, freq_unit="MHz"):
    """
    The check of postprocess.boxes_explained, in TIME: the residual of the old fit, the residual of
    the new fit (normally one more pulse), and the change in the model between them, each with the
    old fit's significant boxes outlined -- white where the change explains leftover signal, red
    where it does not, dashed grey where the model hardly changed inside the box. Each box is
    numbered and listed with its efficiency and the SNR of the model change inside it.

    burst:      the ConditionedData both fits used
    M_FD_old, M_FD_new: the two models, as passed to boxes_explained
    check:      the dict boxes_explained returned
    vlim, pct:  colour limits (shared by all three panels); default the `pct` percentiles of the
                old residual
    rebaseline: subtract each channel's median over time (display only; see the module note)
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
        panels = [(name, _subtract_median(X)) for name, X in panels]

    t_c, t_e = _time_axis(burst)
    ch = _channel_layout(burst, freq_unit)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(panels[0][1], pct)

    def style(b):
        if b["removed"]:
            return "white", "-"
        if np.isfinite(b["efficiency"]) and b["change_snr"] >= 5.0:
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
    lines = [f"{rank}: efficiency {b['efficiency']:+.2f}, model change SNR {b['change_snr']:.1f} -> "
             f"{'explained' if b['removed'] else 'not explained'}"
             for rank, b in enumerate(check["boxes"], start=1)]
    axes[0][0].text(0.99, 0.99, "\n".join(lines) if lines else "no significant box",
                    transform=axes[0][0].transAxes, ha="right", va="top", fontsize=7, color="k",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    axes[0][0].set_ylabel(ch.label)
    verdict = ("the new pulse fits leftover signal" if check["any_removed"]
               else "the new pulse fits none of the leftover signal")
    fig.suptitle(_title(burst, f"Boxes explained by the new fit: {verdict} (white = explained, "
                               f"red = not explained, dashed = model unchanged there)",
                        np.asarray(burst.freq_channels).size, ch.gaps), fontsize=10)
    fig.tight_layout()
    plt.show()
    return fig
