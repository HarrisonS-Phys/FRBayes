"""
plotting.py - Plots of prepared bursts, of models evaluated on the same grid, and of fits.

Single plots take the burst for its grid (channels, times, retained Fourier bins, noise) and plot
its data by default; pass X_fd to plot anything else on that grid instead -- a model, a single
pulse, or a residual. Each has the waterfall in the middle, the channel-summed profile (or
spectrum) above it, and a marginal along the frequency channels to its right.

    plot_burst_TD(burst)                                    data, in time
    plot_burst_TD(burst, M_FD, title="Model")               model, in time
    plot_burst_FD(burst, burst.data_fd - M_FD, "Residual")  residual, in Fourier space

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
    params = {k: jnp.asarray(samples[k][draw_idx]).reshape(n_pulses) for k in MODEL_PARAMS if k in samples}
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
    their edges. Both break across flagged channels."""
    v = _on_channels(values, ch)
    if marginal == "hist":
        return ax.stairs(v, ch.edges, orientation="horizontal", baseline=None, **kw)
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
