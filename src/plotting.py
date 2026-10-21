"""
plotting.py - Plots of prepared bursts, and of models evaluated on the same grid.

Both plots take the burst for its grid (channels, times, retained Fourier bins, noise) and, by
default, plot its data. Pass X_fd to plot anything else on that grid instead -- a model, a single
pulse, or a residual:

    M_FD = jnp.sum(jax.vmap(single_pulse_fn)(params), axis=0)     # (n_chan, n_fourier)
    plot_burst_TD(burst)                                           # data, in time
    plot_burst_TD(burst, M_FD, title="Model")                      # model, in time
    plot_burst_FD(burst, burst.data_fd - M_FD, title="Residual")   # residual, in Fourier space

To compare two plots by eye, give them the same colour limits with vlim=(low, high).

The data of a prepared burst live in the Fourier domain, on the retained Fourier bins only. To show
them in time, each bin is put back at its own index on the full rfft grid and inverted; irfft of the
truncated array would reinterpret bin k as frequency k / (n_time dt), which is not where it came from.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def _title(burst, title, n_chan):
    return (f"{title}  ({n_chan} channels x {burst.n_time} samples, "
            f"dt = {burst.dt * 1e6:.1f} us)")


def plot_burst_TD(burst, X_fd=None, title="Data", vlim=None, pct=(1.0, 99.0)):
    """
    Frequency-summed profile and waterfall in TIME, from Fourier-domain values on the burst's grid.

    burst: the ConditionedData whose grid is used, e.g. as returned by data.prepare_burst
    X_fd:  what to plot, shape (n_chan, n_fourier): a model, a residual, ... Default burst.data_fd.
    title: text at the top; the number of channels, samples and the time resolution are appended,
           so bursts prepared with different downsampling can be compared side by side
    vlim:  (low, high) colour limits of the waterfall. Default: the `pct` percentiles of what is
           plotted. Pass the same vlim to two plots to compare them, e.g. data and model.
    pct:   lower and upper percentiles that set the colour limits when vlim is not given

    Returns the figure.
    """
    X_fd = burst.data_fd if X_fd is None else X_fd
    X_td = np.asarray(to_time_domain(burst, jnp.asarray(X_fd)))
    time = np.asarray(burst.times)
    freq = np.asarray(burst.freq_channels).reshape(-1)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(X_td, pct)

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05, figure=fig)
    ax_ts = fig.add_subplot(gs[0])
    ax_wf = fig.add_subplot(gs[1], sharex=ax_ts)

    ax_ts.plot(time, np.nansum(X_td, axis=0), color="k", lw=0.8)
    ax_ts.tick_params(labelbottom=False)
    ax_ts.set_title(_title(burst, title, X_td.shape[0]))

    ax_wf.imshow(X_td, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis",
                 vmin=vmin, vmax=vmax, extent=(time[0], time[-1], freq[0], freq[-1]))
    ax_wf.set_xlabel("Time [s]")
    ax_wf.set_ylabel("Frequency [MHz]")
    plt.show()
    return fig


def plot_burst_FD(burst, X_fd=None, title="Data", vlim=None, pct=(1.0, 99.0)):
    """
    The burst in FOURIER space, in units of the noise: the amplitude |X| / sigma_noise_fd of every
    channel and retained Fourier bin, which is what the likelihood weighs. Only the amplitude is
    shown; the phase carries the arrival times.

    Top panel: the same for the frequency-summed series, |sum_c X_c| / sqrt(sum_c sigma_c^2), i.e.
    the Fourier transform of the profile shown by plot_burst_TD.

    burst: the ConditionedData whose grid and noise are used, e.g. as returned by data.prepare_burst
    X_fd:  what to plot, shape (n_chan, n_fourier): a model, a residual, ... Default burst.data_fd.
    title: text at the top; the number of channels, samples and the time resolution are appended
    vlim:  (low, high) colour limits of the waterfall, in noise units. Default: the `pct`
           percentiles of what is plotted. Pass the same vlim to two plots to compare them.
    pct:   lower and upper percentiles that set the colour limits when vlim is not given

    Returns the figure.
    """
    X_fd = np.asarray(burst.data_fd if X_fd is None else X_fd)
    sigma = np.broadcast_to(np.asarray(burst.sigma_noise_fd), X_fd.shape)
    amp = np.abs(X_fd) / sigma
    summed = np.abs(np.nansum(X_fd, axis=0)) / np.sqrt(np.nansum(sigma ** 2, axis=0))

    f_kHz = np.asarray(burst.fourier_freqs).reshape(-1) / 1e3
    freq = np.asarray(burst.freq_channels).reshape(-1)
    vmin, vmax = vlim if vlim is not None else np.nanpercentile(amp, pct)

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05, figure=fig)
    ax_ts = fig.add_subplot(gs[0])
    ax_wf = fig.add_subplot(gs[1], sharex=ax_ts)

    ax_ts.plot(f_kHz, summed, color="k", lw=0.8)
    ax_ts.set_ylabel("|X| / noise")
    ax_ts.tick_params(labelbottom=False)
    ax_ts.set_title(_title(burst, title, X_fd.shape[0]))

    # pcolormesh places each bin at its own Fourier frequency, even if the retained bins are uneven
    ax_wf.pcolormesh(f_kHz, freq, amp, shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_wf.set_xlabel("Fourier frequency [kHz]")
    ax_wf.set_ylabel("Frequency [MHz]")
    plt.show()
    return fig
