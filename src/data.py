"""
data.py - Conditioning of dedispersed dynamic spectra for the Fourier-domain FRB model.

The entry point is `prepare_burst`, which loads a dedispersed burst, averages its frequency
channels, transforms to the Fourier domain, truncates the Fourier axis, and builds the
orthogonal spectral basis. It returns a `ConditionedData` holding everything the model and the
likelihood need, so neither has to know how the conditioning was done.

CONVENTIONS (must match model.py and likelihood.py):
1. FOURIER TRANSFORM: d_tilde = dt * rfft(d), so the model's templates share the convention.
2. NOISE: `sigma_noise_fd` is the standard deviation of the real (equivalently, imaginary) part of
   each retained Fourier bin, dt * varsigma_a * sqrt(N / 2) for white noise.
3. RETAINED BINS: DC is dropped, because it is set by each channel's baseline level, which the
   model does not include; the Fourier truncation removes Nyquist along with the rest of the
   upper band. `fourier_freqs` is exactly the set of bins kept, and the likelihood uses all of
   them, so the choice made here is the only place the bin selection lives.
4. TIMES are returned relative to the segment start, as model.py requires for float32.
"""

from dataclasses import dataclass
from typing import Tuple
import jax.numpy as jnp


@dataclass(frozen=True)
class ConditionedData:
    """Everything the model and likelihood need from a conditioned burst."""

    # --- passed to the FD likelihood ---
    data_fd: jnp.ndarray  # complex (nfreq, nfourier), dt * rfft of tmediann-subtracted data, retained bins only
    sigma_noise_fd: jnp.ndarray  # (nfreq, 1) std of the real/imaginary part of each Fourier bin

    # --- passed to setup_model ---
    freq_channels: jnp.ndarray  # (nfreq, 1) block centre frequencies (MHz)
    times: jnp.ndarray  # (ntime,) times relative to the segment start (s)
    fourier_freqs: jnp.ndarray  # (nfourier,) the retained Fourier bins (Hz)
    freq_channels_ref: float  # reference frequency nu_r (MHz)
    channel_width: jnp.ndarray  # (nfreq, 1) width of each block (MHz), for intra-channel smearing

    # --- the orthogonal spectral basis for SED ---
    spectral_basis: Tuple[jnp.ndarray, jnp.ndarray]  # (P1, P2), passed to make_orthogonal_poly_sed
    spectral_coeff_map: dict  # converts sampled c1, c2 back to gamma, beta
    channel_weights: jnp.ndarray  # (nfreq,) metric the basis was orthogonalized in

    # --- retained for plotting and diagnostics ---
    data_td: jnp.ndarray  # (nfreq, ntime) median-subtracted time-domain data
    sigma_noise_td: jnp.ndarray  # (nfreq, 1) per-channel time-domain noise std
    dt: float  # time bin width (s)
    n_time: int  # number of time samples in the segment
    freq_span: jnp.ndarray  # (nfreq,) spread of channel centres within each block (MHz)
    native_dnu: float  # native channel width before averaging (MHz)
    n_noise_bins: int  # time bins used for the off-pulse noise estimate


def block_average_channels(power, freq, n_avg):
    """
    Averages consecutive blocks of n_avg channels.

    Bad channels are assumed to have been DISCARDED before this point, so the array holds only
    good channels and every block contains exactly n_avg of them. That keeps the number of
    averaged samples per block uniform, and with it the quality of the Gaussian approximation to
    the chi'^2 statistics (m -> n_avg * m). The cost is that a block spanning a discarded region
    covers a wider range of real frequency than n_avg contiguous channels, which `freq_span`
    reports and which feeds the intra-channel smearing width. A few channels may be dropped from
    the top of the band if they do not fill a block; the top is where the SED is faintest.

    Returns (power_blocks, freq_blocks, freq_span, n_dropped).
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
    Off-pulse median and MAD estimate of the noise per channel, in the time domain.

    Only bins BEFORE the burst should be used. The scattering tail extends after the arrival time and
    is far longer than the intrinsic width, especially at the bottom of the band, so trailing
    bins cannot be assumed burst-free.

    toa:              burst arrival time (s, relative to the segment start), from an upstream
                      pipeline. The interval [0, toa] is the usable lead-in.
    exclude_fraction: fraction of that lead-in, nearest the TOA, that is discarded. The TOA has
                      its own uncertainty and the pulse has width, so flux reaches back before
                      the quoted arrival time; discarding the last part of the lead-in keeps it
                      out of the noise estimate.
    noise_fraction:   fallback when toa is None, using the first fraction of the segment. Only
                      safe when the burst is known to arrive later than that: a burst inside the
                      window inflates the noise estimate severalfold, which silently rescales
                      the likelihood, the channel weights and the meaning of s_opt.

    The estimate is always attempted, however few bins are available, and the count is returned
    so a caller can decide whether it is enough.

    Returns (median_noise_td, sigma_noise_td, n_noise_bins), the first two (nfreq, 1). The _td
    suffix marks these as time-domain quantities; prepare_burst converts sigma_noise_td to the
    Fourier-domain per-bin std, `sigma_noise_fd`, which is what the likelihood uses.
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
    mad = jnp.median(jnp.abs(off_pulse - median_noise_td), axis=1, keepdims=True)

    # 1.4826 = 1 / Phi^-1(3/4), converting the median absolute deviation to a standard deviation
    # for Gaussian noise. MAD is used rather than the sample std so that an unremoved spike or a
    # sliver of burst in the window cannot inflate the estimate.
    sigma_noise_td = mad * 1.4826

    # Bad channels are assumed already discarded, so every channel should yield a usable noise
    # estimate. A non-positive or non-finite value means something upstream is wrong -- a dead
    # channel left in the file, or a noise window so short that the MAD collapses to zero -- and
    # it would otherwise propagate silently as an infinite sigma and NaN channel weights.
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
    Keeps the lowest 1 / fd_downsample of the Fourier bins, and drops DC.

    The template's Gaussian factor exp(-2 pi^2 sigma^2 f^2) kills it well below Nyquist, so the
    upper bins are noise-dominated and carry no information about any parameter. Unlike
    downsampling in time this is an exact low-pass with no aliasing, and it leaves the number of
    averaged samples per bin, and hence the Gaussian approximation, untouched.

    Returns (data_fd, fourier_freqs) restricted to the retained bins.
    """
    n_keep = fourier_freqs_full.size // fd_downsample
    keep = slice(1, n_keep)
    return data_fd[:, keep], fourier_freqs_full[keep]

def estimate_channel_weights(data_fd, sigma_noise_fd):
    """
    w_a proportional to (optimal SNR in channel a)^2, normalized to sum to 1.

    These are the weights the orthogonal SED basis is built from, and measuring them requires
    the data: that is why this lives in data.py rather than model.py. They define the metric in
    which the basis polynomials are orthogonalized, and it has to be the metric the likelihood
    itself uses, i.e. where the burst actually carries signal-to-noise. Weighting every channel
    equally instead leaves the gamma-beta degeneracy largely intact, because the SED spans
    orders of magnitude across the band.

    Measured in the Fourier domain as excess power over the noise expectation. Every retained
    bin contributes, so there is no burst window to choose and no threshold to tune: the noise
    adds a known mean (2 sigma^2 per bin, from the real and imaginary parts) which is subtracted
    rather than excluded. A time-domain estimate instead needs a window, since summing over the
    whole segment buries the burst in noise, and a window that clips the scattering tail biases
    the weights toward the top of the band where the tail is shortest.

    Summing over every retained bin also means the per-channel estimate is not noisy enough to
    need smoothing along the frequency axis; the metric only has to be approximately right.
    """
    d = jnp.asarray(data_fd)
    s = jnp.asarray(sigma_noise_fd).reshape(-1)
    n_bins = d.shape[-1]

    # E|d|^2 = 2 sigma^2 per bin under noise alone, so subtract that to isolate the signal power.
    # estimate_noise guarantees sigma is finite and positive, so no masking is needed here.
    excess_power = jnp.sum(jnp.abs(d) ** 2, axis=-1) - 2.0 * n_bins * s ** 2

    # Negative values are channels consistent with pure noise
    weights = jnp.maximum(excess_power, 0.0) / s ** 2

    return weights / weights.sum()

def dispersion_orthogonal_reference(freq, weights):
    """
    Reference frequency for which the dispersion delay shape (nu^-2) is orthogonal to a constant
    offset under the channel weights, which decouples DM from the arrival time: without it the
    two are strongly anticorrelated, since both shift the pulse in time and their shapes are
    similar over a factor-of-two band. t_0 is then the arrival time at the frequency carrying
    the weighted-mean dispersion delay.
    """
    nu_inv2_mean = jnp.sum(weights * jnp.asarray(freq) ** -2) / jnp.sum(weights)
    return float(nu_inv2_mean ** -0.5)


def orthogonal_spectral_basis(freq, weights, freq_ref):
    """
    Gram-Schmidt orthogonalization of {1, x, x^2} in the weighted metric, x = ln(nu/nu_r),
    returning the two non-constant basis functions normalized to unit weighted RMS.

    Orthogonalizing against the constant decouples the spectral shape from the fluence, since
    a constant in ln F is exactly a rescaling of A. The returned coefficients convert the
    sampled c1, c2 back to the conventional gamma, beta:
        gamma = gamma_c1 * c1 + gamma_c2 * c2,   beta = beta_c2 * c2.

    Note the basis itself is invariant to freq_ref: shifting nu_r shifts x by a constant, and the
    basis is orthogonal to constants. Only gamma and beta depend on the reference.
    """
    x = jnp.log(jnp.asarray(freq).reshape(-1) / freq_ref)
    w = jnp.asarray(weights).reshape(-1)
    ip = lambda u, v: jnp.sum(w * u * v)

    P0 = jnp.ones_like(x)
    P1 = x - ip(x, P0) / ip(P0, P0) * P0
    P2 = x ** 2 - ip(x ** 2, P0) / ip(P0, P0) * P0 - ip(x ** 2, P1) / ip(P1, P1) * P1

    # Unit weighted RMS, so the sampled coefficients are O(1)
    P1 /= jnp.sqrt(ip(P1, P1) / w.sum())
    P2 /= jnp.sqrt(ip(P2, P2) / w.sum())

    # P1, P2 are polynomials in x, so the map back to gamma, beta is exact
    a1, b1 = jnp.polyfit(x, P1, 1)
    a2, b2, d2 = jnp.polyfit(x, P2, 2)
    return P1, P2, dict(gamma_c1=a1, gamma_c2=b2, beta_c2=a2)


def prepare_burst(path, event_id=None, n_avg=8, fd_downsample=4, toa=None, exclude_fraction=0.2,
                  noise_fraction=0.2, verbose=True):
    """
    Loads a dedispersed burst and conditions it for the Fourier-domain model.

    event_id:       identifier for the burst, carried through to the saved fit so a result can be
                    traced back to its input. Not used in the conditioning itself.
    path:           .npz holding 'power' (nfreq, ntime), 'freq' (nfreq,), 'time' (ntime,). The
                    data are assumed already dedispersed, and bad channels already DISCARDED
                    rather than marked, so only good channels are present. If bad-channel
                    removal ever moves inside this function, `block_average_channels` needs the
                    list of bad channels rather than a pre-filtered array: which channels
                    survive determines where the block boundaries fall, and hence each block's
                    frequency span and intra-channel smearing.
    n_avg:          frequency channels averaged per block
    fd_downsample:  keep the lowest 1 / fd_downsample of the Fourier bins
    toa:            burst arrival time (s, relative to the segment start) for the off-pulse
                    noise estimate; see `estimate_noise` for the fallback when it is None
    exclude_fraction: fraction of the pre-TOA lead-in, nearest the TOA, discarded from the
                    noise estimate
    noise_fraction: fraction of the segment used for the noise estimate when toa is None

    Returns a ConditionedData.
    """
    raw = jnp.load(path)
    power, freq, time = raw["power"], raw["freq"], raw["time"]

    # Order channels from the bottom of the band upward
    freq = freq[::-1]
    power = power[::-1, :]
    native_dnu = float(jnp.median(jnp.abs(jnp.diff(freq))))

    power, freq, freq_span, n_dropped = block_average_channels(power, freq, n_avg)
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
        data_td, dt, toa, exclude_fraction, noise_fraction, verbose
    )
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
    P1, P2, spectral_coeff_map = orthogonal_spectral_basis(freq, channel_weights, freq_channels_ref)
    if verbose:
        print(f"weighted mean frequency {float(jnp.sum(channel_weights * jnp.asarray(freq))):.1f} MHz, "
              f"reference frequency {freq_channels_ref:.1f} MHz")

    # freq_span measures centre to centre, one channel short of the block's full width, hence
    # the + native_dnu. Blocks vary in width since they can straddle discarded channels.
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
        spectral_basis=(P1, P2),
        spectral_coeff_map=spectral_coeff_map,
        channel_weights=channel_weights,
        data_td=data_td,
        sigma_noise_td=sigma_noise_td,
        dt=dt,
        n_time=n_time,
        n_noise_bins=n_noise_bins,
        freq_span=freq_span,
        native_dnu=native_dnu,
    )
