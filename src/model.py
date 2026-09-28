"""
NOTE ON PRECISION & GPU COMPATIBILITY:
This module uses float32 for speed and broad GPU compatibility (where FP64 is often 
slow or unsupported). Because float32 only retains ~7 decimal digits of precision, 
inputs must stay small to prevent rounding away tiny time delays:

1. TIMING: Pass relative timestamps (times[0] ≈ 0). Large absolute times (MJD/Unix) 
   round off fraction-of-a-second delays and distort phase.
2. DM: Fit a residual DM around a dedispersed baseline. Large absolute DMs cause 
   multi-second delays that hit the same float32 precision limits.
"""

from dataclasses import dataclass
from typing import Callable
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
# 2. Spectral Energy Distribution (SED) Functions (F)
# -------------------------------------------------------------------

def running_power_law_sed(params_dict: dict, grid: FDGrid, constants: ModelConstants) -> jnp.ndarray:
    gamma = jnp.float32(params_dict["gamma"])  # Spectral index for burst component
    beta = jnp.float32(params_dict["beta"])  # Spectral running for burst component
    det_response = jnp.float32(constants.det_response)

    kappa = gamma + (beta * grid.log_normalized_freqs_channels)  # Running power law of SED
    F = (grid.normalized_freqs_channels ** kappa) * det_response
    return F

def make_orthogonal_poly_sed(P1: jnp.ndarray, P2: jnp.ndarray) -> Callable:
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
    """
    P1 = jnp.asarray(P1, dtype=jnp.float32).reshape(-1, 1)
    P2 = jnp.asarray(P2, dtype=jnp.float32).reshape(-1, 1)

    def orthogonal_poly_sed(params_dict: dict, grid: FDGrid, constants: ModelConstants) -> jnp.ndarray:
        c1 = jnp.float32(params_dict["c1"])  # coefficient of the linear basis function
        c2 = jnp.float32(params_dict["c2"])  # coefficient of the quadratic basis function
        det_response = jnp.float32(constants.det_response)

        return jnp.exp(c1 * P1 + c2 * P2) * det_response

    return orthogonal_poly_sed

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
# 4. Main Model Orchestrator
# -------------------------------------------------------------------

def setup_model(
    constants_dict: dict,
    freqs_channels: jnp.ndarray,
    times: jnp.ndarray,
    sed_fn: Callable = running_power_law_sed,
    pulse_fn: Callable = scattered_gaussian_fd_pulse,
    fourier_freqs=None
):
    """
    Setting up the necessary constants for the model itself. The model will be written as a closure within the setup method.

    constants_dict: dict of each constant in the model
    freqs_channels: array of frequency channels in the data
    times: array of time bins in the data
    """
    ### We are setting up constant values here
    constants = ModelConstants(**constants_dict)
    # channel_width may be a scalar or a per-channel array (see ModelConstants), so reduce
    # the comparison before combining it
    if constants.DM_incoherent > 0 and bool(jnp.all(jnp.asarray(constants.channel_width) == 0)):
        raise ValueError("DM_incoherent > 0 but channel_width = 0: intra-channel smearing would be silently omitted.")
    grid = build_fd_grid(freqs_channels, times, constants, fourier_freqs)

    def single_pulse_model(params_dict):
        """
        Model of a single pulse, which will be summed in the forward model for multi-pulse signals.

        params_dict: dict of each parameter of the pulse
        """
        # Linear amplitude scaling; defaults to 1.0 if 'A' is omitted e.g. in fluence-marginalized likelihood
        A = jnp.float32(params_dict.get("A", 1.0))

        F = sed_fn(params_dict, grid, constants)
        T = pulse_fn(params_dict, grid, constants)

        M = A * F * T
        return M

    # Recorded so a saved fit can say which SED and pulse forms produced it; the returned
    # closure is always named single_pulse_model, which says nothing about its contents
    single_pulse_model.sed_fn = sed_fn.__name__
    single_pulse_model.pulse_fn = pulse_fn.__name__

    return single_pulse_model
