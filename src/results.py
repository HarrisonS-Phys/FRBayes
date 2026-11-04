"""
results.py - Saving and loading fits of the FRB model.

A saved run holds the posterior samples and everything needed to interpret them: the priors,
the reference frequency, the orthogonal spectral basis, which likelihood was used, and a
reference to the data. The DATA ITSELF IS NOT SAVED -- only an identifier, path and content
hash, so a run can be traced back to its input and checked against it without duplicating
large files across the catalogue.
"""

import hashlib
from pathlib import Path

import numpy as np
import xarray as xr
from arviz_base import from_dict
import arviz as az


def _fn_name(fn):
    """Name of a function for the run record; netcdf attributes cannot hold the function itself."""
    if fn is None:
        return ""
    if isinstance(fn, str):           # model.py stores the SED and pulse functions by name already
        return fn
    return getattr(fn, "__name__", type(fn).__name__)


def describe_priors(priors):
    """
    Serializable description of the priors, one string per parameter.

    NumPyro distributions have no informative repr -- it gives the object address -- so each is
    described by its class name and its own arguments, which is what has to be reproducible.
    """
    record = {}
    for name, d in priors.items():
        args = {k: float(np.asarray(getattr(d, k))) for k in d.arg_constraints if hasattr(d, k)}
        record[f"prior.{name}"] = f"{type(d).__name__}({', '.join(f'{k}={v:g}' for k, v in args.items())})"
    return record


def describe_data(path, event_id=None, n_bytes=None):
    """
    Identifier, location and content hash of the analyzed file.

    The hash is what makes the reference meaningful: an event ID says which burst, but not which
    version of its data, and reprocessing upstream changes the file without changing the name.

    n_bytes: hash only the first n_bytes of the file, for speed. None hashes all of it, which is
             the safer default; a few hundred MB takes under a second.
    """
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        digest.update(fh.read(n_bytes) if n_bytes else fh.read())

    return {
        "data.event_id": str(event_id) if event_id is not None else "",
        "data.path": str(path.resolve()),
        "data.sha256": digest.hexdigest(),
        "data.bytes_hashed": str(n_bytes) if n_bytes else "all",
    }


def save_run(path, mcmc, burst, priors, data_path, single_pulse_fn=None,
             event_id=None, hash_bytes=None, extra_samples=None, **extra):
    """
    Writes a fit to netcdf: posterior samples, plus the conditioning needed to interpret them.

    mcmc:            the finished MCMC object; samples are taken with group_by_chain=True, so the
                     leading two axes are (chain, draw) with no shape guessing
    burst:           the ConditionedData the fit used
    priors:          the priors dictionary of the fit, as returned by run.set_priors. For a
                     marginalized fit this still contains 'A_snr', so the amplitude prior that was
                     integrated over is recorded too.
    data_path:       the file that was analyzed; its contents are hashed, not stored
    single_pulse_fn: the model closure the fit used; only the names of its SED and pulse
                     functions are recorded
    extra_samples:   further posterior arrays to store beside the sampled sites, e.g. fluences
                     drawn after a marginalized fit (run.posthoc_samples returns these). Each must
                     already be shaped (n_chains, n_draws, ...) to match mcmc's own samples, and
                     must not share a name with a sampled site
    extra:           any further scalars worth recording (n_pulses, ...), stored as strings

    The reference frequency is recorded because gamma and tau_ref are defined relative to it and
    it differs per burst; the spectral basis because c1 and c2 are meaningless without it.
    """
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    n_chains, n_draws = next(iter(samples.values())).shape[:2]

    for name, value in (extra_samples or {}).items():
        if name in samples:
            raise ValueError(
                f"extra_samples['{name}'] would overwrite the sampled site of the same name."
            )
        value = np.asarray(value)
        if value.shape[:2] != (n_chains, n_draws):
            raise ValueError(
                f"extra_samples['{name}'] has leading shape {value.shape[:2]}, expected "
                f"({n_chains}, {n_draws}) to match the sampled sites."
            )
        samples[name] = value

    idata = from_dict({"posterior": samples})

    idata.posterior.attrs.update(describe_priors(priors))
    idata.posterior.attrs.update(describe_data(data_path, event_id, hash_bytes))
    idata.posterior.attrs.update({
        "likelihood": mcmc.sampler.model.__name__,
        "seed": str(getattr(mcmc, "seed", "")),       # set by run.run_fit, so the run can be reproduced
        "sed_fn": _fn_name(getattr(single_pulse_fn, "sed_fn", None)),
        "pulse_fn": _fn_name(getattr(single_pulse_fn, "pulse_fn", None)),
        "freq_channels_ref": f"{burst.freq_channels_ref:.6f}",
        "n_noise_bins": str(burst.n_noise_bins),
        "shared_params": ",".join(sorted(k.removesuffix("_shared") for k in samples
                                         if k.endswith("_shared"))), # Parameters sampled once for the whole burst, found from their '<name>_shared' sites
        **{k: str(v) for k, v in extra.items()},
        **{f"constant.{k}": str(v)
           for k, v in vars(getattr(single_pulse_fn, "constants", None) or {}).items()
           if np.ndim(v) == 0},
    })

    P1, P2 = burst.spectral_basis
    freq = np.asarray(burst.freq_channels).reshape(-1)
    idata["conditioning"] = xr.Dataset(
        {
            "P1": ("freq_channel", np.asarray(P1).reshape(-1)),
            "P2": ("freq_channel", np.asarray(P2).reshape(-1)),
            "channel_weights": ("freq_channel", np.asarray(burst.channel_weights).reshape(-1)),
            "channel_width": ("freq_channel", np.asarray(burst.channel_width).reshape(-1)),
            "sigma_noise_fd": ("freq_channel", np.asarray(burst.sigma_noise_fd).reshape(-1)),
            "fourier_freqs": ("fourier_freq", np.asarray(burst.fourier_freqs).reshape(-1)),
            **{f"coeff_{k}": float(v) for k, v in burst.spectral_coeff_map.items()},
        },
        coords={"freq_channel": freq},
    )

    idata.to_netcdf(str(path), engine="h5netcdf")
    return idata


def load_run(path):
    """
    Reads a saved fit back. Returns the DataTree: `["posterior"]` holds the samples and, in its
    attrs, the priors, likelihood and data reference; `["conditioning"]` holds the spectral basis
    and per-channel quantities needed to rebuild the model.
    """
    return xr.open_datatree(str(path), engine="h5netcdf")


def check_data_unchanged(run, data_path, n_bytes=None):
    """
    Confirms the file at `data_path` is the one a saved run was fit to, by comparing content
    hashes. Returns True if they match; raises if the run holds no hash to compare against.
    """
    stored = run["posterior"].attrs.get("data.sha256")
    if not stored:
        raise ValueError("this run has no stored data hash to compare against.")
    return stored == describe_data(data_path, n_bytes=n_bytes)["data.sha256"]

def to_idata(mcmc, **kwargs):
    """az.from_numpyro, working around ArviZ's assumption that max_tree_depth is a single number."""
    depth = mcmc.sampler._max_tree_depth
    if isinstance(depth, tuple):
        mcmc.sampler._max_tree_depth = depth[1]      # the sampling-phase depth, which applied to the draws
    try:
        return az.from_numpyro(mcmc, **kwargs)
    finally:
        mcmc.sampler._max_tree_depth = depth
