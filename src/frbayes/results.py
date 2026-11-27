"""
results.py - Saving and loading fits of the FRB model.

A saved run holds the posterior samples and everything needed to interpret them: the priors, the
reference frequency, the per-channel quantities the model was built from, which likelihood was
used, and a reference to the data. The DATA ITSELF IS NOT SAVED -- only an identifier, path and
content hash, so a run can be traced back to its input and checked against it without duplicating
large files across the catalogue.
"""

import hashlib
from pathlib import Path

import numpy as np
import xarray as xr
from arviz_base import from_dict
import arviz as az


def _fn_name(fn):
    """
    Name of a function for the run record; netcdf attributes cannot hold the function itself.

    fn: the function, its name (model.setup_model stores the SED and pulse forms by name), or None
    """
    if fn is None:
        return ""
    if isinstance(fn, str):
        return fn
    return getattr(fn, "__name__", type(fn).__name__)


def describe_priors(priors):
    """
    Serializable description of the priors, one string per parameter. NumPyro distributions have no
    informative repr (it gives the object address), so each is described by its class name and its
    own arguments, which is what has to be reproducible.

    priors: the priors dict, as returned by run.set_priors
    """
    record = {}
    for name, d in priors.items():
        args = {k: float(np.asarray(getattr(d, k))) for k in d.arg_constraints if hasattr(d, k)}
        record[f"prior.{name}"] = f"{type(d).__name__}({', '.join(f'{k}={v:g}' for k, v in args.items())})"
    return record


def describe_data(path, event_id=None, n_bytes=None):
    """
    Identifier, location and content hash of the analyzed file. The hash is what makes the
    reference meaningful: an event ID says which burst, but not which version of its data, and
    reprocessing upstream changes the file without changing the name.

    path:     the file that was analyzed
    event_id: identifier of the burst; None records an empty string
    n_bytes:  hash only the first n_bytes of the file, for speed; None hashes all of it, the safer
              default (a few hundred MB takes under a second)
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


def _dedispersion_value(given, from_burst, name):
    """
    One of the two dedispersion values for save_run: the one given, else the burst's (from its
    .npz); raises if both are set and differ, or if neither is.

    given:      the value passed to save_run, or None
    from_burst: the burst's value, or None
    name:       the value's name, for the error messages
    """
    if given is not None and from_burst is not None and float(given) != float(from_burst):
        raise ValueError(f"{name} = {given} differs from the burst's {name} = {from_burst} (read from its .npz).")
    value = given if given is not None else from_burst
    if value is None:
        raise ValueError(f"{name} must be given, here or as '{name}' in the burst's .npz: the DM (pc cm^-3) the "
                         f"data were dedispersed with, and the frequency (MHz, np.inf for infinite frequency) "
                         f"it aligned the channels to. The fitted DM is only the residual on top of the first, "
                         f"and the fitted arrival times are on the telescope's clock only at the second.")
    return float(value)


def save_run(path, mcmc, burst, priors, single_pulse_fn=None, hash_bytes=None, extra_samples=None,
             dm_dedispersion=None, dedispersion_ref_freq=None, **extra):
    """
    Writes a fit to netcdf: the posterior samples, with the priors, the data reference (the burst's
    event_id and data_path, and the file's content hash), the seeds of the NUTS run and of the maxL
    fit it started from ('seed', 'maxL_seed'; empty if none) and the model's scalar constants as
    attributes, and the per-channel quantities the model was built from in the 'conditioning'
    group. When NUTS was started from a maxL fit (run.run_NUTS with maxL_start), that fit is kept
    in the 'maxL' group: every site's value, pulses in the fit's own order, with its best ln L
    (including the repulsion prior) and seed as attributes. The reference frequency is recorded because gamma and tau_ref are defined relative to
    it and it differs per burst. The posterior also gets 'DM_total' = dm_dedispersion + DM, draw by
    draw: the dedispersion is a known shift applied to the data and the fit measures what remains,
    so the spread of DM_total is the spread of the residual DM. Returns the DataTree written.

    path:            output file (netcdf)
    mcmc:            the finished MCMC object from run.run_NUTS; samples are taken with
                     group_by_chain=True, so the leading two axes are (chain, draw)
    burst:           the ConditionedData the fit used; its event_id and data_path identify the data,
                     whose contents are hashed, not stored
    priors:          the priors dict of the fit, as returned by run.set_priors
    single_pulse_fn: the model closure the fit used; the names of its SED and pulse forms and its
                     scalar constants are recorded
    hash_bytes:      hash only the first hash_bytes of the data file (describe_data); None hashes all of it
    extra_samples:   further posterior arrays to store beside the sampled sites, each already shaped
                     (n_chains, n_draws, ...) and not sharing a name with a sampled site
    dm_dedispersion: the DM (pc cm^-3) the data were dedispersed with before this fit, e.g. from
                     DM_phase; the fitted 'DM' is only the residual on top of it. None takes the
                     burst's (from its .npz); one of the two is required. Recorded as the attribute
                     'DM_dedispersion'
    dedispersion_ref_freq:
                     the frequency (MHz, np.inf for infinite frequency) the dedispersion
                     aligned the channels to; None takes the burst's, and one of the two is required.
                     Recorded as 'dedispersion_ref_freq'. The data's time
                     axis keeps the telescope's clock only at this frequency nu_dd, so the fitted
                     arrival time t at the model's reference nu_r is t + k DM_dedispersion
                     (nu_r^-2 - nu_dd^-2) on the telescope's clock
    extra:           any further scalars worth recording (n_pulses, ...), stored as strings
    """
    dm_dedispersion = _dedispersion_value(dm_dedispersion, getattr(burst, "dm_dedispersion", None),
                                          "dm_dedispersion")
    dedispersion_ref_freq = _dedispersion_value(dedispersion_ref_freq, getattr(burst, "dedispersion_ref_freq", None),
                                                "dedispersion_ref_freq")

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    n_chains, n_draws = next(iter(samples.values())).shape[:2]
    if "DM_total" in samples or "DM_total" in (extra_samples or {}):
        raise ValueError("'DM_total' is computed by save_run and cannot be a sampled site or extra sample.")
    if "DM" in samples:
        samples["DM_total"] = dm_dedispersion + samples["DM"]      # residual DM on top of the dedispersion DM

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
    idata.posterior.attrs.update(describe_data(burst.data_path, burst.event_id or None, hash_bytes))
    idata.posterior.attrs.update({
        "likelihood": mcmc.sampler.model.__name__,
        "seed": str(getattr(mcmc, "seed", "")),       # set by run.run_NUTS, so the run can be reproduced
        # the seed of the maxL fit the chains started from, also needed to reproduce the run
        "maxL_seed": "" if getattr(mcmc, "maxL_seed", None) is None else str(mcmc.maxL_seed),
        "sed_fn": _fn_name(getattr(single_pulse_fn, "sed_fn", None)),
        "pulse_fn": _fn_name(getattr(single_pulse_fn, "pulse_fn", None)),
        "freq_channels_ref": f"{burst.freq_channels_ref:.6f}",
        "DM_dedispersion": repr(dm_dedispersion),      # pc cm^-3, applied to the data before the fit
        "dedispersion_ref_freq": repr(dedispersion_ref_freq),   # MHz the channels were aligned to (inf allowed)
        "n_noise_bins": str(burst.n_noise_bins),
        "shared_params": ",".join(sorted(k.removesuffix("_shared") for k in samples
                                         if k.endswith("_shared"))), # Parameters sampled once for the whole burst, found from their '<name>_shared' sites
        **{k: str(v) for k, v in extra.items()},
        **{f"constant.{k}": str(v)
           for k, v in vars(getattr(single_pulse_fn, "constants", None) or {}).items()
           if np.ndim(v) == 0},
    })

    freq = np.asarray(burst.freq_channels).reshape(-1)
    conditioning = {
        "channel_weights": ("freq_channel", np.asarray(burst.channel_weights).reshape(-1)),
        "channel_width": ("freq_channel", np.asarray(burst.channel_width).reshape(-1)),
        "sigma_noise_fd": ("freq_channel", np.asarray(burst.sigma_noise_fd).reshape(-1)),
        "fourier_freqs": ("fourier_freq", np.asarray(burst.fourier_freqs).reshape(-1)),
    }
    if burst.freq_members is not None:          # raw channels averaged into each block, for the smearing
        conditioning["freq_members"] = (("freq_channel", "member"), np.asarray(burst.freq_members))
    idata["conditioning"] = xr.Dataset(conditioning, coords={"freq_channel": freq})

    # The maxL fit the chains started from (run.run_NUTS with maxL_start): every site's value at it,
    # pulses in the fit's own order (not sorted by arrival time), with its best ln L and seed
    maxL_fit = getattr(mcmc, "maxL_fit", None)
    if maxL_fit is not None:
        from .postprocess import PAIR_SITES
        n_pulses = int(np.size(maxL_fit["map"]["mean_arrival_time"]))
        variables = {}
        for name, value in maxL_fit["map"].items():
            value = np.asarray(value)
            if name in PAIR_SITES:
                dims = ("pulse_pair",)
            elif value.shape == (n_pulses,):
                dims = ("pulse",)
            else:
                dims = tuple(f"{name}_dim_{i}" for i in range(value.ndim))
            variables[name] = (dims, value)
        idata["maxL"] = xr.Dataset(variables, attrs={
            "logL": "" if maxL_fit["logL"] is None else repr(maxL_fit["logL"]),   # includes the repulsion prior
            "seed": "" if getattr(mcmc, "maxL_seed", None) is None else str(mcmc.maxL_seed),
        })

    idata.to_netcdf(str(path), engine="h5netcdf")
    return idata


def load_run(path):
    """
    Reads a saved fit back into memory and closes the file, so the file can be overwritten (e.g. by
    running save_run again) while the returned tree is still in use. Returns the DataTree:
    ["posterior"] holds the samples and, in its attrs, the priors, likelihood, model constants and
    data reference; ["conditioning"] holds the per-channel quantities needed to rebuild the model;
    ["maxL"], when NUTS started from a maxL fit, holds that fit.

    path: the file save_run wrote
    """
    tree = xr.open_datatree(str(path), engine="h5netcdf")
    try:
        return tree.load()
    finally:
        tree.close()


def check_data_unchanged(run, data_path, n_bytes=None):
    """
    Confirms the file at data_path is the one a saved run was fit to, by comparing content hashes.
    Returns True if they match; raises if the run holds no hash to compare against.

    run:       the DataTree load_run returned
    data_path: the file to check
    n_bytes:   hash only its first n_bytes; must match the hash_bytes the run was saved with
    """
    stored = run["posterior"].attrs.get("data.sha256")
    if not stored:
        raise ValueError("this run has no stored data hash to compare against.")
    return stored == describe_data(data_path, n_bytes=n_bytes)["data.sha256"]


def to_idata(mcmc, **kwargs):
    """
    az.from_numpyro, working around ArviZ's assumption that max_tree_depth is a single number.

    mcmc:   the finished MCMC object
    kwargs: passed to az.from_numpyro
    """
    depth = mcmc.sampler._max_tree_depth
    if isinstance(depth, tuple):
        mcmc.sampler._max_tree_depth = depth[1]      # the sampling-phase depth, which applied to the draws
    try:
        return az.from_numpyro(mcmc, **kwargs)
    finally:
        mcmc.sampler._max_tree_depth = depth
