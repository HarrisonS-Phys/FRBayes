# FRBayes

**This is an alpha release.** The interface may change between versions, and results should be
checked by eye (the plots are there for that); please report problems on the repository's issue
tracker.

Bayesian parameter estimation for fast radio bursts, with JAX and NumPyro. FRBayes fits a
multi-pulse model of a dedispersed dynamic spectrum in the Fourier domain: a maximum-likelihood
search finds the number of pulses, and NUTS samples the posterior of every pulse's arrival time,
width, spectrum and fluence, and of the scattering time and residual DM, which the pulses share by
default. It is built for an NVIDIA GPU; it also runs on a CPU, but much more slowly, so the CPU
suits small tests rather than real fits.

## Requirements

- Linux, with Python 3.13
- An NVIDIA GPU with a driver that supports CUDA 12, **highly recommended**: the fits are written
  for it, and run much more slowly on a CPU. The CUDA libraries themselves are installed by the
  `gpu` extra; nothing else needs installing system-wide.

## Installation

### From PyPI

In a project managed with [uv](https://docs.astral.sh/uv/) (create one with `uv init`):

```bash
uv add "frbayes[gpu]"          # NVIDIA GPU (recommended)
uv add frbayes                 # CPU only
```

In an environment uv does not manage, such as a conda environment or a plain virtual environment:

```bash
pip install "frbayes[gpu]"     # NVIDIA GPU (recommended)
pip install frbayes            # CPU only
```

Use one tool per environment. In a uv project, add packages only with `uv add`: `uv sync` makes the
project's environment match its lockfile exactly, so anything installed there with `pip` is removed
or overwritten at the next sync.

The GPU support is an extra, not part of the default install, because it adds a few GB of NVIDIA
CUDA libraries that install only on Linux with an NVIDIA GPU.

### From source, for development

```bash
git clone <repository URL> FRBayes
cd FRBayes
uv sync --extra gpu --extra dev     # leave out --extra gpu on a machine without an NVIDIA GPU
```

Run `uv sync` in the folder that contains `pyproject.toml`. It creates the environment in `.venv`,
installs FRBayes into it in editable mode (changes to `src/frbayes/` take effect without
reinstalling), and installs the extras you name: `gpu` (JAX's CUDA 12 support) and `dev` (pytest,
Jupyter, ruff, seaborn and ipykernel). Name every extra you want each time: `uv sync` removes the
packages of any extra left out, so syncing without `--extra gpu` takes away the GPU support.

To use the environment from Jupyter, register it as a kernel once:

```bash
uv run python -m ipykernel install --user --name frbayes --display-name "FRBayes"
```

and choose the "FRBayes" kernel in the notebook.

### Checking the installation

```bash
uv run python -c "from frbayes.devices import use_device; use_device('auto'); import jax; print(jax.devices())"
```

prints `[CudaDevice(id=0)]` on a GPU machine with the `gpu` extra installed, and a list of CPU
devices otherwise.

## Choosing the device

JAX fixes its devices when it starts, so choose the device first, before anything imports jax
(including the FRBayes modules other than `devices`):

```python
from frbayes.devices import use_device
use_device("auto")        # "gpu", "cpu", or "auto": the GPU if one is usable, otherwise the CPU
```

- `"gpu"` raises if no GPU is visible or JAX's CUDA support is missing, rather than falling back to
  the CPU without saying so. All chains then run together on the GPU.
- `"cpu"` restricts the process to the cores it was allocated and runs up to that many chains in
  parallel. On a cluster that caps a job's CPU time without restricting which cores it uses, this
  stops JAX from starting a thread for every core of the node, which can slow a fit by orders of
  magnitude.

By default JAX reserves 75% of the GPU's memory when it starts. To reserve less, e.g. on a shared
GPU, set this before importing jax:

```python
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")
```

To switch device, restart the Python session or notebook kernel.

## Input data

One burst per `.npz` file, holding

| key     | shape             | contents                                        |
|---------|-------------------|-------------------------------------------------|
| `power` | (n_chan, n_time)  | the dynamic spectrum, already dedispersed       |
| `freq`  | (n_chan,)         | channel centre frequencies, in **MHz**          |
| `time`  | (n_time,)         | sample times, in **seconds**                    |

The units matter: the dispersion constant and every prior assume MHz and seconds. Channels may be
in any order. Bad channels must be **removed** from all three arrays, not zeroed or masked.

The priors also need the telescope's band, `band=(bottom, top)` in MHz, which sets the bound of the
spectral index prior. It is the full band, not the channels that survived, so the prior is the same
for every burst.

## Usage

The example notebook in `examples/` walks through a full fit, from loading the data to saving the
posterior. Every function's docstring describes its arguments.

## Known limitations

- The likelihood has no self-noise term: the extra noise a bright burst adds while it is on is
  ignored, so uncertainties on bright bursts are too small and the pulse search may add pulses
  too readily.
- The noise is assumed white (the same at every Fourier frequency), which may not hold for real data.
- The noise level is estimated from the data before the burst and then treated as exactly known;
  its own uncertainty is not included.
- A parameter whose posterior sits against a prior bound (an unresolved width at the narrowest
  allowed value, or a scattering time at the longest the window allows) is not flagged
  automatically; such values are limits, not measurements.

## License

MIT; see `LICENSE`.
