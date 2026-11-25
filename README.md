# FRBayes

**This is an alpha release.** The interface may change between versions, and results should be
checked by eye (the plots are there for that); please report problems on the repository's issue
tracker.

Bayesian parameter estimation for fast radio bursts, with JAX and NumPyro. FRBayes fits a
multi-pulse model of a dedispersed dynamic spectrum in the Fourier domain: a maximum-likelihood
search finds the number of pulses, and NUTS samples the posterior of every pulse's arrival time,
width, spectrum and fluence, and of the scattering time and residual DM, which the pulses share by
default. FRBayes can fit both baseband data, coherently dedispersed, and L1 intensity data,
incoherently dedispersed. It is built for an NVIDIA GPU; it also runs on a CPU, but much more
slowly, so the CPU suits small tests rather than real fits.

## Requirements

- Linux
- git, to install from GitHub
- An NVIDIA GPU with a driver that supports CUDA 12, **strongly recommended**: the fits are written
  for it, and run much more slowly on a CPU. The CUDA libraries themselves are installed with
  FRBayes; nothing else needs installing system-wide.

FRBayes needs Python 3.13; uv, below, installs it for you if your system does not have it.

## Installation

FRBayes is installed with [uv](https://docs.astral.sh/uv/), a tool that installs Python packages
into a separate environment for each project, so that FRBayes gets the versions of JAX, NumPyro and
the other packages it needs without changing anything else on your system.

### 1. Install uv

Check whether uv is already installed with `uv --version`. If it is not:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then open a new terminal, or run `source $HOME/.local/bin/env`, so that the `uv` command is found.
If `curl` is not available, `pip install --user uv` also works.

### 2. Create a project

A uv project is a folder for your own analysis, your notebooks and scripts, together with the list
of packages it needs. Create one, with Python 3.13, and move into it:

```bash
uv init --python 3.13 my-frb-analysis
cd my-frb-analysis
```

This creates the folder with `pyproject.toml`, the project's list of packages, a few starter files
(`main.py`, `README.md`, `.python-version`) that you can ignore, and an empty git repository. uv
downloads Python 3.13 if your system does not have it.

### 3. Add FRBayes

From inside the project folder:

```bash
uv add "frbayes[gpu] @ git+https://github.com/<owner>/FRBayes@v0.1.1"    # NVIDIA GPU (strongly recommended)
uv add "frbayes @ git+https://github.com/<owner>/FRBayes@v0.1.1"         # CPU only
```

This creates the project's virtual environment, a private Python installation in the folder
`.venv`, and installs FRBayes into it with every package it needs, at versions that work together.
Nothing is installed outside the project folder. The exact versions are recorded in `uv.lock`, so
`uv sync` can recreate the same environment later or on another machine.

- `@v0.1.1` installs that release of FRBayes.
- `[gpu]` also installs JAX's NVIDIA CUDA libraries, a few GB, which install only on Linux with an
  NVIDIA GPU; leave it out on a machine without one.
- FRBayes is not on PyPI yet, so it is installed straight from its GitHub repository. If the
  repository is private, you need access to it, and git must be able to sign in to GitHub; to use
  an SSH key, write `git+ssh://git@github.com/<owner>/FRBayes@v0.1.1` instead.

### 4. Use the environment

Run a script inside the environment with `uv run`, from the project folder:

```bash
uv run python my_script.py
```

or activate the environment, so that `python` in that terminal is the project's:

```bash
source .venv/bin/activate
```

To use FRBayes in Jupyter, register the environment as a Jupyter kernel once:

```bash
uv add ipykernel
uv run python -m ipykernel install --user --name frbayes --display-name "FRBayes"
```

and choose the "FRBayes" kernel in your notebook. If you do not have Jupyter already, install it in
the project with `uv add jupyterlab` and start it with `uv run jupyter lab`.

Add any other package the same way, with `uv add <package>`, not `pip install`: uv keeps the
environment matching the project's list, and the next time it updates the environment it removes
or replaces anything installed with pip.

### 5. Check the installation

From the project folder:

```bash
uv run python -c "from frbayes.devices import use_device; use_device('auto'); import jax; print(jax.devices())"
```

prints `[CudaDevice(id=0)]` on a GPU machine with `[gpu]` installed, and a list of CPU devices
otherwise.

### Without uv

In an environment uv does not manage, such as a conda environment, install with pip instead, and
run the check above with `python` in place of `uv run python`:

```bash
pip install "frbayes[gpu] @ git+https://github.com/<owner>/FRBayes@v0.1.1"    # NVIDIA GPU (strongly recommended)
pip install "frbayes @ git+https://github.com/<owner>/FRBayes@v0.1.1"         # CPU only
```

Use one tool per environment: do not install into a uv project with pip, or into a conda
environment with uv.

### From source, for development

```bash
git clone https://github.com/<owner>/FRBayes
cd FRBayes
uv sync --extra gpu --extra dev     # leave out --extra gpu on a machine without an NVIDIA GPU
```

This needs Python 3.13 already installed on the system, since FRBayes's own `pyproject.toml` tells
uv to use the system's Python. Run `uv sync` in the folder that contains `pyproject.toml`. It
creates the environment in `.venv`, installs FRBayes into it in editable mode (changes to
`src/frbayes/` take effect without reinstalling), and installs the extras you name: `gpu` (JAX's
CUDA 12 support) and `dev` (pytest, Jupyter, ruff, seaborn and ipykernel). Name every extra you want
each time: `uv sync` removes the packages of any extra left out, so syncing without `--extra gpu`
takes away the GPU support. Register the Jupyter kernel as in step 4, without the `uv add
ipykernel`, which the `dev` extra already installs.

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

| key                     | shape            | contents                                                         |
|-------------------------|------------------|------------------------------------------------------------------|
| `power`                 | (n_chan, n_time) | the dynamic spectrum, already dedispersed                        |
| `freq`                  | (n_chan,)        | channel centre frequencies, in **MHz**                           |
| `time`                  | (n_time,)        | sample times, in **seconds**                                     |
| `dm_dedispersion`       | scalar, optional | the DM the data were dedispersed with (pc cm⁻³)                  |
| `dedispersion_ref_freq` | scalar, optional | the frequency the channels were aligned to (MHz; `inf` for infinite frequency) |

The units matter: the dispersion constant and every prior assume MHz and seconds. Channels may be
in any order. Bad channels must be **removed** from all three arrays, not zeroed or masked. The two
optional values record how the data were dedispersed, which `results.save_run` needs; without them in
the file, pass them to `save_run` instead.

The priors also need the telescope's band, `band=(bottom, top)` in MHz, which sets the bound of the
spectral index prior. It is the full band, not the channels that survived, so the prior is the same
for every burst.

## Usage

The examples demonstrate fitting baseband data, on the CHIME/FRB burst FRB 20181221A. Its data file
is not included in the repository, so run the two notebooks in this order:

1. `examples/notebooks/dedisperse_baseband.ipynb` coherently dedisperses the burst's CHIME/FRB
   baseband data and writes it to `examples/data/` in the `.npz` format above. It needs the
   CHIME/FRB `baseband_analysis` package and access to the CHIME/FRB baseband files.
2. `examples/notebooks/tutorial.ipynb` then walks through a full fit of that file, from loading the
   data to saving the posterior.

L1 intensity data are fitted the same way, except that the model constant `DM_incoherent` (step 5 of
the tutorial) is the DM the data were dedispersed with, not 0, so that the smearing within each
channel is modelled. Every function's docstring describes its arguments.

## Known limitations

- The likelihood has no self-noise term: the extra noise a bright burst adds while it is on is
  ignored, so uncertainties on bright bursts are too small and the pulse search may add pulses
  too readily.
- The noise is assumed white (the same at every Fourier frequency), which may not hold for real data.
- The noise level is estimated from the data before the burst and then treated as exactly known;
  its own uncertainty is not included.
- The telescope's beam response is not modelled (`det_response` is a constant, 1 by default), so the
  fitted spectra and fluences are those of the burst as seen through the beam, frequency structure
  of the beam included.
- A parameter whose posterior sits against a prior bound (an unresolved width at the narrowest
  allowed value, or a scattering time at the longest the window allows) is not flagged
  automatically; such values are limits, not measurements.
- Fitted arrival times are measured from the start of the data segment. The segment's start time is
  not kept, so they cannot yet be converted to absolute times.

## License

MIT; see `LICENSE`.
