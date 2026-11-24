"""
FRBayes - Bayesian fitting of fast radio bursts in the Fourier domain, with JAX and NumPyro.

Choose the device first, before anything imports jax:

    from frbayes.devices import use_device
    use_device("auto")

then import the modules you need, e.g. `from frbayes import data, model, run`. This file imports
none of them, so importing frbayes does not start JAX.
"""

from importlib.metadata import version

__version__ = version("FRBayes")      # from pyproject.toml, so the version is set in one place
