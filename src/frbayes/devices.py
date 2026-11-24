"""
devices.py - Choosing whether FRBayes runs on the CPU or a GPU.

JAX fixes its devices when it starts, so use_device has to run BEFORE jax is imported: as the
first lines of a script or notebook, before importing jax, numpyro or any FRBayes module that
imports them (model, likelihood, run, ...). This module does not import jax itself.

    from frbayes.devices import use_device
    use_device("auto")          # or "gpu" / "cpu"

    import jax, numpyro
    from frbayes.run import run_NUTS, ...

run.run_NUTS then picks the matching chain_method on its own.

Why the CPU setup does more than select the CPU: JAX sizes its thread pools from the cores the
process is allowed to run on. Some clusters cap a job's CPU TIME (a container quota, cgroup
cpu.max) instead of restricting which cores it runs on, so every core of the node stays visible
-- e.g. 96 when 8 are allocated. JAX then starts threads for all 96, the quota is used up early
in every scheduling period, and the whole process is paused until the next one: a 2-minute fit
took hours. use_device("cpu") first restricts the process to as many cores as it was allocated.
"""

import importlib.util
import os
import sys
from pathlib import Path


def _cgroup_cpu_quota():
    """Cores allowed by the container's CPU-time quota; None if no quota is set or none is found."""
    try:                    # cgroup v2: "<quota> <period>" in microseconds, or "max <period>"
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else int(quota) // int(period)
    except (OSError, ValueError):
        pass
    try:                    # cgroup v1: quota is -1 when there is none
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if quota <= 0 else quota // period
    except (OSError, ValueError):
        return None


def allocated_cpus() -> int:
    """
    Number of CPU cores this process may actually use: the smaller of the cores it is allowed to
    run on and the container's CPU quota, when there is one.

    os.cpu_count() is not used: it counts every core of the node, allocated or not.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    quota = _cgroup_cpu_quota()
    if quota is not None:
        n = min(n, quota)
    return max(1, n)


def _gpu_visible() -> bool:
    return Path("/dev/nvidiactl").exists() or Path("/proc/driver/nvidia/gpus").is_dir()


def _jax_cuda_installed() -> bool:
    return any(importlib.util.find_spec(f"jax_cuda{v}_plugin") for v in (12, 13))


def gpu_available() -> bool:
    """True if an NVIDIA GPU is visible AND JAX's CUDA support is installed (pip install '.[gpu]')."""
    return _gpu_visible() and _jax_cuda_installed()


def use_device(device: str = "auto", n_cpu: int = None) -> str:
    """
    Sets JAX up to run on `device` and returns the device chosen, "gpu" or "cpu".

    device: "gpu":  raises if no GPU is visible or JAX's CUDA support is missing, rather than
                    falling back to the CPU without saying so. All chains then run as one batch
                    on the GPU (chain_method="vectorized").
            "cpu":  restricts the process to n_cpu cores and makes one JAX device per core, so up
                    to n_cpu chains run in parallel (chain_method="parallel").
            "auto": "gpu" if gpu_available(), otherwise "cpu".
    n_cpu:  cores to use on the CPU; default allocated_cpus(). Ignored on the GPU.

    Must be called before jax is imported; raises otherwise, since JAX would ignore the settings.
    """
    if "jax" in sys.modules:
        raise RuntimeError(
            "use_device must run before jax is imported, because JAX fixes its devices when it "
            "starts. Restart the kernel and make use_device the first thing that runs."
        )

    device = device.lower()
    if device == "auto":
        device = "gpu" if gpu_available() else "cpu"

    if device == "gpu":
        if not _gpu_visible():
            raise RuntimeError("no NVIDIA GPU is visible on this machine; use_device('cpu') instead.")
        if not _jax_cuda_installed():
            raise RuntimeError("JAX's CUDA support is not installed: pip install -e '.[gpu]'.")
        os.environ["JAX_PLATFORMS"] = "cuda"
        print("FRBayes: running on the GPU")

    elif device == "cpu":
        n = allocated_cpus() if n_cpu is None else int(n_cpu)
        if hasattr(os, "sched_setaffinity"):        # Linux; JAX sizes its thread pools from this
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:n])
        os.environ["JAX_PLATFORMS"] = "cpu"
        other_flags = [f for f in os.environ.get("XLA_FLAGS", "").split()
                       if not f.startswith("--xla_force_host_platform_device_count")]
        os.environ["XLA_FLAGS"] = " ".join(other_flags + [f"--xla_force_host_platform_device_count={n}"])
        print(f"FRBayes: running on the CPU with {n} cores (up to {n} chains in parallel)")

    else:
        raise ValueError(f"device must be 'gpu', 'cpu' or 'auto'; got {device!r}.")

    return device
