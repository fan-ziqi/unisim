"""Backend-neutral physics contracts for UniSim.

The distribution is named ``unisim-core`` while the public Python namespace is
``unisim``. Engine adapters are optional and are loaded explicitly by callers.
"""

from .adapters import ADAPTER_SPECS, AdapterSpec, adapter_spec
from .benchmark import BenchmarkCase, BenchmarkResult
from .conformance import assert_backend_conformance
from .contract import (
    BackendCapability,
    BackendError,
    CameraCfg,
    DebugOverlayGetter,
    DebugPrimitive,
    SimBackend,
    UnsupportedCapabilityError,
    validate_debug_overlays,
)
from .factory import create_backend
from .fake import FakeBackend
from .optional import OptionalDependencyError
from .registry import (
    ENTRY_POINT_GROUP,
    PLUGIN_API_VERSION,
    BackendRegistration,
    BackendRegistrationError,
    discover_backends,
    register_backend,
)

__all__ = [
    "BackendCapability",
    "BackendError",
    "BackendRegistration",
    "BackendRegistrationError",
    "BenchmarkCase",
    "BenchmarkResult",
    "ADAPTER_SPECS",
    "AdapterSpec",
    "CameraCfg",
    "DebugOverlayGetter",
    "DebugPrimitive",
    "ENTRY_POINT_GROUP",
    "FakeBackend",
    "DrakeBackend",
    "MJWarpBackend",
    "MjwarpBackend",
    "NewtonBackend",
    "NewtonDependencyError",
    "SuperDexBackend",
    "SuperDexDependencyError",
    "GenesisBackend",
    "IsaacGymBackend",
    "IsaacGymDependencyError",
    "IsaacSimBackend",
    "IsaacSimDependencyError",
    "OptionalDependencyError",
    "PLUGIN_API_VERSION",
    "SubprocessBackend",
    "MjcfSubprocessBackend",
    "SubprocessWorkerError",
    "MuJoCoBackend",
    "MotrixBackend",
    "SimBackend",
    "UnsupportedCapabilityError",
    "assert_backend_conformance",
    "adapter_spec",
    "create_backend",
    "discover_backends",
    "register_backend",
    "validate_debug_overlays",
]


def __getattr__(name: str):
    """Resolve optional adapter exports without importing engine SDKs eagerly."""
    modules = {
        "DrakeBackend": (".backend.drake", "DrakeBackend"),
        "GenesisBackend": (".backend.genesis", "GenesisBackend"),
        "IsaacGymBackend": (".backend.isaacgym", "IsaacGymBackend"),
        "IsaacGymDependencyError": (".backend.isaacgym.dependencies", "IsaacGymDependencyError"),
        "IsaacSimBackend": (".backend.isaacsim", "IsaacSimBackend"),
        "IsaacSimDependencyError": (".backend.isaacsim.dependencies", "IsaacSimDependencyError"),
        "MJWarpBackend": (".backend.mjwarp", "MjwarpBackend"),
        "MjwarpBackend": (".backend.mjwarp", "MjwarpBackend"),
        "NewtonBackend": (".backend.newton", "NewtonBackend"),
        "NewtonDependencyError": (".backend.newton", "NewtonDependencyError"),
        "SuperDexBackend": (".backend.superdex", "SuperDexBackend"),
        "SuperDexDependencyError": (".backend.superdex", "SuperDexDependencyError"),
        "MotrixBackend": (".backend.motrix", "MotrixBackend"),
        "MuJoCoBackend": (".backend.mujoco", "MuJoCoBackend"),
        # ``SubprocessBackend`` was the name used by the first extraction
        # draft. Keep it as a documented alias while the concrete shared host
        # implementation is named ``MjcfSubprocessBackend`` internally.
        "SubprocessBackend": (".backend.subprocess_ipc.backend", "MjcfSubprocessBackend"),
        "MjcfSubprocessBackend": (".backend.subprocess_ipc.backend", "MjcfSubprocessBackend"),
        "SubprocessWorkerError": (".backend.subprocess_ipc.backend", "SubprocessWorkerError"),
    }
    target = modules.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
