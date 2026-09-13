"""Versioned discovery for third-party :class:`~unisim.SimBackend` providers.

The built-in adapter manifest intentionally remains static.  Third-party
providers are instead installed under the ``unisim.backends`` Python entry
point group and are loaded only when requested.  A provider cannot mutate the
built-in factory table and must return a typed registration with the current
plugin API version.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from .contract import BackendCapability, BackendError, SimBackend

ENTRY_POINT_GROUP = "unisim.backends"
PLUGIN_API_VERSION = 1

_BACKEND_NAME = re.compile(r"[a-z][a-z0-9_-]*\Z")
_RESERVED_NAMES = frozenset(
    {
        "fake",
        "mujoco",
        "motrix",
        "drake",
        "mjwarp",
        "newton",
        "superdex",
        "genesis",
        "isaacgym",
        "isaacsim",
    }
)


class BackendRegistrationError(BackendError):
    """Raised when a third-party backend registration is invalid or ambiguous."""


BackendFactory = Callable[..., SimBackend]


@dataclass(frozen=True, slots=True)
class BackendRegistration:
    """A versioned, lazy third-party backend provider declaration.

    ``factory`` receives the same ``(scene, num_envs, sim_dt, **kwargs)``
    request that :func:`unisim.create_backend` receives.  Its returned backend
    must identify itself with ``backend_type == name`` and must expose every
    capability declared here.  The declaration is metadata only: Euler (or
    another provider) remains the owner of its model, state, controls and
    contacts.
    """

    name: str
    factory: BackendFactory
    capabilities: frozenset[BackendCapability]
    api_version: int = PLUGIN_API_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _BACKEND_NAME.fullmatch(self.name) is None:
            raise BackendRegistrationError(
                "backend registration name must match [a-z][a-z0-9_-]*"
            )
        if self.name in _RESERVED_NAMES:
            raise BackendRegistrationError(
                f"backend registration name {self.name!r} is reserved by UniSim"
            )
        if not callable(self.factory):
            raise BackendRegistrationError("backend registration factory must be callable")
        if self.api_version != PLUGIN_API_VERSION:
            raise BackendRegistrationError(
                f"backend registration {self.name!r} declares API version {self.api_version}; "
                f"UniSim requires {PLUGIN_API_VERSION}"
            )
        capabilities = frozenset(self.capabilities)
        unknown = capabilities.difference(BackendCapability)
        if unknown:
            raise BackendRegistrationError(
                f"backend registration {self.name!r} declares unknown capabilities: "
                f"{sorted(map(str, unknown))}"
            )
        object.__setattr__(self, "capabilities", capabilities)


_REGISTERED: dict[str, BackendRegistration] = {}
_LOADED_ENTRY_POINTS: set[str] = set()


def register_backend(registration: BackendRegistration) -> BackendRegistration:
    """Register one provider explicitly, rejecting conflicting registrations.

    Re-registering the exact same immutable declaration is idempotent; a
    different provider under an existing name fails closed.  Providers should
    normally be surfaced through the :data:`ENTRY_POINT_GROUP` entry point
    group instead of being imported eagerly by an application.
    """
    if not isinstance(registration, BackendRegistration):
        raise TypeError("registration must be a BackendRegistration")
    current = _REGISTERED.get(registration.name)
    if current is not None:
        if current == registration:
            return current
        raise BackendRegistrationError(
            f"UniSim backend {registration.name!r} is already registered by a different provider"
        )
    _REGISTERED[registration.name] = registration
    return registration


def discover_backends() -> tuple[BackendRegistration, ...]:
    """Load every installed third-party provider deliberately and return it.

    This is an explicit cold-path operation.  Normal backend creation only
    loads the entry point whose name was requested, preserving UniSim's lazy
    optional-runtime import boundary.
    """
    for entry_point in _entry_points():
        _load_entry_point(entry_point)
    return tuple(_REGISTERED[name] for name in sorted(_REGISTERED))


def backend_registration(name: str) -> BackendRegistration | None:
    """Return a registration, lazily loading only its matching entry point."""
    registered = _REGISTERED.get(name)
    if registered is not None:
        return registered
    matches = [entry_point for entry_point in _entry_points() if entry_point.name == name]
    if len(matches) > 1:
        rendered = ", ".join(sorted(str(entry_point.value) for entry_point in matches))
        raise BackendRegistrationError(
            f"multiple UniSim providers claim backend {name!r}: {rendered}"
        )
    if not matches:
        return None
    return _load_entry_point(matches[0])


def create_registered_backend(
    registration: BackendRegistration,
    scene: Any | None,
    num_envs: int,
    sim_dt: float,
    kwargs: dict[str, Any],
) -> SimBackend:
    """Create and validate one registered backend without adapting its state."""
    backend = registration.factory(scene, num_envs, sim_dt, **kwargs)
    if not isinstance(backend, SimBackend):
        raise BackendRegistrationError(
            f"backend {registration.name!r} factory returned "
            f"{type(backend).__name__}, not SimBackend"
        )
    if backend.backend_type != registration.name:
        raise BackendRegistrationError(
            f"backend {registration.name!r} factory returned backend_type "
            f"{backend.backend_type!r}"
        )
    missing = registration.capabilities.difference(backend.capabilities)
    if missing:
        raise BackendRegistrationError(
            f"backend {registration.name!r} did not expose declared capabilities: "
            f"{sorted(capability.value for capability in missing)}"
        )
    return backend


def _entry_points() -> tuple[Any, ...]:
    """Return entry points across Python 3.10+ metadata API variants."""
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return tuple(entry_points.select(group=ENTRY_POINT_GROUP))
    if hasattr(entry_points, "get"):
        return tuple(entry_points.get(ENTRY_POINT_GROUP, ()))
    return tuple(entry_points)


def _load_entry_point(entry_point: Any) -> BackendRegistration:
    key = f"{entry_point.name}={entry_point.value}"
    existing = _REGISTERED.get(entry_point.name)
    if key in _LOADED_ENTRY_POINTS:
        if existing is None:
            raise BackendRegistrationError(
                f"UniSim provider entry point {entry_point.name!r} did not register a backend"
            )
        return existing
    _LOADED_ENTRY_POINTS.add(key)
    try:
        provider = entry_point.load()
    except Exception as exc:
        raise BackendRegistrationError(
            f"failed to load UniSim provider entry point {entry_point.name!r}: {exc}"
        ) from exc
    if not callable(provider):
        raise BackendRegistrationError(
            f"UniSim provider entry point {entry_point.name!r} must resolve to a callable"
        )
    registration = provider()
    if not isinstance(registration, BackendRegistration):
        raise BackendRegistrationError(
            f"UniSim provider entry point {entry_point.name!r} returned "
            "a value other than BackendRegistration"
        )
    if registration.name != entry_point.name:
        raise BackendRegistrationError(
            f"UniSim provider entry point {entry_point.name!r} registered "
            f"{registration.name!r} instead"
        )
    return register_backend(registration)


def _reset_registry_for_test() -> None:
    """Reset process-local provider state for focused deterministic tests."""
    _REGISTERED.clear()
    _LOADED_ENTRY_POINTS.clear()


__all__ = [
    "BackendFactory",
    "BackendRegistration",
    "BackendRegistrationError",
    "ENTRY_POINT_GROUP",
    "PLUGIN_API_VERSION",
    "backend_registration",
    "create_registered_backend",
    "discover_backends",
    "register_backend",
]
