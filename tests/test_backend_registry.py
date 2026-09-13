from __future__ import annotations

from dataclasses import dataclass

import pytest

import unisim.registry as registry
from unisim import (
    BackendCapability,
    BackendRegistration,
    BackendRegistrationError,
    FakeBackend,
    create_backend,
    discover_backends,
    register_backend,
)


class _EulerFakeBackend(FakeBackend):
    backend_type = "euler"


def _euler_factory(scene, num_envs, sim_dt, **kwargs):
    del scene, sim_dt
    assert kwargs.pop("body_state_required") is False
    assert not kwargs
    return _EulerFakeBackend(num_envs=num_envs)


def _euler_registration() -> BackendRegistration:
    return BackendRegistration(
        name="euler",
        factory=_euler_factory,
        capabilities=frozenset(
            {
                BackendCapability.RESET,
                BackendCapability.SELECTED_RESET,
                BackendCapability.STATE_READ,
                BackendCapability.STATE_WRITE,
            }
        ),
    )


@dataclass(frozen=True)
class _EntryPoint:
    name: str
    value: str
    provider: object

    def load(self):
        return self.provider


@pytest.fixture(autouse=True)
def _clear_registry():
    registry._reset_registry_for_test()
    yield
    registry._reset_registry_for_test()


def test_requested_entry_point_loads_lazily_and_factory_preserves_raw_kwargs(monkeypatch):
    loaded: list[str] = []

    def provider():
        loaded.append("euler")
        return _euler_registration()

    entry_point = _EntryPoint("euler", "euler_unisim.plugin:provider", provider)
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: [entry_point])

    backend = create_backend("euler", None, 3, 0.002)

    assert backend.backend_type == "euler"
    assert backend.num_envs == 3
    assert loaded == ["euler"]


def test_discovery_is_explicit_and_reports_registered_providers(monkeypatch):
    entry_point = _EntryPoint("euler", "euler_unisim.plugin:provider", _euler_registration)
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: [entry_point])

    assert discover_backends() == (_euler_registration(),)


def test_duplicate_entry_points_fail_closed(monkeypatch):
    entries = [
        _EntryPoint("euler", "one:provider", _euler_registration),
        _EntryPoint("euler", "two:provider", _euler_registration),
    ]
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: entries)

    with pytest.raises(BackendRegistrationError, match="multiple UniSim providers"):
        create_backend("euler")


def test_invalid_registration_and_wrong_backend_identity_fail_closed():
    with pytest.raises(BackendRegistrationError, match="reserved"):
        BackendRegistration("mujoco", _euler_factory, frozenset())
    with pytest.raises(BackendRegistrationError, match="API version"):
        BackendRegistration("euler", _euler_factory, frozenset(), api_version=2)

    def wrong_factory(scene, num_envs, sim_dt, **kwargs):
        del scene, num_envs, sim_dt, kwargs
        return FakeBackend()

    register_backend(BackendRegistration("euler", wrong_factory, frozenset()))
    with pytest.raises(BackendRegistrationError, match="backend_type"):
        create_backend("euler")


def test_unknown_backend_remains_closed_without_matching_entry_point(monkeypatch):
    monkeypatch.setattr(registry.metadata, "entry_points", lambda: [])

    with pytest.raises(ValueError, match="unknown UniSim backend"):
        create_backend("not-euler")
