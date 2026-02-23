"""Shared fixtures for Corvus tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def _no_sops(monkeypatch):
    """Ensure tests never try to invoke SOPS."""
    monkeypatch.setenv("CORVUS_USE_SOPS", "false")


@pytest.fixture()
def project_root():
    """Return the project root path."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
