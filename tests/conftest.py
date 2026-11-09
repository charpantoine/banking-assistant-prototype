"""Shared test setup. The .env file is loaded by config.py, which bank_api imports."""
import importlib

import pytest

import bank_api


@pytest.fixture(autouse=True)
def fresh_bank():
    """Reload the mocked bank so each test starts from the JSON data."""
    importlib.reload(bank_api)
