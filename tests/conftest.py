"""Shared fixtures for catalyst-cloud tests."""

import pytest


FAKE_API_KEY = "cn_live_test_key_1234567890"
FAKE_BASE_URL = "https://test.catalyst-neuromorphic.com"


@pytest.fixture
def api_key():
    return FAKE_API_KEY


@pytest.fixture
def base_url():
    return FAKE_BASE_URL
