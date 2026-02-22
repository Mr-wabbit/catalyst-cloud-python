"""Tests for the Catalyst Cloud client."""

import pytest
import requests_mock

from catalyst_cloud import Client, CatalystCloudError


FAKE_KEY = "cn_live_test_key_1234567890"
BASE = "https://test.catalyst-neuromorphic.com"


class TestClientInit:
    def test_headers_set(self):
        c = Client(FAKE_KEY, base_url=BASE)
        assert c._session.headers["X-API-Key"] == FAKE_KEY
        assert c._session.headers["Content-Type"] == "application/json"

    def test_base_url_trailing_slash_stripped(self):
        c = Client(FAKE_KEY, base_url=BASE + "/")
        assert c.base_url == BASE

    def test_default_timeout(self):
        c = Client(FAKE_KEY, base_url=BASE)
        assert c.timeout == 30


class TestSignup:
    def test_signup_success(self):
        with requests_mock.Mocker() as m:
            m.post(f"{BASE}/v1/signup", json={
                "api_key": "cn_live_new_key",
                "tier": "free",
                "limits": {"jobs_per_day": 10},
            })
            result = Client.signup("test@example.com", base_url=BASE)
            assert result["api_key"] == "cn_live_new_key"
            assert result["tier"] == "free"

    def test_signup_error(self):
        with requests_mock.Mocker() as m:
            m.post(f"{BASE}/v1/signup", status_code=400, json={
                "detail": "Email already registered",
            })
            with pytest.raises(CatalystCloudError) as exc:
                Client.signup("duplicate@example.com", base_url=BASE)
            assert exc.value.status_code == 400
            assert "already registered" in exc.value.detail


class TestCreateNetwork:
    def test_create_network(self):
        with requests_mock.Mocker() as m:
            m.post(f"{BASE}/v1/networks", json={
                "network_id": "net_123",
                "total_neurons": 150,
                "populations": [],
                "connections": [],
            })
            c = Client(FAKE_KEY, base_url=BASE)
            result = c.create_network(
                populations=[{"label": "input", "size": 100}],
                connections=[{"source": "input", "target": "output", "weight": 500}],
            )
            assert result["network_id"] == "net_123"
            assert result["total_neurons"] == 150


class TestJobs:
    def test_submit_job(self):
        with requests_mock.Mocker() as m:
            m.post(f"{BASE}/v1/jobs", json={
                "job_id": "job_456",
                "status": "queued",
            })
            c = Client(FAKE_KEY, base_url=BASE)
            result = c.submit_job("net_123", timesteps=1000)
            assert result["job_id"] == "job_456"
            assert result["status"] == "queued"

    def test_get_job(self):
        with requests_mock.Mocker() as m:
            m.get(f"{BASE}/v1/jobs/job_456", json={
                "status": "completed",
                "result": {"total_spikes": 42},
            })
            c = Client(FAKE_KEY, base_url=BASE)
            result = c.get_job("job_456")
            assert result["status"] == "completed"

    def test_get_spikes(self):
        with requests_mock.Mocker() as m:
            m.get(f"{BASE}/v1/jobs/job_456/spikes", json={
                "spike_trains": {"input": {0: [10, 20]}},
            })
            c = Client(FAKE_KEY, base_url=BASE)
            result = c.get_spikes("job_456")
            assert "input" in result["spike_trains"]


class TestSimulate:
    def test_simulate_blocking(self):
        with requests_mock.Mocker() as m:
            m.post(f"{BASE}/v1/jobs", json={
                "job_id": "job_789",
                "status": "queued",
            })
            m.get(f"{BASE}/v1/jobs/job_789", [
                {"json": {"status": "running"}},
                {"json": {"status": "completed", "result": {"total_spikes": 100}}},
            ])
            c = Client(FAKE_KEY, base_url=BASE)
            result = c.simulate("net_123", timesteps=500, poll_interval=0.01)
            assert result["status"] == "completed"
            assert result["result"]["total_spikes"] == 100

    def test_simulate_job_failed(self):
        with requests_mock.Mocker() as m:
            m.post(f"{BASE}/v1/jobs", json={
                "job_id": "job_fail",
                "status": "queued",
            })
            m.get(f"{BASE}/v1/jobs/job_fail", json={
                "status": "failed",
                "error_message": "Out of memory",
            })
            c = Client(FAKE_KEY, base_url=BASE)
            with pytest.raises(CatalystCloudError) as exc:
                c.simulate("net_123", timesteps=500, poll_interval=0.01)
            assert "Out of memory" in str(exc.value)


class TestUsage:
    def test_usage(self):
        with requests_mock.Mocker() as m:
            m.get(f"{BASE}/v1/usage", json={
                "jobs_today": 3,
                "compute_seconds_today": 12.5,
            })
            c = Client(FAKE_KEY, base_url=BASE)
            result = c.usage()
            assert result["jobs_today"] == 3


class TestErrorHandling:
    def test_500_error(self):
        with requests_mock.Mocker() as m:
            m.get(f"{BASE}/v1/usage", status_code=500, text="Internal Server Error")
            c = Client(FAKE_KEY, base_url=BASE)
            with pytest.raises(CatalystCloudError) as exc:
                c.usage()
            assert exc.value.status_code == 500

    def test_401_error(self):
        with requests_mock.Mocker() as m:
            m.get(f"{BASE}/v1/usage", status_code=401, json={
                "detail": "Invalid API key",
            })
            c = Client(FAKE_KEY, base_url=BASE)
            with pytest.raises(CatalystCloudError) as exc:
                c.usage()
            assert "Invalid API key" in exc.value.detail
