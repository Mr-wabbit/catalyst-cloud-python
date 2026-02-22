"""Catalyst Cloud API client."""

import time
import requests
from typing import Optional


class CatalystCloudError(Exception):
    """Raised when the API returns an error."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class Client:
    """Client for the Catalyst Cloud neuromorphic compute API.

    Args:
        api_key: Your API key (starts with ``cn_live_``).
        base_url: API base URL. Defaults to the production endpoint.
        timeout: Request timeout in seconds.
    """

    DEFAULT_URL = "https://api.catalyst-neuromorphic.com"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_URL,
        timeout: int = 30,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            }
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        kwargs.setdefault("timeout", self.timeout)
        resp = self._session.request(method, f"{self.base_url}{path}", **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise CatalystCloudError(resp.status_code, detail)
        return resp.json()

    # -- Signup (no auth required) --

    @classmethod
    def signup(
        cls, email: str, tier: str = "free", base_url: str = DEFAULT_URL
    ) -> dict:
        """Create a new account and get an API key.

        Args:
            email: Your email address.
            tier: Pricing tier (``free``, ``researcher``, ``startup``, ``enterprise``).
            base_url: API base URL.

        Returns:
            Dict with ``api_key``, ``tier``, ``limits``, and optional ``checkout_url``.
        """
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/signup",
            json={"email": email, "tier": tier},
            timeout=15,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise CatalystCloudError(resp.status_code, detail)
        return resp.json()

    # -- Networks --

    def create_network(
        self,
        populations: list[dict],
        connections: Optional[list[dict]] = None,
    ) -> dict:
        """Define a spiking neural network.

        Args:
            populations: List of population dicts with ``label``, ``size``,
                and optional ``params`` (e.g. ``{"threshold": 1000}``).
            connections: List of connection dicts with ``source``, ``target``,
                ``topology``, ``weight``, etc.

        Returns:
            Dict with ``network_id``, ``total_neurons``, ``populations``, ``connections``.
        """
        return self._request(
            "POST",
            "/v1/networks",
            json={
                "populations": populations,
                "connections": connections or [],
            },
        )

    # -- Jobs --

    def submit_job(
        self,
        network_id: str,
        timesteps: int,
        stimuli: Optional[list[dict]] = None,
        learning: Optional[dict] = None,
        rewards: Optional[list[dict]] = None,
    ) -> dict:
        """Submit a simulation job (non-blocking).

        Args:
            network_id: ID from :meth:`create_network`.
            timesteps: Number of simulation timesteps.
            stimuli: List of stimulus dicts with ``population`` and ``current``.
            learning: Optional learning config dict.
            rewards: Optional list of reward dicts.

        Returns:
            Dict with ``job_id`` and ``status`` (``queued``).
        """
        body = {
            "network_id": network_id,
            "timesteps": timesteps,
            "stimuli": stimuli or [],
            "rewards": rewards or [],
        }
        if learning:
            body["learning"] = learning
        return self._request("POST", "/v1/jobs", json=body)

    def get_job(self, job_id: str) -> dict:
        """Get job status and summary results.

        Returns:
            Dict with ``status``, ``result`` (firing rates, spike counts),
            ``compute_seconds``, etc.
        """
        return self._request("GET", f"/v1/jobs/{job_id}")

    def get_spikes(self, job_id: str) -> dict:
        """Get full spike train data (population-local indices).

        Returns:
            Dict with ``spike_trains`` keyed by population label,
            each containing neuron index -> list of spike times.
        """
        return self._request("GET", f"/v1/jobs/{job_id}/spikes")

    def simulate(
        self,
        network_id: str,
        timesteps: int,
        stimuli: Optional[list[dict]] = None,
        learning: Optional[dict] = None,
        rewards: Optional[list[dict]] = None,
        poll_interval: float = 0.5,
        max_wait: float = 300,
    ) -> dict:
        """Submit a job and wait for completion (blocking).

        Convenience method that calls :meth:`submit_job` then polls
        :meth:`get_job` until the job completes or fails.

        Args:
            network_id: Network ID.
            timesteps: Simulation timesteps.
            stimuli: Stimuli list.
            learning: Learning config.
            rewards: Rewards list.
            poll_interval: Seconds between status checks.
            max_wait: Maximum seconds to wait before raising TimeoutError.

        Returns:
            Completed job dict with ``result`` containing firing rates,
            spike count timeseries, etc.

        Raises:
            TimeoutError: If job doesn't complete within ``max_wait``.
            CatalystCloudError: If the job fails.
        """
        job = self.submit_job(network_id, timesteps, stimuli, learning, rewards)
        job_id = job["job_id"]

        start = time.monotonic()
        while True:
            result = self.get_job(job_id)
            status = result["status"]

            if status == "completed":
                return result
            if status == "failed":
                raise CatalystCloudError(500, result.get("error_message", "Job failed"))

            if time.monotonic() - start > max_wait:
                raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")

            time.sleep(poll_interval)

    # -- Usage --

    def usage(self) -> dict:
        """Get usage statistics for the current billing period.

        Returns:
            Dict with ``jobs_today``, ``compute_seconds_today``,
            ``estimated_cost``, etc.
        """
        return self._request("GET", "/v1/usage")
