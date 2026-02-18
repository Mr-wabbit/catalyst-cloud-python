# catalyst-cloud

Python client for the [Catalyst Cloud](https://catalyst-neuromorphic.com/cloud) neuromorphic compute API.

Run spiking neural network simulations in the cloud. No hardware, no SDK install, no setup.

## Install

```bash
pip install catalyst-cloud
```

## Quick start

```python
import catalyst_cloud as cc

# 1. Sign up (once)
account = cc.Client.signup("you@lab.edu")
print(account["api_key"])  # Save this

# 2. Create a client
client = cc.Client("cn_live_...")

# 3. Define a network
net = client.create_network(
    populations=[
        {"label": "input", "size": 100, "params": {"threshold": 1000}},
        {"label": "hidden", "size": 50},
    ],
    connections=[
        {"source": "input", "target": "hidden", "topology": "random_sparse",
         "weight": 500, "p": 0.3},
    ],
)

# 4. Run simulation (blocking)
job = client.simulate(
    net["network_id"],
    timesteps=1000,
    stimuli=[{"population": "input", "current": 5000}],
)

print(f"Total spikes: {job['result']['total_spikes']}")
print(f"Firing rates: {job['result']['firing_rates']}")

# 5. Get full spike trains
spikes = client.get_spikes(job["job_id"])
for pop, trains in spikes["spike_trains"].items():
    print(f"{pop}: {len(trains)} neurons fired")
```

## Features

- **Hardware-accurate**: Full Loihi 2 parity — LIF neurons, dendritic compartments, STDP, 3-factor learning
- **5 topologies**: all-to-all, one-to-one, random sparse, fixed fan-in, fixed fan-out
- **Simple**: JSON in, spikes out. No boilerplate, no dependencies beyond `requests`
- **Fast**: 1,000 neurons x 1,000 timesteps in under a second

## API reference

### `Client.signup(email, tier="free")`
Create account, get API key. Class method, no auth needed.

### `Client(api_key)`
Create authenticated client.

### `client.create_network(populations, connections)`
Define a spiking neural network. Returns `network_id`.

### `client.simulate(network_id, timesteps, stimuli)`
Submit job and wait for results (blocking). Returns completed job with firing rates and spike counts.

### `client.submit_job(...)` / `client.get_job(job_id)`
Non-blocking submit + poll.

### `client.get_spikes(job_id)`
Full spike trains indexed by population label and neuron index.

### `client.usage()`
Current billing period stats.

## Pricing

| Tier | Monthly | Compute | Neurons |
|------|---------|---------|---------|
| Free | £0 | £0 | 1,024 |
| Researcher | £0 | £18/hr | 32,768 |
| Startup | £49 | £14.40/hr | 131,072 |
| Enterprise | £199 | £10.80/hr | 131,072 |

## Links

- [Cloud landing page](https://catalyst-neuromorphic.com/cloud)
- [API docs](https://catalyst-neuromorphic.com/cloud/docs)
- [Interactive API docs](https://api.catalyst-neuromorphic.com/docs)
- [Pricing](https://catalyst-neuromorphic.com/cloud/pricing)
