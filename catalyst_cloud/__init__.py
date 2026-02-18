"""Catalyst Cloud — Python client for the neuromorphic compute API.

Run spiking neural network simulations in the cloud. No hardware required.

    import catalyst_cloud as cc

    client = cc.Client("cn_live_...")
    net = client.create_network(
        populations=[
            {"label": "input", "size": 100, "params": {"threshold": 1000}},
            {"label": "output", "size": 50},
        ],
        connections=[
            {"source": "input", "target": "output", "topology": "all_to_all", "weight": 500},
        ],
    )
    job = client.simulate(net["network_id"], timesteps=1000, stimuli=[
        {"population": "input", "current": 5000},
    ])
    print(job["result"]["firing_rates"])
"""

from .client import Client, CatalystCloudError

__version__ = "0.1.0"
__all__ = ["Client", "CatalystCloudError"]
