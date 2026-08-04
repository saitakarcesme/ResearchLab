from __future__ import annotations

from typing import Any

import backend.cloud_providers as cloud


def test_shadeform_offer_and_lifecycle_paths(monkeypatch) -> None:
    calls: list[tuple[str, str, Any]] = []

    def fake_request(url: str, api_key: str, *, method: str = "GET", body=None, headers=None):
        calls.append((url, method, body))
        if url.endswith("/instances/types?available=true"):
            return {"instance_types": [{"cloud": "example", "shade_instance_type": "A6000", "hourly_price": 0.4, "configuration": {"gpu_type": "A6000", "vram_per_gpu_in_gb": 48}, "availability": [{"region": "eu-1", "available": True}]}]}
        if url.endswith("/instances/create"):
            return {"id": "instance-1"}
        return {"id": "instance-1", "status": "active", "ip": "203.0.113.4", "ssh_user": "shadeform", "ssh_port": 22}

    monkeypatch.setattr(cloud, "_request", fake_request)
    client = cloud.ShadeformProvider("key")
    offer = client.offers()[0]
    assert offer["vram_gb"] == 48
    created = client.create(offer["id"], name="Lab", ssh_key_id="ssh-1", max_spend_usd=2)
    assert created["external_id"] == "instance-1"
    assert calls[-1][0].endswith("/instances/create")
    assert calls[-1][2]["auto_delete"] == {"spend_threshold": "2"}


def test_runpod_live_offer_query(monkeypatch) -> None:
    def fake_request(url: str, api_key: str, *, method: str = "GET", body=None, headers=None):
        assert "graphql" in url
        return {"data": {"gpuTypes": [{"id": "NVIDIA GeForce RTX 3090", "displayName": "RTX 3090", "memoryInGb": 24, "lowestPrice": {"stockStatus": "High", "uninterruptablePrice": 0.3}}]}}

    monkeypatch.setattr(cloud, "_request", fake_request)
    assert cloud.RunPodProvider("key").offers()[0]["hourly_price_usd"] == 0.3
