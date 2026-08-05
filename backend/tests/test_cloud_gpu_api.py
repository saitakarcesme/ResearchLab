from __future__ import annotations

from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app
from backend import cloud_providers


def test_cloud_account_secret_is_encrypted_and_not_returned(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/cloud/accounts",
            json={
                "provider": "shadeform",
                "name": "Personal Shadeform",
                "api_key": "secret-test-key",
                "budget_usd": 20,
                "ssh_key_id": "ssh-key-id",
                "workspace_path": "/workspace/researchlab",
            },
        )
        assert response.status_code == 201
        account = response.json()
        assert "api_key" not in account
        assert "secret_ref" not in account
        assert account["billing_url"].startswith("https://")

        listed = client.get("/api/cloud/accounts").json()
        assert listed == [account]
        assert list((settings.data_dir / "secrets").glob("*.bin"))

        deleted = client.delete(f"/api/cloud/accounts/{account['id']}")
        assert deleted.status_code == 204
        assert not list((settings.data_dir / "secrets").glob("*.bin"))


def test_vast_marketplace_is_exposed_without_persisting_a_key(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(
        cloud_providers.VastProvider,
        "public_offers",
        classmethod(lambda cls: [{
            "id": "42", "name": "RTX 3090 · PL", "gpu": "RTX 3090", "vram_gb": 24,
            "region": "Poland, PL", "hourly_price_usd": 0.17, "available": True,
            "gpu_count": 1, "reliability": 0.998,
        }]),
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/cloud/vast-offers").json()[0]["gpu"] == "RTX 3090"
        payment = client.get("/api/cloud/payment-config").json()
        assert payment["enabled"] is False


def test_vast_account_is_accepted_and_secret_stays_private(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/cloud/accounts",
            json={
                "provider": "vast", "name": "Vast.ai", "api_key": "vast-secret-key",
                "budget_usd": 50, "workspace_path": "/workspace/researchlab",
            },
        )
        assert response.status_code == 201
        account = response.json()
        assert account["provider"] == "vast"
        assert "secret_ref" not in account
        assert account["billing_url"].startswith("https://")
